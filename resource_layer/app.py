"""Resource Layer process (:8300) - NR server + digital twin.

Serves NR-01..NR-13 under /nr/v1 and publishes nr.notify.* (TelemetryNotify, ResourceAlarmNotify,
ComputeJobStatusNotify) to the Network Layer. Only callers holding the NR service token are accepted.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from common import config
from common.bus import make_bus
from common.envelope import Envelope, TraceBook, check, ep, iso, make, now, respond
from common.nr_messages import NR_BODY_MODELS, rname, rtype
from common.refpoint import ServerSide, check_service_token, new_id, read_envelope
from common.events import get_hub, install_events
from common.tracing import MessageLog

from .comm_twin import RanCoreTwin
from .pools import ComputePool, Job, StoragePool
from .tool_host import ToolHost
from .ue_edge_sim import UeEdgeSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)   # 1 Hz telemetry notifies would flood the log
log = logging.getLogger("resource_layer")
SELF = ep("resource", "ResourceLayer", "res-1")
NET = ep("network", "PerformanceMonitor")


class ResourceLayer:
    def __init__(self):
        self.tracer = MessageLog("resource")
        self.server = ServerSide(self.tracer, "NR")
        self.bus = make_bus("resource", {"nr.notify.*": config.NETWORK_URL}, tracer=self.tracer)
        self.twin = RanCoreTwin(lambda cells: UeEdgeSimulator(config.CLIENT_CSV, cells, seed=config.RANDOM_SEED),
                                config.BS_CSV, seed=config.RANDOM_SEED)
        self.compute = ComputePool()
        self.compute.listener = self._job_event
        self.storage = StoragePool()
        self.tools = ToolHost()
        self.reservations: dict[str, dict[str, Any]] = {}
        self.traces = TraceBook()          # correlationId -> trace of the network's requests for that flow
        self.subscriptions: dict[str, dict[str, Any]] = {}
        self.seq = 0
        self.alarm_last: dict[str, float] = {}
        self.active_alarms: dict[str, str] = {}
        self._tasks: list[asyncio.Task] = []

    # ---- background loops ---------------------------------------------------------
    async def start(self) -> None:
        self._tasks.append(asyncio.create_task(self._tick_loop()))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await self.bus.close()

    async def _tick_loop(self) -> None:
        while True:
            t0 = time.time()
            try:
                self.twin.step()
                await self._expire_reservations()
                await self._telemetry()
                await self._alarms()
            except Exception:
                log.exception("tick failed")
            await asyncio.sleep(max(0.05, config.TELEMETRY_PERIOD_SEC - (time.time() - t0)))

    async def _publish(self, msg_type: str, corr: str, body: dict, dest=NET, priority: int = 3) -> None:
        env = make(msg_type, "NR", corr, body, SELF, dest, priority=priority, trace=self.traces.get(corr))
        await self.bus.publish(f"nr.notify.{msg_type}", env)

    async def _telemetry(self) -> None:
        nowt = time.time()
        scen = self.twin.ue.scenario.flags()
        for sub in list(self.subscriptions.values()):
            if nowt - sub["_last"] < sub["periodSec"] - 1e-3:
                continue
            sub["_last"] = nowt
            scope = sub.get("scope") or {}
            for rid in sub["resources"]:
                if rtype(rid) == "comm" and rname(rid) in self.twin.cells:
                    cell = rname(rid)
                    iid = scope.get("intentId")
                    if iid and iid not in self.twin.cells[cell].bindings:
                        iid = None
                    values = self.twin.sample(cell, intent_id=iid, slice_type=scope.get("sliceType"))
                    kp = sub.get("kpis")
                    if kp:
                        values = {k: v for k, v in values.items() if k in kp or k == "load"}
                    feats = self.twin.features(cell)
                elif rtype(rid) == "comp" and rname(rid) in self.compute.nodes:
                    n = self.compute.nodes[rname(rid)]
                    values, feats = {"cpu_load": n.used_cpu / n.cpu, "gpu_load": (n.used_gpu / n.gpu) if n.gpu else 0.0}, None
                else:
                    continue
                self.seq += 1
                body = {"subscriptionId": sub["subscriptionId"], "resourceId": rid, "intentId": scope.get("intentId"),
                        "sampleTime": iso(), "seq": self.seq, "values": values, "scenario": scen}
                if feats:
                    body["features"] = feats
                await self._publish("TelemetryNotify", scope.get("intentId") or sub["subscriptionId"], body)

    async def _alarm(self, rid: str, severity: str, typ: str, details: dict, intents: list[str]) -> None:
        key = f"{rid}:{typ}"
        if severity != "CLEARED" and time.time() - self.alarm_last.get(key, 0) < 10:
            return
        log.info("NR ResourceAlarmNotify %s %s %s intents=%s %s", severity, typ, rid, intents, details or "")
        self.alarm_last[key] = time.time()
        if severity == "CLEARED":
            self.active_alarms.pop(key, None)
        else:
            self.active_alarms[key] = severity
        aid = new_id("alm")
        await self._publish("ResourceAlarmNotify", intents[0] if intents else aid,
                            {"alarmId": aid, "resourceId": rid, "severity": severity, "type": typ, "details": details,
                             "affectedIntents": intents, "raisedAt": iso()}, priority=2)

    async def _alarms(self) -> None:
        for c in self.twin.cells.values():
            rid = f"comm:{c.cell_id}"
            intents = list(c.bindings)
            if not c.up:
                await self._alarm(rid, "CRITICAL", "CELL_DOWN", {}, intents)
                continue
            if f"{rid}:CELL_DOWN" in self.active_alarms:
                await self._alarm(rid, "CLEARED", "CELL_DOWN", {}, intents)
            prb = c.last.get("prb_utilization", 0.0)
            if prb > 0.9:
                await self._alarm(rid, "MAJOR", "THRESHOLD_CROSSED",
                                  {"kpi": "prb_utilization", "value": round(prb, 3), "threshold": 0.9}, intents)
            for iid, k in c.last.get("intents", {}).items():
                if k.get("load", 0) > 1.0:
                    await self._alarm(f"{rid}/{iid}", "MAJOR", "CAPACITY_EXHAUSTED",
                                      {"kpi": "load", "value": round(k["load"], 3), "threshold": 1.0}, [iid])
                    self.twin.count_violation(c.bindings[iid].slice, "load")
        for n in self.compute.nodes.values():
            if n.load > 0.9:
                await self._alarm(f"comp:{n.node_id}", "MAJOR", "NODE_OVERLOAD",
                                  {"kpi": "load", "value": round(n.load, 3), "threshold": 0.9}, [])

    async def _job_event(self, job: Job) -> None:
        await self._publish("ComputeJobStatusNotify", job.job_id,
                            {"jobId": job.job_id, "state": job.state, "node": f"comp:{job.node}",
                             "outputRef": job.output_ref, "reason": job.reason})

    # ---- reservations -------------------------------------------------------------
    def _reserved_share(self, cell: str, exclude_intent: Optional[str]) -> float:
        tot = 0.0
        for r in self.reservations.values():
            if r.get("intentId") == exclude_intent and exclude_intent:
                continue
            for it in r["items"]:
                if it["kind"] == "PRB_SHARE" and rname(it["resourceId"]) == cell:
                    held = it["amount"] / 100.0
                    tot += max(0.0, held - self.twin.intent_share(cell, r.get("intentId")))
        return tot

    def reserve(self, body: dict) -> tuple[str, Optional[str], list[dict]]:
        granted = []
        iid = body.get("intentId")
        for it in body["items"]:
            t, name = rtype(it["resourceId"]), rname(it["resourceId"])
            if t == "comm":
                if name not in self.twin.cells:
                    return "REJECTED", f"unknown resource {it['resourceId']}", []
                free = self.twin.free_share(name, exclude_intent=iid) - self._reserved_share(name, iid)
                if it["kind"] != "PRB_SHARE" or it["amount"] / 100.0 > free + 1e-9:
                    return "REJECTED", (f"INSUFFICIENT_RESOURCES: {it['resourceId']} free PRB share "
                                        f"{free*100:.1f}% < requested {it['amount']}%"), []
            elif t == "comp":
                if name == "*" or name not in self.compute.nodes:
                    name = self.compute.pick_node(it["amount"] if it["kind"] == "CPU" else 0,
                                                  it["amount"] if it["kind"] == "GPU" else 0) or ""
                if not self.compute.can_reserve(name, it["kind"], it["amount"]):
                    return "REJECTED", f"INSUFFICIENT_RESOURCES: compute {it['resourceId']}", []
                it = {**it, "resourceId": f"comp:{name}"}
            elif t == "stor":
                if self.storage.used_gb() + sum(self.storage.reserved_gb.values()) + it["amount"] > self.storage.quota_gb:
                    return "REJECTED", "INSUFFICIENT_RESOURCES: storage quota", []
            elif t not in ("tool", "ue"):
                return "INVALID", f"bad resource id {it['resourceId']}", []
            granted.append(it)
        for it in granted:
            t, name = rtype(it["resourceId"]), rname(it["resourceId"])
            if t == "comp":
                self.compute.reserve(name, it["kind"], it["amount"])
            elif t == "stor":
                self.storage.reserved_gb[body["reservationId"]] = it["amount"]
        return "SUCCESS", None, granted

    def release(self, rid: str) -> bool:
        r = self.reservations.pop(rid, None)
        if not r:
            return False
        for it in r["items"]:
            if rtype(it["resourceId"]) == "comp":
                self.compute.reserve(rname(it["resourceId"]), it["kind"], it["amount"], -1)
        self.storage.reserved_gb.pop(rid, None)
        return True

    async def _expire_reservations(self) -> None:
        t = time.time()
        for rid, r in list(self.reservations.items()):
            if not r.get("committed") and r["expiresAtTs"] < t:
                self.release(rid)
                log.info("reservation %s expired (no ConfigApply before expiresAt)", rid)


rl: ResourceLayer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rl
    rl = ResourceLayer()
    await rl.start()
    yield
    await rl.stop()


app = FastAPI(title="Resource Layer (NR server, digital twin)", lifespan=lifespan)
install_events(app, get_hub("resource"))      # GET /events (SSE) for the dashboard


async def _inbound(request: Request, expected: set[str]) -> tuple[Optional[Envelope], Optional[Response]]:
    env = await read_envelope(request)
    if isinstance(env, Response):
        return None, env
    transport = f"http {request.method} {request.url.path}"
    cached = rl.server.inbound(env, transport)
    if cached is not None:
        return None, rl.server.outbound(env, cached, transport)
    if not check_service_token(request):
        return None, rl.server.outbound(env, respond(env, "UNAUTHORIZED", reason="NR service token required"), transport)
    err = check(env, "NR", expected)
    if err:
        return None, rl.server.outbound(env, err, transport)
    rl.traces.remember(env.header.correlationId, env.header.trace)
    model = NR_BODY_MODELS.get(env.msg_type)
    if model:
        try:
            model.model_validate(env.body)
        except ValidationError as e:
            return None, rl.server.outbound(env, respond(env, "INVALID", reason=str(e.errors()[:3])), transport)
    return env, None


def _out(request: Request, env: Envelope, resp: Envelope) -> Response:
    h, b = env.header, env.body
    detail = b.get("target") or b.get("reservationId") or b.get("toolId") or b.get("kind") or ""
    log.info("NR %s from %s/%s corr=%s %s -> %s%s", h.msgType, h.source.layer, h.source.entity, h.correlationId, detail,
             resp.status, f" ({resp.result.reason})" if resp.result and resp.result.reason else "")
    return rl.server.outbound(env, resp, f"http {request.method} {request.url.path}")


# NR-01 ---------------------------------------------------------------------------
@app.get("/nr/v1/resources")
async def inventory(request: Request, type: Optional[str] = None):
    body = await request.body()
    if not body:  # allow plain GET for humans; synthesize an envelope
        if not check_service_token(request):
            return JSONResponse({"error": "UNAUTHORIZED"}, status_code=401)
        env = make("ResourceInventoryQuery", "NR", new_id("inv"), {"type": type}, ep("network", "ResourceController"), SELF)
        rl.server.inbound(env, "http GET /nr/v1/resources")
    else:
        env, early = await _inbound(request, {"ResourceInventoryQuery"})
        if early:
            return early
    t = type or env.body.get("type")
    items: list[dict] = []
    if t in (None, "comm"):
        items += rl.twin.inventory()
    if t in (None, "comp"):
        items += rl.compute.inventory()
    if t in (None, "stor"):
        items += rl.storage.inventory()
    if t in (None, "tool"):
        items += rl.tools.inventory()
    if t in (None, "ue"):
        items += rl.twin.ue.inventory()
    return _out(request, env, respond(env, "SUCCESS", {"resources": items, "count": len(items)}))


# NR-02 / NR-03 -----------------------------------------------------------------------
@app.post("/nr/v1/reservations")
async def reserve(request: Request):
    env, early = await _inbound(request, {"ResourceReserveRequest"})
    if early:
        return early
    b = env.body
    if b["reservationId"] in rl.reservations:
        rl.release(b["reservationId"])
    status, reason, granted = rl.reserve(b)
    if status != "SUCCESS":
        return _out(request, env, respond(env, status, {"reservationId": b["reservationId"]}, reason=reason))
    exp = now() + timedelta(seconds=config.RESERVATION_HOLD_SEC)
    rl.reservations[b["reservationId"]] = {"reservationId": b["reservationId"], "intentId": b.get("intentId"),
                                           "items": granted, "priority": b.get("priority", 3),
                                           "preemptible": b.get("preemptible", False),
                                           "expiresAtTs": exp.timestamp(), "committed": False}
    return _out(request, env, respond(env, "SUCCESS", {"reservationId": b["reservationId"], "granted": granted,
                                                       "expiresAt": iso(exp)}))


@app.delete("/nr/v1/reservations/{reservation_id}")
async def release(reservation_id: str, request: Request):
    env, early = await _inbound(request, {"ResourceReleaseRequest"})
    if early:
        return early
    ok = rl.release(reservation_id)
    return _out(request, env, respond(env, "SUCCESS" if ok else "REJECTED", {"reservationId": reservation_id},
                                      reason=None if ok else "unknown reservation"))


# NR-04 / NR-05 / NR-06 -----------------------------------------------------------------
@app.post("/nr/v1/configs:validate")
async def validate(request: Request):
    env, early = await _inbound(request, {"ConfigValidateRequest"})
    if early:
        return early
    res = await asyncio.to_thread(rl.twin.validate, env.body, int(env.body.get("simulateSec", 60)))
    if not res["ok"]:
        return _out(request, env, respond(env, "REJECTED", {"configId": env.body["configId"]}, reason=res["reason"]))
    return _out(request, env, respond(env, "SUCCESS", {"configId": env.body["configId"], "predicted": res["predicted"],
                                                       "sideEffects": res["sideEffects"]}))


@app.post("/nr/v1/configs")
async def apply(request: Request):
    env, early = await _inbound(request, {"ConfigApplyRequest"})
    if early:
        return early
    b = env.body
    err = rl.twin.check(b)
    if err:
        return _out(request, env, respond(env, "REJECTED", {"configId": b["configId"]}, reason=err))
    rsv = b.get("reservationId")
    if rsv:
        if rsv not in rl.reservations:
            return _out(request, env, respond(env, "REJECTED", {"configId": b["configId"]},
                                              reason=f"reservation {rsv} unknown or expired"))
        rl.reservations[rsv]["committed"] = True
        share = (b.get("config") or {}).get("intentShare")
        if share is not None:   # the binding consumes the reservation: held amount follows the applied share
            for it in rl.reservations[rsv]["items"]:
                if it["kind"] == "PRB_SHARE" and it["resourceId"] == b["target"]:
                    it["amount"] = float(share)
    out = rl.twin.apply(b)
    return _out(request, env, respond(env, "SUCCESS", {**out, "appliedAt": iso()}))


@app.post("/nr/v1/configs/{config_id}:rollback")
async def rollback(config_id: str, request: Request):
    env, early = await _inbound(request, {"ConfigRollbackRequest"})
    if early:
        return early
    ok = rl.twin.rollback(config_id)
    return _out(request, env, respond(env, "SUCCESS" if ok else "REJECTED", {"configId": config_id, "rolledBackAt": iso()},
                                      reason=None if ok else "nothing to roll back"))


# NR-07 ---------------------------------------------------------------------------------
@app.post("/nr/v1/telemetry/subscriptions")
async def subscribe(request: Request):
    env, early = await _inbound(request, {"TelemetrySubscribeRequest"})
    if early:
        return early
    b = dict(env.body)
    if b.get("cancel"):
        rl.subscriptions.pop(b["subscriptionId"], None)
        return _out(request, env, respond(env, "SUCCESS", {"subscriptionId": b["subscriptionId"], "cancelled": True}))
    b["_last"] = 0.0
    b["periodSec"] = max(config.TELEMETRY_PERIOD_SEC, float(b.get("periodSec", 1)))
    rl.subscriptions[b["subscriptionId"]] = b
    return _out(request, env, respond(env, "SUCCESS", {"subscriptionId": b["subscriptionId"]}))


# NR-10 ---------------------------------------------------------------------------------
@app.post("/nr/v1/tools/{tool_id}:invoke")
async def invoke(tool_id: str, request: Request):
    env, early = await _inbound(request, {"ToolInvokeRequest"})
    if early:
        return early
    b = env.body
    res = await rl.tools.invoke(tool_id, b.get("operation", "generate"), b.get("arguments") or {}, int(b.get("timeoutMs", 15000)))
    status = res.pop("status")
    reason = res.pop("reason", None)
    return _out(request, env, respond(env, status, {"invocationId": b["invocationId"], **res}, reason=reason))


# NR-11 ---------------------------------------------------------------------------------
@app.post("/nr/v1/compute/jobs")
async def compute_job(request: Request):
    env, early = await _inbound(request, {"ComputeJobRequest"})
    if early:
        return early
    b = env.body
    if b.get("cancel"):
        job = await rl.compute.cancel(b["jobId"])
        return _out(request, env, respond(env, "SUCCESS" if job else "REJECTED", {"jobId": b["jobId"],
                                                                                  "state": job.state if job else None}))
    job = await rl.compute.submit(b)
    if job.state == "FAILED":
        return _out(request, env, respond(env, "REJECTED", {"jobId": job.job_id}, reason=job.reason))
    return _out(request, env, respond(env, "ACCEPTED", {"jobId": job.job_id, "state": job.state, "node": f"comp:{job.node}"}))


# NR-13 ---------------------------------------------------------------------------------
def _storage_env(request: Request, bucket: str, key: str) -> Envelope:
    return make("StorageObjectRequest", "NR", request.headers.get("X-Correlation-Id", f"{bucket}/{key}"),
                {"bucket": bucket, "key": key, "contentType": request.headers.get("Content-Type", "")},
                ep("network", "Repository"), SELF, trace=_traceparent(request))


def _traceparent(request: Request):
    """W3C traceparent header (00-<traceId>-<spanId>-<flags>) -> the caller's trace context."""
    from common.envelope import Trace
    tp = (request.headers.get("traceparent") or "").split("-")
    return Trace(traceId=tp[1], spanId=tp[2]) if len(tp) == 4 else None


@app.put("/nr/v1/storage/{bucket}/{key:path}")
async def storage_put(bucket: str, key: str, request: Request):
    env = _storage_env(request, bucket, key)
    if request.headers.get("X-Msg-Id"):
        env.header.msgId = request.headers["X-Msg-Id"]
    rl.tracer.record("in", env, transport=f"http PUT {request.url.path}")
    if not check_service_token(request):
        return rl.server.outbound(env, respond(env, "UNAUTHORIZED"), "http PUT storage")
    meta = {"contentType": request.headers.get("Content-Type", "application/octet-stream"),
            "retentionHours": request.headers.get("X-Retention-Hours"), "intentId": request.headers.get("X-Intent-Id")}
    m = rl.storage.put(bucket, key, await request.body(), {k: v for k, v in meta.items() if v})
    return rl.server.outbound(env, respond(env, "SUCCESS", {"object": m}), "http PUT storage")


@app.get("/nr/v1/storage/{bucket}/{key:path}")
async def storage_get(bucket: str, key: str, request: Request):
    env = _storage_env(request, bucket, key)
    rl.tracer.record("in", env, transport=f"http GET {request.url.path}")
    if not check_service_token(request):
        return JSONResponse({"error": "UNAUTHORIZED"}, status_code=401)
    got = rl.storage.get_any(bucket, key)
    resp = respond(env, "SUCCESS" if got else "REJECTED", {"bucket": bucket, "key": key},
                   reason=None if got else "not found")
    rl.tracer.record("out", resp, transport="http GET storage")
    if not got:
        return JSONResponse(resp.dump(), status_code=404)
    data, meta = got
    return Response(data, media_type=str(meta.get("contentType") or "application/octet-stream"),
                    headers={"X-Msg-Id": resp.header.msgId, "X-Correlation-Id": env.header.correlationId,
                             "X-Object-Size": str(len(data))})


# ---- simulator control (outside the NR catalogue) ---------------------------------------
@app.post("/sim/scenario")
async def set_scenario(payload: dict[str, Any]):
    sc = rl.twin.ue.scenario
    for k in ("is_emergency", "is_special_event", "is_iot_surge"):
        if k in payload:
            setattr(sc, k, bool(payload[k]))
    if "cell_down" in payload:
        sc.cell_down = [c.split(":")[-1] for c in payload["cell_down"]]
    if "load_factor" in payload:
        sc.load_factor = float(payload["load_factor"])
    if "fail_tool" in payload:
        rl.tools.fail_next.add(payload["fail_tool"])
    return {"scenario": {**sc.flags(), "cell_down": sc.cell_down, "load_factor": sc.load_factor}}


@app.get("/sim/state")
async def sim_state():
    return {"t": rl.twin.t, "scenario": {**rl.twin.ue.scenario.flags(), "cell_down": rl.twin.ue.scenario.cell_down,
                                          "load_factor": rl.twin.ue.scenario.load_factor},
            "cells": rl.twin.inventory(), "reservations": [{k: v for k, v in r.items()} for r in rl.reservations.values()],
            "subscriptions": len(rl.subscriptions), "violations": {f"{s}/{k}": n for (s, k), n in rl.twin.violations.items()},
            "activeAlarms": rl.active_alarms}


@app.get("/trace")
async def trace(correlationId: Optional[str] = None, limit: int = 500):
    return rl.tracer.query(correlationId, limit=limit)


@app.get("/health")
async def health():
    return {"layer": "resource", "ok": True, "cells": len(rl.twin.cells)}
