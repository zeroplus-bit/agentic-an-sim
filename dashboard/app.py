"""Operations dashboard gateway (:8400) - a separate process that FOLLOWS the three layers and never sits in a flow.

    python -m uvicorn dashboard.app:app --port 8400        (run_all.py starts it as the 4th process)
    open http://127.0.0.1:8400/

In:   GET <layer>/events  (SSE, common/events.py)      every traced message, agent-bus/MCP tool row and log record
      GET <layer> state APIs every 2 s                  intents, metrics, agents, tools, conflicts, cells
Out:  GET /dash/stream    (SSE)  snapshot + live events + incident updates + state
      GET /dash/api/snapshot | events | incidents | trace/{traceId}/mermaid | logs/{layer}
      POST /dash/api/scenario | conflicts/{id}/decide | intents | review/{key}/ack   (operator actions, proxied)

Each event is mapped to one closed-loop stage (DETECT, REPORT, EXECUTE, COORDINATE, RESOLVE, or FLOW for the rest of
the signalling) and events of one intent between drift and recovery form an incident, so the dashboard shows
detect -> report -> execute -> coordinate -> resolve as one row with its timings (MTTA, MTTR) and who decided.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import Body, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from common import config
from common.tracing import to_mermaid

log = logging.getLogger("dashboard")
STATIC = Path(__file__).resolve().parent / "static"
LAYERS = {"service": config.SERVICE_URL, "network": config.NETWORK_URL, "resource": config.RESOURCE_URL}
STATE_EVERY = 2.0
RING = 20000
QUIET_SEC = 15.0

EXECUTE_MSGS = {"ConfigValidateRequest", "ConfigValidateResponse", "ConfigApplyRequest", "ConfigApplyResponse",
                "ConfigRollbackRequest", "ConfigRollbackResponse", "ResourceReserveRequest", "ResourceReserveResponse",
                "ResourceReleaseRequest", "ResourceReleaseResponse", "ResourceInventoryQuery", "ResourceInventoryResponse"}
OK_STATUS = {None, "SUCCESS", "ACCEPTED", "VALID", "OK", "FEASIBLE", "COMPLETED", "DONE"}


def _epoch(at: Optional[str]) -> float:
    if not at:
        return time.time()
    try:
        return datetime.fromisoformat(at.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


class Incident:
    def __init__(self, iid: str, intent_id: str, cause: str, at: float, cell: Optional[str]):
        self.id = iid
        self.intent_id = intent_id
        self.cause = cause
        self.cells = {cell} if cell else set()
        self.opened = at
        self.stages: dict[str, float] = {"DETECT": at}       # first time each stage was seen
        self.counts: Counter = Counter()
        self.first_action: Optional[float] = None
        self.resolved: Optional[float] = None
        self.state = "OPEN"                                  # OPEN | ESCALATED | RESOLVED | CLOSED
        self.degraded = False                                # the intent actually went DEGRADED (SLA impact)
        self.outcome: Optional[str] = None
        self.escalations = 0
        self.decisions: list[str] = []
        self.human_review = False
        self.actions: list[str] = []
        self.trace_id: Optional[str] = None
        self.events: deque = deque(maxlen=400)
        self.last = at
        self.active_at = at                                  # last detect/execute/coordinate or warning event

    def view(self) -> dict[str, Any]:
        return {"id": self.id, "intentId": self.intent_id, "cause": self.cause, "cells": sorted(self.cells),
                "openedAt": self.opened, "stages": {k: round(v - self.opened, 2) for k, v in self.stages.items()},
                "counts": dict(self.counts), "state": self.state, "escalations": self.escalations,
                "decisions": self.decisions[-5:], "humanReview": self.human_review, "actions": self.actions[-8:],
                "mtta": round(self.first_action - self.opened, 2) if self.first_action else None,
                "mttr": round(self.resolved - self.opened, 2) if self.resolved else None,
                "traceId": self.trace_id, "lastAt": self.last, "events": len(self.events),
                "degraded": self.degraded, "outcome": self.outcome}


class Gateway:
    def __init__(self):
        self.events: deque = deque(maxlen=RING)
        self.seq = 0
        self.cursor: dict[str, int] = {k: 0 for k in LAYERS}
        self.boot: dict[str, Optional[str]] = {k: None for k in LAYERS}
        self.link: dict[str, dict[str, Any]] = {k: {"up": False, "since": None, "events": 0, "error": None} for k in LAYERS}
        self.layer_counts: dict[str, dict[str, int]] = {k: {} for k in LAYERS}
        self.state: dict[str, Any] = {}
        self.incidents: dict[str, Incident] = {}
        self.open_by_intent: dict[str, str] = {}
        self.n_incidents = 0
        self.intent_trace: dict[str, str] = {}
        self.known_intents: set[str] = set()
        self.acked: set[str] = set()
        self.clients: set[asyncio.Queue] = set()
        self.http: Optional[httpx.AsyncClient] = None
        self.tasks: list[asyncio.Task] = []

    # ---- following the layers -------------------------------------------------------------------------
    async def follow(self, layer: str, url: str) -> None:
        while True:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(5, read=30)) as h:
                    async with h.stream("GET", f"{url}/events", params={"since": self.cursor[layer]}) as r:
                        r.raise_for_status()
                        self.link[layer].update(up=True, since=time.time(), error=None)
                        self.broadcast("link", {"layer": layer, **self.link[layer]})
                        kind, data = None, []
                        async for line in r.aiter_lines():
                            if line.startswith("event:"):
                                kind = line[6:].strip()
                            elif line.startswith("data:"):
                                data.append(line[5:].lstrip())
                            elif line == "":
                                if kind and data:
                                    self.on_layer_event(layer, kind, json.loads("\n".join(data)))
                                kind, data = None, []
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                err = type(e).__name__
            except asyncio.CancelledError:
                raise
            except Exception as e:  # keep following whatever happens
                log.exception("follow %s", layer)
                err = repr(e)
            if self.link[layer]["up"] or self.link[layer]["error"] != err:
                self.link[layer].update(up=False, error=err)
                self.broadcast("link", {"layer": layer, **self.link[layer]})
            await asyncio.sleep(1.5)

    def on_layer_event(self, layer: str, kind: str, d: dict[str, Any]) -> None:
        if kind == "hello":
            if self.boot[layer] != d.get("boot"):          # new process: its sequence starts again
                if self.boot[layer] is not None:
                    self.cursor[layer] = 0
                self.boot[layer] = d.get("boot")
            return
        if kind == "stats":
            self.layer_counts[layer] = d.get("counts", {})
            return
        if kind != "event":
            return
        self.cursor[layer] = max(self.cursor[layer], int(d.get("seq", 0)))
        self.link[layer]["events"] += 1
        ev = self.normalise(layer, d)
        if ev is None:
            return
        self.seq += 1
        ev["id"] = self.seq
        self.events.append(ev)
        inc = self.correlate(ev)
        self.broadcast("event", ev)
        if inc is not None:
            self.broadcast("incident", inc.view())

    # ---- event model ----------------------------------------------------------------------------------
    def normalise(self, layer: str, d: dict[str, Any]) -> Optional[dict[str, Any]]:
        kind = d.get("kind")
        b = d.get("brief") or {}
        mt = d.get("msgType") or ""
        if kind == "msg" and d.get("dir") == "in":
            return None                                  # every message is shown once, by its sender ("out")
        ev: dict[str, Any] = {"at": d.get("at"), "t": _epoch(d.get("at")), "layer": layer, "kind": kind,
                              "msgType": mt or None, "refPoint": d.get("refPoint"), "from": d.get("from"),
                              "to": d.get("to"), "status": d.get("status") or b.get("result"), "msgId": d.get("msgId"),
                              "correlationId": d.get("correlationId"), "traceId": d.get("traceId"),
                              "transport": d.get("transport"), "dir": d.get("dir"), "brief": b or None}
        corr = d.get("correlationId")
        intent = b.get("intentId") or d.get("intentId")
        if not intent and corr and (corr in self.known_intents or corr.startswith("intent-")):
            intent = corr
        if mt == "IntentSubmitRequest" and corr:
            self.known_intents.add(corr)
        ev["intentId"] = intent
        stage, sev, summary = "FLOW", "info", mt
        if kind == "log":
            lvl = d.get("level", "INFO")
            stage = d.get("stage") or "LOG"
            sev = {"ERROR": "critical", "CRITICAL": "critical", "WARNING": "warn"}.get(lvl, "info")
            summary = d.get("msg", "")
            ev.update(level=lvl, logger=d.get("logger"), cell=d.get("cell"), exc=d.get("exc"))
        elif mt == "ResourceAlarmNotify":
            cleared = b.get("severity") == "CLEARED"
            stage = "RESOLVE" if cleared else "DETECT"
            sev = "ok" if cleared else ("critical" if b.get("severity") in ("CRITICAL", "MAJOR") else "warn")
            summary = f"NR-09 {b.get('type')} {b.get('severity')} on {b.get('resourceId')}"
            ev["affected"] = b.get("affectedIntents") or []
        elif mt == "IntentStatusNotify":
            st, prev = b.get("state"), b.get("previousState")
            stage = "RESOLVE" if st == "FULFILLED" and prev == "DEGRADED" else "REPORT"
            sev = {"DEGRADED": "warn", "FAILED": "critical", "REJECTED": "critical", "FULFILLED": "ok"}.get(st, "info")
            summary = f"SN-04 {prev or '-'} → {st}" + (f": {b.get('reason')}" if b.get("reason") else "")
        elif mt == "PerformanceReportNotify":
            stage = "REPORT"
            sev = "info" if b.get("met") is not False else "warn"
            summary = "SN-07 report: " + ("SLA met" if b.get("met") else "violated " + ", ".join(b.get("violations") or []))
        elif mt == "ConflictNotify":
            stage = "COORDINATE"
            sev = "warn" if b.get("resolution") == "NEEDS_DECISION" else "info"
            summary = f"SN-14 {b.get('resolution')} on {b.get('resource') or b.get('resourceId') or ''} intents={b.get('intents')}"
            if b.get("options"):
                summary += f" options={b['options']}"
        elif mt in EXECUTE_MSGS:
            stage = "EXECUTE"
            sev = "warn" if (ev["status"] not in OK_STATUS or "Rollback" in mt) else "info"
            summary = f"{d.get('refPoint')} {mt}" + (f" [{ev['status']}]" if ev["status"] else "")
            if b.get("resultReason"):
                summary += f": {b['resultReason']}"
        elif mt.startswith("ToolCall"):
            stage, kind = "EXECUTE", "tool"
            summary = f"MCP {mt.split(' ', 1)[-1]} by {(d.get('from') or '').split('/')[-1]}"
        elif mt.startswith("ToolDenied"):
            stage, kind, sev = "EXECUTE", "tool", "critical"
            summary = f"AuthGuard denied {mt.split(' ', 1)[-1]} to {(d.get('from') or '').split('/')[-1]}"
        elif mt == "AgentStateNotify":
            summary = f"SN-09 agent {b.get('agentId')} {b.get('state')}"
        elif mt.startswith("Agent"):
            stage, kind = ("COORDINATE" if "Negotiate" in mt else "EXECUTE"), "agent"
            summary = f"{mt} {(d.get('from') or '').split('/')[-1]} → {(d.get('to') or '').split('/')[-1]}"
            if b.get("explanation"):
                summary += f": {b['explanation']}"
        elif mt:
            summary = f"{d.get('refPoint') or ''} {mt}" + (f" [{ev['status']}]" if ev["status"] else "")
            if ev["status"] and ev["status"] not in OK_STATUS:
                sev = "warn"
        ev.update(kind=kind, stage=stage, severity=sev, summary=summary)
        if intent and ev["traceId"]:
            self.intent_trace.setdefault(intent, ev["traceId"])
        if intent and not ev["traceId"]:
            ev["traceId"] = self.intent_trace.get(intent)
        return ev

    # ---- incidents ------------------------------------------------------------------------------------
    def correlate(self, ev: dict[str, Any]) -> Optional[Incident]:
        intents = [ev["intentId"]] if ev.get("intentId") else []
        if ev.get("msgType") == "ResourceAlarmNotify":
            intents = ev.get("affected") or intents
        touched = None
        for iid in intents:
            inc = self.incidents.get(self.open_by_intent.get(iid, ""))
            opens = (ev["stage"] == "DETECT" or
                     (ev.get("msgType") == "IntentStatusNotify" and (ev.get("brief") or {}).get("state") == "DEGRADED"))
            if inc is None and opens and ev["severity"] != "ok":
                self.n_incidents += 1
                inc = Incident(f"INC-{self.n_incidents:04d}", iid, ev["summary"][:160], ev["t"], ev.get("cell"))
                self.incidents[inc.id] = inc
                self.open_by_intent[iid] = inc.id
            if inc is None:
                continue
            self._attach(inc, ev)
            touched = inc
        return touched

    def _attach(self, inc: Incident, ev: dict[str, Any]) -> None:
        st, mt, b = ev["stage"], ev.get("msgType") or "", ev.get("brief") or {}
        inc.events.append(ev["id"])
        inc.counts[st] += 1
        inc.last = ev["t"]
        if st in ("DETECT", "EXECUTE", "COORDINATE") or ev["severity"] in ("warn", "critical"):
            inc.active_at = ev["t"]
        inc.trace_id = inc.trace_id or ev.get("traceId")
        if ev.get("cell"):
            inc.cells.add(ev["cell"])
        if st not in inc.stages and st != "RESOLVE":     # RESOLVE is set when the intent is FULFILLED again
            inc.stages[st] = ev["t"]
        if mt == "IntentStatusNotify" and b.get("state") == "DEGRADED":
            inc.degraded = True
        if st == "EXECUTE" and inc.first_action is None and (mt == "ConfigApplyRequest" or
                                                             (ev["kind"] == "log" and "assurance task" in ev["summary"])):
            inc.first_action = ev["t"]
        if mt == "ConfigApplyRequest":
            inc.actions.append(f"NR-05 apply {b.get('configId') or ''}".strip() + (" (steering)" if b.get("mobility") else ""))
        if mt == "ConfigRollbackRequest":
            inc.actions.append("NR-06 rollback")
        if mt == "ConflictNotify" and b.get("resolution") == "NEEDS_DECISION":
            inc.escalations += 1
            inc.state = "ESCALATED"
        if ev["kind"] == "log" and ev.get("logger") == "orchestrator":
            inc.decisions.append(ev["summary"])
            inc.human_review = inc.human_review or "human review" in ev["summary"]
        resolved = (mt == "IntentStatusNotify" and b.get("state") == "FULFILLED") or \
                   (ev["kind"] == "log" and ev["stage"] == "RESOLVE" and "recovered" in ev["summary"])
        if resolved and inc.state in ("OPEN", "ESCALATED"):
            inc.state, inc.resolved = "RESOLVED", ev["t"]
            inc.outcome = "recovered" if inc.degraded or inc.counts.get("EXECUTE") else "no SLA impact"
            inc.stages["RESOLVE"] = ev["t"]          # the step shows when the intent was FULFILLED again
            self.open_by_intent.pop(inc.intent_id, None)
        if mt == "IntentStatusNotify" and b.get("state") in ("TERMINATED", "FAILED") and inc.state != "RESOLVED":
            inc.state, inc.resolved = "CLOSED", ev["t"]
            self.open_by_intent.pop(inc.intent_id, None)

    # ---- state polling --------------------------------------------------------------------------------
    async def poll(self) -> None:
        async def get(url: str) -> Any:
            try:
                r = await self.http.get(url, timeout=3)
                return r.json() if r.status_code == 200 else None
            except (httpx.HTTPError, ValueError):
                return None

        while True:
            res = await asyncio.gather(
                get(f"{config.NETWORK_URL}/status/intents"), get(f"{config.NETWORK_URL}/metrics"),
                get(f"{config.NETWORK_URL}/status/agents"), get(f"{config.NETWORK_URL}/status/tools"),
                get(f"{config.SERVICE_URL}/api/conflicts"), get(f"{config.RESOURCE_URL}/sim/state"),
                get(f"{config.SERVICE_URL}/api/models/intent-nlu"))
            keys = ("intents", "metrics", "agents", "tools", "conflicts", "sim", "nlu")
            self.state = {k: v for k, v in zip(keys, res)}
            for it in self.state.get("intents") or []:
                self.known_intents.add(it["intentId"])
            self.expire(self.state.get("intents"))
            self.broadcast("state", self.view_state())
            await asyncio.sleep(STATE_EVERY)

    def expire(self, intents: Optional[list[dict[str, Any]]]) -> None:
        """An alarm that never took the intent out of FULFILLED (handled before any SLA impact, or none at all) is
        closed once the intent has been quiet and FULFILLED for QUIET_SEC."""
        if intents is None:
            return
        states = {i["intentId"]: i["state"] for i in intents}
        now = time.time()
        for iid, inc_id in list(self.open_by_intent.items()):
            inc = self.incidents[inc_id]
            if states.get(iid) == "FULFILLED" and not inc.degraded and now - inc.active_at > QUIET_SEC:
                inc.state, inc.resolved = "CLOSED", inc.last
                inc.outcome = "handled before SLA impact" if inc.counts.get("EXECUTE") else "no SLA impact"
                self.open_by_intent.pop(iid, None)
                self.broadcast("incident", inc.view())

    def review_queue(self) -> list[dict[str, Any]]:
        conf = self.state.get("conflicts") or {}
        out = []
        for d in conf.get("decisions") or []:
            if d.get("humanReview"):
                key = f"{d.get('conflictId') or 'status'}:{d.get('intentId')}:{int(d.get('at', 0))}"
                out.append({"key": key, "kind": "HUMAN_REVIEW", "acked": key in self.acked, **d})
        for c in conf.get("conflicts") or []:
            if c.get("resolution") == "NEEDS_DECISION" and not c.get("decided"):
                out.append({"key": c["conflictId"], "kind": "PENDING_DECISION", "acked": False, **c})
        return out[-30:]

    def kpis(self) -> dict[str, Any]:
        intents = self.state.get("intents") or []
        metrics = self.state.get("metrics") or {}
        sim = self.state.get("sim") or {}
        ratios = [r["fulfilmentRatio"] for r in metrics.get("intents", []) if r.get("fulfilmentRatio") is not None]
        incs = list(self.incidents.values())
        done = [i for i in incs if i.state == "RESOLVED"]
        mtta = [i.first_action - i.opened for i in incs if i.first_action]
        mttr = [i.resolved - i.opened for i in done]
        autonomous = [i for i in done if not i.human_review]
        queue = [q for q in self.review_queue() if not q["acked"]]
        alarms = sim.get("activeAlarms") or {}
        cells = sim.get("cells") or []
        util = [c.get("prbUtilization", 0) for c in cells if c.get("up")]
        return {"intents": len(intents), "byState": dict(Counter(i["state"] for i in intents)),
                "sla": round(sum(ratios) / len(ratios), 3) if ratios else None,
                "activeAlarms": len(alarms) if isinstance(alarms, (dict, list)) else 0,
                "cellsDown": sum(1 for c in cells if not c.get("up", True)),
                "openIncidents": sum(1 for i in incs if i.state in ("OPEN", "ESCALATED")),
                "incidents": len(incs), "resolved": len(done),
                "mtta": round(sum(mtta) / len(mtta), 2) if mtta else None,
                "mttr": round(sum(mttr) / len(mttr), 2) if mttr else None,
                "autonomy": round(len(autonomous) / len(done), 3) if done else None,
                "humanQueue": len(queue), "avgPrbUtil": round(sum(util) / len(util), 3) if util else None,
                "telemetry": sum(c.get("TelemetryNotify", 0) for c in self.layer_counts.values()) // 2}

    def view_state(self) -> dict[str, Any]:
        s = self.state
        return {"kpis": self.kpis(), "intents": s.get("intents") or [], "agents": s.get("agents") or [],
                "tools": [t for t in (s.get("tools") or []) if t.get("stats", {}).get("calls") or t.get("stats", {}).get("denied")],
                "sim": {k: (s.get("sim") or {}).get(k) for k in ("t", "scenario", "cells", "activeAlarms")},
                "assurance": (s.get("metrics") or {}).get("assurance"), "actions": (s.get("metrics") or {}).get("actions"),
                "efficiency": (s.get("metrics") or {}).get("resourceEfficiency"),
                "conflicts": (s.get("conflicts") or {}).get("conflicts", [])[-15:],
                "decisions": (s.get("conflicts") or {}).get("decisions", [])[-15:],
                "review": self.review_queue(), "links": self.link, "layerCounts": self.layer_counts,
                "nlu": {k: (s.get("nlu") or {}).get(k) for k in ("active", "shadow", "versions")} if s.get("nlu") else None,
                "fulfilment": {r["intentId"]: r.get("fulfilmentRatio") for r in (s.get("metrics") or {}).get("intents", [])},
                "now": time.time()}

    def snapshot(self, n_events: int = 400) -> dict[str, Any]:
        return {"state": self.view_state(), "incidents": [i.view() for i in list(self.incidents.values())[-50:]],
                "events": list(self.events)[-n_events:]}

    # ---- browser fan-out ------------------------------------------------------------------------------
    def broadcast(self, kind: str, data: Any) -> None:
        msg = f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
        for q in list(self.clients):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:              # slow browser: drop it, it reconnects and gets a fresh snapshot
                self.clients.discard(q)

    def query(self, intent: Optional[str] = None, trace: Optional[str] = None, layer: Optional[str] = None,
              stage: Optional[str] = None, kind: Optional[str] = None, q: Optional[str] = None,
              incident: Optional[str] = None, limit: int = 500) -> list[dict[str, Any]]:
        ids = set(self.incidents[incident].events) if incident and incident in self.incidents else None
        out = []
        for e in reversed(self.events):
            if ids is not None and e["id"] not in ids:
                continue
            if intent and e.get("intentId") != intent:
                continue
            if trace and e.get("traceId") != trace:
                continue
            if layer and e["layer"] not in layer.split(","):
                continue
            if stage and e["stage"] not in stage.split(","):
                continue
            if kind and e["kind"] not in kind.split(","):
                continue
            if q and q.lower() not in json.dumps(e, ensure_ascii=False, default=str).lower():
                continue
            out.append(e)
            if len(out) >= limit:
                break
        return out[::-1]


gw = Gateway()


@asynccontextmanager
async def lifespan(app: FastAPI):
    gw.http = httpx.AsyncClient()
    gw.tasks = [asyncio.create_task(gw.follow(k, u)) for k, u in LAYERS.items()] + [asyncio.create_task(gw.poll())]
    yield
    for t in gw.tasks:
        t.cancel()
    await gw.http.aclose()


app = FastAPI(title="AAN operations dashboard", lifespan=lifespan)


@app.get("/")
@app.get("/dash")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/dash/stream")
async def stream(request: Request):
    q: asyncio.Queue = asyncio.Queue(maxsize=5000)

    async def gen():
        gw.clients.add(q)
        try:
            yield f"event: snapshot\ndata: {json.dumps(gw.snapshot(), ensure_ascii=False, default=str)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    yield await asyncio.wait_for(q.get(), timeout=10)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                if q not in gw.clients:             # dropped as too slow
                    break
        finally:
            gw.clients.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/dash/api/snapshot")
async def snapshot(events: int = 100):
    return gw.snapshot(events)


@app.get("/dash/api/events")
async def events(intentId: Optional[str] = None, traceId: Optional[str] = None, layer: Optional[str] = None,
                 stage: Optional[str] = None, kind: Optional[str] = None, q: Optional[str] = None,
                 incident: Optional[str] = None, limit: int = 500):
    return gw.query(intentId, traceId, layer, stage, kind, q, incident, min(limit, 5000))


@app.get("/dash/api/incidents")
async def incidents():
    return [i.view() for i in gw.incidents.values()]


@app.get("/dash/api/trace/{trace_id}/mermaid", response_class=PlainTextResponse)
async def trace_mermaid(trace_id: str, incident: Optional[str] = None):
    rows = [e for e in gw.query(trace=trace_id, incident=incident, limit=5000) if e["kind"] in ("msg", "internal", "tool", "agent")]
    if not rows and incident in gw.incidents:
        rows = [e for e in gw.query(incident=incident, limit=5000) if e["kind"] in ("msg", "internal", "tool", "agent")]
    rows = [{**e, "dir": e.get("dir") or "internal", "from": e.get("from") or "?/?", "to": e.get("to") or "?/?"}
            for e in rows if e.get("msgType")][-120:]
    return to_mermaid(rows)


@app.get("/dash/api/logs/{layer}", response_class=PlainTextResponse)
async def process_log(layer: str, lines: int = 300):
    if layer not in (*LAYERS, "dashboard"):
        return PlainTextResponse("unknown layer", status_code=404)
    p = config.LOG_DIR / f"{layer}.log"
    if not p.exists():
        return ""
    with p.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 200_000))
        text = f.read().decode("utf-8", "replace")
    return "\n".join(text.splitlines()[-lines:])


@app.post("/dash/api/scenario")
async def scenario(payload: dict[str, Any] = Body(...)):
    log.warning("operator scenario injection: %s", payload)
    r = await gw.http.post(f"{config.RESOURCE_URL}/sim/scenario", json=payload, timeout=5)
    return r.json()


DEMO_INTENTS = [
    {"utterance": "Remote surgery link at Hospital A under 5 ms, 50 Mbps, 99.999%", "priority": 2,
     "cells": ["gNB-012-1", "gNB-012-2"]},
    {"utterance": "Stadium 4K live streaming for fans, 200 Mbps, latency under 50 ms", "priority": 3,
     "cells": ["gNB-013-1", "gNB-013-2"]},
    {"utterance": "Smart meters at the port, 5000 devices, 5 Mbps, latency up to 200 ms", "priority": 3,
     "cells": ["gNB-015-1", "gNB-015-2"]},
]


async def run_demo_scenario() -> None:
    async def set_resource(payload: dict[str, Any]) -> dict[str, Any]:
        r = await gw.http.post(f"{config.RESOURCE_URL}/sim/scenario", json=payload, timeout=5)
        return r.json()

    try:
        log.warning("comprehensive demo scenario started", extra={"stage": "DETECT"})
        await set_resource({"is_emergency": False, "is_special_event": False, "is_iot_surge": False,
                            "cell_down": [], "load_factor": 1.0})

        for payload in DEMO_INTENTS:
            r = await gw.http.post(f"{config.SERVICE_URL}/api/intents", json=payload, timeout=60)
            r.raise_for_status()
        await asyncio.sleep(10)

        await set_resource({"is_emergency": True, "load_factor": 1.3})
        log.warning("comprehensive demo: emergency traffic and URLLC load increase",
                    extra={"stage": "DETECT"})
        await asyncio.sleep(8)

        await set_resource({"cell_down": ["gNB-012-1"], "load_factor": 1.6})
        log.warning("comprehensive demo: gNB-012-1 cell outage injected",
                    extra={"stage": "DETECT", "cell": "gNB-012-1"})
        await asyncio.sleep(30)

        await set_resource({"is_emergency": False, "is_special_event": False, "is_iot_surge": False,
                            "cell_down": [], "load_factor": 1.0})
        log.info("comprehensive demo: scenario cleared and recovery phase started",
                 extra={"stage": "RESOLVE"})
    except Exception:
        log.exception("comprehensive demo scenario failed")


@app.post("/dash/api/scenario/demo")
async def demo_scenario():
    if getattr(gw, "demo_task", None) and not gw.demo_task.done():
        return {"started": False, "running": True, "message": "종합 시나리오가 이미 실행 중입니다."}
    gw.demo_task = asyncio.create_task(run_demo_scenario())
    return {"started": True, "running": True,
            "message": "종합 시나리오 시작: 기본 인텐트 → 응급 부하 → 셀 장애 → 부하 상승 → 복구"}


@app.post("/dash/api/intents")
async def submit(payload: dict[str, Any] = Body(...)):
    r = await gw.http.post(f"{config.SERVICE_URL}/api/intents", json=payload, timeout=60)
    return r.json()


@app.post("/dash/api/conflicts/{conflict_id}/decide")
async def decide(conflict_id: str, payload: dict[str, Any] = Body(...)):
    r = await gw.http.post(f"{config.SERVICE_URL}/api/conflicts/{conflict_id}:decide", json=payload, timeout=30)
    return r.json()


@app.post("/dash/api/review/{key:path}/ack")
async def ack(key: str):
    gw.acked.add(key)
    gw.broadcast("state", gw.view_state())
    return {"acked": key}


@app.get("/health")
async def health():
    return {"layer": "dashboard", "ok": True, "links": {k: v["up"] for k, v in gw.link.items()},
            "events": len(gw.events), "incidents": len(gw.incidents), "clients": len(gw.clients)}
