"""Service Layer process (:8100) - SN client, user/operator REST API (/api), receives sn.notify.*.

Operators (or an LLM agent through mcp_servers/sn_mcp_server.py) use /api; everything that reaches the
network goes over the SN reference point.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from common import config
from common.bus import InProcBus
from common.envelope import Envelope, TraceBook, ep
from common.refpoint import RefPointClient, check_bus_token, new_id
from common.sn_messages import SN_CATALOGUE
from common.nr_messages import NR_CATALOGUE
from common.events import get_hub, install_events
from common.tracing import MessageLog, to_mermaid

from .lcm.intent_nlu import IntentNluLifecycle
from .lcm.managers import AgentLifecycleManager, DataManager, ModelLifecycleManager, ResourceManager
from .orchestration.connectivity import ConnectivityManager
from .orchestration.identity import IdentityManager
from .orchestration.intent_processor import IntentProcessor
from .orchestration.monitor import ServiceMonitor
from .orchestration.orchestrator_agent import ServiceOrchestratorAgent
from .orchestration.provisioner import ServiceProvisioner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("service_layer")


class ServiceLayer:
    def __init__(self):
        self.tracer = MessageLog("service")
        self.bus = InProcBus("service", tracer=self.tracer)
        self.sn = RefPointClient(config.NETWORK_URL, "SN", ep("service", "ServiceLayer", "svc-1"), self.tracer)
        self.identity = IdentityManager()
        self.connectivity = ConnectivityManager()
        self.intents = IntentProcessor()
        self.traces = TraceBook()          # intentId -> trace of the intent's flow
        self.intent_lcm = IntentNluLifecycle(self.intents)
        self.provisioner = ServiceProvisioner(self)
        self.monitor = ServiceMonitor(self)
        self.orchestrator = ServiceOrchestratorAgent(self)
        self.data = DataManager(self)
        self.resources = ResourceManager(self)
        self.agents = AgentLifecycleManager(self)
        self.models = ModelLifecycleManager(self)
        self.http = httpx.AsyncClient(timeout=30)
        self.onboarded: list[str] = []

    async def start(self) -> None:
        self.bus.subscribe("sn.notify.IntentStatusNotify", self.monitor.on_status)
        self.bus.subscribe("sn.notify.PerformanceReportNotify", self.monitor.on_report)
        self.bus.subscribe("sn.notify.AgentStateNotify", self.agents.on_state)
        self.bus.subscribe("sn.notify.ModelPerformanceNotify", self.models.on_performance)
        self.bus.subscribe("sn.notify.ConflictNotify", self.orchestrator.on_conflict)
        asyncio.create_task(self._bootstrap())

    async def _bootstrap(self) -> None:
        """Flow 2 at start-up: onboard the default agents once the Network Layer answers."""
        for _ in range(120):
            try:
                r = await self.http.get(f"{config.NETWORK_URL}/health")
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
        self.onboarded = await self.agents.onboard_defaults()
        log.info("default agents: %s", self.onboarded)

    def new_intent_id(self) -> str:
        return new_id("intent")


sl: ServiceLayer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sl
    sl = ServiceLayer()
    await sl.start()
    yield
    await sl.sn.close()
    await sl.http.aclose()


app = FastAPI(title="Service Layer (SN client, operator API)", lifespan=lifespan)
install_events(app, get_hub("service"))      # GET /events (SSE) for the dashboard


def _result(env: Envelope) -> dict[str, Any]:
    return {"status": env.status, "reason": env.result.reason if env.result else None,
            **{k: v for k, v in env.body.items() if k != "result"}}


def _build_intent(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    tenant = payload.get("tenantId", "tenant-a")
    iid = payload.get("intentId") or sl.new_intent_id()
    if payload.get("intent"):
        intent = {**payload["intent"], "intentId": iid, "tenantId": tenant}
        prio = int(payload.get("priority") or 3)
    else:
        intent, prio = sl.intents.parse(payload["utterance"], iid, tenant)
    devices = intent.pop("_devices", None)
    if not payload.get("intent") and iid != "intent-preview":
        sl.intent_lcm.observe(iid, payload["utterance"], intent)
    if payload.get("cells"):
        intent.setdefault("context", {})["area"] = {"type": "cellList", "cells": payload["cells"]}
    if payload.get("expectations"):
        intent["expectations"] = payload["expectations"]
    sl.connectivity.attach(intent, intent.get("utterance", ""), devices)
    if payload.get("priority"):
        prio = int(payload["priority"])
    return intent, prio


# ---- intents ------------------------------------------------------------------------------------
@app.post("/api/intents:parse")
async def parse_intent(payload: dict[str, Any] = Body(...)):
    intent, prio = _build_intent({**payload, "intentId": payload.get("intentId") or "intent-preview"})
    return {"intent": intent, "priority": prio}


@app.post("/api/intents:feasibility")
async def feasibility(payload: dict[str, Any] = Body(...)):
    intent, prio = _build_intent({**payload, "intentId": payload.get("intentId") or new_id("feas")})
    r = await sl.provisioner.feasibility(intent, prio)
    return {"intent": intent, "priority": prio, "feasibility": _result(r)}


@app.post("/api/intents")
async def submit_intent(payload: dict[str, Any] = Body(...)):
    """{utterance, tenantId?, priority?, cells?, expectations?, checkFeasibility?}  or  {intent: {...}}"""
    intent, prio = _build_intent(payload)
    out: dict[str, Any] = {"intent": intent, "priority": prio}
    val = intent.get("semantic", {}).get("validation") or {}
    if val.get("blocking") and not payload.get("force"):
        # stated KPIs are invalid or contradict the slice type: ask instead of provisioning
        out.update({"submitted": False, "status": "CLARIFICATION_REQUIRED",
                    "reason": "; ".join(v["message"] for v in val["violations"] if v["severity"] in ("error", "conflict")),
                    "hint": "rephrase, pass an explicit intent, or resubmit with force=true"})
        return out
    if payload.get("checkFeasibility"):
        f = await sl.provisioner.feasibility(intent, prio)
        out["feasibility"] = _result(f)
        if not f.body.get("feasible"):
            out["submitted"] = False
            return out
    r = await sl.provisioner.submit(intent, prio)
    out["submitted"] = True
    out["response"] = _result(r)
    return out


@app.get("/api/intents")
async def list_intents():
    return [{**x, "fulfilment": sl.monitor.fulfilment(x["intentId"])} for x in sl.provisioner.view()]


@app.get("/api/intents/{intent_id}")
async def get_intent(intent_id: str, network: bool = True):
    v = sl.provisioner.view(intent_id)
    if not v:
        return JSONResponse({"error": "unknown intent"}, status_code=404)
    v["fulfilmentRatio"] = sl.monitor.fulfilment(intent_id)
    v["latestReports"] = sl.monitor.latest(intent_id, 3)
    if network:   # observability view of the Network Layer (not an SN message)
        try:
            r = await sl.http.get(f"{config.NETWORK_URL}/status/intents/{intent_id}")
            v["networkView"] = r.json() if r.status_code == 200 else None
        except httpx.HTTPError:
            v["networkView"] = None
    return v


@app.patch("/api/intents/{intent_id}")
async def modify_intent(intent_id: str, payload: dict[str, Any] = Body(...)):
    if intent_id not in sl.provisioner.instances:
        return JSONResponse({"error": "unknown intent"}, status_code=404)
    changes = {k: v for k, v in payload.items() if k in ("expectations", "priority", "constraints", "reportPolicy", "context")}
    if payload.get("cells"):
        changes["context"] = {"area": {"type": "cellList", "cells": payload["cells"]}}
    r = await sl.provisioner.modify(intent_id, changes)
    return _result(r)


@app.delete("/api/intents/{intent_id}")
async def cancel_intent(intent_id: str, reason: Optional[str] = None):
    if intent_id not in sl.provisioner.instances:
        return JSONResponse({"error": "unknown intent"}, status_code=404)
    return _result(await sl.provisioner.cancel(intent_id, reason))


@app.get("/api/intents/{intent_id}/reports")
async def reports(intent_id: str, n: int = 20):
    return {"fulfilment": sl.monitor.fulfilment(intent_id), "reports": sl.monitor.latest(intent_id, n)}


@app.post("/api/intents/{intent_id}/subscriptions")
async def subscribe(intent_id: str, payload: dict[str, Any] = Body(default={})):
    return _result(await sl.monitor.subscribe(intent_id, float(payload.get("periodSec", 10)), payload.get("kpis")))


# ---- agents / models / policies ------------------------------------------------------------------------
@app.get("/api/agents")
async def agents():
    try:
        net = (await sl.http.get(f"{config.NETWORK_URL}/status/agents")).json()
    except (httpx.HTTPError, ValueError):
        net = None
    return {"states": sl.agents.states, "descriptors": sl.agents.descriptors, "networkView": net}


@app.post("/api/agents")
async def deploy_agent(payload: dict[str, Any] = Body(...)):
    r, _ = await sl.agents.deploy(payload["agentDescriptor"] if "agentDescriptor" in payload else payload,
                                  payload.get("action", "INSTANTIATE"))
    return _result(r)


@app.post("/api/agents:external")
async def onboard_external(payload: dict[str, Any] = Body(...)):
    return await sl.agents.onboard_external(payload["agentId"], payload.get("skills", []), payload.get("permissions"),
                                            payload.get("tools"))


@app.delete("/api/agents/{agent_id}")
async def terminate_agent(agent_id: str):
    return _result(await sl.agents.terminate(agent_id))


@app.get("/api/models")
async def models():
    try:
        net = (await sl.http.get(f"{config.NETWORK_URL}/status/models")).json()
    except (httpx.HTTPError, ValueError):
        net = None
    return {**sl.models.view(), "intentNlu": sl.intent_lcm.view(), "networkView": net}


@app.get("/api/models/intent-nlu")
async def intent_nlu_status():
    return sl.intent_lcm.view()


@app.post("/api/models/intent-nlu/shadow")
async def intent_nlu_shadow(payload: dict[str, Any] = Body(...)):
    return sl.intent_lcm.start_shadow(str(payload["version"]), bool(payload.get("autoPromote", False)))


@app.post("/api/models/intent-nlu/promote")
async def intent_nlu_promote(payload: dict[str, Any] = Body(...)):
    return sl.intent_lcm.promote(str(payload["version"]), payload.get("reason", "operator"))


@app.post("/api/models/intent-nlu/rollback")
async def intent_nlu_rollback():
    return sl.intent_lcm.rollback()


@app.post("/api/intents/{intent_id}/feedback")
async def intent_feedback(intent_id: str, payload: dict[str, Any] = Body(...)):
    """Operator-confirmed interpretation: {slice, expectations?: {kpi: value}, devices?}"""
    return sl.intent_lcm.feedback(intent_id, payload["slice"], payload.get("expectations"), payload.get("devices"),
                                  payload.get("operator", "operator"))


@app.post("/api/models/deployments")
async def deploy_model(payload: dict[str, Any] = Body(...)):
    r = await sl.models.deploy(payload.get("name", "lstm_single"), payload["version"],
                               float(payload.get("trafficShare", 0.2)), payload.get("rollbackOn"),
                               bool(payload.get("autoPromote", False)))
    return _result(r)


@app.put("/api/policies/resource/{tenant_id}")
async def resource_policy(tenant_id: str, payload: dict[str, Any] = Body(...)):
    return _result(await sl.resources.set_policy(tenant_id, payload))


@app.put("/api/policies/data/{policy_id}")
async def data_policy(policy_id: str, payload: dict[str, Any] = Body(...)):
    return _result(await sl.data.set_policy(policy_id, payload))


@app.get("/api/offers")
async def offers():
    return sl.resources.offers


@app.get("/api/conflicts")
async def conflicts():
    return sl.orchestrator.view()


@app.post("/api/conflicts/{conflict_id}:decide")
async def decide(conflict_id: str, payload: dict[str, Any] = Body(...)):
    return await sl.orchestrator.decide(conflict_id, payload["action"], payload.get("intentId"))


@app.get("/api/areas")
async def areas():
    return sl.connectivity.view()


@app.get("/api/catalogue")
async def catalogue():
    return {"SN": SN_CATALOGUE, "NR": NR_CATALOGUE}


# ---- evaluation & traces --------------------------------------------------------------------------------
@app.get("/api/metrics")
async def metrics():
    try:
        net = (await sl.http.get(f"{config.NETWORK_URL}/metrics")).json()
    except (httpx.HTTPError, ValueError):
        net = {}
    rows = []
    for x in sl.provisioner.view():
        n = next((r for r in net.get("intents", []) if r["intentId"] == x["intentId"]), {})
        f = sl.monitor.fulfilment(x["intentId"])
        rows.append({"intentId": x["intentId"], "state": x["state"], "sliceType": x["sliceType"],
                     "intentFulfilmentRatio": f["ratio"], "reportWindows": f["windows"],
                     "timeToFulfilSec": x["timeToFulfilSec"], "timeToRecoverSec": n.get("recoverTimesSec"),
                     "signalling": n.get("signalling")})
    return {"intents": rows, "resourceEfficiency": net.get("resourceEfficiency"), "conflicts": net.get("conflicts"),
            "signallingTotals": {"service": dict(sl.tracer.totals), "network": net.get("signallingTotals")},
            "memory": net.get("memory"), "actions": net.get("actions")}


@app.get("/api/trace/{correlation_id}")
async def trace(correlation_id: str, format: str = "json"):
    rows = sl.tracer.query(correlation_id, limit=5000)
    for url in (config.NETWORK_URL, config.RESOURCE_URL):
        try:
            rows += (await sl.http.get(f"{url}/trace", params={"correlationId": correlation_id, "limit": 5000})).json()
        except httpx.HTTPError:
            pass
    rows.sort(key=lambda r: r.get("at", ""))
    if format == "mermaid":
        return PlainTextResponse(to_mermaid(rows))
    return rows


# ---- simulation control (outside SN/NR: drives the digital twin's scenarios) ---------------------------
@app.post("/api/sim/scenario")
async def scenario(payload: dict[str, Any] = Body(...)):
    return (await sl.http.post(f"{config.RESOURCE_URL}/sim/scenario", json=payload)).json()


@app.get("/api/sim/state")
async def sim_state():
    return (await sl.http.get(f"{config.RESOURCE_URL}/sim/state")).json()


# ---- notifications ---------------------------------------------------------------------------------------
@app.post("/bus/inbox")
async def bus_inbox(payload: dict[str, Any], request: Request):
    if not check_bus_token(request):
        return JSONResponse({"error": "bad bus token"}, status_code=401)
    env = Envelope.model_validate(payload["envelope"])
    if env.header.refPoint != "SN" or not payload["topic"].startswith("sn.notify."):
        return JSONResponse({"error": "only sn.notify.* accepted"}, status_code=400)
    return {"delivered": await sl.bus.deliver_local(payload["topic"], env)}


@app.get("/health")
async def health():
    return {"layer": "service", "ok": True, "onboarded": sl.onboarded, "intents": len(sl.provisioner.instances)}
