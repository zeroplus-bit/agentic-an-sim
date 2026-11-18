"""Network Layer process (:8200) - SN server, NR client, agents, and the MCP tool endpoint (/tools/mcp)."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import ValidationError

from common.envelope import Envelope, check, respond
from common.refpoint import check_bus_token, read_envelope
from common.sn_messages import SN_BODY_MODELS
from common.events import get_hub, install_events
from common.tracing import to_mermaid

from .core import NetworkLayer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
nl = NetworkLayer()
_mcp_app = nl.tools.mcp_http_app()          # creates the MCP session manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with nl.tools.mcp.session_manager.run():
        await nl.start()
        yield
        await nl.stop()


app = FastAPI(title="Network Layer (SN server, agents, MCP tools)", lifespan=lifespan)
app.mount("/tools", _mcp_app)
install_events(app, get_hub("network"))      # GET /events (SSE) for the dashboard


async def _sn(request: Request, expected: set[str], handler: Callable[[Envelope], Awaitable[Envelope]]) -> Response:
    env = await read_envelope(request)
    if isinstance(env, Response):
        return env
    transport = f"http {request.method} {request.url.path}"
    cached = nl.server.inbound(env, transport)
    if cached is not None:
        return nl.server.outbound(env, cached, transport)
    err = check(env, "SN", expected)
    if err is None and env.header.auth is None:
        err = respond(env, "UNAUTHORIZED", reason="auth required on SN")
    if err is None and env.msg_type in SN_BODY_MODELS:
        body = dict(env.body)
        if "intentId" in request.path_params and env.msg_type in ("IntentModifyRequest", "IntentCancelRequest"):
            body.setdefault("intentId", request.path_params["intentId"])
        try:
            SN_BODY_MODELS[env.msg_type].model_validate(body)
        except ValidationError as e:
            err = respond(env, "INVALID", reason=str(e.errors()[:3]))
    if err is not None:
        return nl.server.outbound(env, err, transport)
    nl.traces.remember(env.header.correlationId, env.header.trace)   # e.g. SN-05 before the intent record exists
    try:
        resp = await handler(env)
    except Exception as e:  # never leak a stack trace over SN
        logging.exception("SN handler failed")
        resp = respond(env, "FAILED", reason=f"{type(e).__name__}: {e}")
    return nl.server.outbound(env, resp, transport)


# ---- SN reference point ------------------------------------------------------------------
@app.post("/sn/v1/intents")                                     # SN-01
async def intent_submit(request: Request):
    return await _sn(request, {"IntentSubmitRequest"}, nl.mediator.submit)


@app.patch("/sn/v1/intents/{intentId}")                         # SN-02
async def intent_modify(intentId: str, request: Request):
    return await _sn(request, {"IntentModifyRequest"}, lambda e: nl.mediator.modify(e, intentId))


@app.delete("/sn/v1/intents/{intentId}")                        # SN-03
async def intent_cancel(intentId: str, request: Request):
    return await _sn(request, {"IntentCancelRequest"}, lambda e: nl.mediator.cancel(e, intentId))


@app.post("/sn/v1/intents:feasibility")                         # SN-05
async def intent_feasibility(request: Request):
    return await _sn(request, {"IntentFeasibilityQuery"}, nl.prevalidator.feasibility)


@app.post("/sn/v1/subscriptions")                               # SN-06
async def monitoring_subscribe(request: Request):
    return await _sn(request, {"MonitoringSubscribeRequest"}, nl.reports.subscribe)


@app.post("/sn/v1/agents")                                      # SN-08
async def agent_deploy(request: Request):
    async def h(env: Envelope) -> Envelope:
        _, err = nl.auth.verify_sn(env, "sn:agent:write")
        if err:
            return respond(env, "UNAUTHORIZED", reason=err)
        nl.spawn(nl.registry.deploy(env))
        return respond(env, "ACCEPTED", {"agentId": env.body["agentDescriptor"]["agentId"],
                                         "action": env.body.get("action", "INSTANTIATE")})
    return await _sn(request, {"AgentDeployRequest"}, h)


@app.post("/sn/v1/models/deployments")                          # SN-10
async def model_deploy(request: Request):
    return await _sn(request, {"ModelDeployRequest"}, nl.model_ctrl.deploy)


@app.put("/sn/v1/data-policies/{policyId}")                     # SN-12
async def data_policy(policyId: str, request: Request):
    return await _sn(request, {"DataPolicyRequest"}, lambda e: nl.data_ctrl.set_policy(e, policyId))


@app.put("/sn/v1/resource-policies/{tenantId}")                 # SN-13
async def resource_policy(tenantId: str, request: Request):
    return await _sn(request, {"ResourcePolicyRequest"}, lambda e: nl.resource_ctrl.set_policy(e, tenantId))


# ---- bus bridge (NR notifications from the Resource Layer) ---------------------------------
@app.post("/bus/inbox")
async def bus_inbox(payload: dict[str, Any], request: Request):
    if not check_bus_token(request):
        return JSONResponse({"error": "bad bus token"}, status_code=401)
    env = Envelope.model_validate(payload["envelope"])
    if env.header.refPoint != "NR" or not payload["topic"].startswith("nr.notify."):
        return JSONResponse({"error": "only nr.notify.* accepted"}, status_code=400)
    n = await nl.bus.deliver_local(payload["topic"], env)
    return {"delivered": n}


# ---- observability ----------------------------------------------------------------------------
@app.get("/status/intents")
async def intents():
    return [r.view() for r in nl.intents.values()]


@app.get("/status/intents/{intentId}")
async def intent(intentId: str):
    r = nl.intents.get(intentId)
    if not r:
        return JSONResponse({"error": "unknown intent"}, status_code=404)
    return {**r.view(), "tasks": [{**nl.tasks[t].info(), "output": nl.tasks[t].output} for t in r.tasks if t in nl.tasks],
            "recentSamples": {c: list(d)[-5:] for c, d in r.samples.items()},
            "episodes": nl.memory.recall(intentId, limit=10)}


@app.get("/status/agents")
async def agents():
    return nl.registry.view()


@app.get("/status/tools")
async def tools():
    return nl.tools.list()


@app.get("/status/models")
async def models():
    return {**nl.model_ctrl.view(), "executor": nl.model_exec.view()}


@app.get("/metrics")
async def metrics():
    rows = []
    for r in nl.intents.values():
        rows.append({"intentId": r.intent_id, "state": r.state, "priority": r.priority,
                     "timeToFulfilSec": round(r.fulfilled_at - r.created_at, 3) if r.fulfilled_at else None,
                     "recoverTimesSec": [round(x, 2) for x in r.recover_times],
                     "fulfilmentRatio": round(r.windows_met / r.windows_total, 3) if r.windows_total else None,
                     "reportWindows": r.windows_total, "signalling": nl.tracer.overhead(r.intent_id)})
    return {"intents": rows, "resourceEfficiency": nl.perf_mon.efficiency(), "conflicts": nl.conflict.stats(),
            "memory": nl.memory.stats(), "signallingTotals": dict(nl.tracer.totals),
            "actions": {"applied": nl.actions.applied, "rolledBack": nl.actions.rolled_back},
            "assurance": nl.perf_ctrl.view(),
            "reportsSent": nl.reports.sent, "agentBusMessages": nl.agent_bus.messages,
            "telemetrySamples": nl.perf_mon.samples_seen}


@app.get("/trace")
async def trace(correlationId: Optional[str] = None, limit: int = 500):
    return nl.tracer.query(correlationId, limit=limit)


@app.get("/trace/{correlationId}/mermaid", response_class=PlainTextResponse)
async def trace_mermaid(correlationId: str):
    return to_mermaid(nl.tracer.query(correlationId, limit=5000))


@app.get("/health")
async def health():
    return {"layer": "network", "ok": True, "intents": len(nl.intents),
            "agents": {a["agentId"]: a["state"] for a in nl.registry.view()}}
