"""SN MCP server - lets an LLM agent (Claude Desktop / Claude Code / any MCP client) act as the Service Layer's
Agent-based Orchestration block: express intents in natural language, watch fulfilment, resolve conflicts,
run agent/model lifecycle - all over the SN reference point via the Service Layer API.

Run (stdio):            python -m mcp_servers.sn_mcp_server
Run (streamable HTTP):  python -m mcp_servers.sn_mcp_server --http --port 8110
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

SERVICE_URL = os.environ.get("AANS_SERVICE_URL", "http://127.0.0.1:8100")
DASHBOARD_URL = os.environ.get("AANS_DASHBOARD_URL", "http://127.0.0.1:8400")

mcp = FastMCP(
    "aan-service-layer",
    instructions=(
        "You operate an Agentic AI Autonomous Network simulator (ITU-T style 3-layer framework: Service, Network, "
        "Resource layers joined by the SN and NR reference points). Express service needs as intents in natural "
        "language with measurable targets (e.g. 'URLLC for Hospital A, under 5 ms, 50 Mbps, 99.999%'). "
        "Check feasibility first for demanding intents, then submit, then watch state "
        "(RECEIVED->VALIDATING->PLANNING->DEPLOYING->FULFILLED<->DEGRADED->TERMINATED) and reports. "
        "Resolve NEEDS_DECISION conflicts with decide_conflict. Simulation scenarios can be injected with set_scenario."),
)


async def _req(method: str, path: str, **kw) -> Any:
    async with httpx.AsyncClient(base_url=SERVICE_URL, timeout=60) as c:
        r = await c.request(method, path, **kw)
        try:
            return r.json()
        except ValueError:
            return {"status": r.status_code, "text": r.text}


@mcp.tool()
async def parse_intent(utterance: str, tenantId: str = "tenant-a") -> dict:
    """Preview how an utterance becomes a structured intent (slice type, expectations, area/cells, priority). Nothing is sent."""
    return await _req("POST", "/api/intents:parse", json={"utterance": utterance, "tenantId": tenantId})


@mcp.tool()
async def check_feasibility(utterance: str, tenantId: str = "tenant-a", cells: Optional[list[str]] = None) -> dict:
    """SN-05 IntentFeasibilityQuery: predicted KPIs, cost per hour and alternatives before committing."""
    return await _req("POST", "/api/intents:feasibility", json={"utterance": utterance, "tenantId": tenantId, "cells": cells})


@mcp.tool()
async def submit_intent(utterance: str, tenantId: str = "tenant-a", priority: Optional[int] = None,
                        cells: Optional[list[str]] = None, expectations: Optional[list[dict]] = None,
                        checkFeasibility: bool = False, force: bool = False) -> dict:
    """SN-01 IntentSubmitRequest from natural language. priority 1 = emergency .. 5 = best effort.
    expectations override parsed ones, e.g. [{"kpi":"latency_ms","op":"<=","value":5,"percentile":99}].
    Returns status CLARIFICATION_REQUIRED (nothing provisioned) when the stated KPIs are invalid or contradict the
    slice type; ask the user, then resubmit rephrased, with explicit expectations, or with force=true.
    Also check intent.semantic.needsClarification / classifier.predictionSet: more than one slice type means the
    utterance is ambiguous at the calibrated error level."""
    return await _req("POST", "/api/intents", json={"utterance": utterance, "tenantId": tenantId, "priority": priority,
                                                    "cells": cells, "expectations": expectations,
                                                    "checkFeasibility": checkFeasibility, "force": force})


@mcp.tool()
async def list_intents() -> list:
    """All intents with state, slice type, cells, time-to-fulfil and fulfilment ratio."""
    return await _req("GET", "/api/intents")


@mcp.tool()
async def get_intent(intentId: str) -> dict:
    """One intent: Service Layer view, status events, latest SN-07 reports and the Network Layer's view
    (tasks, agents, PRB shares, applied configs, recent telemetry, learning episodes)."""
    return await _req("GET", f"/api/intents/{intentId}")


@mcp.tool()
async def get_reports(intentId: str, n: int = 10) -> dict:
    """Latest SN-07 PerformanceReportNotify windows (KPIs, compliance, forecast, explanation)."""
    return await _req("GET", f"/api/intents/{intentId}/reports", params={"n": n})


@mcp.tool()
async def modify_intent(intentId: str, expectations: Optional[list[dict]] = None, priority: Optional[int] = None,
                        cells: Optional[list[str]] = None) -> dict:
    """SN-02 IntentModifyRequest (re-plans the slice)."""
    return await _req("PATCH", f"/api/intents/{intentId}",
                      json={k: v for k, v in {"expectations": expectations, "priority": priority, "cells": cells}.items() if v})


@mcp.tool()
async def cancel_intent(intentId: str, reason: str = "cancelled by operator") -> dict:
    """SN-03 IntentCancelRequest (rolls back configs, releases reservations)."""
    return await _req("DELETE", f"/api/intents/{intentId}", params={"reason": reason})


@mcp.tool()
async def list_conflicts() -> dict:
    """SN-14 ConflictNotify received from Conflict mgmt and the decisions taken."""
    return await _req("GET", "/api/conflicts")


@mcp.tool()
async def decide_conflict(conflictId: str, action: str, intentId: Optional[str] = None) -> dict:
    """Resolve a NEEDS_DECISION conflict. action: MOVE_TO_OTHER_CELLS | LOWER_PRIORITY | CANCEL | ACCEPT_PARTIAL."""
    return await _req("POST", f"/api/conflicts/{conflictId}:decide", json={"action": action, "intentId": intentId})


@mcp.tool()
async def list_agents() -> dict:
    """Agent descriptors, SN-09 states and the Network Layer registry (skills, tools, permissions, load)."""
    return await _req("GET", "/api/agents")


@mcp.tool()
async def deploy_agent(agentId: str, role: str, skills: list[str], permissions: list[str],
                       tools: Optional[list[str]] = None, cpu: float = 1) -> dict:
    """SN-08 AgentDeployRequest. role: RAN_SLICE_ALLOCATOR | KPI_ASSURANCE | INTENT_PLANNER | EXTERNAL."""
    return await _req("POST", "/api/agents", json={"agentDescriptor": {
        "agentId": agentId, "role": role, "skills": skills, "permissions": permissions, "tools": tools or [],
        "resources": {"cpu": cpu, "memoryMb": 512}}})


@mcp.tool()
async def onboard_external_agent(agentId: str, skills: list[str], permissions: Optional[list[str]] = None) -> dict:
    """Onboard yourself (or another external LLM agent) into the Network Layer: returns an agent token and the
    MCP URL of the Network Layer ToolRegistry (NR operations, LSTM/DQN, conflict check, memory)."""
    return await _req("POST", "/api/agents:external", json={"agentId": agentId, "skills": skills, "permissions": permissions})


@mcp.tool()
async def terminate_agent(agentId: str) -> dict:
    """SN-08 with action TERMINATE."""
    return await _req("DELETE", f"/api/agents/{agentId}")


@mcp.tool()
async def list_models() -> dict:
    """Model catalogue, deployments (SN-10/11 state) and live canary metrics (MAE, inference p95)."""
    return await _req("GET", "/api/models")


@mcp.tool()
async def deploy_model(version: str, name: str = "lstm_single", trafficShare: float = 0.2,
                       rollbackMae: float = 0.2, autoPromote: bool = False) -> dict:
    """SN-10 ModelDeployRequest with a canary strategy; trafficShare 1.0 promotes. Rolls back when MAE > rollbackMae."""
    return await _req("POST", "/api/models/deployments", json={"name": name, "version": version, "trafficShare": trafficShare,
                                                              "rollbackOn": {"metric": "mae", "op": ">", "value": rollbackMae},
                                                              "autoPromote": autoPromote})


@mcp.tool()
async def confirm_intent_interpretation(intentId: str, slice: str, expectations: Optional[dict] = None,
                                        devices: Optional[int] = None) -> dict:
    """Operator ground truth for a submitted utterance (slice URLLC|eMBB|mMTC, optional stated KPIs as
    {"latency_ms": 5, "bandwidth_mbps": 50, "reliability_percent": 99.999}, optional device count). Feeds the
    intent-NLU lifecycle (shadow comparison, McNemar decision) and datasets/intent_nlu/feedback.jsonl."""
    return await _req("POST", f"/api/intents/{intentId}/feedback",
                      json={"slice": slice, "expectations": expectations, "devices": devices})


@mcp.tool()
async def intent_model_status() -> dict:
    """Intent-NLU lifecycle: primary version, available versions (1.0 keyword .. 2.2 TF joint), shadow state,
    accuracy on operator-confirmed intents, decision rule and history."""
    return await _req("GET", "/api/models/intent-nlu")


@mcp.tool()
async def manage_intent_model(action: str, version: Optional[str] = None, autoPromote: bool = False) -> dict:
    """action = shadow (run `version` alongside the primary on every submitted utterance), promote (make
    `version` primary) or rollback (restore the previous primary)."""
    if action == "rollback":
        return await _req("POST", "/api/models/intent-nlu/rollback")
    if action not in ("shadow", "promote") or not version:
        return {"ok": False, "reason": "action must be shadow|promote (with version) or rollback"}
    return await _req("POST", f"/api/models/intent-nlu/{action}", json={"version": version, "autoPromote": autoPromote})


@mcp.tool()
async def set_resource_policy(tenantId: str, prbSharePercent: float, computeCpu: float = 8, storageGb: float = 50,
                              overbooking: float = 1.0) -> dict:
    """SN-13 ResourcePolicyRequest (per-tenant PRB share quota per cell)."""
    return await _req("PUT", f"/api/policies/resource/{tenantId}",
                      json={"quota": {"prbSharePercent": prbSharePercent, "computeCpu": computeCpu, "storageGb": storageGb},
                            "overbooking": overbooking, "preemption": "PRIORITY"})


@mcp.tool()
async def set_data_policy(policyId: str, sources: list[str], periodSec: float = 5, kpis: Optional[list[str]] = None) -> dict:
    """SN-12 DataPolicyRequest. sources: cell | slice | compute."""
    return await _req("PUT", f"/api/policies/data/{policyId}",
                      json={"sources": sources, "periodSec": periodSec, "kpis": kpis or [], "retentionHours": 24,
                            "anonymize": True})


@mcp.tool()
async def get_metrics() -> dict:
    """Evaluation metrics: fulfilment ratio, time to fulfil/recover, signalling overhead, PRB efficiency, conflict rate."""
    return await _req("GET", "/api/metrics")


@mcp.tool()
async def get_trace(correlationId: str, mermaid: bool = True) -> str:
    """All SN/NR messages of one intent across the three layers, as a Mermaid sequence diagram (or JSON)."""
    r = await _req("GET", f"/api/trace/{correlationId}", params={"format": "mermaid" if mermaid else "json"})
    return r.get("text") if isinstance(r, dict) and "text" in r else json.dumps(r)[:60000]


@mcp.tool()
async def set_scenario(is_emergency: Optional[bool] = None, is_special_event: Optional[bool] = None,
                       is_iot_surge: Optional[bool] = None, cell_down: Optional[list[str]] = None,
                       load_factor: Optional[float] = None) -> dict:
    """Simulation control of the digital twin (outside SN/NR): emergency (URLLC x2), special event (eMBB x1.5),
    IoT surge (mMTC x1.8), cells down, global load factor."""
    payload = {k: v for k, v in {"is_emergency": is_emergency, "is_special_event": is_special_event,
                                 "is_iot_surge": is_iot_surge, "cell_down": cell_down, "load_factor": load_factor}.items()
               if v is not None}
    return await _req("POST", "/api/sim/scenario", json=payload)


@mcp.tool()
async def get_network_state() -> dict:
    """Digital-twin state: cells (PRB split, dedicated shares, utilization), reservations, alarms."""
    return await _req("GET", "/api/sim/state")


@mcp.resource("aan://catalogue", description="SN-01..SN-14 and NR-01..NR-13 message catalogue")
async def catalogue() -> str:
    return json.dumps(await _req("GET", "/api/catalogue"), indent=1)


@mcp.tool()
async def get_dashboard_snapshot(events: int = 30, incidentId: Optional[str] = None) -> dict:
    """Operations dashboard view (needs the dashboard process, :8400): KPIs (SLA, MTTA/MTTR, autonomy ratio, human
    review queue), open/resolved incidents with their detect->report->execute->coordinate->resolve timings, and the
    latest events. With incidentId: that incident's events only."""
    try:
        async with httpx.AsyncClient(base_url=DASHBOARD_URL, timeout=10) as c:
            if incidentId:
                return {"incidentId": incidentId,
                        "events": (await c.get("/dash/api/events", params={"incident": incidentId, "limit": 300})).json()}
            snap = (await c.get("/dash/api/snapshot", params={"events": events})).json()
    except httpx.HTTPError as e:
        return {"error": f"dashboard not reachable at {DASHBOARD_URL}: {e}"}
    st = snap["state"]
    return {"kpis": st["kpis"], "incidents": snap["incidents"][-15:], "review": st["review"],
            "links": {k: v.get("up") for k, v in st["links"].items()},
            "events": [{k: e.get(k) for k in ("at", "layer", "stage", "severity", "intentId", "summary")}
                       for e in snap["events"]]}


@mcp.resource("aan://dashboard", description="Operations dashboard snapshot: KPIs, incidents, human review queue")
async def dashboard() -> str:
    return json.dumps(await get_dashboard_snapshot(events=20), indent=1, default=str)


@mcp.resource("aan://areas", description="Named areas of the simulated city and their cells")
async def areas() -> str:
    return json.dumps(await _req("GET", "/api/areas"), indent=1)


@mcp.prompt()
def operate_intent(goal: str) -> str:
    """Closed-loop operation of one service goal."""
    return (f"Goal: {goal}\n1. parse_intent and check_feasibility.\n2. submit_intent (adjust expectations/priority if "
            "infeasible).\n3. Poll get_intent until FULFILLED, REJECTED or FAILED.\n4. Inspect get_reports; if DEGRADED, "
            "look at list_conflicts and decide.\n5. Summarise KPIs vs expectations and the agents' actions.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--port", type=int, default=8110)
    a = ap.parse_args()
    if a.http:
        mcp.settings.port = a.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
