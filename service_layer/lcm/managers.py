"""Service LCM blocks: DataManager (SN-12), ResourceManager (SN-13), AgentLifecycleManager (SN-08/09),
ModelLifecycleManager (SN-10/11)."""
from __future__ import annotations

import json
import time
from collections import deque
from typing import Any, Optional

from common import config
from common.envelope import Auth, Envelope, ep
from common.refpoint import new_id


def _op_auth(sl) -> Auth:
    return Auth(tenantId=None, token=sl.identity.operator_token())


# ---- Data Management --------------------------------------------------------------------------
class DataManager:
    """Data catalogue and collection policies pushed to the Network Layer (starts from database/*.py)."""
    CATALOGUE = {
        "cell": ["latency_ms", "bandwidth_mbps", "prb_utilization", "active_ues"],
        "slice": ["latency_ms", "reliability_percent", "bandwidth_mbps", "utilization", "load", "prb_utilization", "active_ues"],
        "compute": ["cpu_load", "gpu_load"],
        "ue": ["active_ues"],
    }

    def __init__(self, sl):
        self.sl = sl
        self.policies: dict[str, dict[str, Any]] = {}

    async def set_policy(self, policy_id: str, body: dict[str, Any]) -> Envelope:
        body = {"policyId": policy_id, **body}
        env = self.sl.sn.envelope("DataPolicyRequest", policy_id, body, ep("network", "DataController"),
                                  auth=_op_auth(self.sl), source=ep("service", "DataManager"))
        r = await self.sl.sn.send("PUT", f"/sn/v1/data-policies/{policy_id}", env)
        self.policies[policy_id] = {**body, "status": r.status, "result": r.body}
        return r


# ---- Resource Management ----------------------------------------------------------------------
class ResourceManager:
    """Resource quotas and vendor/provider offers per tenant (starts from vendor_registry/registry.py)."""

    def __init__(self, sl):
        self.sl = sl
        self.policies: dict[str, dict[str, Any]] = {}
        self.offers = self._load_offers()

    @staticmethod
    def _load_offers() -> list[dict[str, Any]]:
        try:
            data = json.loads(config.VENDOR_OFFERINGS.read_text(encoding="utf-8"))
            return [{"offeringId": k, "vendorId": v.get("vendor_id"), "name": v.get("name"),
                     "sliceType": v.get("slice_type"), "qos": v.get("qos_parameters"),
                     "pricePerHour": v.get("price_per_hour")} for k, v in data.items()]
        except Exception:
            return []

    async def set_policy(self, tenant_id: str, body: dict[str, Any]) -> Envelope:
        body = {"tenantId": tenant_id, **body}
        env = self.sl.sn.envelope("ResourcePolicyRequest", tenant_id, body, ep("network", "ResourceController"),
                                  auth=_op_auth(self.sl), source=ep("service", "ResourceManager"))
        r = await self.sl.sn.send("PUT", f"/sn/v1/resource-policies/{tenant_id}", env)
        self.policies[tenant_id] = {**body, "status": r.status}
        return r


# ---- Agent LCM ---------------------------------------------------------------------------------
BASE_TOOLS = ["intent.get", "model.lstm.forecast", "model.dqn.infer", "decision.translate", "nr.resource.reserve",
              "nr.config.validate", "nr.config.apply", "conflict.check"]
DEFAULT_AGENTS: list[dict[str, Any]] = [
    {"agentId": "agent-ran-01", "role": "RAN_SLICE_ALLOCATOR", "version": "1.2.0",
     "skills": ["ALLOCATE_SLICE", "REBALANCE_PRB"],
     "tools": BASE_TOOLS,
     "policy": {"llm": config.GEMINI_MODEL, "model": "dqn_slice_alloc:1.0", "forecast": "lstm_single"},
     "resources": {"cpu": 1, "memoryMb": 512},
     "permissions": ["intent:read", "model:infer", "nr:resource:write", "nr:config:validate", "nr:config:write",
                     "nr:telemetry:read"]},
    {"agentId": "agent-assure-01", "role": "KPI_ASSURANCE", "version": "1.0.0", "skills": ["ASSURE_KPI"],
     "tools": BASE_TOOLS + ["nr.telemetry.window", "nr.resource.inventory", "memory.recall", "memory.store_episode"],
     "policy": {"model": "dqn_slice_alloc:1.0"}, "resources": {"cpu": 1, "memoryMb": 512},
     "permissions": ["intent:read", "model:infer", "nr:resource:read", "nr:resource:write", "nr:config:validate",
                     "nr:config:write", "nr:telemetry:read", "memory:rw"]},   # resource:read = neighbour cells for load balancing
    {"agentId": "agent-planner-01", "role": "INTENT_PLANNER", "version": "1.0.0", "skills": ["DECOMPOSE_INTENT", "EXPLAIN"],
     "tools": ["intent.get", "agent.discover", "llm.generate"],
     "policy": {"llm": config.GEMINI_MODEL}, "resources": {"cpu": 1, "memoryMb": 256},
     "permissions": ["intent:read", "agent:cooperate", "tool:llm:invoke"]},
]
ALL_AGENT_PERMS = ["intent:read", "model:infer", "nr:resource:read", "nr:resource:write", "nr:config:validate",
                   "nr:config:write", "nr:telemetry:read", "nr:telemetry:subscribe", "tool:llm:invoke",
                   "nr:compute:write", "nr:storage:rw", "agent:cooperate", "memory:rw"]


class AgentLifecycleManager:
    """Onboard, instantiate, scale and terminate agents; owns agent descriptors (Flow 2)."""

    def __init__(self, sl):
        self.sl = sl
        self.descriptors: dict[str, dict[str, Any]] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.history: deque = deque(maxlen=500)

    async def deploy(self, descriptor: dict[str, Any], action: str = "INSTANTIATE") -> tuple[Envelope, Optional[str]]:
        aid = descriptor["agentId"]
        token = self.sl.identity.agent_token(aid, descriptor.get("permissions") or [])
        self.descriptors[aid] = descriptor
        env = self.sl.sn.envelope("AgentDeployRequest", aid, {"agentDescriptor": descriptor, "action": action,
                                                               "agentToken": token},
                                  ep("network", "AgentRegistry"), auth=_op_auth(self.sl),
                                  source=ep("service", "AgentLifecycleManager"))
        r = await self.sl.sn.send("POST", "/sn/v1/agents", env)
        return r, token

    async def terminate(self, agent_id: str) -> Envelope:
        d = self.descriptors.get(agent_id) or {"agentId": agent_id, "role": "EXTERNAL", "skills": []}
        r, _ = await self.deploy(d, "TERMINATE")
        return r

    async def onboard_defaults(self) -> list[str]:
        out = []
        for d in DEFAULT_AGENTS:
            r, _ = await self.deploy(d)
            out.append(f"{d['agentId']}:{r.status}")
        return out

    async def onboard_external(self, agent_id: str, skills: list[str], permissions: Optional[list[str]] = None,
                               tools: Optional[list[str]] = None) -> dict[str, Any]:
        d = {"agentId": agent_id, "role": "EXTERNAL", "version": "1.0.0", "skills": skills,
             "tools": tools or ["*"], "permissions": permissions or ALL_AGENT_PERMS,
             "resources": {"cpu": 0.5, "memoryMb": 128}, "policy": {"llm": "external"}}
        r, token = await self.deploy(d)
        return {"status": r.status, "agentId": agent_id, "agentToken": token,
                "mcpUrl": f"{config.NETWORK_URL}/tools/mcp",
                "howTo": "connect an MCP client (streamable HTTP) with header 'Authorization: Bearer <agentToken>'"}

    async def on_state(self, topic: str, env: Envelope) -> None:
        b = env.body
        self.states[b["agentId"]] = {**b, "at": time.time()}
        self.history.append({"at": time.time(), **b})


# ---- AI/ML Model LCM ------------------------------------------------------------------------------
class ModelLifecycleManager:
    """Model catalogue, versioning, deploy/rollback requests, training-data lineage (Flow 4)."""

    def __init__(self, sl):
        self.sl = sl
        self.catalogue = {
            "lstm_single": {
                "1.0": {"artifactUri": "repo://models/lstm_single/1.0", "framework": "keras-h5 (numpy runtime)",
                        "source": str(config.LSTM_MODEL_PATH), "lineage": "data/training (repo), synthetic scenarios",
                        "inputSpec": {"shape": [10, 11]}, "outputSpec": {"shape": [3]}},
                "2.0": {"artifactUri": "repo://models/lstm_single/2.0", "framework": "keras-h5 (numpy runtime)",
                        "source": str(config.LSTM_MODEL_V2_PATH), "lineage": "best_model checkpoint of the same run",
                        "inputSpec": {"shape": [10, 11]}, "outputSpec": {"shape": [3]}},
            },
            "dqn_slice_alloc": {"1.0": {"artifactUri": "inproc://dqn", "framework": "numpy (port of dqn_agent.py)",
                                        "lineage": "online replay buffer from assurance episodes"}},
        }
        self.deployments: dict[str, dict[str, Any]] = {}

    async def deploy(self, name: str, version: str, traffic_share: float = 0.2,
                     rollback_on: Optional[dict[str, Any]] = None, auto_promote: bool = False,
                     deployment_id: Optional[str] = None) -> Envelope:
        meta = self.catalogue.get(name, {}).get(version) or {"artifactUri": f"repo://models/{name}/{version}"}
        did = deployment_id or new_id("dep")
        body = {"deploymentId": did,
                "model": {"name": name, "version": version, "artifactUri": meta["artifactUri"],
                          "framework": meta.get("framework", "numpy"), "inputSpec": meta.get("inputSpec", {}),
                          "outputSpec": meta.get("outputSpec", {})},
                "strategy": {"type": "CANARY" if traffic_share < 1 else "ALL_AT_ONCE", "trafficShare": traffic_share,
                             "rollbackOn": rollback_on or {"metric": "mae", "op": ">", "value": 0.2}}}
        self.deployments[did] = {"request": body, "autoPromote": auto_promote, "notifications": [], "state": "REQUESTED"}
        env = self.sl.sn.envelope("ModelDeployRequest", did, body, ep("network", "ModelController"),
                                  auth=_op_auth(self.sl), source=ep("service", "ModelLifecycleManager"))
        r = await self.sl.sn.send("POST", "/sn/v1/models/deployments", env)
        self.deployments[did]["state"] = r.status
        return r

    async def on_performance(self, topic: str, env: Envelope) -> None:
        b = env.body
        d = self.deployments.setdefault(b["deploymentId"], {"notifications": [], "autoPromote": False})
        d["state"] = b["state"]
        d["notifications"].append({"at": time.time(), **b})
        d["notifications"] = d["notifications"][-50:]
        req = d.get("request")
        if (d.get("autoPromote") and b["state"] == "ACTIVE" and req and req["strategy"]["trafficShare"] < 1
                and not b.get("drift") and b["metrics"].get("samples", 0) >= 20
                and len([n for n in d["notifications"] if n["state"] == "ACTIVE"]) >= 3):
            d["autoPromote"] = False
            m = req["model"]
            await self.deploy(m["name"], m["version"], 1.0, deployment_id=f"{b['deploymentId']}-promote")

    def view(self) -> dict[str, Any]:
        return {"catalogue": self.catalogue,
                "deployments": {k: {"state": v.get("state"), "autoPromote": v.get("autoPromote"),
                                    "model": (v.get("request") or {}).get("model", {}).get("version"),
                                    "last": v["notifications"][-1] if v["notifications"] else None}
                                for k, v in self.deployments.items()}}
