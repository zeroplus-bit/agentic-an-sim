"""Registration/Discovery (AgentRegistry): agent cards (id, skills, tools, endpoint) and discovery by skill.

Follows the pattern of 5G-Marketplace vendor_registry/registry.py. Implements Flow 2 (agent onboarding):
SN-08 -> NR-02 (comp:*) -> NR-11 AGENT_RUNTIME -> register card -> bind permissions -> SN-09 REGISTERED/ACTIVE.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from common import config
from common.envelope import Envelope
from common.refpoint import new_id

log = logging.getLogger("registry")


class AgentRegistry:
    def __init__(self, nl):
        self.nl = nl
        self.cards: dict[str, dict[str, Any]] = {}

    def agent_classes(self) -> dict[str, type]:
        from ..agents.assurance_agent import AssuranceAgent
        from ..agents.base import ExternalAgent
        from ..agents.planner_agent import PlannerAgent
        from ..agents.ran_allocator_agent import RanAllocatorAgent
        return {"RAN_SLICE_ALLOCATOR": RanAllocatorAgent, "KPI_ASSURANCE": AssuranceAgent,
                "INTENT_PLANNER": PlannerAgent, "EXTERNAL": ExternalAgent}

    async def notify(self, agent_id: str, state: str, corr: Optional[str] = None) -> None:
        card = self.cards[agent_id]
        card["state"] = state
        await self.nl.sn_notify("AgentStateNotify", corr or agent_id,
                                {"agentId": agent_id, "state": state, "endpoint": card.get("endpoint"),
                                 "load": round(card.get("load", 0.0), 3)},
                                "AgentRegistry", "AgentLifecycleManager")

    async def deploy(self, env: Envelope) -> None:
        """Background part of SN-08 (after ACCEPTED)."""
        b = env.body
        d = b["agentDescriptor"]
        aid = d["agentId"]
        corr = env.header.correlationId
        if b.get("action") == "TERMINATE":
            await self.terminate(aid, corr)
            return
        if aid in self.cards and self.cards[aid]["state"] != "TERMINATED":
            # re-deploy = upgrade descriptor in place
            self.cards[aid]["descriptor"] = d
            await self.notify(aid, self.cards[aid]["state"], corr)
            return
        res = d.get("resources") or {}
        rsv = new_id("rsv-agent")
        r = await self.nl.resource_ctrl.reserve(None, [{"resourceId": "comp:*", "kind": "CPU", "amount": res.get("cpu", 1),
                                                        "unit": "core"}], priority=env.header.priority, corr=corr,
                                                reservation_id=rsv, tenant=None)
        if r.status != "SUCCESS":
            self.cards[aid] = {"descriptor": d, "state": "TERMINATED", "reason": (r.result.reason if r.result else "")}
            await self.notify(aid, "TERMINATED", corr)
            return
        node = r.body["granted"][0]["resourceId"]
        job_id = f"job-agent-{aid}"
        j = await self.nl.nr_call("ComputeJobRequest", "POST", "/nr/v1/compute/jobs",
                                  {"jobId": job_id, "kind": "AGENT_RUNTIME", "model": (d.get("policy") or {}).get("model"),
                                   "resources": {"cpu": res.get("cpu", 1), "gpu": res.get("gpu", 0)}, "node": node},
                                  corr, "AgentRegistry")
        cls = self.agent_classes().get(d["role"], self.agent_classes()["EXTERNAL"])
        endpoint = d.get("endpoint") or (f"bus://agents/{aid}" if cls.__name__ != "ExternalAgent"
                                         else f"mcp://{config.NETWORK_URL.split('//')[-1]}/tools/mcp")
        self.cards[aid] = {"descriptor": d, "state": "REGISTERED", "endpoint": endpoint, "load": 0.0,
                           "reservationId": rsv, "jobId": job_id, "node": node, "instance": None,
                           "jobStatus": j.status}
        err = self.nl.auth.bind_agent(aid, b.get("agentToken"), d.get("permissions") or [])
        if err:
            self.cards[aid]["reason"] = err
            await self.terminate(aid, corr)
            return
        await self.notify(aid, "REGISTERED", corr)
        inst = cls(self.nl, d, b.get("agentToken"))
        await inst.start()
        self.cards[aid]["instance"] = inst
        await self.notify(aid, "ACTIVE", corr)
        log.info("agent %s ACTIVE on %s (%s)", aid, node, d["role"])

    async def terminate(self, agent_id: str, corr: str) -> None:
        card = self.cards.get(agent_id)
        if not card:
            return
        inst = card.get("instance")
        if inst:
            await inst.stop()
        if card.get("jobId"):
            await self.nl.nr_call("ComputeJobRequest", "POST", "/nr/v1/compute/jobs",
                                  {"jobId": card["jobId"], "kind": "AGENT_RUNTIME", "cancel": True}, corr, "AgentRegistry")
        if card.get("reservationId"):
            await self.nl.resource_ctrl.release(card["reservationId"], corr)
        self.nl.auth.unbind(agent_id)
        card["instance"] = None
        await self.notify(agent_id, "TERMINATED", corr)

    def discover(self, skill: str) -> list[dict[str, Any]]:
        found = [dict(agentId=aid, **{k: v for k, v in c.items() if k != "instance"})
                 for aid, c in self.cards.items()
                 if c["state"] in ("ACTIVE", "BUSY") and skill in c["descriptor"].get("skills", [])]
        return sorted(found, key=lambda c: (c["state"] == "BUSY", c.get("load", 0)))

    def instance(self, agent_id: str):
        c = self.cards.get(agent_id)
        return c.get("instance") if c else None

    def view(self) -> list[dict[str, Any]]:
        return [{"agentId": aid, "state": c["state"], "role": c["descriptor"].get("role"),
                 "skills": c["descriptor"].get("skills"), "tools": c["descriptor"].get("tools"),
                 "endpoint": c.get("endpoint"), "load": round(c.get("load", 0.0), 3), "node": c.get("node"),
                 "permissions": self.nl.auth.agent_perms.get(aid, []), "reason": c.get("reason"),
                 "stats": c["instance"].stats if c.get("instance") else None}
                for aid, c in self.cards.items()]
