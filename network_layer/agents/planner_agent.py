"""Planner agent (skills DECOMPOSE_INTENT, EXPLAIN): splits an intent into tasks and writes explanations.
Uses an LLM only as a Tool Resource (llm_generate -> NR-10); the plan itself stays deterministic."""
from __future__ import annotations

from typing import Any

from ..core import TaskRecord
from .base import Agent


class PlannerAgent(Agent):
    def __init__(self, nl, descriptor, token):
        super().__init__(nl, descriptor, token)
        self.handlers = {"DECOMPOSE_INTENT": self.decompose, "EXPLAIN": self.explain}

    async def decompose(self, task: TaskRecord) -> dict[str, Any]:
        iid = task.input["intentId"]
        intent = await self.call("intent_get", intentId=iid)
        plan = [{"type": "ALLOCATE_SLICE", "skill": "ALLOCATE_SLICE", "cells": intent["cells"]}]
        kpis = [e["kpi"] for e in intent["expectations"]]
        plan.append({"type": "ASSURE_KPI", "skill": "ASSURE_KPI", "kpis": kpis})
        if intent["sliceType"] == "URLLC" or intent["priority"] <= 2:
            plan.append({"type": "PROTECT", "skill": "ASSURE_KPI", "note": "priority scheduling + preemption allowed"})
        plan.append({"type": "REPORT", "skill": "REPORT"})
        agents = {p["skill"]: [a["agentId"] for a in (await self.call("agent_discover", skill=p["skill"]))["agents"]]
                  for p in plan if p["skill"] != "REPORT"}
        txt = await self.call("llm_generate", intentId=iid, kind="plan",
                              prompt=f"Plan for intent {iid} ({intent['sliceType']}, {kpis}) on {intent['cells']}: "
                                     + ", ".join(p["type"] for p in plan))
        return {"ok": True, "plan": plan, "agents": agents, "explanation": txt.get("text")}

    async def explain(self, task: TaskRecord) -> dict[str, Any]:
        iid = task.input["intentId"]
        intent = await self.call("intent_get", intentId=iid)
        r = await self.call("llm_generate", intentId=iid, kind="explain", prompt=f"Explain the state of {iid}",
                            facts={"state": intent["state"], "kpis": intent["fulfilment"], "action": intent["explanation"]})
        return {"ok": True, "explanation": r.get("text")}
