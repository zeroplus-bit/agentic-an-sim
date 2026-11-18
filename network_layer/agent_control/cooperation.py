"""Agent Cooperation/Interaction (AgentBus): agent-to-agent messages (task delegation, negotiation).

Messages ride the internal bus on topics agents.<agentId>.<kind>, so they appear in the trace.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from common.envelope import Envelope, ep, make

from ..core import TaskRecord

log = logging.getLogger("agent_bus")


class AgentBus:
    def __init__(self, nl):
        self.nl = nl
        self.messages = 0

    async def _post(self, kind: str, sender: str, receiver: str, corr: str, body: dict[str, Any]) -> None:
        # internal to the Network Layer: logged in the trace, but not counted as SN/NR signalling
        self.messages += 1
        self.nl.tracer.record_raw({"dir": "internal", "transport": f"agentbus:agents.{receiver}.{kind.lower()}",
                                   "correlationId": corr, "msgType": f"Agent{kind}", "from": f"network/{sender}",
                                   "to": f"network/{receiver}", "body": body})

    async def on_message(self, topic: str, env: Envelope) -> None:
        pass  # in-process agents are invoked directly by delegate(); this hook keeps a trace of the traffic

    async def delegate(self, task: TaskRecord, skill: str, sender: str = "IntentMediator",
                       timeout: float = 120.0, prefer: Optional[str] = None) -> dict[str, Any]:
        cards = self.nl.registry.discover(skill)
        if prefer:
            cards.sort(key=lambda c: c["agentId"] != prefer)
        if not cards:
            self.nl.tracker.update_task(task, "FAILED", output={"error": f"NO_AGENT_FOR_SKILL:{skill}"})
            return {"ok": False, "error": f"NO_AGENT_FOR_SKILL:{skill}"}
        card = cards[0]
        aid = card["agentId"]
        inst = self.nl.registry.instance(aid)
        if inst is None:
            self.nl.tracker.update_task(task, "FAILED", output={"error": f"agent {aid} is external (endpoint {card['endpoint']})"})
            return {"ok": False, "error": f"agent {aid} has no in-process runtime"}
        self.nl.tracker.update_task(task, "ASSIGNED", agent_id=aid)
        await self._post("TaskDelegate", sender, aid, task.intent_id or task.task_id,
                         {"taskId": task.task_id, "type": task.type, "skill": skill, "input": task.input})
        reg = self.nl.registry.cards[aid]
        reg["load"] = min(1.0, reg.get("load", 0) + 0.35)
        if reg["state"] == "ACTIVE":
            await self.nl.registry.notify(aid, "BUSY", task.intent_id)
        try:
            self.nl.tracker.update_task(task, "RUNNING")
            out = await asyncio.wait_for(inst.handle(task), timeout)
        except Exception as e:
            log.exception("agent %s failed task %s", aid, task.task_id)
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.nl.tracker.update_task(task, "COMPLETED" if out.get("ok") else "FAILED", output=out)
        await self._post("TaskResult", aid, sender, task.intent_id or task.task_id,
                         {"taskId": task.task_id, "ok": out.get("ok"), "summary": out.get("explanation") or out.get("error")})
        reg["load"] = max(0.0, reg.get("load", 0) - 0.35)
        if reg["state"] == "BUSY" and reg["load"] < 0.05:
            await self.nl.registry.notify(aid, "ACTIVE", task.intent_id)
        return out

    async def negotiate(self, requester: str, holder_intent: str, cell: str, ask_share: float, corr: str) -> dict[str, Any]:
        """Ask the assurance agent of a holder intent whether it can yield PRB share."""
        cards = self.nl.registry.discover("ASSURE_KPI")
        if not cards:
            return {"accepted": False, "reason": "no assurance agent"}
        aid = cards[0]["agentId"]
        await self._post("Negotiate", requester, aid, corr, {"holderIntent": holder_intent, "cell": cell, "askShare": ask_share})
        inst = self.nl.registry.instance(aid)
        ans = await inst.on_negotiate(holder_intent, cell, ask_share) if inst else {"accepted": False}
        await self._post("NegotiateResult", aid, requester, corr, ans)
        return ans
