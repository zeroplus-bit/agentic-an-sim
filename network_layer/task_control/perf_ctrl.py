"""Performance Control (PerformanceController): choose corrective action on SLA drift by assigning an
assurance task to an agent (Flow 3 step 4); close the episode for DQN retraining when KPIs recover.

Escalation: when the agent reports CAPACITY_CEILING (the share cannot grow, every cell is down) or the attempts
run out, the Service Layer is told AT ONCE over SN-14 (ConflictNotify NEEDS_DECISION with the shortfall and the
options) instead of retrying silently; further retries back off to ESCALATION_RETRY_SEC until the decision or a
change in load. Reclaim (scale-down) tasks go to the same agent with mode "reclaim".
"""
from __future__ import annotations

import logging
import time
from typing import Any


log = logging.getLogger("perf_ctrl")
MAX_ATTEMPTS = 4
ESCALATION_RETRY_SEC = 30.0


class PerformanceController:
    def __init__(self, nl):
        self.nl = nl
        self.pending_episode: dict[str, dict[str, Any]] = {}     # intentId -> last corrective action
        self.attempts: dict[str, int] = {}
        self.escalations: list[dict[str, Any]] = []
        self.reclaims: list[dict[str, Any]] = []

    async def on_drift(self, rec, violations: list[str], cell: str, cause: str) -> None:
        if rec.assurance_running or rec.state != "DEGRADED":
            return
        if rec.escalated_at and time.time() - rec.escalated_at < ESCALATION_RETRY_SEC:
            return
        n = self.attempts.get(rec.intent_id, 0)
        if n >= MAX_ATTEMPTS:
            if n == MAX_ATTEMPTS:
                self.attempts[rec.intent_id] = n + 1
                rec.explanation = f"assurance gave up after {n} attempts; escalated to Service Layer"
                await self.escalate(rec, cell, {"reason": "ATTEMPTS_EXHAUSTED", "attempts": n}, violations)
            return
        self.attempts[rec.intent_id] = n + 1
        rec.assurance_running = True
        rec.last_action_at = time.time()
        try:
            task = self.nl.tracker.new_task(rec.intent_id, "ASSURE_KPI",
                                            {"intentId": rec.intent_id, "cell": cell, "violations": violations,
                                             "cause": cause, "mode": "correct", "attempt": n + 1})
            log.info("assurance task %s (attempt %d, %s on %s) -> ASSURE_KPI", task.task_id, n + 1, cause, cell,
                     extra={"intentId": rec.intent_id, "stage": "EXECUTE", "cell": cell})
            out = await self.nl.agent_bus.delegate(task, "ASSURE_KPI")
            log.log(logging.INFO if out.get("ok") else logging.WARNING, "assurance %s by %s: %s",
                    "done" if out.get("ok") else "failed", task.agent_id,
                    str(out.get("explanation") or out.get("error"))[:300],
                    extra={"intentId": rec.intent_id, "stage": "EXECUTE", "cell": cell})
            if out.get("ok"):
                rec.explanation = out.get("explanation", "")
                if out.get("episode"):
                    self.pending_episode[rec.intent_id] = {**out["episode"], "taskId": task.task_id,
                                                           "agentId": task.agent_id, "at": time.time()}
            else:
                rec.explanation = f"assurance failed: {out.get('error')}"
                if out.get("ceiling") and not str(out.get("error", "")).startswith("NEEDS_DECISION"):
                    await self.escalate(rec, cell, out["ceiling"], violations)
                elif out.get("ceiling"):
                    rec.escalated_at = time.time()     # ConflictManager already sent SN-14 NEEDS_DECISION
        finally:
            rec.assurance_running = False
            rec.last_action_at = time.time()
            rec.recent_ok.clear()

    async def escalate(self, rec, cell: str, ceiling: dict[str, Any], violations: list[str]) -> None:
        """SN-14 NEEDS_DECISION: the network cannot meet the intent by itself; options for the orchestrator."""
        rec.escalated_at = time.time()
        cid = self.nl.conflict._record([rec.intent_id], f"comm:{cell}/prb", "NEEDS_DECISION",
                                       options=[{"action": "ACCEPT_PARTIAL", "intentId": rec.intent_id,
                                                 "maxShare": ceiling.get("max")},
                                                {"action": "RELAX_EXPECTATIONS", "intentId": rec.intent_id},
                                                {"action": "CANCEL", "intentId": rec.intent_id}])
        body = {"conflictId": cid, "intents": [rec.intent_id], "resource": f"comm:{cell}/prb",
                "resolution": "NEEDS_DECISION", "cause": "NETWORK_CANNOT_MEET_INTENT",
                "detail": {**ceiling, "violations": violations[:3]},
                "options": [{"action": "ACCEPT_PARTIAL", "intentId": rec.intent_id, "maxShare": ceiling.get("max")},
                            {"action": "RELAX_EXPECTATIONS", "intentId": rec.intent_id},
                            {"action": "CANCEL", "intentId": rec.intent_id}]}
        self.escalations.append({"at": time.time(), "intentId": rec.intent_id, "cell": cell, **ceiling, "conflictId": cid})
        log.warning("intent %s escalated over SN-14 NEEDS_DECISION (%s)", rec.intent_id, ceiling,
                    extra={"intentId": rec.intent_id, "stage": "COORDINATE", "cell": cell})
        await self.nl.sn_notify("ConflictNotify", rec.intent_id, body, "PerformanceController", "ServiceOrchestratorAgent",
                                priority=rec.priority)

    async def on_reclaim(self, rec, cell: str) -> None:
        if rec.assurance_running or rec.state != "FULFILLED":
            return
        rec.assurance_running = True
        try:
            task = self.nl.tracker.new_task(rec.intent_id, "ASSURE_KPI",
                                            {"intentId": rec.intent_id, "cell": cell, "mode": "reclaim", "cause": "RECLAIM"})
            out = await self.nl.agent_bus.delegate(task, "ASSURE_KPI")
            if out.get("ok") and not out.get("noop"):
                rec.explanation = out.get("explanation", "")
                rec.last_action_at = time.time()
                self.reclaims.append({"at": time.time(), "intentId": rec.intent_id, "cell": cell,
                                      "explanation": rec.explanation})
                log.info("reclaim on %s: %s", cell, rec.explanation[:200],
                         extra={"intentId": rec.intent_id, "stage": "RESOLVE", "cell": cell})
        finally:
            rec.assurance_running = False
            rec.reclaimed_at = time.time()

    async def on_recovered(self, rec) -> None:
        self.attempts.pop(rec.intent_id, None)
        rec.escalated_at = None
        ep = self.pending_episode.pop(rec.intent_id, None)
        if not ep:
            return
        dt = time.time() - ep["at"]
        reward = 5.0 - 0.05 * dt - 0.05 * float(ep.get("shareDelta", 0))
        self.close_episode(rec.intent_id, ep, reward, "RECOVERED")

    def close_episode(self, intent_id: str, ep: dict[str, Any], reward: float, outcome: str) -> None:
        nxt = ep.get("nextState") or ep["state"]
        self.nl.memory.store_episode(intent_id, ep.get("agentId") or "", ep.get("taskId", ""), ep["state"],
                                     int(ep["action"]), reward, nxt, outcome, {"shareDelta": ep.get("shareDelta")})
        self.nl.model_exec.learn(ep["state"], int(ep["action"]), reward, nxt)

    def on_rolled_back(self, intent_id: str) -> None:
        ep = self.pending_episode.pop(intent_id, None)
        if ep:
            self.close_episode(intent_id, ep, -5.0, "ROLLED_BACK")

    def view(self) -> dict[str, Any]:
        return {"escalations": self.escalations[-20:], "reclaims": self.reclaims[-20:]}
