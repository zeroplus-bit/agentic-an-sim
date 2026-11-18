"""Task Status mgmt (TaskTracker): intent/task state machine; source of SN-04 IntentStatusNotify."""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from common.refpoint import new_id
from common.sn_messages import INTENT_TRANSITIONS

from ..core import IntentRecord, TaskRecord

log = logging.getLogger("task_tracker")


class TaskTracker:
    def __init__(self, nl):
        self.nl = nl

    async def set_state(self, rec: IntentRecord, new: str, reason: Optional[str] = None,
                        explanation: Optional[str] = None, fulfilment: Optional[dict[str, Any]] = None) -> bool:
        old = rec.state
        if new == old:
            return False
        if new not in INTENT_TRANSITIONS.get(old, set()):
            log.warning("illegal transition %s -> %s for %s", old, new, rec.intent_id)
            return False
        rec.previous, rec.state = old, new
        t = time.time()
        if reason:
            rec.reason = reason
        if explanation:
            rec.explanation = explanation
        if fulfilment is not None:
            rec.fulfilment = fulfilment
        if new == "FULFILLED":
            if rec.fulfilled_at is None:
                rec.fulfilled_at = t
            if rec.drift_since is not None:
                rec.recover_times.append(t - rec.drift_since)
                rec.drift_since = None
        if new == "DEGRADED" and rec.drift_since is None:
            rec.drift_since = t
        rec.history.append({"at": t, "from": old, "to": new, "reason": reason})
        body = {"intentId": rec.intent_id, "state": new, "previousState": old,
                "serviceInstanceId": rec.service_instance_id,
                "tasks": [self.nl.tasks[tid].info() for tid in rec.tasks if tid in self.nl.tasks],
                "fulfilment": rec.fulfilment, "explanation": rec.explanation or None, "reason": reason}
        await self.nl.sn_notify("IntentStatusNotify", rec.intent_id, body, "TaskTracker", "ServiceMonitor",
                                priority=rec.priority, trace=rec.trace)
        return True

    def new_task(self, intent_id: Optional[str], type_: str, input_: Optional[dict[str, Any]] = None) -> TaskRecord:
        t = TaskRecord(new_id("task", 3), intent_id, type_, input_ or {})
        self.nl.tasks[t.task_id] = t
        if intent_id and intent_id in self.nl.intents:
            self.nl.intents[intent_id].tasks.append(t.task_id)
        return t

    def update_task(self, task: TaskRecord, state: str, agent_id: Optional[str] = None,
                    output: Optional[dict[str, Any]] = None) -> None:
        task.state = state
        task.updated = time.time()
        if agent_id:
            task.agent_id = agent_id
        if output is not None:
            task.output = output
        if state in ("COMPLETED", "FAILED"):
            self.nl.spawn(self.nl.repo.save_task(task))
