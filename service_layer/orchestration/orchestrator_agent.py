"""Agent-based Orchestration (ServiceOrchestratorAgent): decides which service-level intents to issue, modify
or cancel. Built-in policy is rule-based; an external LLM agent (e.g. Claude through the SN MCP server,
mcp_servers/sn_mcp_server.py) can take this role by reading conflicts/reports and calling the same operations.
Starts from 5G-Marketplace ai_agent/agent.py::AIAgent.

Built-in decision policy for SN-14 NEEDS_DECISION (no human in the loop unless stated):
  * an intent is never relocated outside the area it asked for (a hospital link must not move to stadium cells);
    MOVE_TO_OTHER_CELLS only picks unused cells of the SAME named area.
  * otherwise ACCEPT_PARTIAL: the service keeps running on the best share the network can give, the tenant is told,
    and the network recovers it by itself when demand drops. For priority 1-2 intents the decision is also
    flagged humanReview (a critical service is running degraded) - the one case that asks for a person.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

from common.envelope import Envelope


log = logging.getLogger("orchestrator")


class ServiceOrchestratorAgent:
    def __init__(self, sl, auto: bool = True):
        self.sl = sl
        self.auto = auto
        self.conflicts: dict[str, dict[str, Any]] = {}
        self.decisions: deque = deque(maxlen=200)

    async def on_conflict(self, topic: str, env: Envelope) -> None:
        b = env.body
        self.conflicts[b["conflictId"]] = {**b, "at": time.time(), "decided": b["resolution"] != "NEEDS_DECISION"}
        if b["resolution"] == "NEEDS_DECISION" and self.auto:
            target = b["intents"][0]
            if self._same_area_free_cells(target):
                r = await self.decide(b["conflictId"], "MOVE_TO_OTHER_CELLS")
                if r.get("ok"):
                    return
            await self.decide(b["conflictId"], "ACCEPT_PARTIAL")

    def _area_of(self, intent_id: str) -> tuple[str | None, list[str]]:
        from .connectivity import AREAS
        inst = self.sl.provisioner.instances.get(intent_id)
        cells = set((inst["intent"].get("context") or {}).get("area", {}).get("cells", [])) if inst else set()
        for name, a in AREAS.items():
            if cells & set(a["cells"]):
                return name, a["cells"]
        return None, []

    def _same_area_free_cells(self, intent_id: str) -> list[str]:
        _, area_cells = self._area_of(intent_id)
        used = {cell for k, x in self.sl.provisioner.instances.items() if x["state"] in ("FULFILLED", "DEGRADED")
                for cell in (x["intent"].get("context") or {}).get("area", {}).get("cells", [])}
        return [c for c in area_cells if c not in used]

    async def decide(self, conflict_id: str, action: str, intent_id: str | None = None) -> dict[str, Any]:
        c = self.conflicts.get(conflict_id)
        if not c:
            return {"ok": False, "error": "unknown conflict"}
        target = intent_id or c["intents"][0]
        inst = self.sl.provisioner.instances.get(target)
        if not inst:
            return {"ok": False, "error": f"intent {target} not owned by this Service Layer"}
        res: dict[str, Any] = {"conflictId": conflict_id, "action": action, "intentId": target}
        if action == "MOVE_TO_OTHER_CELLS":
            same = self._same_area_free_cells(target)
            free = [same] if same else []
            if not free:
                res.update(ok=False, error=f"no free cells in the intent's own area ({self._area_of(target)[0]}); "
                                           "not relocating to another area")
            else:
                for _ in range(50):       # let the Network Layer settle the current attempt first
                    if inst["state"] in ("FULFILLED", "DEGRADED", "REJECTED", "FAILED", "TERMINATED"):
                        break
                    await asyncio.sleep(0.2)
                if inst["state"] in ("FULFILLED", "DEGRADED"):
                    r = await self.sl.provisioner.modify(target, {"context": {"area": {"type": "cellList", "cells": free[0]}}})
                    res.update(ok=r.status == "ACCEPTED", newCells=free[0], status=r.status, how="SN-02 modify")
                else:                     # rejected -> resubmit a copy of the intent on the free cells
                    import copy
                    new = copy.deepcopy(inst["intent"])
                    new["intentId"] = f"{target}-r{sum(1 for k in self.sl.provisioner.instances if k.startswith(target))}"
                    new.setdefault("context", {})["area"] = {"type": "cellList", "cells": free[0]}
                    r = await self.sl.provisioner.submit(new, inst["priority"])
                    res.update(ok=r.status == "ACCEPTED", newCells=free[0], status=r.status, how="SN-01 resubmit",
                               newIntentId=new["intentId"])
        elif action == "LOWER_PRIORITY":
            r = await self.sl.provisioner.modify(target, {"priority": min(5, inst["priority"] + 1)})
            res.update(ok=r.status == "ACCEPTED", status=r.status)
        elif action == "CANCEL":
            r = await self.sl.provisioner.cancel(target, f"conflict {conflict_id}")
            res.update(ok=r.status == "ACCEPTED", status=r.status)
        else:
            res.update(ok=True, action="ACCEPT_PARTIAL", note="running on the best available share; tenant informed; "
                                                              "the network restores it when demand drops",
                       humanReview=inst["priority"] <= 2, detail=c.get("detail"))
        c["decided"] = True
        c["decision"] = res
        self.decisions.append({"at": time.time(), **res})
        log.warning("conflict %s decided: %s for %s%s", conflict_id, res.get("action"), target,
                    " (human review requested)" if res.get("humanReview") else "",
                    extra={"intentId": target, "stage": "COORDINATE"})
        return res

    async def on_status(self, b: dict[str, Any]) -> None:
        if b["state"] in ("REJECTED", "FAILED"):
            reason = str(b.get("reason") or "")
            admission = any(k in reason for k in ("CONFLICT", "INSUFFICIENT", "RESERVATION_REJECTED", "PREDICTED_KPIS_BREAK"))
            self.decisions.append({"at": time.time(), "intentId": b["intentId"], "observed": b["state"], "reason": reason,
                                   "action": "ADMISSION_CONTROL (tenant told why)" if admission else "NONE (reported to operator)",
                                   "humanReview": not admission})

    def view(self) -> dict[str, Any]:
        return {"auto": self.auto, "conflicts": list(self.conflicts.values())[-50:], "decisions": list(self.decisions)[-50:]}
