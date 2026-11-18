"""Conflict mgmt (ConflictManager): detect conflicting intents/actions on the same resource; resolve by priority.

priority 1 = emergency ... 5 = best effort. A higher-priority intent may preempt PRB share from
lower-priority intents on the same cell (SN-14 PREEMPTED_LOWER_PRIORITY); equal/higher priority
holders produce SN-14 NEEDS_DECISION for the Service Layer's orchestration agent.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from common.envelope import Envelope
from common.refpoint import new_id

CELL_BUDGET = 90.0          # % of a cell's PRBs that may be dedicated to intents
MIN_SHARE = 5.0             # never preempt an intent below this share


class ConflictManager:
    def __init__(self, nl):
        self.nl = nl
        self.conflicts: list[dict[str, Any]] = []
        self.intents_seen = 0

    # ---- estimation --------------------------------------------------------------------
    def used(self, cell: str, exclude: Optional[str] = None) -> dict[str, float]:
        return {iid: r.shares.get(cell, 0.0) for iid, r in self.nl.intents.items()
                if iid != exclude and r.active and r.shares.get(cell)}

    def free(self, cell: str, exclude: Optional[str] = None) -> float:
        return CELL_BUDGET - sum(self.used(cell, exclude).values())

    # ---- SN-01 immediate check ------------------------------------------------------------
    def check_new(self, intent: dict[str, Any], priority: int) -> Optional[str]:
        self.intents_seen += 1
        need = self.nl.translator.estimate_share(intent, target_load=0.72)
        for cell in (intent.get("context") or {}).get("area", {}).get("cells") or []:
            free = self.free(cell)
            if need <= free:
                continue
            holders = self.used(cell)
            preemptable = sum(max(0.0, s - MIN_SHARE) for iid, s in holders.items()
                              if self.nl.intents[iid].priority > priority)
            if need <= free + preemptable:
                continue
            blocker = min(holders, key=lambda i: self.nl.intents[i].priority) if holders else None
            self._record([intent["intentId"]] + ([blocker] if blocker else []), f"comm:{cell}/prb",
                         "REJECTED_AT_SUBMIT", notified=False)
            return f"CONFLICT_WITH:{blocker}" if blocker else f"INSUFFICIENT_RESOURCES:{cell}"
        return None

    # ---- change check (Flow 3 step 5) ---------------------------------------------------------
    async def check_change(self, intent_id: str, cell: str, new_share: float, corr: str) -> dict[str, Any]:
        rec = self.nl.intents[intent_id]
        free = self.free(cell, exclude=intent_id)
        if new_share <= free + 1e-9:
            return {"ok": True}
        deficit = new_share - free
        holders = self.used(cell, exclude=intent_id)
        lower = sorted([i for i in holders if self.nl.intents[i].priority > rec.priority],
                       key=lambda i: -self.nl.intents[i].priority)
        plan = []
        for iid in lower:
            take = min(deficit, max(0.0, holders[iid] - MIN_SHARE))
            if take > 0:
                plan.append({"intentId": iid, "cell": cell, "from": holders[iid], "to": math.floor(holders[iid] - take)})
                deficit -= take
            if deficit <= 1e-9:
                break
        if deficit <= 1e-9:
            cid = self._record([intent_id] + [p["intentId"] for p in plan], f"comm:{cell}/prb", "PREEMPTED_LOWER_PRIORITY",
                               options=plan)
            await self.nl.sn_notify("ConflictNotify", intent_id,
                                    {"conflictId": cid, "intents": [intent_id] + [p["intentId"] for p in plan],
                                     "resource": f"comm:{cell}/prb", "resolution": "PREEMPTED_LOWER_PRIORITY",
                                     "options": plan}, "ConflictManager", "ServiceOrchestratorAgent", priority=rec.priority)
            return {"ok": True, "preempt": plan, "conflictId": cid}
        others = [i for i in holders if i not in [p["intentId"] for p in plan]]
        options = [{"action": "ACCEPT_PARTIAL", "intentId": intent_id, "maxShare": round(free, 1)},
                   *[{"action": "LOWER_PRIORITY_OR_SHRINK", "intentId": i, "currentShare": holders[i]} for i in others],
                   {"action": "MOVE_TO_OTHER_CELLS", "intentId": intent_id}]
        cid = self._record([intent_id] + others, f"comm:{cell}/prb", "NEEDS_DECISION", options=options)
        await self.nl.sn_notify("ConflictNotify", intent_id,
                                {"conflictId": cid, "intents": [intent_id] + others, "resource": f"comm:{cell}/prb",
                                 "resolution": "NEEDS_DECISION", "options": options},
                                "ConflictManager", "ServiceOrchestratorAgent", priority=rec.priority)
        return {"ok": False, "conflictId": cid, "maxShare": free, "holders": others}

    def _record(self, intents: list[str], resource: str, resolution: str, options=None, notified: bool = True) -> str:
        cid = new_id("cfl")
        self.conflicts.append({"conflictId": cid, "intents": intents, "resource": resource,
                               "resolution": resolution, "options": options or [], "notified": notified})
        return cid

    async def on_alarm(self, topic: str, env: Envelope) -> None:
        b = env.body
        if b.get("type") == "CAPACITY_EXHAUSTED" and len(b.get("affectedIntents", [])) > 1:
            self._record(b["affectedIntents"], b["resourceId"], "DETECTED_FROM_ALARM", notified=False)

    def stats(self) -> dict[str, Any]:
        n = len(self.conflicts)
        auto = sum(1 for c in self.conflicts if c["resolution"] in ("PREEMPTED_LOWER_PRIORITY", "REJECTED_AT_SUBMIT"))
        return {"conflicts": n, "intents": self.intents_seen,
                "conflictsPer100Intents": round(100 * n / max(1, self.intents_seen), 2),
                "resolvedWithoutServiceLayer": round(auto / n, 3) if n else None, "recent": self.conflicts[-10:]}
