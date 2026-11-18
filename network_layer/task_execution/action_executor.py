"""Action Execution (ActionExecutor): send NR configuration (NR-05) and handle rollback (NR-06), including the
30 s post-change watch of Flow 3 step 6. Starts from slice_selection/engine.py::deploy_slice."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from common import config
from common.envelope import Envelope
from common.refpoint import new_id

log = logging.getLogger("action_executor")


class ActionExecutor:
    def __init__(self, nl):
        self.nl = nl
        self.applied = 0
        self.rolled_back = 0

    async def apply(self, a: dict[str, Any], agent_id: str) -> dict[str, Any]:
        iid = a.get("intentId")
        rec = self.nl.intents.get(iid or "")
        cell = a["target"].split(":", 1)[-1]
        cfg = {k: v for k, v in a["config"].items() if k not in ("rationale", "currentSharePercent")}
        body = {"configId": a.get("configId") or new_id("cfg"),
                "reservationId": a.get("reservationId") or (rec.reservation_id if rec else None),
                "intentId": iid, "target": f"comm:{cell}", "config": cfg,
                "origin": {"agentId": agent_id, **(a.get("origin") or {})}}
        before = self._compliance(rec, cell, since=time.time() - 15) if rec else None
        r = await self.nl.nr_call("ConfigApplyRequest", "POST", "/nr/v1/configs", body, iid or agent_id, "ActionExecutor",
                                  priority=rec.priority if rec else 3)
        if r.status != "SUCCESS":
            return {"ok": False, "status": r.status, "reason": r.result.reason if r.result else None}
        self.applied += 1
        entry = {"configId": body["configId"], "cell": cell, "rollbackToken": r.body.get("rollbackToken"),
                 "share": cfg.get("intentShare"), "allocation": cfg.get("allocation"), "isolation": cfg.get("isolation"),
                 "at": time.time(),
                 "by": agent_id, "origin": body["origin"]}
        if "mobility" in cfg:
            entry["mobility"] = cfg["mobility"]
        if rec:
            rec.configs.append(entry)
            if cfg.get("intentShare") is not None:
                rec.shares[cell] = float(cfg["intentShare"])
            if "mobility" in cfg:
                self._set_steering(rec, cell, cfg["mobility"])
            rec.last_action_at = time.time()
            if rec.state == "DEGRADED" or (a.get("origin") or {}).get("reclaim"):
                # corrective change or scale-down -> watch and roll back if KPIs get worse
                self.nl.spawn(self._watch(rec, entry, before))
        return {"ok": True, "status": "SUCCESS", "configId": body["configId"], "rollbackToken": entry["rollbackToken"],
                "appliedAt": r.body.get("appliedAt")}

    async def rollback(self, config_id: str, corr: str, reason: Optional[str] = None, record: bool = True) -> Envelope:
        r = await self.nl.nr_call("ConfigRollbackRequest", "POST", f"/nr/v1/configs/{config_id}:rollback",
                                  {"configId": config_id, "rollbackToken": f"rb-{config_id}", "reason": reason},
                                  corr, "ActionExecutor")
        if r.status == "SUCCESS":
            self.rolled_back += 1
        if record and r.status == "SUCCESS":
            rec = self.nl.intents.get(corr)
            if rec:
                gone = [c for c in rec.configs if c["configId"] == config_id]
                rec.configs = [c for c in rec.configs if c["configId"] != config_id]
                for g in gone:
                    prev = [c for c in rec.configs if c["cell"] == g["cell"] and c.get("share") is not None]
                    if prev:
                        rec.shares[g["cell"]] = float(prev[-1]["share"])
                    else:
                        rec.shares.pop(g["cell"], None)
                    if "mobility" in g:      # restore the steering that was in place before this config
                        mob = [c["mobility"] for c in rec.configs if c["cell"] == g["cell"] and "mobility" in c]
                        self._set_steering(rec, g["cell"], mob[-1] if mob else {"offload": {}})
        return r

    @staticmethod
    def _set_steering(rec, cell: str, mobility: dict[str, Any]) -> None:
        old = rec.steering.pop(cell, None)
        if old:
            rec.shares.pop(old["target"], None)
        for t, o in (mobility.get("offload") or {}).items():
            rec.steering[cell] = {"target": t, "fraction": float(o["fraction"]), "share": float(o["share"])}
            rec.shares[t] = float(o["share"])

    @staticmethod
    def _compliance(rec, cell: str, since: float) -> Optional[float]:
        if rec is None:
            return None
        xs = [s for s in rec.samples.get(cell, []) if s["t"] >= since]
        if not xs:
            return None
        ok = sum(1 for s in xs if rec_ok(rec, s))
        return ok / len(xs)

    async def _watch(self, rec, entry: dict[str, Any], before: Optional[float]) -> None:
        await asyncio.sleep(config.ROLLBACK_WATCH_SEC)
        if entry not in rec.configs or rec.state in ("TERMINATED", "REJECTED", "FAILED"):
            return
        after = self._compliance(rec, entry["cell"], since=entry["at"] + 2)
        if before is None or after is None:
            return
        if after < before - 0.1:
            log.warning("intent %s: KPIs worse after %s (%.2f -> %.2f); rolling back", rec.intent_id, entry["configId"],
                        before, after, extra={"intentId": rec.intent_id, "stage": "EXECUTE"})
            await self.rollback(entry["configId"], rec.intent_id, f"KPIs worse within {config.ROLLBACK_WATCH_SEC:.0f}s")
            rec.explanation = f"rolled back {entry['configId']}: compliance {before:.2f} -> {after:.2f}"
            self.nl.perf_ctrl.on_rolled_back(rec.intent_id)


def rec_ok(rec, sample: dict[str, Any]) -> bool:
    from ..task_control.perf_monitor import PerformanceMonitor
    return PerformanceMonitor.evaluate(rec.expectations, sample)[0]
