"""RAN slice allocator agent (skills ALLOCATE_SLICE, REBALANCE_PRB) - Flow 1.

For each proposal: LSTM forecast + DQN action -> Decision Translation -> NR-02 reserve -> NR-04 validate;
if the predicted KPIs break the intent, re-propose with the feedback (up to MAX_PROPOSALS), else NR-05 apply.
Every step is an MCP tool call on the Network Layer ToolRegistry.
"""
from __future__ import annotations

import contextlib
from typing import Any

from common import config

from ..core import TaskRecord
from .base import Agent


class RanAllocatorAgent(Agent):
    def __init__(self, nl, descriptor, token):
        super().__init__(nl, descriptor, token)
        self.handlers = {"ALLOCATE_SLICE": self.allocate, "REBALANCE_PRB": self.allocate}

    async def _preempt(self, plan: list[dict[str, Any]]) -> list[str]:
        done = []
        for p in plan:
            cfg = await self.call("decision_translate", intentId=p["intentId"], cell=p["cell"], sharePercent=p["to"])
            r = await self.call("nr_config_apply", intentId=p["intentId"], target=p["cell"], config=cfg,
                                origin={"reason": "preempted by higher priority"})
            if r.get("ok"):
                done.append(f"{p['intentId']}@{p['cell']} {p['from']:.0f}%->{p['to']:.0f}%")
        return done

    async def allocate(self, task: TaskRecord) -> dict[str, Any]:
        iid = task.input["intentId"]
        intent = await self.call("intent_get", intentId=iid)
        cells, st = intent["cells"], intent["sliceType"]
        feedback, history, last_pred, preempted = None, [], {}, []
        for attempt in range(1, config.MAX_PROPOSALS + 1):
            async with contextlib.AsyncExitStack() as locks:
                for c in sorted(cells):      # fixed order: no deadlock with other agents
                    await locks.enter_async_context(self.nl.cell_lock(c))
                out, feedback = await self._attempt(iid, intent, cells, st, attempt, feedback, history, last_pred, preempted)
            if out is not None:
                return out
        return {"ok": False, "error": f"PREDICTED_KPIS_BREAK_INTENT after {config.MAX_PROPOSALS} proposals: "
                                      f"{history[-1]['violations'] if history else ''}",
                "predicted": _worst([p for p in last_pred.values() if p]), "history": history,
                "explanation": "pre-validation rejected every proposal"}

    async def _attempt(self, iid, intent, cells, st, attempt, feedback, history, last_pred, preempted):
        """One proposal under the cells' locks -> (result or None to try again, feedback for the next proposal)."""
        proposals = {}
        for cell in cells:
            lstm = await self.call("model_lstm_forecast", cell=cell)
            dqn = await self.call("model_dqn_infer", cell=cell)
            cfg = await self.call("decision_translate", intentId=iid, cell=cell, lstmAllocation=lstm.get("vector"),
                                  dqnAction=dqn.get("action"), attempt=attempt, feedback=feedback)
            proposals[cell] = {"config": cfg, "lstm": lstm, "dqn": dqn}
        items = [{"resourceId": f"comm:{c}", "kind": "PRB_SHARE", "amount": p["config"]["intentShare"], "unit": "percent"}
                 for c, p in proposals.items()]
        rsv = await self.call("nr_resource_reserve", intentId=iid, items=items, priority=intent["priority"])
        if rsv.get("status") != "SUCCESS":
            fixed = False
            for c, p in proposals.items():
                cc = await self.call("conflict_check", intentId=iid, cell=c, sharePercent=p["config"]["intentShare"])
                if cc.get("preempt"):
                    preempted += await self._preempt(cc["preempt"])
                    fixed = True
                elif not cc.get("ok"):
                    return {"ok": False, "error": f"CONFLICT:{cc.get('conflictId')} on {c}: {rsv.get('reason')}",
                            "explanation": f"no PRB share available on {c}; SN-14 NEEDS_DECISION sent"}, None
            if fixed:
                rsv = await self.call("nr_resource_reserve", intentId=iid, items=items, priority=intent["priority"])
            if rsv.get("status") != "SUCCESS":
                return {"ok": False, "error": f"RESERVATION_REJECTED: {rsv.get('reason')}"}, None
        verdicts, feedback = {}, None
        for c, p in proposals.items():
            v = await self.call("nr_config_validate", intentId=iid, target=c, config=p["config"], simulateSec=60)
            verdicts[c] = v
            last_pred[c] = v.get("predicted")
            if not v.get("ok"):
                feedback = {"violations": v.get("violations") or [v.get("reason") or "validation rejected"],
                            "predicted": v.get("predicted")}
                history.append({"attempt": attempt, "cell": c, "share": p["config"]["intentShare"], **feedback})
                break
        if feedback:
            return None, feedback
        applied = []
        for c, p in proposals.items():
            r = await self.call("nr_config_apply", intentId=iid, target=c, config=p["config"],
                                origin={"decision": {"model": p["dqn"].get("model"), "action": p["dqn"].get("action"),
                                                     "mode": p["dqn"].get("mode")},
                                        "forecast": {"model": p["lstm"].get("model"), "allocation": p["lstm"].get("allocation")}})
            if not r.get("ok"):
                if "exceeds" in str(r.get("reason")):      # cell changed under us: re-propose on fresh state
                    history.append({"attempt": attempt, "cell": c, "applyFailed": r.get("reason")})
                    return None, {"violations": [f"apply rejected: {r.get('reason')}"]}
                return {"ok": False, "error": f"APPLY_FAILED on {c}: {r.get('reason')}"}, None
            applied.append({"cell": c, "configId": r["configId"], "share": p["config"]["intentShare"],
                            "allocation": p["config"]["allocation"]})
        any_p = next(iter(proposals.values()))
        worst = _worst([verdicts[c]["predicted"] for c in proposals])
        expl = (f"{'Dedicated' if any_p['config']['isolation'] == 'dedicated' else 'Shared'} {st} slice on "
                f"{len(cells)} cell(s), {any_p['config']['intentShare']:.0f}% PRB share; "
                f"DQN action {any_p['dqn'].get('action')} ({any_p['dqn'].get('mode')}), "
                f"LSTM {any_p['lstm'].get('model')}; proposal {attempt}/{config.MAX_PROPOSALS}"
                + (f"; preempted {', '.join(preempted)}" if preempted else ""))
        return {"ok": True, "configs": applied, "predicted": worst, "attempts": attempt, "history": history,
                "explanation": expl}, None


def _worst(preds: list[dict[str, Any]]) -> dict[str, Any]:
    if not preds:
        return {}
    out = {}
    for k in preds[0]:
        vals = [p[k] for p in preds if k in p]
        out[k] = max(vals) if k.startswith("latency") or k == "prb_utilization" else min(vals)
    return out
