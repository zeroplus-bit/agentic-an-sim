"""Decision Translation (DecisionTranslator): turn agent/model decisions into concrete NR config
(PRB share, slice params). Starts from dqn_agent.py::action_to_allocation."""
from __future__ import annotations

import math
import re
from typing import Any, Optional

import numpy as np

from ..task_execution.ml import action_to_allocation

SLICES = ("eMBB", "URLLC", "mMTC")
CELL_CAPACITY_MBPS = 400.0
DEFAULT_GBR = {"eMBB": 100.0, "URLLC": 20.0, "mMTC": 5.0}
DEFAULT_LATENCY = {"eMBB": 50.0, "URLLC": 10.0, "mMTC": 200.0}
TARGET_LOAD = {"URLLC": 0.72, "eMBB": 0.8, "mMTC": 0.85}
FIVE_QI = {"eMBB": 9, "URLLC": 82, "mMTC": 8}
SST = {"eMBB": 1, "URLLC": 2, "mMTC": 3}
DEMAND_FACTOR = 1.15          # the twin's per-intent demand = gbr * 1.15


def _exp(intent: dict[str, Any], kpi: str) -> Optional[float]:
    for e in intent.get("expectations") or []:
        if e["kpi"] == kpi:
            return float(e["value"])
    return None


class DecisionTranslator:
    def __init__(self, nl):
        self.nl = nl

    @staticmethod
    def gbr(intent: dict[str, Any]) -> float:
        st = intent["semantic"]["serviceCategory"]
        return _exp(intent, "bandwidth_mbps") or DEFAULT_GBR[st]

    def estimate_share(self, intent: dict[str, Any], target_load: float) -> float:
        demand = self.gbr(intent) * DEMAND_FACTOR
        return float(min(80.0, max(2.0, math.ceil(demand / (CELL_CAPACITY_MBPS * target_load) * 100))))

    def translate(self, intent: dict[str, Any], cell: str, cell_info: dict[str, Any], priority: int = 3,
                  lstm_alloc: Optional[list[float]] = None, dqn_alloc: Optional[list[float]] = None, attempt: int = 1,
                  target_load: Optional[float] = None, feedback: Optional[dict[str, Any]] = None,
                  share: Optional[float] = None, current_isolation: Optional[str] = None) -> dict[str, Any]:
        st = intent["semantic"]["serviceCategory"]
        i = SLICES.index(st)
        tl = target_load or (TARGET_LOAD[st] - 0.08 * (attempt - 1))
        why = []
        if feedback:
            v = " ".join(feedback.get("violations") or [])
            if "latency" in v or "reliability" in v:
                tl -= 0.1
                why.append("lower target load after latency/reliability miss")
            if "bandwidth" in v:
                tl -= 0.05
                why.append("more headroom after bandwidth miss")
        tl = max(0.35, tl)
        share = float(share) if share else self.estimate_share(intent, tl)

        cur = cell_info.get("allocation") or {"eMBB": .4, "URLLC": .4, "mMTC": .2}
        vecs, weights = [np.array([cur[s] for s in SLICES])], [0.5]
        if lstm_alloc is not None:
            vecs.append(np.asarray(lstm_alloc, float))
            weights.append(0.25)
            why.append("LSTM forecast blended")
        if dqn_alloc is not None:
            vecs.append(np.asarray(dqn_alloc, float))
            weights.append(0.25)
            why.append("DQN action blended")
        a = sum(w * v for w, v in zip(weights, vecs)) / sum(weights)
        a = a / a.sum()
        iid = intent["intentId"]
        ded_other = sum(x["sharePercent"] for k, x in (cell_info.get("intents") or {}).items()
                        if k != iid and not k.startswith(f"{iid}@") and x.get("slice") == st) / 100.0
        need = ded_other + share / 100.0 + 0.10            # keep 10% of the cell for the slice's other users
        if a[i] < need:
            a[i] = min(0.9, need)
            rest = [j for j in range(3) if j != i]
            remain = 1.0 - a[i]
            tot = sum(a[j] for j in rest)
            for j in rest:
                a[j] = max(0.05, remain * a[j] / tot)
            a = a / a.sum()
            if a[i] < need - 1e-6:          # renormalisation ate the headroom
                a[i] = need
                for j in rest:
                    a[j] = (1 - need) / 2
            why.append(f"{st} slice raised to {a[i]*100:.0f}% to host {share:.0f}% dedicated share")
        digits = re.sub(r"\D", "", iid) or "1"
        constraints = intent.get("constraints") or {}
        iso = constraints.get("isolation") or ("dedicated" if st == "URLLC" else "shared")
        if iso == "shared" and current_isolation == "dedicated":
            iso = "dedicated"          # an earlier escalation stands (a resize or reclaim must not undo it)
            why.append("keeps the dedicated share an earlier KPI miss escalated to")
        elif iso == "shared" and feedback and feedback.get("violations"):
            iso = "dedicated"          # a shared slice missed its KPIs -> escalate to a dedicated PRB share
            why.append("escalated shared -> dedicated after KPI miss")
        return {
            "slice": {"sNssai": {"sst": SST[st], "sd": f"{int(digits) % 0xFFFFFF:06X}"}, "type": st},
            "allocation": {s: round(float(a[k]), 4) for k, s in enumerate(SLICES)},
            "qos": {"5qi": FIVE_QI[st], "latencyBudgetMs": _exp(intent, "latency_ms") or DEFAULT_LATENCY[st],
                    "gbrMbps": self.gbr(intent)},
            "scheduling": {"policy": "PRIORITY" if (st == "URLLC" or priority <= 2) else "WFQ",
                           "preemption": st == "URLLC" or priority <= 2},
            "intentShare": share, "isolation": "dedicated" if iso == "dedicated" else "shared",
            "demandMbps": self.gbr(intent), "ueGroup": (intent.get("context") or {}).get("ueGroup"),
            "targetLoad": round(tl, 3), "rationale": "; ".join(why) or "rule-based sizing",
        }

    async def translate_for_agent(self, a: dict[str, Any]) -> dict[str, Any]:
        rec = self.nl.intents.get(a["intentId"])
        if not rec:
            return {"error": "unknown intent"}
        info = await self.nl.resource_ctrl.cell_info(a["cell"], max_age=0.0)
        dqn_alloc = action_to_allocation(int(a["dqnAction"])).tolist() if a.get("dqnAction") is not None else None
        last = next((c for c in reversed(rec.configs) if c["cell"] == a["cell"]), None)
        bound = (info.get("intents") or {}).get(rec.intent_id) or {}
        cur_iso = bound.get("isolation") or (last or {}).get("isolation")
        cfg = self.translate(rec.intent, a["cell"], info, rec.priority, a.get("lstmAllocation"), dqn_alloc,
                             int(a.get("attempt", 1)), a.get("targetLoad"), a.get("feedback"), a.get("sharePercent"),
                             current_isolation=cur_iso)
        off = a.get("offload")
        if off is not None:            # mobility load balancing: {} = undo, {target, fraction, share} = steer
            mob: dict[str, Any] = {}
            if off:
                tinfo = await self.nl.resource_ctrl.cell_info(off["target"], max_age=0.0)
                tcfg = self.translate(rec.intent, off["target"], tinfo, rec.priority, share=float(off["share"]),
                                      current_isolation="dedicated")
                mob[off["target"]] = {"fraction": round(float(off["fraction"]), 3), "share": float(off["share"]),
                                      "allocation": tcfg["allocation"]}
                cfg["rationale"] += f"; steer {float(off['fraction']):.0%} of UEs to {off['target']} ({off['share']:.0f}% PRB)"
            cfg["mobility"] = {"offload": mob}
        cfg["currentSharePercent"] = rec.shares.get(a["cell"])
        return cfg
