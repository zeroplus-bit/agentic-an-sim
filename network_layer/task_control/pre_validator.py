"""Pre-validation (PreValidator): dry-run a config on a twin copy (NR-04) and reject it if predicted KPIs
break the intent. Also answers SN-05 IntentFeasibilityQuery. Starts from ndt/simulator.py (what-if mode)."""
from __future__ import annotations

from typing import Any, Optional

from common.envelope import Envelope, respond
from common.refpoint import new_id
from common.sn_messages import Expectation, Intent

PRICE_PER_SHARE_HOUR = 5.25      # cost units per PRB % per cell per hour (simulated vendor offer)


def predicted_value(exp: dict[str, Any], predicted: dict[str, Any]) -> Optional[float]:
    k = exp["kpi"]
    if k == "latency_ms" and exp.get("percentile"):
        return predicted.get("latency_ms_p99", predicted.get("latency_ms"))
    return predicted.get(k)


def compare(expectations: list[dict[str, Any]], predicted: dict[str, Any]) -> list[str]:
    out = []
    for e in expectations:
        v = predicted_value(e, predicted)
        if v is None:
            continue
        if not Expectation(**e).met_by(v):
            out.append(f"{e['kpi']} {v:g} !{e['op']} {e['value']:g}")
    return out


class PreValidator:
    def __init__(self, nl):
        self.nl = nl

    async def validate(self, body: dict[str, Any], corr: str, priority: int = 3) -> Envelope:
        return await self.nl.nr_call("ConfigValidateRequest", "POST", "/nr/v1/configs:validate", body, corr,
                                     "PreValidator", priority=priority)

    async def validate_for_agent(self, a: dict[str, Any], agent_id: str) -> dict[str, Any]:
        rec = self.nl.intents.get(a.get("intentId") or "")
        target = a["target"] if a["target"].startswith("comm:") else f"comm:{a['target']}"
        body = {"configId": new_id("cfg"), "reservationId": a.get("reservationId") or (rec.reservation_id if rec else None),
                "intentId": a.get("intentId"), "target": target, "config": a["config"],
                "origin": {"agentId": agent_id, **(a.get("origin") or {})}, "simulateSec": int(a.get("simulateSec", 60))}
        r = await self.validate(body, a.get("intentId") or agent_id, rec.priority if rec else 3)
        if r.status != "SUCCESS":
            return {"ok": False, "status": r.status, "reason": r.result.reason if r.result else None, "configId": body["configId"]}
        pred = r.body.get("predicted", {})
        viol = compare(rec.expectations, pred) if rec else []
        return {"ok": not viol, "status": "SUCCESS", "configId": body["configId"], "predicted": pred,
                "violations": viol, "sideEffects": r.body.get("sideEffects", [])}

    async def feasibility(self, env: Envelope) -> Envelope:
        _, err = self.nl.auth.verify_sn(env, "sn:intent:read")
        if err:
            return respond(env, "UNAUTHORIZED", reason=err)
        intent = Intent.model_validate(env.body["intent"]).model_dump(mode="json", exclude_none=True)
        cells = (intent.get("context") or {}).get("area", {}).get("cells") or await self.nl.mediator._pick_cells(intent)
        tr = self.nl.translator
        attempts, best = [], None
        for tl in (None, 0.6, 0.5):
            preds, viols, share = [], [], None
            for cell in cells[:2]:
                info = await self.nl.resource_ctrl.cell_info(cell)
                if not info:
                    return respond(env, "SUCCESS", {"feasible": False, "reason": f"unknown cell {cell}"})
                cfg = tr.translate(intent, cell, info, env.header.priority, target_load=tl)
                share = cfg["intentShare"]
                r = await self.validate({"configId": new_id("cfg-feas"), "intentId": intent["intentId"],
                                         "target": f"comm:{cell}", "config": cfg, "simulateSec": 30},
                                        env.header.correlationId, env.header.priority)
                if r.status != "SUCCESS":
                    viols.append(r.result.reason if r.result else "validation rejected")
                    continue
                preds.append(r.body["predicted"])
                viols += compare(intent["expectations"], r.body["predicted"])
            worst = _worst(preds)
            item = {"targetLoad": tl or "default", "sharePercentPerCell": share, "predicted": worst, "violations": viols,
                    "estimatedCostPerHour": round((share or 0) * len(cells) * PRICE_PER_SHARE_HOUR, 2)}
            attempts.append(item)
            if not viols:
                best = item
                break
        ok = best is not None
        chosen = best or attempts[0]
        max_cost = (intent.get("constraints") or {}).get("maxCostPerHour")
        if ok and max_cost and chosen["estimatedCostPerHour"] > max_cost:
            ok = False
            chosen["violations"] = [f"cost {chosen['estimatedCostPerHour']} > maxCostPerHour {max_cost}"]
        conf = round(max(0.3, 0.92 - 0.1 * (len(attempts) - 1) - (0.0 if ok else 0.3)), 2)
        return respond(env, "SUCCESS", {"feasible": ok, "predicted": chosen["predicted"], "confidence": conf,
                                        "estimatedCostPerHour": chosen["estimatedCostPerHour"], "cells": cells,
                                        "violations": chosen["violations"],
                                        "alternatives": [a for a in attempts if a is not chosen]})


def _worst(preds: list[dict[str, Any]]) -> dict[str, Any]:
    if not preds:
        return {}
    out = {}
    for k in preds[0]:
        vals = [p[k] for p in preds if k in p]
        out[k] = max(vals) if k.startswith("latency") or k == "prb_utilization" else min(vals)
    return out
