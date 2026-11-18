"""KPI assurance agent (skill ASSURE_KPI) - Flow 3 closed loop.

mode "correct" (drift): read the telemetry window, ask the DQN (and LSTM) for a new allocation, size a larger PRB
share from the measured load, check conflicts (may preempt lower-priority intents -> SN-14), validate (NR-04) and
apply (NR-05). ActionExecutor watches the change and rolls back (NR-06) if KPIs get worse.
  * the drifting cell is DOWN -> fail over: correct the intent's surviving cells, which now carry the handed-over
    load; all cells down -> CAPACITY_CEILING (nothing a PRB change can fix) for escalation.
  * the share cannot grow any further (per-intent cap or cell budget) -> CAPACITY_CEILING with the shortfall, so
    PerformanceController escalates at once instead of retrying.
  * before CAPACITY_CEILING is reported, mobility load balancing is tried: steer up to 40% of the intent's UEs in
    the cell to the least loaded neighbour cell with free PRB budget (a dedicated share there), validated on the
    twin together with the source cell; only if no neighbour can take them is the Service Layer asked.
mode "reclaim" (KPIs met, load well under target): first undo steering once the UEs fit back into the own share,
then shrink the share back toward what the measured load needs, never below the share the intent was first
fulfilled with (scale-down; returns PRBs to the cell).
Every read-translate-reserve-validate-apply runs under the cell's lock (no lost updates between agents).
"""
from __future__ import annotations

import asyncio
import math
from typing import Any, Optional

import numpy as np

from ..core import TaskRecord
from .base import Agent

TARGET_LOAD = 0.65
MAX_INTENT_SHARE = 80.0          # translator keeps >= 10% of the cell for the slice's other users
MAX_OFFLOAD = 0.4                # edge UEs that can be steered (the twin enforces the same bound)


class AssuranceAgent(Agent):
    def __init__(self, nl, descriptor, token):
        super().__init__(nl, descriptor, token)
        self.handlers = {"ASSURE_KPI": self.assure}

    async def _load(self, iid: str, cell: str) -> float:
        win = await self.call("nr_telemetry_window", intentId=iid, cell=cell, n=10)
        samples = (win.get("samples") or {}).get(cell, [])
        loads = [s["load"] for s in samples[-3:] if "load" in s]
        return float(np.max(loads)) if loads else 1.2       # recent peak: a mean lags a ramping surge

    async def assure(self, task: TaskRecord) -> dict[str, Any]:
        inp = task.input
        mode = inp.get("mode")
        if mode == "watch":
            return {"ok": True, "explanation": "watching"}
        iid, cell = inp["intentId"], inp["cell"]
        rec = self.nl.intents.get(iid)
        if mode == "reclaim":
            return await self.reclaim(iid, cell)
        down = set(rec.down_cells) if rec else set()
        if cell in down or inp.get("cause", "").startswith("ALARM_CELL_DOWN"):
            survivors = [c for c in (rec.cells if rec else []) if c not in down and c != cell]
            if not survivors:
                return {"ok": False, "error": "CAPACITY_CEILING: every cell of the intent is down",
                        "ceiling": {"cell": cell, "reason": "ALL_CELLS_DOWN"}}
            notes, last = [f"{cell} down: failing over to {', '.join(survivors)}"], None
            for sc in survivors:
                load = await self._load(iid, sc)
                if load < TARGET_LOAD + 0.1:
                    notes.append(f"{sc} load {load:.2f} already within target")
                    continue
                last = await self._correct(iid, sc, inp, load)
                notes.append(last.get("explanation") or last.get("error", ""))
                if not last.get("ok"):
                    return {**last, "explanation": "; ".join(notes)}
            return {"ok": True, "explanation": "; ".join(notes), **({"configId": last["configId"],
                    "episode": last.get("episode")} if last and last.get("ok") else {})}
        load = await self._load(iid, cell)
        if load < 0.5 and (rec.shares.get(cell) if rec else 0):
            return {"ok": False, "error": f"load {load:.2f} is low; drift not caused by PRB shortage ({inp.get('cause')})"}
        out = await self._correct(iid, cell, inp, load)
        if out.get("ok") and rec:
            # the intent's other cells usually see the same surge: correct them in the same task instead of
            # waiting for their own drift round (one recovery instead of one per cell)
            notes = [out["explanation"]]
            for oc in [c for c in rec.cells if c != cell and c not in down]:
                ol = await self._load(iid, oc)
                if ol > TARGET_LOAD + 0.05:
                    o2 = await self._correct(iid, oc, inp, ol)
                    notes.append(o2.get("explanation") or o2.get("error", ""))
            out["explanation"] = "; ".join(notes)
        return out

    async def _correct(self, iid: str, cell: str, inp: dict[str, Any], load: float) -> dict[str, Any]:
        async with self.nl.cell_lock(cell):
            intent = await self.call("intent_get", intentId=iid)
            cur = float((intent.get("shares") or {}).get(cell) or 0.0)
            if cur >= MAX_INTENT_SHARE - 0.5:
                need = math.ceil(cur * load / TARGET_LOAD)
                lb = await self._offload(iid, cell, load, cur, intent)
                if lb.get("ok"):
                    return lb
                return {"ok": False, "lbTried": lb.get("explanation"),
                        "error": f"CAPACITY_CEILING: {cell} share {cur:.0f}% is at the {MAX_INTENT_SHARE:.0f}% cap; "
                                 f"load {load:.2f} needs ~{need}%; load balancing: {lb.get('explanation')}",
                        "ceiling": {"cell": cell, "share": cur, "needed": need, "max": MAX_INTENT_SHARE, "load": round(load, 3)}}
            dqn = await self.call("model_dqn_infer", cell=cell, violations=len(inp.get("violations") or []))
            lstm = await self.call("model_lstm_forecast", cell=cell)
            new = float(min(MAX_INTENT_SHARE, max(cur + 5, math.ceil(cur * load / TARGET_LOAD))))
            notes = []
            rec = self.nl.intents.get(iid)
            free = self.nl.conflict.free(cell, exclude=iid)
            lower = [h for h in self.nl.conflict.used(cell, exclude=iid)
                     if rec and self.nl.intents[h].priority > rec.priority]
            if new > free + 1e-9 and not lower and free <= cur + 0.5:
                lb = await self._offload(iid, cell, load, cur, intent)
                if lb.get("ok"):
                    return lb
                notes.append(f"cell budget exhausted ({free:.0f}% free); load balancing: {lb.get('explanation')}")
            for attempt in range(1, 4):
                cc = await self.call("conflict_check", intentId=iid, cell=cell, sharePercent=new)
                if not cc.get("ok"):
                    cap = float(cc.get("maxShare") or 0)
                    if cap <= cur + 0.5:
                        return {"ok": False, "error": f"NEEDS_DECISION ({cc.get('conflictId')}): no free PRB on {cell}",
                                "explanation": "conflict escalated to Service Layer (SN-14)",
                                "ceiling": {"cell": cell, "share": cur, "needed": new, "max": cap, "load": round(load, 3),
                                            "conflictId": cc.get("conflictId")}}
                    new = math.floor(cap)
                    notes.append(f"capped at free share {new:.0f}%")
                for p in cc.get("preempt") or []:
                    pcfg = await self.call("decision_translate", intentId=p["intentId"], cell=p["cell"], sharePercent=p["to"])
                    await self.call("nr_config_apply", intentId=p["intentId"], target=p["cell"], config=pcfg,
                                    origin={"reason": f"preempted by {iid}"})
                    notes.append(f"preempted {p['intentId']} {p['from']:.0f}%->{p['to']:.0f}%")
                cfg = await self.call("decision_translate", intentId=iid, cell=cell, lstmAllocation=lstm.get("vector"),
                                      dqnAction=dqn.get("action"), sharePercent=new,
                                      feedback={"violations": inp.get("violations") or []})
                shares = dict(intent.get("shares") or {})
                shares[cell] = new
                items = [{"resourceId": f"comm:{c}", "kind": "PRB_SHARE", "amount": s, "unit": "percent"} for c, s in shares.items()]
                rsv = await self.call("nr_resource_reserve", intentId=iid, items=items, priority=intent["priority"])
                if rsv.get("status") != "SUCCESS":
                    notes.append(f"reserve {new:.0f}% rejected: {rsv.get('reason')}")
                    new = max(cur + 2, new - 5)
                    continue
                v = await self.call("nr_config_validate", intentId=iid, target=cell, config=cfg, simulateSec=30)
                if not v.get("ok"):
                    notes.append(f"{new:.0f}% predicted {v.get('violations') or v.get('reason')}")
                    if new >= MAX_INTENT_SHARE - 0.5:
                        break
                    new = min(MAX_INTENT_SHARE, new + 5)
                    continue
                r = await self.call("nr_config_apply", intentId=iid, target=cell, config=cfg,
                                    origin={"decision": {"model": dqn.get("model"), "action": dqn.get("action"),
                                                         "mode": dqn.get("mode")}, "reason": inp.get("cause")})
                if not r.get("ok"):
                    return {"ok": False, "error": f"apply failed: {r.get('reason')}"}
                expl = (f"{cell}: PRB share {cur:.0f}% -> {new:.0f}% (measured load {load:.2f}, target {TARGET_LOAD}); "
                        f"DQN action {dqn.get('action')} ({dqn.get('mode')}); predicted {v.get('predicted')}"
                        + (f"; {'; '.join(notes)}" if notes else ""))
                return {"ok": True, "configId": r["configId"], "explanation": expl,
                        "episode": {"state": dqn.get("state"), "action": dqn.get("action"), "shareDelta": new - cur}}
            need = math.ceil(cur * load / TARGET_LOAD)
            if new >= MAX_INTENT_SHARE - 0.5:
                lb = await self._offload(iid, cell, load, cur, intent)
                if lb.get("ok"):
                    return lb
                notes.append(f"load balancing: {lb.get('explanation')}")
            return {"ok": False, "error": f"CAPACITY_CEILING: no validated correction up to {new:.0f}% on {cell} "
                                          f"(load {load:.2f} needs ~{need}%)", "explanation": "; ".join(notes),
                    "ceiling": {"cell": cell, "share": cur, "needed": need, "max": new, "load": round(load, 3)}}

    async def _offload(self, iid: str, cell: str, load: float, cur: float, intent: dict[str, Any]) -> dict[str, Any]:
        """Mobility load balancing: steer edge UEs of the intent in `cell` to a neighbour cell."""
        rec = self.nl.intents.get(iid)
        frac = min(MAX_OFFLOAD, 1.0 - TARGET_LOAD / max(load, 1e-6))
        if frac < 0.05:
            return {"ok": False, "explanation": f"load {load:.2f} too close to target for steering"}
        inv = await self.call("nr_resource_inventory", type="comm")
        cells = {x["resourceId"].split(":", 1)[1]: x for x in inv.get("resources", [])}
        own = set(rec.cells) if rec else set()
        cand = [n for n in cells.get(cell, {}).get("neighbours", []) if n in cells and cells[n]["up"] and n not in own]
        cand.sort(key=lambda n: (cells[n]["prbUtilization"], -cells[n]["freeSharePercent"]))
        tried = []
        for t in cand:
            free = min(self.nl.conflict.free(t, exclude=iid), cells[t]["freeSharePercent"])
            share = float(math.ceil(frac * load * cur / TARGET_LOAD))
            lock = self.nl.cell_lock(t)
            try:
                await asyncio.wait_for(lock.acquire(), timeout=2.0)
            except asyncio.TimeoutError:
                tried.append(f"{t} busy")
                continue
            try:
                for _ in range(3):
                    if share > free:
                        tried.append(f"{t} free {free:.0f}% < {share:.0f}%")
                        break
                    off = {"target": t, "fraction": round(frac, 3), "share": share}
                    cfg = await self.call("decision_translate", intentId=iid, cell=cell, sharePercent=cur, offload=off)
                    shares = dict(intent.get("shares") or {})
                    shares[t] = share
                    items = [{"resourceId": f"comm:{c}", "kind": "PRB_SHARE", "amount": s_, "unit": "percent"}
                             for c, s_ in shares.items()]
                    rsv = await self.call("nr_resource_reserve", intentId=iid, items=items, priority=intent["priority"])
                    if rsv.get("status") != "SUCCESS":
                        tried.append(f"{t} reserve {share:.0f}%: {rsv.get('reason')}")
                        break
                    v = await self.call("nr_config_validate", intentId=iid, target=cell, config=cfg, simulateSec=30)
                    if not v.get("ok"):
                        tried.append(f"{t} {share:.0f}%: predicted {v.get('violations') or v.get('reason')}")
                        share = float(math.ceil(share * 1.2))      # edge UEs need more PRBs than the estimate
                        continue
                    r = await self.call("nr_config_apply", intentId=iid, target=cell, config=cfg,
                                        origin={"reason": "mobility load balancing", "offload": off})
                    if not r.get("ok"):
                        tried.append(f"{t} apply: {r.get('reason')}")
                        break
                    return {"ok": True, "configId": r["configId"], "steered": off,
                            "explanation": f"{cell} at {cur:.0f}% cap (load {load:.2f}): steered {frac:.0%} of UEs to "
                                           f"neighbour {t} with {share:.0f}% PRB; predicted {v.get('predicted')}"}
            finally:
                lock.release()
        return {"ok": False, "explanation": "no neighbour could take the load: " + ("; ".join(tried) or "no candidate")}

    async def _undo_offload(self, iid: str, cell: str, steer: dict[str, Any], load: float, cur: float,
                            intent: dict[str, Any]) -> Optional[dict[str, Any]]:
        from ..task_control.decision_translator import TARGET_LOAD as PROVISION_LOAD
        rec = self.nl.intents.get(iid)
        total = load / (1.0 - steer["fraction"])        # load if the steered UEs come back
        if cur * total / PROVISION_LOAD[rec.slice_type if rec else "eMBB"] > cur:
            return None
        cfg = await self.call("decision_translate", intentId=iid, cell=cell, sharePercent=cur, offload={})
        v = await self.call("nr_config_validate", intentId=iid, target=cell, config=cfg, simulateSec=30)
        if not v.get("ok"):
            return {"ok": True, "noop": True, "explanation": f"{cell}: steering kept, return predicted "
                                                             f"{v.get('violations') or v.get('reason')}"}
        r = await self.call("nr_config_apply", intentId=iid, target=cell, config=cfg,
                            origin={"reason": "reclaim: steering undone", "reclaim": True})
        if not r.get("ok"):
            return {"ok": False, "error": f"undo steering failed: {r.get('reason')}"}
        shares = {c: s_ for c, s_ in (intent.get("shares") or {}).items() if c != steer["target"]}
        items = [{"resourceId": f"comm:{c}", "kind": "PRB_SHARE", "amount": s_, "unit": "percent"} for c, s_ in shares.items()]
        await self.call("nr_resource_reserve", intentId=iid, items=items, priority=intent["priority"])
        return {"ok": True, "configId": r["configId"],
                "explanation": f"{cell}: steered UEs back from {steer['target']} (load {total:.2f} if all served here); "
                               f"released {steer['share']:.0f}% on {steer['target']}"}

    async def reclaim(self, iid: str, cell: str) -> dict[str, Any]:
        rec = self.nl.intents.get(iid)
        steer = dict(rec.steering.get(cell)) if rec and rec.steering.get(cell) else None
        if steer:
            async with self.nl.cell_lock(cell):
                intent = await self.call("intent_get", intentId=iid)
                cur = float((intent.get("shares") or {}).get(cell) or 0.0)
                load = await self._load(iid, cell)
                out = await self._undo_offload(iid, cell, steer, load, cur, intent)
            return out or {"ok": True, "noop": True, "explanation": f"{cell}: steering still needed"}
        async with self.nl.cell_lock(cell):
            intent = await self.call("intent_get", intentId=iid)
            cur = float((intent.get("shares") or {}).get(cell) or 0.0)
            floor_ = float((rec.baseline_shares if rec else {}).get(cell, 0.0))
            load = await self._load(iid, cell)
            from ..task_control.decision_translator import TARGET_LOAD as PROVISION_LOAD
            target = float(max(floor_, math.ceil(cur * load / PROVISION_LOAD[rec.slice_type if rec else "eMBB"])))
            if target > cur - 3:
                return {"ok": True, "explanation": f"{cell}: nothing to reclaim (share {cur:.0f}%, load {load:.2f})", "noop": True}
            cfg = await self.call("decision_translate", intentId=iid, cell=cell, sharePercent=target)
            v = await self.call("nr_config_validate", intentId=iid, target=cell, config=cfg, simulateSec=30)
            if not v.get("ok"):
                return {"ok": True, "noop": True, "explanation": f"{cell}: reclaim to {target:.0f}% predicted "
                                                                 f"{v.get('violations') or v.get('reason')}; kept {cur:.0f}%"}
            shares = dict(intent.get("shares") or {})
            shares[cell] = target
            items = [{"resourceId": f"comm:{c}", "kind": "PRB_SHARE", "amount": s, "unit": "percent"} for c, s in shares.items()]
            r = await self.call("nr_config_apply", intentId=iid, target=cell, config=cfg,
                                origin={"reason": "reclaim", "reclaim": True})
            if not r.get("ok"):
                return {"ok": False, "error": f"reclaim apply failed: {r.get('reason')}"}
            await self.call("nr_resource_reserve", intentId=iid, items=items, priority=intent["priority"])
            return {"ok": True, "configId": r["configId"],
                    "explanation": f"{cell}: reclaimed PRB share {cur:.0f}% -> {target:.0f}% (load {load:.2f}, floor {floor_:.0f}%)"}

    async def on_negotiate(self, holder_intent: str, cell: str, ask_share: float) -> dict[str, Any]:
        rec = self.nl.intents.get(holder_intent)
        if not rec or rec.state != "FULFILLED":
            return {"accepted": False, "reason": "holder not stable"}
        loads = [s.get("load", 1) for s in list(rec.samples.get(cell, []))[-5:]]
        if loads and max(loads) < 0.6:
            give = min(ask_share, rec.shares.get(cell, 0) * (1 - max(loads) / 0.7))
            return {"accepted": give > 0, "share": round(give, 1)}
        return {"accepted": False, "reason": "holder needs its share"}
