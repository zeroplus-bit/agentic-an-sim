"""Performance Monitoring (PerformanceMonitor): compare NR-08 telemetry with intent expectations and raise
drift events (Flow 3 steps 1-3). Starts from 5G-Marketplace compliance_monitor/monitor.py."""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Optional

import numpy as np

import logging

from common.envelope import Envelope
from common.nr_messages import rname
from common.sn_messages import Expectation

log = logging.getLogger("perf_monitor")
SLICES = ("eMBB", "URLLC", "mMTC")
RETRY_SEC = 5.0             # still degraded after a correction -> try again this soon (was 15 s)
RECLAIM_HOLD_SEC = 20.0      # KPIs met and no change for this long before shares are scaled back
# reclaim sizes back to the PROVISIONING target load of the slice (decision_translator.TARGET_LOAD); corrections
# aim lower (0.65) for headroom while KPIs are at risk
RECLAIM_MIN_STEP = 3.0       # only shrink when the measured load frees at least this many PRB % points


class PerformanceMonitor:
    def __init__(self, nl):
        self.nl = nl
        self.cells: dict[str, deque] = {}
        self.slice_util: dict[str, deque] = {s: deque(maxlen=2000) for s in SLICES}
        self.prb: deque = deque(maxlen=5000)
        self.samples_seen = 0

    # ---- evaluation ------------------------------------------------------------------
    @staticmethod
    def evaluate(expectations: list[dict[str, Any]], values: dict[str, float]) -> tuple[bool, list[str]]:
        viol = []
        for e in expectations:
            v = values.get(e["kpi"])
            if v is None:
                continue
            if not Expectation(**e).met_by(v):
                viol.append(f"{e['kpi']} {v:.4g} !{e['op']} {e['value']:g}")
        return not viol, viol

    async def on_telemetry(self, topic: str, env: Envelope) -> None:
        b = env.body
        self.samples_seen += 1
        cell = rname(b["resourceId"])
        iid = b.get("intentId")
        if not iid:
            if b["resourceId"].startswith("comm:"):
                self.cells.setdefault(cell, deque(maxlen=300)).append({"t": time.time(), **b["values"]})
                f = b.get("features") or {}
                for s, k in zip(SLICES, ("embb_util", "urllc_util", "mmtc_util")):
                    if k in f:
                        self.slice_util[s].append(f[k])
                if "prb_utilization" in b["values"]:
                    self.prb.append(b["values"]["prb_utilization"])
            return
        rec = self.nl.intents.get(iid)
        if not rec or rec.state not in ("DEPLOYING", "FULFILLED", "DEGRADED"):
            return
        s = {"t": time.time(), "cell": cell, **b["values"]}
        rec.samples.setdefault(cell, deque(maxlen=600)).append(s)
        if cell in rec.down_cells and set(rec.cells) - rec.down_cells:
            return        # an out-of-service cell's sample says nothing about service on the surviving cells
        ok, viol = self.evaluate(rec.expectations, b["values"])
        rec.recent_ok.append(ok)
        rec.fulfilment = {**{k: round(v, 5) for k, v in b["values"].items()
                             if k in ("latency_ms", "reliability_percent", "bandwidth_mbps")},
                          "met": ok, "cell": cell, "source": "telemetry"}
        if rec.state == "FULFILLED":
            if not rec.baseline_shares and rec.shares:
                rec.baseline_shares = dict(rec.shares)
            bad = list(rec.recent_ok).count(False)
            fc = self.forecast_breach(rec, cell)
            if bad >= 3:
                await self.drift(rec, viol, cell, "VIOLATION")
            elif fc:
                await self.drift(rec, [fc], cell, "FORECAST_BREACH")
            elif ok and self._reclaimable(rec, cell):
                rec.reclaimed_at = time.time()
                self.nl.spawn(self.nl.perf_ctrl.on_reclaim(rec, cell))
        elif rec.state == "DEGRADED" and not rec.assurance_running:
            recent = list(rec.recent_ok)
            if len(recent) >= 3 and all(recent[-3:]):
                await self.nl.tracker.set_state(rec, "FULFILLED", reason="KPIs recovered",
                                                fulfilment=rec.fulfilment)
                log.info("KPIs recovered on %s", cell, extra={"intentId": rec.intent_id, "stage": "RESOLVE", "cell": cell})
                await self.nl.perf_ctrl.on_recovered(rec)
            elif time.time() - rec.last_action_at > RETRY_SEC and not ok:
                self.nl.spawn(self.nl.perf_ctrl.on_drift(rec, viol, cell, "STILL_DEGRADED"))

    @staticmethod
    def _reclaimable(rec, cell: str) -> bool:
        """Scale-down: share above the first-fulfilled share, KPIs met and load low for RECLAIM_HOLD_SEC."""
        cur, base = rec.shares.get(cell, 0.0), rec.baseline_shares.get(cell, 0.0)
        now = time.time()
        steer = rec.steering.get(cell)
        if rec.assurance_running or (cur <= base + 2 and not steer) or \
                now - max(rec.last_action_at, rec.reclaimed_at) < RECLAIM_HOLD_SEC:
            return False
        xs = [x for x in rec.samples.get(cell, []) if x["t"] >= now - RECLAIM_HOLD_SEC]
        if len(xs) < 10 or not all(rec_ok for rec_ok in list(rec.recent_ok)):
            return False
        loads = [x["load"] for x in xs if "load" in x]
        if not loads:
            return False
        from .decision_translator import TARGET_LOAD
        if steer:     # would all the steered UEs fit back into the own share at the provisioning target?
            return cur * max(loads) / (1 - steer["fraction"]) / TARGET_LOAD[rec.slice_type] <= cur
        need = max(base, math.ceil(cur * max(loads) / TARGET_LOAD[rec.slice_type]))   # worst load of the window
        return need <= cur - RECLAIM_MIN_STEP

    def forecast_breach(self, rec, cell: str) -> Optional[str]:
        """LSTM/trend forecast: drift before an actual violation (Flow 3 step 2)."""
        lat = [e for e in rec.expectations if e["kpi"] == "latency_ms"]
        if not lat:
            return None
        samples = [x["latency_ms"] for x in list(rec.samples.get(cell, []))[-10:] if "latency_ms" in x]
        if len(samples) < 10:
            return None
        budget = float(lat[0]["value"])
        pred = self.nl.model_exec.kpi_trend(samples, horizon=10)
        if pred > budget and samples[-1] > 0.8 * budget and samples[-1] <= budget:
            return f"forecast latency_ms {pred:.3g} > {budget:g} within 10 s"
        return None

    async def drift(self, rec, viol: list[str], cell: str, cause: str) -> None:
        log.warning("drift detected on %s: %s (%s)", cell, "; ".join(viol)[:200], cause,
                    extra={"intentId": rec.intent_id, "stage": "DETECT", "cell": cell})
        await self.nl.tracker.set_state(rec, "DEGRADED", reason=f"{cause}: {'; '.join(viol)} on {cell}")
        self.nl.spawn(self.nl.perf_ctrl.on_drift(rec, viol, cell, cause))

    async def on_alarm(self, topic: str, env: Envelope) -> None:
        b = env.body
        cell = b["resourceId"].split(":", 1)[1].split("/")[0] if ":" in b["resourceId"] else None
        for iid in b.get("affectedIntents", []):
            rec = self.nl.intents.get(iid)
            if not rec:
                continue
            if b["type"] == "CELL_DOWN":
                if b.get("severity") == "CLEARED":
                    rec.down_cells.discard(cell)
                    continue
                if cell in rec.down_cells:
                    continue          # repeated alarm for an outage already being handled
                rec.down_cells.add(cell)
            if b.get("severity") == "CLEARED":
                continue
            if rec.state == "FULFILLED" and b["type"] in ("CAPACITY_EXHAUSTED", "CELL_DOWN"):
                await self.drift(rec, [f"alarm {b['type']} {b.get('details')}"], cell, f"ALARM_{b['type']}")

    # ---- queries -----------------------------------------------------------------------
    def window(self, intent_id: Optional[str], cell: Optional[str], n: int = 10) -> dict[str, Any]:
        if intent_id and intent_id in self.nl.intents:
            rec = self.nl.intents[intent_id]
            cells = [cell] if cell else list(rec.samples)
            return {"intentId": intent_id, "samples": {c: list(rec.samples.get(c, []))[-n:] for c in cells}}
        if cell:
            return {"cell": cell, "samples": list(self.cells.get(cell, []))[-n:]}
        return {"error": "give intentId or cell"}

    @staticmethod
    def aggregate(samples: list[dict[str, Any]], kpis: list[str]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for k in kpis:
            v = np.array([s[k] for s in samples if k in s], dtype=float)
            if v.size == 0:
                continue
            d = {"avg": round(float(v.mean()), 5)}
            if k == "latency_ms":
                d["p99"] = round(float(np.percentile(v, 99)), 4)
                d["max"] = round(float(v.max()), 4)
            if k in ("bandwidth_mbps", "reliability_percent"):
                d["min"] = round(float(v.min()), 5)
            out[k] = d
        return out

    def efficiency(self) -> dict[str, Any]:
        out = {}
        for s, d in self.slice_util.items():
            if d:
                v = np.array(d)
                out[s] = {"avgUtilization": round(float(v.mean()), 3),
                          "shareInTarget_0.7_0.9": round(float(((v >= 0.7) & (v <= 0.9)).mean()), 3),
                          "shareOverloaded": round(float((v > 1.0).mean()), 3)}
        return {"perSlice": out, "avgPrbUtilization": round(float(np.mean(self.prb)), 3) if self.prb else None,
                "target": [0.7, 0.9]}
