"""Data/Model Execution (ModelExecutor): run LSTM forecasts and DQN policies on telemetry windows.
Starts from ml_orchestrator_demo.py::MLOrchestrator and lstm_web_server.py. Inference is metered on the
Resource Layer's Computing Resources with NR-11 ComputeJobRequest (fire-and-forget)."""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Optional

import numpy as np

from common.envelope import Envelope
from common.nr_messages import rname
from common.refpoint import new_id

from .ml import FEATURES, NumpyDQN, action_to_allocation, make_state

SLICES = ("eMBB", "URLLC", "mMTC")
METER = os.environ.get("AANS_METER_INFERENCE", "1") == "1"


class ModelExecutor:
    def __init__(self, nl):
        self.nl = nl
        self.windows: dict[str, deque] = {}
        self.last_features: dict[str, dict[str, float]] = {}
        self.scenario: dict[str, bool] = {}
        self.dqn = NumpyDQN()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.calls = {"lstm": 0, "dqn": 0}

    async def on_telemetry(self, topic: str, env: Envelope) -> None:
        b = env.body
        if b.get("intentId") or not b.get("features"):
            return
        cell = rname(b["resourceId"])
        vec = np.array([b["features"].get(k, 0.0) for k in FEATURES], dtype=float)
        w = self.windows.setdefault(cell, deque(maxlen=11))
        if len(w) >= 10:
            self.nl.model_ctrl.observe(np.array(list(w)[-10:]), vec)
        w.append(vec)
        self.last_features[cell] = b["features"]
        self.scenario = b.get("scenario") or {}

    def _window(self, cell: str) -> Optional[np.ndarray]:
        w = self.windows.get(cell)
        if not w:
            return None
        rows = list(w)[-10:]
        while len(rows) < 10:
            rows.insert(0, rows[0])
        return np.array(rows)

    def forecast_allocation(self, cell: str, caller: str = "") -> dict[str, Any]:
        win = self._window(cell)
        if win is None:
            return {"cell": cell, "allocation": None, "reason": "no telemetry window yet"}
        rt, ver = self.nl.model_ctrl.route("lstm_single")
        t0 = time.perf_counter()
        pred = rt.predict(win)
        ms = (time.perf_counter() - t0) * 1000
        self.nl.model_ctrl.record_infer("lstm_single", ver, ms)
        self.calls["lstm"] += 1
        if METER:
            self.nl.spawn(self.compute_job("INFERENCE", f"lstm_single:{ver}", f"stor:windows/{cell}/{len(self.windows[cell])}",
                                           {"cpu": 1, "gpu": 0}, caller or cell))
        cur = win[-1]
        return {"cell": cell, "model": f"lstm_single:{ver}", "inferenceMs": round(ms, 3),
                "vector": [round(float(x), 4) for x in pred],
                "allocation": {s: round(float(pred[i]), 4) for i, s in enumerate(SLICES)},
                "current": {s: round(float(cur[3 + i]), 4) for i, s in enumerate(SLICES)},
                "utilization": {s: round(float(cur[6 + i]), 4) for i, s in enumerate(SLICES)},
                "windowLen": len(self.windows[cell])}

    def state(self, cell: str, violations: int = 0) -> Optional[np.ndarray]:
        f = self.last_features.get(cell)
        if not f:
            return None
        return make_state(f, self.scenario, violations)

    def dqn_infer(self, cell: str, violations: int = 0) -> dict[str, Any]:
        s = self.state(cell, violations)
        if s is None:
            return {"cell": cell, "action": None, "reason": "no telemetry yet"}
        a, mode = self.dqn.act(s)
        self.calls["dqn"] += 1
        alloc = action_to_allocation(a)
        return {"cell": cell, "action": a, "mode": mode, "model": f"{self.dqn.name}:{self.dqn.version}",
                "allocation": {x: round(float(alloc[i]), 4) for i, x in enumerate(SLICES)},
                "state": [round(float(v), 5) for v in s], "scenario": self.scenario}

    def learn(self, state, action: int, reward: float, next_state) -> None:
        self.dqn.remember(state, action, reward, next_state, False)
        self.dqn.train(batch_size=32, iters=5)

    @staticmethod
    def kpi_trend(values: list[float], horizon: int = 10) -> float:
        y = np.asarray(values, dtype=float)
        if y.size < 3:
            return float(y[-1]) if y.size else 0.0
        x = np.arange(y.size)
        k, c = np.polyfit(x, y, 1)
        return float(k * (y.size - 1 + horizon) + c)

    def kpi_forecast(self, rec) -> dict[str, Any]:
        out: dict[str, Any] = {"model": f"lstm_single:{self.nl.model_ctrl.primary['lstm_single']}+trend", "horizonSec": 60}
        lat = []
        for cell, d in rec.samples.items():
            vals = [x["latency_ms"] for x in list(d)[-20:] if "latency_ms" in x]
            if len(vals) >= 3:
                lat.append(self.kpi_trend(vals, horizon=60))
        if lat:
            out["latency_ms_p99"] = round(max(lat), 3)
        if rec.cells:
            fc = self.forecast_allocation(rec.cells[0], caller=rec.intent_id) if self._window(rec.cells[0]) is not None else None
            if fc and fc.get("allocation"):
                out["recommendedAllocation"] = fc["allocation"]
        return out

    async def compute_job(self, kind: str, model: Optional[str], input_ref: Optional[str], resources: dict[str, float],
                          corr: str) -> Envelope:
        jid = new_id("job")
        self.jobs[jid] = {"kind": kind, "model": model, "state": "SUBMITTED", "at": time.time()}
        r = await self.nl.nr_call("ComputeJobRequest", "POST", "/nr/v1/compute/jobs",
                                  {"jobId": jid, "kind": kind, "model": model, "inputRef": input_ref, "resources": resources,
                                   "deadlineMs": 500}, corr, "ModelExecutor")
        return r

    async def on_job(self, topic: str, env: Envelope) -> None:
        b = env.body
        if b["jobId"] in self.jobs:
            self.jobs[b["jobId"]]["state"] = b["state"]
            if b["state"] in ("DONE", "FAILED", "CANCELLED") and len(self.jobs) > 2000:
                for k in list(self.jobs)[:1000]:
                    self.jobs.pop(k, None)

    def view(self) -> dict[str, Any]:
        states: dict[str, int] = {}
        for j in self.jobs.values():
            states[j["state"]] = states.get(j["state"], 0) + 1
        return {"cellsWithWindow": len(self.windows), "calls": self.calls, "jobs": states, "dqn": self.dqn.info()}
