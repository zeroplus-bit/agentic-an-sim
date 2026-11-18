"""AI Model Control (ModelController): load, swap and A/B (canary) the models chosen by Model LCM (Flow 4).
Starts from AIAgent._load_models; model paths come from configuration."""
from __future__ import annotations

import asyncio
import logging
import operator
import random
import time
from collections import deque
from typing import Any, Optional

import numpy as np

from common import config
from common.envelope import Envelope, respond
from common.refpoint import new_id

from ..task_execution.ml import KerasLstmNumpy, RuleAllocator

log = logging.getLogger("model_ctrl")
OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


class ModelController:
    def __init__(self, nl):
        self.nl = nl
        self.runtimes: dict[str, dict[str, Any]] = {"lstm_single": {}}
        self.primary: dict[str, str] = {}
        self.canary: dict[str, dict[str, Any]] = {}
        self.deployments: dict[str, dict[str, Any]] = {}
        self.errors: dict[tuple[str, str], deque] = {}
        self.infer_ms: dict[tuple[str, str], deque] = {}
        try:
            self.runtimes["lstm_single"]["1.0"] = KerasLstmNumpy(config.LSTM_MODEL_PATH, config.LSTM_SCALER_PATH,
                                                                 "lstm_single", "1.0")
        except Exception as e:  # missing file / h5py -> rule based
            log.warning("LSTM not loaded (%s); using rule-based allocator", e)
            self.runtimes["lstm_single"]["1.0"] = RuleAllocator()
        self.primary["lstm_single"] = "1.0"

    # ---- serving -----------------------------------------------------------------------
    def route(self, name: str) -> tuple[Any, str]:
        c = self.canary.get(name)
        if c and random.random() < c["share"]:
            return self.runtimes[name][c["version"]], c["version"]
        v = self.primary[name]
        return self.runtimes[name][v], v

    def record_infer(self, name: str, version: str, ms: float) -> None:
        self.infer_ms.setdefault((name, version), deque(maxlen=500)).append(ms)

    def observe(self, window: np.ndarray, new_features: np.ndarray) -> None:
        """Shadow evaluation: label = demand-proportional allocation observed at t+1."""
        alloc, util = new_features[3:6], new_features[6:9]
        label = np.clip(alloc * util, 1e-3, None)
        label = np.clip(label / label.sum(), 0.1, 0.8)
        label = label / label.sum()
        versions = {self.primary["lstm_single"]}
        if "lstm_single" in self.canary:
            versions.add(self.canary["lstm_single"]["version"])
        for v in versions:
            rt = self.runtimes["lstm_single"][v]
            t0 = time.perf_counter()
            pred = rt.predict(window)
            self.record_infer("lstm_single", v, (time.perf_counter() - t0) * 1000)
            self.errors.setdefault(("lstm_single", v), deque(maxlen=300)).append(float(np.mean(np.abs(pred - label))))

    def metrics(self, name: str, version: str) -> dict[str, float]:
        e = self.errors.get((name, version)) or []
        ms = self.infer_ms.get((name, version)) or []
        return {"mae": round(float(np.mean(list(e)[-100:])), 4) if e else None,
                "inferenceMsP95": round(float(np.percentile(list(ms), 95)), 3) if ms else None,
                "samples": len(e)}

    # ---- SN-10 ---------------------------------------------------------------------------
    async def deploy(self, env: Envelope) -> Envelope:
        _, err = self.nl.auth.verify_sn(env, "sn:model:write")
        if err:
            return respond(env, "UNAUTHORIZED", reason=err)
        b = env.body
        if b["model"]["name"] not in self.runtimes:
            return respond(env, "REJECTED", {"deploymentId": b["deploymentId"]},
                           reason=f"no runtime for model family {b['model']['name']}")
        self.deployments[b["deploymentId"]] = {**b, "state": "DEPLOYING", "createdAt": time.time()}
        self.nl.spawn(self._deploy(b, env.header.correlationId))
        return respond(env, "ACCEPTED", {"deploymentId": b["deploymentId"], "state": "DEPLOYING"})

    async def _deploy(self, b: dict[str, Any], corr: str) -> None:
        dep = self.deployments[b["deploymentId"]]
        name, ver = b["model"]["name"], b["model"]["version"]
        strategy = b.get("strategy") or {}
        if ver not in self.runtimes[name]:
            data, meta = await self.nl.repo.get_artifact(b["model"]["artifactUri"], corr)
            if data is None:
                await self._notify(dep, "ROLLED_BACK", {"error": 1.0}, reason=f"artifact {b['model']['artifactUri']} not found")
                return
            job = await self.nl.model_exec.compute_job("MODEL_LOAD", f"{name}:{ver}", b["model"]["artifactUri"],
                                                       {"cpu": 1, "gpu": 0}, corr)
            try:
                rt = await asyncio.to_thread(KerasLstmNumpy, data, config.LSTM_SCALER_PATH, name, ver)
            except Exception as e:
                await self._notify(dep, "ROLLED_BACK", {"error": 1.0}, reason=f"load failed: {e}")
                return
            self.runtimes[name][ver] = rt
            dep["loadJob"] = job.body.get("jobId")
        share = float(strategy.get("trafficShare", 1.0))
        if share >= 1.0 or strategy.get("type") == "ALL_AT_ONCE":
            old = self.primary[name]
            self.primary[name] = ver
            self.canary.pop(name, None)
            dep["state"] = "PROMOTED"
            await self._notify(dep, "PROMOTED", self.metrics(name, ver), reason=f"primary {old} -> {ver}")
            return
        self.canary[name] = {"version": ver, "share": share, "deploymentId": b["deploymentId"],
                             "rollbackOn": strategy.get("rollbackOn"), "since": time.time()}
        dep["state"] = "ACTIVE"
        await self._notify(dep, "ACTIVE", self.metrics(name, ver))

    async def _notify(self, dep: dict[str, Any], state: str, metrics: dict[str, float], drift: bool = False,
                      reason: Optional[str] = None) -> None:
        dep["state"] = state
        dep["metrics"] = metrics
        body = {"deploymentId": dep["deploymentId"], "metrics": {k: v for k, v in metrics.items() if v is not None},
                "drift": drift, "state": state, "model": f"{dep['model']['name']}:{dep['model']['version']}"}
        if reason:
            body["reason"] = reason
        await self.nl.sn_notify("ModelPerformanceNotify", dep["deploymentId"], body, "ModelController",
                                "ModelLifecycleManager")

    async def loop(self) -> None:
        """Periodic SN-11 for canaries and the rollback rule."""
        while True:
            await asyncio.sleep(5)
            for name, c in list(self.canary.items()):
                dep = self.deployments.get(c["deploymentId"])
                if not dep:
                    continue
                m = self.metrics(name, c["version"])
                p = self.metrics(name, self.primary[name])
                m["primaryMae"] = p["mae"]
                if m["samples"] < 5:
                    continue
                drift = bool(p["mae"] is not None and m["mae"] is not None and m["mae"] > 1.5 * p["mae"])
                rule = c.get("rollbackOn") or {}
                fire = False
                if rule:
                    val = m.get(rule.get("metric", "mae"))
                    fire = val is not None and OPS.get(rule.get("op", ">"), operator.gt)(val, float(rule["value"]))
                if fire:
                    self.canary.pop(name, None)
                    await self._notify(dep, "ROLLED_BACK", m, drift,
                                       reason=f"rollback rule {rule} fired; restored {self.primary[name]}")
                else:
                    await self._notify(dep, "ACTIVE", m, drift)

    def view(self) -> dict[str, Any]:
        return {"primary": self.primary, "canary": self.canary,
                "versions": {n: list(v) for n, v in self.runtimes.items()},
                "metrics": {f"{n}:{v}": self.metrics(n, v) for (n, v) in self.errors},
                "deployments": {k: {"state": d["state"], "model": d["model"], "metrics": d.get("metrics")}
                                for k, d in self.deployments.items()},
                "dqn": self.nl.model_exec.dqn.info()}
