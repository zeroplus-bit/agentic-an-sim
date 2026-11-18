"""Data Control (DataController): apply collection policy - which telemetry, at what period (NR-07, SN-12)."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from common.envelope import Envelope, respond
from common.refpoint import new_id

log = logging.getLogger("data_ctrl")
SLICES = ("eMBB", "URLLC", "mMTC")


class DataController:
    def __init__(self, nl):
        self.nl = nl
        self.policies: dict[str, dict[str, Any]] = {}
        self.subscriptions: dict[str, dict[str, Any]] = {}

    async def subscribe(self, resources: list[str], scope: dict[str, Any], period: float, corr: str,
                        sub_id: Optional[str] = None, kpis: Optional[list[str]] = None) -> Envelope:
        sid = sub_id or new_id("tsub")
        body = {"subscriptionId": sid, "resources": resources, "scope": {k: v for k, v in scope.items() if v},
                "periodSec": period, "format": "SAMPLE"}
        if kpis:
            body["kpis"] = kpis
        r = await self.nl.nr_call("TelemetrySubscribeRequest", "POST", "/nr/v1/telemetry/subscriptions", body, corr,
                                  "DataController")
        if r.status == "SUCCESS":
            self.subscriptions[sid] = body
        return r

    async def cancel(self, sub_id: str, corr: str) -> Envelope:
        self.subscriptions.pop(sub_id, None)
        return await self.nl.nr_call("TelemetrySubscribeRequest", "POST", "/nr/v1/telemetry/subscriptions",
                                     {"subscriptionId": sub_id, "resources": [], "cancel": True}, corr, "DataController")

    async def baseline(self) -> None:
        """Default collection policy: every cell, every 2 s (feeds LSTM windows and efficiency metrics)."""
        while True:
            try:
                cells = await self.nl.resource_ctrl.cell_ids()
                if cells:
                    r = await self.subscribe([f"comm:{c}" for c in sorted(cells)], {}, 2.0, "data-policy-default",
                                             sub_id="tsub-baseline")
                    if r.status == "SUCCESS":
                        self.policies.setdefault("default", {"policyId": "default", "sources": ["cell"], "periodSec": 2.0,
                                                             "subscriptions": ["tsub-baseline"]})
                        log.info("baseline telemetry subscription active on %d cells", len(cells))
                        return
            except Exception as e:  # resource layer not up yet
                log.info("waiting for resource layer: %s", e)
            await asyncio.sleep(2)

    async def set_policy(self, env: Envelope, policy_id: str) -> Envelope:
        _, err = self.nl.auth.verify_sn(env, "sn:policy:write")
        if err:
            return respond(env, "UNAUTHORIZED", reason=err)
        b = env.body
        old = self.policies.get(policy_id)
        if old:
            for sid in old.get("subscriptions", []):
                if sid != "tsub-baseline":
                    await self.cancel(sid, policy_id)
        cells = sorted(await self.nl.resource_ctrl.cell_ids())
        period = float(b.get("periodSec", 5))
        subs, notes = [], []
        for src in b.get("sources", ["cell"]):
            if src == "cell":
                sid = f"tsub-{policy_id}-cell"
                await self.subscribe([f"comm:{c}" for c in cells], {}, period, policy_id, sub_id=sid, kpis=b.get("kpis") or None)
                subs.append(sid)
            elif src == "slice":
                for st in SLICES:
                    sid = f"tsub-{policy_id}-{st}"
                    await self.subscribe([f"comm:{c}" for c in cells], {"sliceType": st}, period, policy_id, sub_id=sid,
                                         kpis=b.get("kpis") or None)
                    subs.append(sid)
            elif src == "compute":
                sid = f"tsub-{policy_id}-comp"
                inv = await self.nl.resource_ctrl.inventory("comp", corr=policy_id)
                await self.subscribe([x["resourceId"] for x in inv["resources"]], {}, period, policy_id, sub_id=sid)
                subs.append(sid)
            else:
                notes.append(f"source '{src}' has no telemetry producer in the twin; ignored")
        self.policies[policy_id] = {**b, "policyId": policy_id, "subscriptions": subs}
        return respond(env, "SUCCESS", {"policyId": policy_id, "subscriptions": subs, "notes": notes})
