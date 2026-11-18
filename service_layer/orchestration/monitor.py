"""Service Monitoring (ServiceMonitor): subscribe to intent and KPI reports over SN (SN-04, SN-06, SN-07) and
compute the fulfilment ratio per intent. Starts from routes/dashboard.py and dashboard_api.py."""
from __future__ import annotations

import time
from collections import deque
from typing import Any

from common.envelope import Auth, Envelope, ep
from common.refpoint import new_id


class ServiceMonitor:
    def __init__(self, sl):
        self.sl = sl
        self.reports: dict[str, deque] = {}
        self.status_events: deque = deque(maxlen=2000)

    async def on_status(self, topic: str, env: Envelope) -> None:
        b = env.body
        inst = self.sl.provisioner.instances.get(b["intentId"])
        self.status_events.append({"at": time.time(), **{k: b.get(k) for k in ("intentId", "state", "previousState", "reason")}})
        if not inst:
            return
        inst["state"] = b["state"]
        inst["serviceInstanceId"] = b.get("serviceInstanceId") or inst["serviceInstanceId"]
        inst["events"].append({"at": time.time(), "state": b["state"], "previous": b.get("previousState"),
                               "reason": b.get("reason"), "explanation": b.get("explanation"),
                               "fulfilment": b.get("fulfilment"), "tasks": b.get("tasks")})
        if b["state"] == "FULFILLED" and inst["fulfilledAt"] is None:
            inst["fulfilledAt"] = time.time()
        await self.sl.orchestrator.on_status(b)

    async def on_report(self, topic: str, env: Envelope) -> None:
        b = env.body
        self.reports.setdefault(b["intentId"], deque(maxlen=500)).append({"at": time.time(), **b})

    async def subscribe(self, intent_id: str, period: float = 10, kpis=None) -> Envelope:
        inst = self.sl.provisioner.instances[intent_id]
        body = {"subscriptionId": new_id("sub"), "target": {"intentId": intent_id},
                "kpis": kpis or [e["kpi"] for e in inst["intent"]["expectations"]], "periodSec": period,
                "thresholdEvents": True, "includeExplanation": True}
        env = self.sl.sn.envelope("MonitoringSubscribeRequest", intent_id, body, ep("network", "PerformanceMonitor"),
                                  auth=Auth(tenantId=inst["tenantId"], token=self.sl.identity.tenant_token(inst["tenantId"])),
                                  source=ep("service", "ServiceMonitor"), trace=self.sl.traces.for_flow(intent_id))
        return await self.sl.sn.send("POST", "/sn/v1/subscriptions", env)

    def fulfilment(self, intent_id: str) -> dict[str, Any]:
        rs = [r for r in self.reports.get(intent_id, []) if str(r.get("subscriptionId", "")).startswith("sub-auto-")] \
            or list(self.reports.get(intent_id, []))
        met = sum(1 for r in rs if (r.get("compliance") or {}).get("met"))
        return {"windows": len(rs), "met": met, "ratio": round(met / len(rs), 3) if rs else None}

    def latest(self, intent_id: str, n: int = 5) -> list[dict[str, Any]]:
        return list(self.reports.get(intent_id, []))[-n:]
