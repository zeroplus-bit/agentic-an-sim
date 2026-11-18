"""Service Provisioning (ServiceProvisioner): build a service order from an intent, send SN-01/02/03/05 and
track the service instance. Starts from routes/slice_selection.py::deploy_slice."""
from __future__ import annotations

import time
from typing import Any, Optional

from common.envelope import Auth, Envelope, ep


class ServiceProvisioner:
    def __init__(self, sl):
        self.sl = sl
        self.instances: dict[str, dict[str, Any]] = {}

    def _auth(self, tenant: str) -> Auth:
        return Auth(tenantId=tenant, token=self.sl.identity.tenant_token(tenant))

    async def submit(self, intent: dict[str, Any], priority: int) -> Envelope:
        iid, tenant = intent["intentId"], intent["tenantId"]
        self.instances[iid] = {"intent": intent, "priority": priority, "state": "SUBMITTED", "submittedAt": time.time(),
                               "fulfilledAt": None, "serviceInstanceId": None, "events": [], "tenantId": tenant}
        env = self.sl.sn.envelope("IntentSubmitRequest", iid, {"intent": intent}, ep("network", "IntentMediator"),
                                  priority=priority, auth=self._auth(tenant), source=ep("service", "ServiceProvisioner", "svc-1"),
                                  trace=self.sl.traces.for_flow(iid))
        r = await self.sl.sn.send("POST", "/sn/v1/intents", env)
        inst = self.instances[iid]
        inst["response"] = r.body
        if r.status != "ACCEPTED":
            inst["state"] = r.status
            inst["reason"] = r.result.reason if r.result else None
        else:
            inst["state"] = "RECEIVED"
            if r.body.get("cells"):
                intent.setdefault("context", {}).setdefault("area", {"type": "cellList"})["cells"] = r.body["cells"]
        return r

    async def feasibility(self, intent: dict[str, Any], priority: int) -> Envelope:
        env = self.sl.sn.envelope("IntentFeasibilityQuery", intent["intentId"], {"intent": intent},
                                  ep("network", "PreValidator"), priority=priority, auth=self._auth(intent["tenantId"]),
                                  source=ep("service", "IntentProcessor"), trace=self.sl.traces.for_flow(intent["intentId"]))
        return await self.sl.sn.send("POST", "/sn/v1/intents:feasibility", env)

    async def modify(self, intent_id: str, changes: dict[str, Any]) -> Envelope:
        inst = self.instances[intent_id]
        body = {"intentId": intent_id, **changes}
        env = self.sl.sn.envelope("IntentModifyRequest", intent_id, body, ep("network", "IntentMediator"),
                                  priority=changes.get("priority") or inst["priority"], auth=self._auth(inst["tenantId"]),
                                  source=ep("service", "ServiceProvisioner"), trace=self.sl.traces.for_flow(intent_id))
        r = await self.sl.sn.send("PATCH", f"/sn/v1/intents/{intent_id}", env)
        if r.status == "ACCEPTED":
            for k in ("expectations", "constraints", "reportPolicy"):
                if k in changes:
                    inst["intent"][k] = changes[k]
            if changes.get("context"):
                inst["intent"]["context"] = {**inst["intent"].get("context", {}), **changes["context"]}
            if changes.get("priority"):
                inst["priority"] = changes["priority"]
        return r

    async def cancel(self, intent_id: str, reason: Optional[str] = None) -> Envelope:
        inst = self.instances[intent_id]
        env = self.sl.sn.envelope("IntentCancelRequest", intent_id, {"intentId": intent_id, "reason": reason},
                                  ep("network", "IntentMediator"), priority=inst["priority"],
                                  auth=self._auth(inst["tenantId"]), source=ep("service", "ServiceProvisioner"),
                                  trace=self.sl.traces.for_flow(intent_id))
        return await self.sl.sn.send("DELETE", f"/sn/v1/intents/{intent_id}", env)

    def view(self, intent_id: Optional[str] = None) -> Any:
        def one(iid, x):
            return {"intentId": iid, "state": x["state"], "priority": x["priority"], "tenantId": x["tenantId"],
                    "serviceInstanceId": x["serviceInstanceId"], "utterance": x["intent"].get("utterance"),
                    "sliceType": x["intent"]["semantic"]["serviceCategory"], "expectations": x["intent"]["expectations"],
                    "cells": (x["intent"].get("context") or {}).get("area", {}).get("cells"),
                    "timeToFulfilSec": round(x["fulfilledAt"] - x["submittedAt"], 3) if x["fulfilledAt"] else None,
                    "reason": x.get("reason"), "lastEvent": x["events"][-1] if x["events"] else None}
        if intent_id:
            x = self.instances.get(intent_id)
            return {**one(intent_id, x), "intent": x["intent"], "events": x["events"][-50:]} if x else None
        return [one(i, x) for i, x in self.instances.items()]
