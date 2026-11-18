"""Resource Control (ResourceController): reserve, release and rebalance resources (NR-01/02/03) and
enforce per-tenant resource policies received over SN-13. Starts from ml_orchestrator_demo.py allocation logic."""
from __future__ import annotations

import time
from typing import Any, Optional

from common.envelope import Envelope, ep, make, respond
from common.refpoint import new_id


class ResourceController:
    def __init__(self, nl):
        self.nl = nl
        self.policies: dict[str, dict[str, Any]] = {}
        self._inv: dict[str, tuple[float, dict]] = {}

    async def inventory(self, type_: Optional[str] = None, corr: str = "inventory", max_age: float = 0.0) -> dict[str, Any]:
        key = type_ or "*"
        if max_age and key in self._inv and time.time() - self._inv[key][0] < max_age:
            return self._inv[key][1]
        r = await self.nl.nr_call("ResourceInventoryQuery", "GET", "/nr/v1/resources", {"type": type_}, corr,
                                  "ResourceController")
        data = {"status": r.status, "resources": r.body.get("resources", []), "count": r.body.get("count", 0)}
        if r.status == "SUCCESS":
            self._inv[key] = (time.time(), data)
        return data

    async def cell_ids(self) -> set[str]:
        inv = await self.inventory("comm", max_age=10)
        return {x["resourceId"].split(":", 1)[1] for x in inv["resources"]}

    async def cell_info(self, cell: str, max_age: float = 1.0) -> dict[str, Any]:
        inv = await self.inventory("comm", corr=cell, max_age=max_age)
        for x in inv["resources"]:
            if x["resourceId"] == f"comm:{cell}":
                return x
        return {}

    def quota_check(self, tenant: str, extra_share: float, cell: Optional[str] = None) -> Optional[str]:
        p = self.policies.get(tenant)
        if not p or "prbSharePercent" not in (p.get("quota") or {}):
            return None
        limit = p["quota"]["prbSharePercent"] * float(p.get("overbooking", 1.0))
        for c in ([cell] if cell else {c for r in self.nl.intents.values() for c in r.cells} or {None}):
            used = sum(r.shares.get(c, 0.0) for r in self.nl.intents.values() if r.tenant == tenant and r.active) if c else 0
            if used + extra_share > limit + 1e-9:
                return f"QUOTA_EXCEEDED: tenant {tenant} PRB share {used + extra_share:.1f}% > {limit:.1f}%"
        return None

    async def reserve(self, intent_id: Optional[str], items: list[dict], priority: int, corr: str,
                      reservation_id: Optional[str] = None, tenant: Optional[str] = None) -> Envelope:
        rec = self.nl.intents.get(intent_id or "")
        rid = reservation_id or (rec.reservation_id if rec and rec.reservation_id else new_id("rsv"))
        body = {"reservationId": rid, "intentId": intent_id, "items": items, "priority": priority,
                "preemptible": priority >= 4}
        if rec and rec.intent.get("context", {}).get("timeWindow"):
            body["window"] = rec.intent["context"]["timeWindow"]
        if tenant:
            for it in items:
                if it.get("kind") == "PRB_SHARE":
                    cell = it["resourceId"].split(":", 1)[1]
                    already = rec.shares.get(cell, 0.0) if rec else 0.0
                    err = self.quota_check(tenant, it["amount"] - already, cell)
                    if err:
                        req = make("ResourceReserveRequest", "NR", corr, body, ep("network", "ResourceController"),
                                   ep("resource", "CommunicationResources"))
                        return respond(req, "REJECTED", {"reservationId": rid}, reason=err)
        if rec and rec.state == "PLANNING":
            await self.nl.tracker.set_state(rec, "DEPLOYING")
        r = await self.nl.nr_call("ResourceReserveRequest", "POST", "/nr/v1/reservations", body, corr, "ResourceController",
                                  priority=priority)
        if r.status == "SUCCESS" and rec:
            rec.reservation_id = rid
        return r

    async def release(self, reservation_id: str, corr: str) -> Envelope:
        return await self.nl.nr_call("ResourceReleaseRequest", "DELETE", f"/nr/v1/reservations/{reservation_id}",
                                     {"reservationId": reservation_id}, corr, "ResourceController")

    async def set_policy(self, env: Envelope, tenant_id: str) -> Envelope:
        _, err = self.nl.auth.verify_sn(env, "sn:policy:write")
        if err:
            return respond(env, "UNAUTHORIZED", reason=err)
        self.policies[tenant_id] = {**env.body, "tenantId": tenant_id}
        return respond(env, "SUCCESS", {"tenantId": tenant_id, "policy": self.policies[tenant_id]})
