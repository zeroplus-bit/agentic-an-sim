"""Identification Management (IdentityManager): issue identities and tokens for tenants, services and agents."""
from __future__ import annotations

from typing import Any, Optional

from common import security

TENANT_PERMS = ["sn:intent:write", "sn:intent:read", "sn:monitor:read"]
OPERATOR_PERMS = ["sn:*"]


class IdentityManager:
    def __init__(self):
        self.tenants: dict[str, dict[str, Any]] = {}
        self.agents: dict[str, dict[str, Any]] = {}
        self._operator = security.issue("service-layer", "operator", None, OPERATOR_PERMS)

    def operator_token(self) -> str:
        return self._operator

    def tenant_token(self, tenant_id: str) -> str:
        t = self.tenants.get(tenant_id)
        if not t or not security.verify(t["token"]):
            t = {"tenantId": tenant_id, "token": security.issue(f"tenant/{tenant_id}", "tenant", tenant_id, TENANT_PERMS)}
            self.tenants[tenant_id] = t
        return t["token"]

    def agent_token(self, agent_id: str, perms: list[str], ttl_sec: int = 7 * 24 * 3600) -> str:
        tok = security.issue(agent_id, "agent", None, perms, ttl_sec=ttl_sec)
        self.agents[agent_id] = {"agentId": agent_id, "perms": perms}
        return tok

    def view(self) -> dict[str, Any]:
        return {"tenants": list(self.tenants), "agents": self.agents}

    @staticmethod
    def whoami(token: Optional[str]) -> Optional[dict[str, Any]]:
        return security.verify(token)
