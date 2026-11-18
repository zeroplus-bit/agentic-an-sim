"""Authentication & Authorization (AuthGuard): verifies SN tokens and agent permissions before any task runs."""
from __future__ import annotations

from typing import Any, Optional

from common import security
from common.envelope import Envelope


class AuthGuard:
    def __init__(self, nl):
        self.nl = nl
        self.agent_perms: dict[str, list[str]] = {}          # agentId -> bound permissions

    def verify_sn(self, env: Envelope, perm: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        a = env.header.auth
        claims = security.verify(a.token if a else None)
        if not claims:
            return None, "missing or invalid bearer token"
        if a and a.tenantId and claims.get("tenantId") not in (None, a.tenantId) and claims.get("kind") != "operator":
            return None, f"token tenant {claims.get('tenantId')} != header tenant {a.tenantId}"
        if not security.has_perm(claims, perm):
            return None, f"permission {perm} denied"
        return claims, None

    def bind_agent(self, agent_id: str, token: Optional[str], descriptor_perms: list[str]) -> Optional[str]:
        """Bind an agent's permissions to its token: effective = descriptor perms allowed by the token."""
        claims = security.verify(token)
        if not claims or claims.get("sub") != agent_id or claims.get("kind") != "agent":
            return "agent token invalid or not issued for this agent"
        allowed = [p for p in descriptor_perms if security.has_perm(claims, p)]
        self.agent_perms[agent_id] = allowed
        return None

    def agent_claims(self, token: Optional[str]) -> Optional[dict[str, Any]]:
        claims = security.verify(token)
        if not claims:
            return None
        if claims.get("kind") == "agent":
            sub = claims["sub"]
            card = self.nl.registry.cards.get(sub)
            if not card or card["state"] == "TERMINATED":
                return None
            return {**claims, "perms": self.agent_perms.get(sub, [])}
        if claims.get("kind") == "operator":
            return claims
        return None

    @staticmethod
    def allowed(claims: dict[str, Any], perm: str) -> bool:
        return security.has_perm(claims, perm)

    def unbind(self, agent_id: str) -> None:
        self.agent_perms.pop(agent_id, None)
