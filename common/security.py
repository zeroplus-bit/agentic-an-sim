"""HMAC-signed bearer tokens (JWT-like, dependency free).

IdentityManager (Service Layer) issues them, AuthGuard (Network Layer) verifies them.
Claims: sub, kind (tenant|service|agent|operator), tenantId, perms[], exp.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Optional

from . import config


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue(sub: str, kind: str, tenant_id: Optional[str] = None, perms: Optional[list[str]] = None,
          ttl_sec: int = 24 * 3600, secret: str = config.SN_SECRET) -> str:
    claims = {"sub": sub, "kind": kind, "tenantId": tenant_id, "perms": perms or [],
              "iat": int(time.time()), "exp": int(time.time()) + ttl_sec}
    payload = _b64(json.dumps(claims, separators=(",", ":")).encode())
    sig = _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def verify(token: Optional[str], secret: str = config.SN_SECRET) -> Optional[dict[str, Any]]:
    if not token or "." not in token:
        return None
    if token.lower().startswith("bearer "):
        token = token[7:]
    payload, sig = token.rsplit(".", 1)
    good = _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, good):
        return None
    try:
        claims = json.loads(_unb64(payload))
    except ValueError:
        return None
    if claims.get("exp", 0) < time.time():
        return None
    return claims


def has_perm(claims: dict[str, Any], perm: str) -> bool:
    perms = claims.get("perms") or []
    if "*" in perms or perm in perms:
        return True
    # wildcard  nr:config:*  matches nr:config:write
    return any(p.endswith(":*") and perm.startswith(p[:-1]) for p in perms)
