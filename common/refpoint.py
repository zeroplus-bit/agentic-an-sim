"""HTTP client and server helpers for the two reference points."""
from __future__ import annotations

import itertools
import logging
import threading
from typing import Any, Optional

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

from . import config
from .envelope import STATUS_HTTP, Auth, Deduplicator, Endpoint, Envelope, Trace, make, respond
from .tracing import MessageLog

log = logging.getLogger("refpoint")

_counters: dict[str, itertools.count] = {}
_lock = threading.Lock()


def new_id(prefix: str, width: int = 4) -> str:
    with _lock:
        c = _counters.setdefault(prefix, itertools.count(1))
        return f"{prefix}-{next(c):0{width}d}"


class RefPointClient:
    """Sends envelopes over REST and returns the response envelope (never raises on 4xx/5xx)."""

    def __init__(self, base_url: str, ref_point: str, source: Endpoint, tracer: MessageLog,
                 service_token: Optional[str] = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.ref_point = ref_point
        self.source = source
        self.tracer = tracer
        self.service_token = service_token
        self._client = httpx.AsyncClient(timeout=timeout)

    def envelope(self, msg_type: str, correlation_id: str, body: dict, dest: Endpoint,
                 priority: int = 3, auth: Optional[Auth] = None, trace: Optional[Trace] = None,
                 source: Optional[Endpoint] = None, ttl_ms: int = 30000) -> Envelope:
        return make(msg_type, self.ref_point, correlation_id, body, source or self.source, dest,
                    priority=priority, auth=auth, trace=trace, ttl_ms=ttl_ms)

    async def send(self, method: str, path: str, env: Envelope, params: Optional[dict] = None) -> Envelope:
        self.tracer.record("out", env, transport=f"http {method} {path}")
        headers = {"Content-Type": "application/json"}
        if self.service_token:
            headers["X-Service-Token"] = self.service_token
        if env.header.auth and env.header.auth.token:
            headers["Authorization"] = f"Bearer {env.header.auth.token}"
        try:
            r = await self._client.request(method, f"{self.base_url}{path}", json=env.dump(),
                                           headers=headers, params=params)
            data = r.json()
            resp = Envelope.model_validate(data)
        except Exception as e:  # network error or non-envelope body
            log.warning("%s %s failed: %s", method, path, e)
            resp = respond(env, "FAILED", reason=f"transport error: {e}")
        self.tracer.record("in", resp, transport=f"http {method} {path}")
        return resp

    async def raw(self, method: str, path: str, **kw) -> httpx.Response:
        headers = kw.pop("headers", {}) or {}
        if self.service_token:
            headers["X-Service-Token"] = self.service_token
        return await self._client.request(method, f"{self.base_url}{path}", headers=headers, **kw)

    async def close(self) -> None:
        await self._client.aclose()


def reply(env: Envelope) -> JSONResponse:
    status = (env.body.get("result") or {}).get("status", "SUCCESS")
    return JSONResponse(env.dump(), status_code=STATUS_HTTP.get(status, 200))


async def read_envelope(request: Request) -> Envelope | JSONResponse:
    try:
        data = await request.json()
        return Envelope.model_validate(data)
    except Exception as e:
        return JSONResponse({"error": f"INVALID envelope: {e}"}, status_code=400)


class ServerSide:
    """Common inbound handling: trace, dedup, header check."""

    def __init__(self, tracer: MessageLog, ref_point: str):
        self.tracer = tracer
        self.ref_point = ref_point
        self.dedup = Deduplicator()

    def inbound(self, env: Envelope, transport: str) -> Optional[Envelope]:
        self.tracer.record("in", env, transport=transport)
        cached = self.dedup.get(env.header.msgId)
        if cached is not None:
            return cached
        return None

    def outbound(self, req: Envelope, resp: Envelope, transport: str) -> JSONResponse:
        self.dedup.remember(req.header.msgId, resp)
        self.tracer.record("out", resp, transport=transport)
        return reply(resp)


def check_service_token(request: Request) -> bool:
    return request.headers.get("X-Service-Token") == config.NR_SERVICE_TOKEN


def check_bus_token(request: Request) -> bool:
    return request.headers.get("X-Bus-Token") == config.BUS_TOKEN


def dict_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = dict_merge(out[k], v)
        else:
            out[k] = v
    return out
