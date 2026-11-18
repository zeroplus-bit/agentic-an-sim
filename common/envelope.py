"""Common message envelope shared by the SN and NR reference points.

Every message is ``{"header": {...}, "body": {...}}``. Responses repeat the
request's correlationId, use the ``...Response`` msgType and add ``body.result``.
"""
from __future__ import annotations

import secrets
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION = "1.0"

Status = Literal["ACCEPTED", "SUCCESS", "REJECTED", "FAILED", "INVALID", "UNAUTHORIZED", "EXPIRED"]

STATUS_HTTP = {
    "ACCEPTED": 202,
    "SUCCESS": 200,
    "REJECTED": 409,
    "FAILED": 500,
    "INVALID": 400,
    "UNAUTHORIZED": 401,
    "EXPIRED": 408,
}
STATUS_CODE = {"ACCEPTED": 0, "SUCCESS": 0, "REJECTED": 409, "FAILED": 500,
               "INVALID": 400, "UNAUTHORIZED": 401, "EXPIRED": 408}


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: Optional[datetime] = None) -> str:
    dt = dt or now()
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class Endpoint(BaseModel):
    model_config = ConfigDict(extra="allow")
    layer: Literal["service", "network", "resource"]
    entity: str
    instanceId: Optional[str] = None


class Auth(BaseModel):
    model_config = ConfigDict(extra="allow")
    tenantId: Optional[str] = None
    token: Optional[str] = None


class Trace(BaseModel):
    traceId: str = Field(default_factory=lambda: secrets.token_hex(16))
    spanId: str = Field(default_factory=lambda: secrets.token_hex(8))


class Header(BaseModel):
    model_config = ConfigDict(extra="allow")
    msgId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlationId: str
    msgType: str
    refPoint: Literal["SN", "NR"]
    version: str = PROTOCOL_VERSION
    timestamp: str = Field(default_factory=iso)
    source: Endpoint
    destination: Endpoint
    priority: int = Field(3, ge=1, le=5)
    ttlMs: int = 30000
    auth: Optional[Auth] = None
    trace: Trace = Field(default_factory=Trace)


class Result(BaseModel):
    status: Status
    code: int = 0
    reason: Optional[str] = None


class Envelope(BaseModel):
    header: Header
    body: dict[str, Any] = Field(default_factory=dict)

    # -- helpers -----------------------------------------------------------
    @property
    def msg_type(self) -> str:
        return self.header.msgType

    @property
    def result(self) -> Optional[Result]:
        r = self.body.get("result")
        return Result(**r) if isinstance(r, dict) else None

    @property
    def status(self) -> Optional[str]:
        r = self.body.get("result") or {}
        return r.get("status")

    def dump(self) -> dict:
        return self.model_dump(mode="json", exclude_none=True)


def ep(layer: str, entity: str, instance: Optional[str] = None) -> Endpoint:
    return Endpoint(layer=layer, entity=entity, instanceId=instance)


def make(msg_type: str, ref_point: str, correlation_id: str, body: dict, source: Endpoint,
         destination: Endpoint, priority: int = 3, ttl_ms: int = 30000,
         auth: Optional[Auth] = None, trace: Optional[Trace] = None) -> Envelope:
    hdr = Header(correlationId=correlation_id, msgType=msg_type, refPoint=ref_point,
                 source=source, destination=destination, priority=priority, ttlMs=ttl_ms,
                 auth=auth, trace=child_trace(trace) if trace else Trace())
    return Envelope(header=hdr, body=body)


class TraceBook:
    """correlationId -> the flow's trace context, so every message of one flow (an intent) carries one traceId,
    W3C trace-context style: same traceId, a new spanId per message (child_trace)."""

    def __init__(self, maxlen: int = 5000):
        self._d: dict[str, Trace] = {}
        self.maxlen = maxlen

    def get(self, corr: Optional[str]) -> Optional[Trace]:
        return self._d.get(corr) if corr else None

    def remember(self, corr: Optional[str], trace: Optional[Trace]) -> None:
        if not corr or trace is None or corr in self._d:
            return                       # the first trace seen for a flow is the flow's trace
        if len(self._d) >= self.maxlen:
            self._d.pop(next(iter(self._d)))
        self._d[corr] = trace

    def for_flow(self, corr: str) -> Trace:
        if corr not in self._d:
            self.remember(corr, Trace())
        return self._d[corr]


def child_trace(parent: Optional[Trace]) -> Trace:
    if parent is None:
        return Trace()
    return Trace(traceId=parent.traceId, spanId=secrets.token_hex(8))


def response_type(msg_type: str) -> str:
    for suffix in ("Request", "Query"):
        if msg_type.endswith(suffix):
            return msg_type[: -len(suffix)] + "Response"
    return msg_type + "Response"


def respond(req: Envelope, status: str, body: Optional[dict] = None,
            reason: Optional[str] = None, code: Optional[int] = None) -> Envelope:
    h = req.header
    out = dict(body or {})
    out["result"] = Result(status=status, code=STATUS_CODE[status] if code is None else code,
                           reason=reason).model_dump()
    hdr = Header(correlationId=h.correlationId, msgType=response_type(h.msgType), refPoint=h.refPoint,
                 source=h.destination, destination=h.source, priority=h.priority,
                 ttlMs=h.ttlMs, trace=child_trace(h.trace))
    return Envelope(header=hdr, body=out)


def is_expired(h: Header) -> bool:
    try:
        return parse_iso(h.timestamp) + timedelta(milliseconds=h.ttlMs) < now()
    except ValueError:
        return False


class Deduplicator:
    """Remembers recent msgIds and their responses (at-most-once processing)."""

    def __init__(self, size: int = 5000):
        self._seen: "OrderedDict[str, Optional[Envelope]]" = OrderedDict()
        self._size = size

    def get(self, msg_id: str) -> Optional[Envelope]:
        return self._seen.get(msg_id)

    def seen(self, msg_id: str) -> bool:
        return msg_id in self._seen

    def remember(self, msg_id: str, response: Optional[Envelope]) -> None:
        self._seen[msg_id] = response
        while len(self._seen) > self._size:
            self._seen.popitem(last=False)


def check(env: Envelope, ref_point: str, allowed: Optional[set[str]] = None) -> Optional[Envelope]:
    """Validate the header. Returns an error response, or None if acceptable."""
    h = env.header
    if h.refPoint != ref_point:
        return respond(env, "INVALID", reason=f"refPoint {h.refPoint} not accepted on {ref_point} endpoint")
    if h.version.split(".")[0] != PROTOCOL_VERSION.split(".")[0]:
        return respond(env, "INVALID", reason=f"unsupported major version {h.version}")
    if allowed is not None and h.msgType not in allowed:
        return respond(env, "INVALID", reason=f"msgType {h.msgType} not allowed here (expected {sorted(allowed)})")
    if is_expired(h):
        return respond(env, "EXPIRED", reason="ttlMs elapsed before processing")
    return None
