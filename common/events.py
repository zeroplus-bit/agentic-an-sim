"""Per-layer event hub for the operations dashboard: every traced message (SN/NR/agent bus/MCP tool call) and every
log record of the process goes into one bounded, sequence-numbered ring and is served as Server-Sent Events at
GET /events?since=<seq>.

    hub = get_hub("network");  install_events(app, hub)

The dashboard gateway (dashboard/app.py, :8400) follows the three layer streams, so the layers stay independent of the
dashboard: nothing is pushed, a layer without a subscriber only keeps its ring. High-rate traffic (NR-08 telemetry,
1 Hz x cells) is counted, not stored, so the ring keeps the control-plane story.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import Counter, deque
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import StreamingResponse

from .envelope import iso

NOISY = {"TelemetryNotify"}          # counted per msgType, not streamed


def brief(msg_type: str, body: Optional[dict[str, Any]]) -> dict[str, Any]:
    """The few body fields an operator needs to read a message without opening it (keeps events small)."""
    b = body or {}
    out: dict[str, Any] = {}
    for k in ("intentId", "state", "previousState", "reason", "severity", "type", "resourceId", "resolution",
              "conflictId", "cause", "resource", "agentId", "jobId", "configId", "target", "action"):
        v = b.get(k)
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
    res = b.get("result")
    if isinstance(res, dict):
        out["result"] = res.get("status")
        if res.get("reason"):
            out["resultReason"] = str(res["reason"])[:160]
    comp = b.get("compliance")
    if isinstance(comp, dict):
        out["met"] = comp.get("met")
        if comp.get("violations"):
            out["violations"] = [str(v)[:80] for v in comp["violations"][:3]]
    if isinstance(b.get("affectedIntents"), list):
        out["affectedIntents"] = b["affectedIntents"][:8]
    if isinstance(b.get("intents"), list):
        out["intents"] = b["intents"][:8]
    if isinstance(b.get("options"), list):
        out["options"] = [o.get("action") for o in b["options"] if isinstance(o, dict)][:4]
    if isinstance(b.get("detail"), dict):
        out["detail"] = {k: v for k, v in b["detail"].items() if isinstance(v, (str, int, float, bool))}
    if b.get("explanation"):
        out["explanation"] = str(b["explanation"])[:240]
    cfg = b.get("config")
    if isinstance(cfg, dict):
        shares = cfg.get("prbShare") or cfg.get("shares")
        if isinstance(shares, dict):
            out["shares"] = shares
        if cfg.get("mobility"):
            out["mobility"] = True
    return out


class EventHub:
    def __init__(self, layer: str, maxlen: int = 5000):
        self.layer = layer
        self.boot = uuid.uuid4().hex[:8]           # lets a follower notice a restart (sequence starts again)
        self.seq = 0
        self._ring: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._waiters: set[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = set()
        self.counts: Counter = Counter()           # msgType -> count, including the noisy ones

    def publish(self, ev: dict[str, Any]) -> None:
        mt = ev.get("msgType")
        with self._lock:
            if mt:
                self.counts[mt] += 1
            if mt in NOISY:
                return
            self.seq += 1
            ev = {"seq": self.seq, "layer": self.layer, "at": ev.get("at") or iso(), **ev}
            self._ring.append(ev)
            waiters = list(self._waiters)
        for loop, evt in waiters:
            try:
                loop.call_soon_threadsafe(evt.set)
            except RuntimeError:                   # loop closed
                self._waiters.discard((loop, evt))

    def after(self, seq: int, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            if not self._ring or self._ring[-1]["seq"] <= seq:
                return []
            rows = [e for e in self._ring if e["seq"] > seq]
        return rows[:limit]

    async def stream(self, since: int, request) -> Any:
        loop = asyncio.get_running_loop()
        evt = asyncio.Event()
        key = (loop, evt)
        self._waiters.add(key)
        last = since
        try:
            yield _sse("hello", {"layer": self.layer, "boot": self.boot, "seq": self.seq})
            last_stats = 0.0
            while True:
                if await request.is_disconnected():
                    break
                evt.clear()
                rows = self.after(last)
                for r in rows:
                    last = r["seq"]
                    yield _sse("event", r, r["seq"])
                if time.time() - last_stats > 5:
                    last_stats = time.time()
                    yield _sse("stats", {"layer": self.layer, "boot": self.boot, "seq": self.seq,
                                         "counts": dict(self.counts)})
                if not rows:
                    try:
                        await asyncio.wait_for(evt.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
        finally:
            self._waiters.discard(key)


def _sse(event: str, data: dict[str, Any], eid: Optional[int] = None) -> str:
    head = f"id: {eid}\n" if eid is not None else ""
    return f"{head}event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


class HubLogHandler(logging.Handler):
    """Log records -> hub events (kind=log). `extra={"intentId": ..., "stage": ...}` makes a record part of an incident."""

    SKIP = ("httpx", "httpcore", "uvicorn.access", "mcp.server", "sse_starlette")

    def __init__(self, hub: EventHub, level: int = logging.INFO):
        super().__init__(level)
        self.hub = hub

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(self.SKIP):
            return
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        ev = {"kind": "log", "level": record.levelname, "logger": record.name, "msg": msg[:600]}
        for k in ("intentId", "stage", "cell", "traceId"):
            v = getattr(record, k, None)
            if v is not None:
                ev[k] = v
        if record.exc_info and record.exc_info[1] is not None:
            ev["exc"] = repr(record.exc_info[1])[:300]
        self.hub.publish(ev)


_HUBS: dict[str, EventHub] = {}


def get_hub(layer: str) -> EventHub:
    if layer not in _HUBS:
        _HUBS[layer] = EventHub(layer)
    return _HUBS[layer]


def install_events(app, hub: EventHub) -> None:
    """GET /events (SSE) on a layer app + route this process's log records into the hub."""
    root = logging.getLogger()
    if not any(isinstance(h, HubLogHandler) and h.hub is hub for h in root.handlers):
        root.addHandler(HubLogHandler(hub))

    @app.get("/events")
    async def events(request: Request, since: int = 0):
        last = request.headers.get("last-event-id")
        if last and last.isdigit():
            since = int(last)
        return StreamingResponse(hub.stream(since, request), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/events/recent")
    async def recent(since: int = 0, limit: int = 500):
        return {"layer": hub.layer, "boot": hub.boot, "seq": hub.seq, "events": hub.after(since, limit)}
