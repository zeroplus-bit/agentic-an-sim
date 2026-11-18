"""Publish/subscribe bus for notifications (sn.notify.*, nr.notify.*) and internal agent messages.

* InProcBus      - asyncio fan-out inside one process (default for in-process use and tests)
* HttpBridgeBus  - InProcBus + forwards selected topics to another layer's ``POST /bus/inbox``
                   (lets the three FastAPI processes exchange notifications without a broker)
* NatsBus        - optional, requires ``pip install nats-py``

A notification is the same Envelope as a request, published instead of posted.
"""
from __future__ import annotations

import asyncio
import fnmatch
import logging
from typing import Awaitable, Callable, Optional

import httpx

from . import config
from .envelope import Envelope

log = logging.getLogger("bus")
Handler = Callable[[str, Envelope], Awaitable[None]]


class InProcBus:
    def __init__(self, name: str = "bus", tracer=None):
        self.name = name
        self._subs: list[tuple[str, Handler]] = []
        self.tracer = tracer
        self._tasks: set[asyncio.Task] = set()

    def subscribe(self, pattern: str, handler: Handler) -> None:
        self._subs.append((pattern, handler))

    async def publish(self, topic: str, env: Envelope) -> None:
        if self.tracer:
            self.tracer.record("out", env, transport=f"bus:{topic}")
        await self.deliver_local(topic, env, trace=False)

    async def deliver_local(self, topic: str, env: Envelope, trace: bool = True) -> int:
        if trace and self.tracer:
            self.tracer.record("in", env, transport=f"bus:{topic}")
        n = 0
        for pattern, handler in list(self._subs):
            if fnmatch.fnmatchcase(topic, pattern):
                n += 1
                t = asyncio.create_task(self._safe(handler, topic, env))
                self._tasks.add(t)
                t.add_done_callback(self._tasks.discard)
        return n

    @staticmethod
    async def _safe(handler: Handler, topic: str, env: Envelope) -> None:
        try:
            await handler(topic, env)
        except Exception:  # pragma: no cover - handler bugs must not kill the bus
            log.exception("bus handler failed for %s", topic)

    async def close(self) -> None:
        for t in list(self._tasks):
            t.cancel()


class HttpBridgeBus(InProcBus):
    """Forwards topics matching a route pattern to a remote layer's /bus/inbox."""

    def __init__(self, name: str, routes: dict[str, str], tracer=None):
        super().__init__(name, tracer)
        self.routes = routes                      # pattern -> base url
        self._client = httpx.AsyncClient(timeout=5.0)

    async def publish(self, topic: str, env: Envelope) -> None:
        if self.tracer:
            self.tracer.record("out", env, transport=f"bus:{topic}")
        await self.deliver_local(topic, env, trace=False)
        for pattern, url in self.routes.items():
            if fnmatch.fnmatchcase(topic, pattern):
                try:
                    await self._client.post(f"{url}/bus/inbox", json={"topic": topic, "envelope": env.dump()},
                                            headers={"X-Bus-Token": config.BUS_TOKEN})
                except httpx.HTTPError as e:
                    log.warning("bus bridge %s -> %s failed: %s", topic, url, e)

    async def close(self) -> None:
        await super().close()
        await self._client.aclose()


class NatsBus(InProcBus):  # pragma: no cover - optional dependency
    def __init__(self, name: str, url: str = config.NATS_URL, tracer=None):
        super().__init__(name, tracer)
        self.url = url
        self._nc = None

    async def connect(self) -> None:
        import nats  # type: ignore
        self._nc = await nats.connect(self.url)

    def subscribe(self, pattern: str, handler: Handler) -> None:
        super().subscribe(pattern, handler)
        subject = pattern.replace("*", ">") if pattern.endswith("*") else pattern

        async def _cb(msg):
            env = Envelope.model_validate_json(msg.data)
            await self.deliver_local(msg.subject, env)

        if self._nc is not None:
            asyncio.get_event_loop().create_task(self._nc.subscribe(subject, cb=_cb))

    async def publish(self, topic: str, env: Envelope) -> None:
        if self.tracer:
            self.tracer.record("out", env, transport=f"bus:{topic}")
        if self._nc is None:
            await self.connect()
        await self._nc.publish(topic, env.model_dump_json(exclude_none=True).encode())


def make_bus(name: str, routes: Optional[dict[str, str]] = None, tracer=None) -> InProcBus:
    kind = config.BUS_KIND
    if kind == "nats":
        return NatsBus(name, tracer=tracer)
    if kind == "http" and routes:
        return HttpBridgeBus(name, routes, tracer=tracer)
    return InProcBus(name, tracer=tracer)
