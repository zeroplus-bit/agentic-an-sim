"""Agent runtime base class. Agents are registered by Agent Control, run their tasks through Task Execution
and are governed by Task Control. They reach resources ONLY through ToolRegistry tools (MCP)."""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from ..core import TaskRecord

log = logging.getLogger("agents")


class Agent:
    def __init__(self, nl, descriptor: dict[str, Any], token: str):
        self.nl = nl
        self.d = descriptor
        self.id = descriptor["agentId"]
        self.token = token
        self.tools = nl.tools.client(self.id, token)
        self.allowed_tools = set(descriptor.get("tools") or [])
        self.stats = {"tasks": 0, "ok": 0, "failed": 0, "toolCalls": 0, "transport": self.tools.mode}
        self.handlers: dict[str, Callable[[TaskRecord], Awaitable[dict[str, Any]]]] = {}

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        await self.tools.close()

    async def call(self, tool: str, **args) -> Any:
        spec = self.nl.tools.by_mcp_name(tool)
        if self.allowed_tools and "*" not in self.allowed_tools and spec and spec.name not in self.allowed_tools:
            raise PermissionError(f"tool {spec.name} not in agent descriptor of {self.id}")
        self.stats["toolCalls"] += 1
        return await self.tools.call(tool, **args)

    async def handle(self, task: TaskRecord) -> dict[str, Any]:
        h = self.handlers.get(task.type)
        self.stats["tasks"] += 1
        if not h:
            self.stats["failed"] += 1
            return {"ok": False, "error": f"{self.id} cannot handle {task.type}"}
        out = await h(task)
        self.stats["ok" if out.get("ok") else "failed"] += 1
        return out

    async def on_negotiate(self, holder_intent: str, cell: str, ask_share: float) -> dict[str, Any]:
        return {"accepted": False, "reason": "not supported"}


class ExternalAgent(Agent):
    """Card for an agent that runs outside the simulator (e.g. an LLM client such as Claude) and acts through
    the MCP tool endpoint with its agent token. It has no in-process task handler."""

    async def handle(self, task: TaskRecord) -> dict[str, Any]:
        return {"ok": False, "error": f"{self.id} is an external MCP agent; tasks are pulled via MCP tools"}
