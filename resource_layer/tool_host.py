"""Tool Resources - LLM endpoints and other external tools (ToolHost).

Agents never call an LLM directly: they invoke it as a resource over NR-10, so the
simulator can meter, rate-limit and fail tools like any other resource.
API keys live ONLY in this process (env GEMINI_API_KEY), never in messages.
Starts from 5G-Marketplace gemini_integration.py and the Ollama client in ai_agent/agent.py,
with a deterministic offline fallback so the simulator runs without network access.
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx

from common import config


class ToolHost:
    def __init__(self):
        self.tools: dict[str, dict[str, Any]] = {
            "tool:gemini": {"kind": "llm", "operations": ["generate"], "available": bool(os.environ.get("GEMINI_API_KEY"))},
            "tool:ollama": {"kind": "llm", "operations": ["generate"], "available": None},
            "tool:template": {"kind": "llm", "operations": ["generate"], "available": True},
            "tool:calculator": {"kind": "function", "operations": ["eval"], "available": True},
        }
        self.usage: dict[str, dict[str, float]] = {k: {"calls": 0, "tokens": 0, "failures": 0} for k in self.tools}
        self.fail_next: set[str] = set()          # fault injection

    async def invoke(self, tool_id: str, operation: str, args: dict[str, Any], timeout_ms: int) -> dict[str, Any]:
        if tool_id not in self.tools:
            return {"status": "REJECTED", "reason": f"unknown tool {tool_id}"}
        u = self.usage[tool_id]
        u["calls"] += 1
        t0 = time.perf_counter()
        if tool_id in self.fail_next:
            self.fail_next.discard(tool_id)
            u["failures"] += 1
            return {"status": "FAILED", "reason": "injected fault"}
        try:
            if tool_id == "tool:gemini":
                text, used = await self._gemini(args, timeout_ms)
            elif tool_id == "tool:ollama":
                text, used = await self._ollama(args, timeout_ms)
            elif tool_id == "tool:calculator":
                text, used = self._calc(args), tool_id
            else:
                text, used = self._template(args), "tool:template"
        except Exception as e:  # fall back to offline template rather than failing the agent
            u["failures"] += 1
            text, used = self._template(args) + f" [fallback: {type(e).__name__}]", "tool:template"
        tokens = max(1, len(text.split()) * 4 // 3)
        u["tokens"] += tokens
        return {"status": "SUCCESS", "output": {"text": text, "servedBy": used},
                "usage": {"latencyMs": round((time.perf_counter() - t0) * 1000, 1), "tokens": tokens}}

    async def _gemini(self, args: dict[str, Any], timeout_ms: int) -> tuple[str, str]:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            return self._template(args), "tool:template"
        model = args.get("model") or config.GEMINI_MODEL
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        async with httpx.AsyncClient(timeout=timeout_ms / 1000) as c:
            r = await c.post(url, params={"key": key}, json={
                "contents": [{"parts": [{"text": args.get("prompt", "")}]}],
                "generationConfig": {"maxOutputTokens": int(args.get("maxTokens", 512))}})
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"], "tool:gemini"

    async def _ollama(self, args: dict[str, Any], timeout_ms: int) -> tuple[str, str]:
        async with httpx.AsyncClient(timeout=timeout_ms / 1000) as c:
            r = await c.post(f"{config.OLLAMA_URL}/api/generate", json={
                "model": args.get("model") or config.OLLAMA_MODEL, "prompt": args.get("prompt", ""), "stream": False})
            r.raise_for_status()
            return r.json().get("response", ""), "tool:ollama"

    @staticmethod
    def _calc(args: dict[str, Any]) -> str:
        expr = str(args.get("expression", ""))
        if any(ch not in "0123456789+-*/(). " for ch in expr):
            return "ERROR: only arithmetic allowed"
        return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 - sanitized above

    @staticmethod
    def _template(args: dict[str, Any]) -> str:
        """Deterministic offline 'LLM': summarises the structured facts passed by the caller."""
        facts = args.get("facts") or {}
        kind = args.get("kind", "explain")
        if kind == "explain" and facts:
            parts = []
            if "state" in facts:
                parts.append(f"Intent is {facts['state']}")
            if facts.get("violations"):
                parts.append("violations: " + ", ".join(facts["violations"]))
            elif "kpis" in facts:
                parts.append("all expectations met")
            if facts.get("forecast"):
                parts.append(f"forecast {facts['forecast']}")
            if facts.get("action"):
                parts.append(f"last action: {facts['action']}")
            return "; ".join(parts) + "."
        prompt = str(args.get("prompt", ""))
        return (prompt[:200] + "...") if len(prompt) > 200 else (prompt or "OK")

    def inventory(self) -> list[dict[str, Any]]:
        return [{"resourceId": k, "type": "tool", **v, "usage": self.usage[k]} for k, v in self.tools.items()]
