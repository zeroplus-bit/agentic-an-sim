"""Tool/Function (ToolRegistry): callable tools (NR operations, model calls, LLM calls) and their schemas,
exposed to agents over MCP (Model Context Protocol).

* Every tool has a canonical dotted name (nr.config.apply), an MCP name (nr_config_apply), a JSON schema
  (derived by FastMCP from the typed signature) and a required permission.
* The registry is served as an MCP server (streamable HTTP) at  <NETWORK_URL>/tools/mcp.
  In-process agents (RAN allocator, assurance, planner) are MCP clients of it; so can be any external
  LLM agent (e.g. Claude) once it was onboarded through SN-08 and holds an agent token.
* AuthGuard checks the caller's bearer token and the tool permission on EVERY call.
* AANS_TOOL_MODE=direct bypasses the MCP transport for in-process agents (same handlers, same checks).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from common import config
from common.refpoint import new_id

log = logging.getLogger("tools")
Handler = Callable[[dict[str, Any], dict[str, Any]], Awaitable[Any]]


class ToolDenied(PermissionError):
    pass


@dataclass
class ToolSpec:
    name: str
    perm: str
    description: str
    handler: Handler
    nr_message: Optional[str] = None

    @property
    def mcp_name(self) -> str:
        return self.name.replace(".", "_")


class ToolRegistry:
    def __init__(self, nl):
        self.nl = nl
        self.specs: dict[str, ToolSpec] = {}
        self.stats: dict[str, dict[str, float]] = {}
        self._clients: list["ToolClient"] = []
        self._register_builtin()
        self.mcp = self._build_mcp()

    # ---- registry ---------------------------------------------------------------------
    def register(self, name: str, perm: str, description: str, handler: Handler, nr_message: Optional[str] = None):
        self.specs[name] = ToolSpec(name, perm, description, handler, nr_message)
        self.stats[name] = {"calls": 0, "errors": 0, "denied": 0, "totalMs": 0.0}

    def by_mcp_name(self, mcp_name: str) -> Optional[ToolSpec]:
        for s in self.specs.values():
            if s.mcp_name == mcp_name or s.name == mcp_name:
                return s
        return None

    async def invoke(self, name: str, args: dict[str, Any], claims: Optional[dict[str, Any]], via: str) -> Any:
        spec = self.by_mcp_name(name)
        if spec is None:
            raise KeyError(f"unknown tool {name}")
        st = self.stats[spec.name]
        if not claims:
            st["denied"] += 1
            raise ToolDenied("UNAUTHORIZED: missing/invalid agent token")
        if not self.nl.auth.allowed(claims, spec.perm):
            st["denied"] += 1
            self.nl.tracer.record_raw({"dir": "internal", "transport": f"tool:{via}", "msgType": f"ToolDenied {spec.mcp_name}",
                                       "correlationId": args.get("intentId") or claims.get("sub"), "status": "UNAUTHORIZED",
                                       "from": f"network/{claims.get('sub')}", "to": "network/AuthGuard"})
            raise ToolDenied(f"UNAUTHORIZED: {claims.get('sub')} lacks {spec.perm} for {spec.name}")
        t0 = time.perf_counter()
        st["calls"] += 1
        self.nl.tracer.record_raw({"dir": "internal", "transport": f"tool:{via}", "msgType": f"ToolCall {spec.mcp_name}",
                                   "correlationId": args.get("intentId") or args.get("taskId") or claims.get("sub"),
                                   "from": f"network/{claims.get('sub')}", "to": "network/ToolRegistry"})
        try:
            return await spec.handler(args, claims)
        except Exception:
            st["errors"] += 1
            raise
        finally:
            st["totalMs"] += (time.perf_counter() - t0) * 1000

    def list(self) -> list[dict[str, Any]]:
        return [{"name": s.name, "mcpName": s.mcp_name, "permission": s.perm, "description": s.description,
                 "nrMessage": s.nr_message, "stats": self.stats[s.name]} for s in self.specs.values()]

    def client(self, agent_id: str, token: Optional[str], mode: Optional[str] = None) -> "ToolClient":
        c = ToolClient(self, agent_id, token, mode or config.TOOL_MODE)
        self._clients.append(c)
        return c

    async def close(self) -> None:
        for c in self._clients:
            await c.close()

    # ---- built-in tools ----------------------------------------------------------------
    def _register_builtin(self) -> None:
        nl = self.nl
        R = self.register

        async def intent_get(a, c):
            rec = nl.intents.get(a["intentId"])
            return rec.view() if rec else {"error": "unknown intent"}

        async def intent_list(a, c):
            return {"intents": [{"intentId": i, "state": r.state, "priority": r.priority, "sliceType": r.slice_type,
                                 "cells": r.cells, "shares": r.shares} for i, r in nl.intents.items()]}

        async def inventory(a, c):
            return await nl.resource_ctrl.inventory(a.get("type"), corr=a.get("intentId") or "inventory")

        async def reserve(a, c):
            rec = nl.intents.get(a.get("intentId") or "")
            r = await nl.resource_ctrl.reserve(a.get("intentId"), a["items"], priority=int(a.get("priority", 3)),
                                               corr=a.get("intentId") or c["sub"], reservation_id=a.get("reservationId"),
                                               tenant=rec.tenant if rec else None)
            return {"status": r.status, "reason": r.result.reason if r.result else None, **r.body}

        async def release(a, c):
            r = await nl.resource_ctrl.release(a["reservationId"], a.get("intentId") or c["sub"])
            return {"status": r.status}

        async def validate(a, c):
            return await nl.prevalidator.validate_for_agent(a, c["sub"])

        async def apply(a, c):
            return await nl.actions.apply(a, c["sub"])

        async def rollback(a, c):
            r = await nl.actions.rollback(a["configId"], a.get("intentId") or c["sub"], a.get("reason"))
            return {"status": r.status, "reason": r.result.reason if r.result else None}

        async def tsub(a, c):
            r = await nl.data_ctrl.subscribe(a["resources"], {"intentId": a.get("intentId"), "sliceType": a.get("sliceType")},
                                             float(a.get("periodSec", 1)), a.get("intentId") or c["sub"])
            return {"status": r.status, "subscriptionId": r.body.get("subscriptionId")}

        async def twindow(a, c):
            return nl.perf_mon.window(a.get("intentId"), a.get("cell"), int(a.get("n", 10)))

        async def llm(a, c):
            r = await nl.reports.llm(a.get("prompt", ""), facts=a.get("facts"), kind=a.get("kind", "explain"),
                                     tool_id=a.get("toolId", "tool:gemini"), caller={"agentId": c["sub"]},
                                     corr=a.get("intentId") or c["sub"], max_tokens=int(a.get("maxTokens", 512)))
            return r

        async def compute(a, c):
            r = await nl.model_exec.compute_job(a.get("kind", "INFERENCE"), a.get("model"), a.get("inputRef"),
                                                a.get("resources") or {"cpu": 1, "gpu": 0}, a.get("intentId") or c["sub"])
            return {"status": r.status, **{k: v for k, v in r.body.items() if k != "result"}}

        async def sput(a, c):
            return await nl.repo.put(a["bucket"], a["key"], a["content"].encode(), a.get("contentType", "application/json"),
                                     corr=a.get("intentId") or c["sub"])

        async def sget(a, c):
            data, meta = await nl.repo.get(a["bucket"], a["key"], corr=a.get("intentId") or c["sub"])
            if data is None:
                return {"found": False}
            try:
                text = data.decode()
            except UnicodeDecodeError:
                text = f"<{len(data)} bytes binary>"
            return {"found": True, "size": len(data), "content": text[:20000], "meta": meta}

        async def lstm(a, c):
            return nl.model_exec.forecast_allocation(a["cell"], caller=c["sub"])

        async def dqn(a, c):
            return nl.model_exec.dqn_infer(a["cell"], int(a.get("violations", 0)))

        async def translate(a, c):
            return await nl.translator.translate_for_agent(a)

        async def conflict(a, c):
            return await nl.conflict.check_change(a["intentId"], a["cell"], float(a["sharePercent"]), a["intentId"])

        async def discover(a, c):
            return {"agents": [{k: v for k, v in x.items() if k not in ("descriptor",)} | {"skills": x["descriptor"]["skills"]}
                               for x in nl.registry.discover(a["skill"])]}

        async def delegate(a, c):
            task = nl.tracker.new_task(a.get("intentId"), a["taskType"], a.get("input") or {})
            out = await nl.agent_bus.delegate(task, a["skill"], sender=c["sub"])
            return {"taskId": task.task_id, **out}

        async def recall(a, c):
            return {"episodes": nl.memory.recall(a.get("intentId"), a.get("agentId"), int(a.get("limit", 10)))}

        async def store(a, c):
            eid = nl.memory.store_episode(a["intentId"], c["sub"], a.get("taskId", ""), a["state"], int(a["action"]),
                                          float(a["reward"]), a["nextState"], a.get("outcome", ""), a.get("details"))
            return {"episodeId": eid}

        R("intent.get", "intent:read", "Read one intent as the Network Layer sees it (state, expectations, cells, shares, configs).", intent_get)
        R("intent.list", "intent:read", "List intents with state, priority and PRB shares per cell.", intent_list)
        R("nr.resource.inventory", "nr:resource:read", "NR-01 ResourceInventoryQuery: cells, compute nodes, storage, tools, UE groups.", inventory, "NR-01")
        R("nr.resource.reserve", "nr:resource:write", "NR-02 ResourceReserveRequest: reserve PRB share (percent) on cells or CPU/GPU on compute.", reserve, "NR-02")
        R("nr.resource.release", "nr:resource:write", "NR-03 ResourceReleaseRequest.", release, "NR-03")
        R("nr.config.validate", "nr:config:validate", "NR-04 ConfigValidateRequest: dry-run a slice config on a twin copy; returns predicted KPIs and whether the intent's expectations are met.", validate, "NR-04")
        R("nr.config.apply", "nr:config:write", "NR-05 ConfigApplyRequest: apply a validated slice config to the live twin (records a rollback token).", apply, "NR-05")
        R("nr.config.rollback", "nr:config:write", "NR-06 ConfigRollbackRequest.", rollback, "NR-06")
        R("nr.telemetry.subscribe", "nr:telemetry:subscribe", "NR-07 TelemetrySubscribeRequest for cells (optionally scoped to an intent or slice).", tsub, "NR-07")
        R("nr.telemetry.window", "nr:telemetry:read", "Last N NR-08 telemetry samples for an intent or cell (from Performance Monitoring).", twindow)
        R("llm.generate", "tool:llm:invoke", "NR-10 ToolInvokeRequest on tool:gemini / tool:ollama / tool:template (LLM text generation).", llm, "NR-10")
        R("nr.compute.job", "nr:compute:write", "NR-11 ComputeJobRequest (INFERENCE | TRAINING | MODEL_LOAD).", compute, "NR-11")
        R("nr.storage.put", "nr:storage:rw", "NR-13 StorageObjectRequest PUT (text content).", sput, "NR-13")
        R("nr.storage.get", "nr:storage:rw", "NR-13 StorageObjectRequest GET.", sget, "NR-13")
        R("model.lstm.forecast", "model:infer", "LSTM (models/lstm_single) on the cell's last 10 telemetry samples -> recommended [eMBB, URLLC, mMTC] allocation.", lstm)
        R("model.dqn.infer", "model:infer", "DQN policy (dqn_agent.py port) on the cell state -> discrete action 0-26 and its allocation.", dqn)
        R("decision.translate", "model:infer", "Decision Translation: turn model outputs into a concrete NR slice config (PRB split, intent share, 5QI, scheduling).", translate)
        R("conflict.check", "nr:config:validate", "Conflict mgmt: can the intent take sharePercent on this cell? May preempt lower-priority intents (SN-14).", conflict)
        R("agent.discover", "agent:cooperate", "Registration/Discovery: find ACTIVE agents by skill.", discover)
        R("agent.delegate", "agent:cooperate", "Agent Cooperation: create a task and delegate it to an agent with the given skill.", delegate)
        R("memory.recall", "memory:rw", "Memory: recent episodes (state, action, reward, outcome) for an intent or agent.", recall)
        R("memory.store_episode", "memory:rw", "Memory: store an episode for DQN retraining.", store)

    # ---- MCP server ----------------------------------------------------------------------
    def _build_mcp(self) -> FastMCP:
        reg = self
        m = FastMCP("aan-network-tools", host=config.HOST, streamable_http_path="/mcp",
                    instructions=("Network Layer tool registry of the Agentic AI Autonomous Network simulator. "
                                  "Tools act on the Resource Layer only through the NR reference point. "
                                  "Send 'Authorization: Bearer <agent token>' (issued at SN-08 onboarding)."))

        def claims_of(ctx: Context) -> Optional[dict[str, Any]]:
            tok = None
            try:
                req = ctx.request_context.request
                if req is not None:
                    tok = req.headers.get("authorization")
            except Exception:
                pass
            return reg.nl.auth.agent_claims(tok)

        async def call(name: str, args: dict[str, Any], ctx: Context) -> Any:
            args = {k: v for k, v in args.items() if v is not None}
            try:
                return await reg.invoke(name, args, claims_of(ctx), via="mcp")
            except ToolDenied as e:
                raise ToolError(str(e)) from e
            except KeyError as e:
                raise ToolError(f"bad arguments for {name}: missing {e}") from e

        @m.tool(name="intent_get", description=self.specs["intent.get"].description)
        async def intent_get(intentId: str, ctx: Context) -> dict:
            return await call("intent.get", {"intentId": intentId}, ctx)

        @m.tool(name="intent_list", description=self.specs["intent.list"].description)
        async def intent_list(ctx: Context) -> dict:
            return await call("intent.list", {}, ctx)

        @m.tool(name="nr_resource_inventory", description=self.specs["nr.resource.inventory"].description)
        async def nr_resource_inventory(ctx: Context, type: Optional[str] = None) -> dict:
            return await call("nr.resource.inventory", {"type": type}, ctx)

        @m.tool(name="nr_resource_reserve", description=self.specs["nr.resource.reserve"].description)
        async def nr_resource_reserve(items: list[dict], ctx: Context, intentId: Optional[str] = None, priority: int = 3,
                                      reservationId: Optional[str] = None) -> dict:
            return await call("nr.resource.reserve", {"items": items, "intentId": intentId, "priority": priority,
                                                      "reservationId": reservationId}, ctx)

        @m.tool(name="nr_resource_release", description=self.specs["nr.resource.release"].description)
        async def nr_resource_release(reservationId: str, ctx: Context, intentId: Optional[str] = None) -> dict:
            return await call("nr.resource.release", {"reservationId": reservationId, "intentId": intentId}, ctx)

        @m.tool(name="nr_config_validate", description=self.specs["nr.config.validate"].description)
        async def nr_config_validate(target: str, config: dict, ctx: Context, intentId: Optional[str] = None,
                                     reservationId: Optional[str] = None, simulateSec: int = 60,
                                     origin: Optional[dict] = None) -> dict:
            return await call("nr.config.validate", {"target": target, "config": config, "intentId": intentId,
                                                     "reservationId": reservationId, "simulateSec": simulateSec,
                                                     "origin": origin}, ctx)

        @m.tool(name="nr_config_apply", description=self.specs["nr.config.apply"].description)
        async def nr_config_apply(target: str, config: dict, ctx: Context, intentId: Optional[str] = None,
                                  reservationId: Optional[str] = None, configId: Optional[str] = None,
                                  origin: Optional[dict] = None) -> dict:
            return await call("nr.config.apply", {"target": target, "config": config, "intentId": intentId,
                                                  "reservationId": reservationId, "configId": configId,
                                                  "origin": origin}, ctx)

        @m.tool(name="nr_config_rollback", description=self.specs["nr.config.rollback"].description)
        async def nr_config_rollback(configId: str, ctx: Context, intentId: Optional[str] = None,
                                     reason: Optional[str] = None) -> dict:
            return await call("nr.config.rollback", {"configId": configId, "intentId": intentId, "reason": reason}, ctx)

        @m.tool(name="nr_telemetry_subscribe", description=self.specs["nr.telemetry.subscribe"].description)
        async def nr_telemetry_subscribe(resources: list[str], ctx: Context, intentId: Optional[str] = None,
                                         sliceType: Optional[str] = None, periodSec: float = 1.0) -> dict:
            return await call("nr.telemetry.subscribe", {"resources": resources, "intentId": intentId,
                                                         "sliceType": sliceType, "periodSec": periodSec}, ctx)

        @m.tool(name="nr_telemetry_window", description=self.specs["nr.telemetry.window"].description)
        async def nr_telemetry_window(ctx: Context, intentId: Optional[str] = None, cell: Optional[str] = None,
                                      n: int = 10) -> dict:
            return await call("nr.telemetry.window", {"intentId": intentId, "cell": cell, "n": n}, ctx)

        @m.tool(name="llm_generate", description=self.specs["llm.generate"].description)
        async def llm_generate(prompt: str, ctx: Context, toolId: str = "tool:gemini", facts: Optional[dict] = None,
                               kind: str = "explain", maxTokens: int = 512, intentId: Optional[str] = None) -> dict:
            return await call("llm.generate", {"prompt": prompt, "toolId": toolId, "facts": facts, "kind": kind,
                                               "maxTokens": maxTokens, "intentId": intentId}, ctx)

        @m.tool(name="nr_compute_job", description=self.specs["nr.compute.job"].description)
        async def nr_compute_job(ctx: Context, kind: str = "INFERENCE", model: Optional[str] = None,
                                 inputRef: Optional[str] = None, resources: Optional[dict] = None,
                                 intentId: Optional[str] = None) -> dict:
            return await call("nr.compute.job", {"kind": kind, "model": model, "inputRef": inputRef,
                                                 "resources": resources, "intentId": intentId}, ctx)

        @m.tool(name="nr_storage_put", description=self.specs["nr.storage.put"].description)
        async def nr_storage_put(bucket: str, key: str, content: str, ctx: Context,
                                 contentType: str = "application/json", intentId: Optional[str] = None) -> dict:
            return await call("nr.storage.put", {"bucket": bucket, "key": key, "content": content,
                                                 "contentType": contentType, "intentId": intentId}, ctx)

        @m.tool(name="nr_storage_get", description=self.specs["nr.storage.get"].description)
        async def nr_storage_get(bucket: str, key: str, ctx: Context, intentId: Optional[str] = None) -> dict:
            return await call("nr.storage.get", {"bucket": bucket, "key": key, "intentId": intentId}, ctx)

        @m.tool(name="model_lstm_forecast", description=self.specs["model.lstm.forecast"].description)
        async def model_lstm_forecast(cell: str, ctx: Context) -> dict:
            return await call("model.lstm.forecast", {"cell": cell}, ctx)

        @m.tool(name="model_dqn_infer", description=self.specs["model.dqn.infer"].description)
        async def model_dqn_infer(cell: str, ctx: Context, violations: int = 0) -> dict:
            return await call("model.dqn.infer", {"cell": cell, "violations": violations}, ctx)

        @m.tool(name="decision_translate", description=self.specs["decision.translate"].description)
        async def decision_translate(intentId: str, cell: str, ctx: Context,
                                     lstmAllocation: Optional[list[float]] = None, dqnAction: Optional[int] = None,
                                     attempt: int = 1, targetLoad: Optional[float] = None,
                                     feedback: Optional[dict] = None, sharePercent: Optional[float] = None,
                                     offload: Optional[dict] = None) -> dict:
            """offload = {"target": cell, "fraction": 0..0.4, "share": PRB %} steers UEs to a neighbour; {} undoes it."""
            return await call("decision.translate", {"intentId": intentId, "cell": cell, "lstmAllocation": lstmAllocation,
                                                     "dqnAction": dqnAction, "attempt": attempt, "targetLoad": targetLoad,
                                                     "feedback": feedback, "sharePercent": sharePercent,
                                                     "offload": offload}, ctx)

        @m.tool(name="conflict_check", description=self.specs["conflict.check"].description)
        async def conflict_check(intentId: str, cell: str, sharePercent: float, ctx: Context) -> dict:
            return await call("conflict.check", {"intentId": intentId, "cell": cell, "sharePercent": sharePercent}, ctx)

        @m.tool(name="agent_discover", description=self.specs["agent.discover"].description)
        async def agent_discover(skill: str, ctx: Context) -> dict:
            return await call("agent.discover", {"skill": skill}, ctx)

        @m.tool(name="agent_delegate", description=self.specs["agent.delegate"].description)
        async def agent_delegate(skill: str, taskType: str, ctx: Context, intentId: Optional[str] = None,
                                 input: Optional[dict] = None) -> dict:
            return await call("agent.delegate", {"skill": skill, "taskType": taskType, "intentId": intentId,
                                                 "input": input}, ctx)

        @m.tool(name="memory_recall", description=self.specs["memory.recall"].description)
        async def memory_recall(ctx: Context, intentId: Optional[str] = None, agentId: Optional[str] = None,
                                limit: int = 10) -> dict:
            return await call("memory.recall", {"intentId": intentId, "agentId": agentId, "limit": limit}, ctx)

        @m.tool(name="memory_store_episode", description=self.specs["memory.store_episode"].description)
        async def memory_store_episode(intentId: str, state: list[float], action: int, reward: float,
                                       nextState: list[float], ctx: Context, taskId: str = "", outcome: str = "",
                                       details: Optional[dict] = None) -> dict:
            return await call("memory.store_episode", {"intentId": intentId, "state": state, "action": action,
                                                       "reward": reward, "nextState": nextState, "taskId": taskId,
                                                       "outcome": outcome, "details": details}, ctx)

        @m.resource("aan://tools/catalogue", description="Tool catalogue with permissions and NR message mapping")
        def catalogue() -> str:
            return json.dumps(reg.list(), indent=1, default=str)

        return m

    def mcp_http_app(self):
        return self.mcp.streamable_http_app()


class ToolCallError(RuntimeError):
    pass


class ToolClient:
    """What an agent holds: call(tool_name, **args). Transport = MCP (default) or direct."""

    def __init__(self, reg: ToolRegistry, agent_id: str, token: Optional[str], mode: str):
        self.reg, self.agent_id, self.token, self.mode = reg, agent_id, token, mode
        self._session: Optional[ClientSession] = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._worker: Optional[asyncio.Task] = None
        self._err: Optional[BaseException] = None
        self.tool_names: list[str] = []

    async def _run(self) -> None:
        url = f"{config.NETWORK_URL}/tools/mcp"
        try:
            async with streamablehttp_client(url, headers={"Authorization": f"Bearer {self.token}"}) as (r, w, _):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    self.tool_names = [t.name for t in (await s.list_tools()).tools]
                    self._session = s
                    self._ready.set()
                    await self._stop.wait()
        except BaseException as e:  # noqa: BLE001
            self._err = e
            self._ready.set()
        finally:
            self._session = None

    async def _ensure(self) -> ClientSession:
        if self._session is not None:
            return self._session
        if self._worker is None or self._worker.done():
            self._ready = asyncio.Event()
            self._err = None
            self._worker = asyncio.create_task(self._run())
        await asyncio.wait_for(self._ready.wait(), 15)
        if self._session is None:
            raise ToolCallError(f"MCP connect failed: {self._err!r}")
        return self._session

    async def call(self, name: str, **args) -> Any:
        args = {k: v for k, v in args.items() if v is not None}
        if self.mode != "mcp":
            claims = self.reg.nl.auth.agent_claims(self.token)
            try:
                return await self.reg.invoke(name, args, claims, via="direct")
            except ToolDenied as e:
                raise ToolCallError(str(e)) from e
        s = await self._ensure()
        res = await s.call_tool(name, _jsonable(args))
        if res.isError:
            raise ToolCallError(" ".join(getattr(c, "text", "") for c in res.content))
        if res.structuredContent is not None:
            sc = res.structuredContent
            return sc["result"] if isinstance(sc, dict) and set(sc) == {"result"} else sc
        texts = [getattr(c, "text", "") for c in res.content]
        try:
            return json.loads(texts[0]) if texts else None
        except (ValueError, IndexError):
            return {"text": " ".join(texts)}

    async def close(self) -> None:
        self._stop.set()
        if self._worker:
            try:
                await asyncio.wait_for(self._worker, 3)
            except (asyncio.TimeoutError, Exception):
                self._worker.cancel()


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
