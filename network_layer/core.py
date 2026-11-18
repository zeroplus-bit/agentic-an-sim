"""Network Layer container: shared state + the 19 functional blocks wired together."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from common import config
from common.bus import make_bus
from common.envelope import Envelope, ep, make
from common.refpoint import RefPointClient, ServerSide
from common.tracing import MessageLog

log = logging.getLogger("network_layer")


@dataclass
class TaskRecord:
    task_id: str
    intent_id: Optional[str]
    type: str
    input: dict[str, Any] = field(default_factory=dict)
    agent_id: Optional[str] = None
    state: str = "PENDING"            # PENDING | ASSIGNED | RUNNING | COMPLETED | FAILED | CANCELLED
    output: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def info(self) -> dict[str, Any]:
        return {"taskId": self.task_id, "agentId": self.agent_id, "type": self.type, "state": self.state}


@dataclass
class IntentRecord:
    intent: dict[str, Any]
    priority: int
    tenant: str
    trace: Any = None
    state: str = "RECEIVED"
    previous: Optional[str] = None
    tasks: list[str] = field(default_factory=list)
    service_instance_id: Optional[str] = None
    reservation_id: Optional[str] = None
    telemetry_sub: Optional[str] = None
    configs: list[dict[str, Any]] = field(default_factory=list)     # applied configs, in order
    shares: dict[str, float] = field(default_factory=dict)          # cell -> PRB % currently bound
    samples: dict[str, deque] = field(default_factory=dict)         # cell -> recent samples
    recent_ok: deque = field(default_factory=lambda: deque(maxlen=5))
    fulfilment: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    reason: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    fulfilled_at: Optional[float] = None
    drift_since: Optional[float] = None
    recover_times: list[float] = field(default_factory=list)
    baseline_shares: dict[str, float] = field(default_factory=dict)  # shares at first FULFILLED (reclaim floor)
    down_cells: set = field(default_factory=set)                      # cells with an active CELL_DOWN alarm
    escalated_at: Optional[float] = None                              # last SN-14 escalation (capacity / give-up)
    reclaimed_at: float = 0.0
    steering: dict[str, dict[str, Any]] = field(default_factory=dict)  # source cell -> {target, fraction, share}
    assurance_running: bool = False
    last_action_at: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    windows_total: int = 0
    windows_met: int = 0

    @property
    def intent_id(self) -> str:
        return self.intent["intentId"]

    @property
    def slice_type(self) -> str:
        return self.intent["semantic"]["serviceCategory"]

    @property
    def cells(self) -> list[str]:
        return list((self.intent.get("context") or {}).get("area", {}).get("cells") or [])

    @property
    def expectations(self) -> list[dict[str, Any]]:
        return self.intent.get("expectations") or []

    @property
    def active(self) -> bool:
        return self.state in ("DEPLOYING", "FULFILLED", "DEGRADED", "PLANNING")

    def view(self) -> dict[str, Any]:
        return {"intentId": self.intent_id, "tenantId": self.tenant, "priority": self.priority, "state": self.state,
                "sliceType": self.slice_type, "cells": self.cells, "expectations": self.expectations,
                "serviceInstanceId": self.service_instance_id, "shares": self.shares, "configs": self.configs,
                "fulfilment": self.fulfilment, "explanation": self.explanation, "reason": self.reason,
                "tasks": self.tasks, "history": self.history[-20:],
                "context": self.intent.get("context"), "constraints": self.intent.get("constraints")}


class NetworkLayer:
    SELF = ep("network", "NetworkLayer", "net-1")

    def __init__(self):
        self.tracer = MessageLog("network")
        self.server = ServerSide(self.tracer, "SN")
        self.bus = make_bus("network", {"sn.notify.*": config.SERVICE_URL}, tracer=self.tracer)
        self.nr = RefPointClient(config.RESOURCE_URL, "NR", self.SELF, self.tracer, service_token=config.NR_SERVICE_TOKEN)
        self.intents: dict[str, IntentRecord] = {}
        self.cell_locks: dict[str, asyncio.Lock] = {}
        from common.envelope import TraceBook
        self.traces = TraceBook()          # correlationId -> trace of an SN request seen before any intent record   # serialises read-translate-reserve-validate-apply per cell
        self.tasks: dict[str, TaskRecord] = {}
        self._bg: set[asyncio.Task] = set()

        from .agent_control.auth import AuthGuard
        from .agent_control.conflict import ConflictManager
        from .agent_control.cooperation import AgentBus
        from .agent_control.mediator import IntentMediator
        from .agent_control.memory import AgentMemory
        from .agent_control.registry import AgentRegistry
        from .agent_control.task_tracker import TaskTracker
        from .agent_control.tools import ToolRegistry
        from .task_control.data_ctrl import DataController
        from .task_control.decision_translator import DecisionTranslator
        from .task_control.model_ctrl import ModelController
        from .task_control.perf_ctrl import PerformanceController
        from .task_control.perf_monitor import PerformanceMonitor
        from .task_control.pre_validator import PreValidator
        from .task_control.resource_ctrl import ResourceController
        from .task_execution.action_executor import ActionExecutor
        from .task_execution.model_executor import ModelExecutor
        from .task_execution.report_gen import ReportGenerator
        from .task_execution.repository import Repository

        # Agent Control
        self.auth = AuthGuard(self)
        self.registry = AgentRegistry(self)
        self.conflict = ConflictManager(self)
        self.tools = ToolRegistry(self)
        self.agent_bus = AgentBus(self)
        self.memory = AgentMemory(self)
        self.mediator = IntentMediator(self)
        self.tracker = TaskTracker(self)
        # Task Execution
        self.model_exec = ModelExecutor(self)
        self.reports = ReportGenerator(self)
        self.repo = Repository(self)
        self.actions = ActionExecutor(self)
        # Task Control
        self.model_ctrl = ModelController(self)
        self.translator = DecisionTranslator(self)
        self.data_ctrl = DataController(self)
        self.prevalidator = PreValidator(self)
        self.resource_ctrl = ResourceController(self)
        self.perf_ctrl = PerformanceController(self)
        self.perf_mon = PerformanceMonitor(self)

    # ---- lifecycle ------------------------------------------------------------------
    async def start(self) -> None:
        self.bus.subscribe("nr.notify.TelemetryNotify", self.perf_mon.on_telemetry)
        self.bus.subscribe("nr.notify.TelemetryNotify", self.model_exec.on_telemetry)
        self.bus.subscribe("nr.notify.ResourceAlarmNotify", self.perf_mon.on_alarm)
        self.bus.subscribe("nr.notify.ResourceAlarmNotify", self.conflict.on_alarm)
        self.bus.subscribe("nr.notify.ComputeJobStatusNotify", self.model_exec.on_job)
        self.bus.subscribe("agents.*", self.agent_bus.on_message)
        self.spawn(self.data_ctrl.baseline())
        self.spawn(self.reports.loop())
        self.spawn(self.model_ctrl.loop())

    async def stop(self) -> None:
        for t in list(self._bg):
            t.cancel()
        await self.tools.close()
        await self.bus.close()
        await self.nr.close()

    def flow_trace(self, corr: Optional[str]):
        rec = self.intents.get(corr or "")
        return (rec.trace if rec and rec.trace else None) or self.traces.get(corr)

    def cell_lock(self, cell: str) -> asyncio.Lock:
        return self.cell_locks.setdefault(cell, asyncio.Lock())

    def spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)
        t.add_done_callback(_log_exc)
        return t

    # ---- reference point helpers --------------------------------------------------------
    async def nr_call(self, msg_type: str, method: str, path: str, body: dict, corr: str, entity: str,
                      priority: int = 3, trace=None) -> Envelope:
        trace = trace or self.flow_trace(corr)
        env = self.nr.envelope(msg_type, corr, body, ep("resource", _NR_DEST.get(msg_type, "ResourceLayer")),
                               priority=priority, source=ep("network", entity), trace=trace)
        return await self.nr.send(method, path, env)

    async def sn_notify(self, msg_type: str, corr: str, body: dict, entity: str, dest_entity: str,
                        priority: int = 3, trace=None) -> None:
        trace = trace or self.flow_trace(corr)
        env = make(msg_type, "SN", corr, body, ep("network", entity), ep("service", dest_entity),
                   priority=priority, trace=trace)
        await self.bus.publish(f"sn.notify.{msg_type}", env)


# receiving entity = the Resource Layer block of the design's NR catalogue (RanCoreTwin, ComputePool, StoragePool and
# ToolHost are the classes that implement Communication, Computing, Storage and Tool Resources)
_NR_DEST = {
    "ResourceInventoryQuery": "ResourceLayer", "ResourceReserveRequest": "CommunicationResources",
    "ResourceReleaseRequest": "CommunicationResources", "ConfigValidateRequest": "CommunicationResources",
    "ConfigApplyRequest": "CommunicationResources", "ConfigRollbackRequest": "CommunicationResources",
    "TelemetrySubscribeRequest": "CommunicationResources", "ToolInvokeRequest": "ToolResources",
    "ComputeJobRequest": "ComputingResources", "StorageObjectRequest": "StorageResources",
}


def _log_exc(t: asyncio.Task) -> None:
    if not t.cancelled() and t.exception():
        log.error("background task failed", exc_info=t.exception())
