"""NR reference point (Network Layer <-> Resource Layer): catalogue and body models NR-01..NR-13.

Resource ids are typed: comm:gNB-012-1, comp:node-3, stor:bucket-1, tool:gemini, ue:group/hospital-a-devices.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

RESOURCE_TYPES = ("comm", "comp", "stor", "tool", "ue")


def rtype(resource_id: str) -> str:
    return resource_id.split(":", 1)[0]


def rname(resource_id: str) -> str:
    return resource_id.split(":", 1)[1] if ":" in resource_id else resource_id


class _M(BaseModel):
    model_config = ConfigDict(extra="allow")


class ResourceInventoryQuery(_M):                    # NR-01 (GET ?type=)
    type: Optional[str] = None


class ReserveItem(_M):
    resourceId: str
    kind: Literal["PRB_SHARE", "CPU", "GPU", "MEMORY_MB", "STORAGE_GB", "TOOL_QUOTA"]
    amount: float
    unit: str = ""


class ResourceReserveRequest(_M):                    # NR-02
    reservationId: str
    intentId: Optional[str] = None
    items: list[ReserveItem]
    window: Optional[dict[str, str]] = None
    preemptible: bool = False
    priority: int = 3


class ResourceReleaseRequest(_M):                    # NR-03
    reservationId: str


class SliceConfig(_M):
    slice: dict[str, Any] = Field(default_factory=dict)          # {"sNssai": {...}, "type": "URLLC"}
    allocation: dict[str, float] = Field(default_factory=dict)   # {"eMBB": .4, "URLLC": .2, "mMTC": .4}
    qos: dict[str, Any] = Field(default_factory=dict)            # {"5qi": 82, "latencyBudgetMs": 5, "gbrMbps": 50}
    scheduling: dict[str, Any] = Field(default_factory=dict)     # {"policy": "PRIORITY", "preemption": True}
    intentShare: Optional[float] = None                           # PRB % dedicated to the intent on this cell
    ueGroup: Optional[str] = None


class ConfigRequest(_M):                             # NR-04 / NR-05
    configId: str
    reservationId: Optional[str] = None
    intentId: Optional[str] = None
    target: str
    config: SliceConfig
    origin: dict[str, Any] = Field(default_factory=dict)
    simulateSec: int = 60


class ConfigRollbackRequest(_M):                     # NR-06
    configId: str
    rollbackToken: Optional[str] = None
    reason: Optional[str] = None


class TelemetrySubscribeRequest(_M):                 # NR-07
    subscriptionId: str
    resources: list[str]
    scope: dict[str, Any] = Field(default_factory=dict)
    kpis: list[str] = Field(default_factory=lambda: ["latency_ms", "reliability_percent", "bandwidth_mbps",
                                                     "prb_utilization", "active_ues"])
    periodSec: float = 1
    format: Literal["SAMPLE", "AGGREGATE"] = "SAMPLE"
    cancel: bool = False


class TelemetryNotify(_M):                           # NR-08
    subscriptionId: str
    resourceId: str
    intentId: Optional[str] = None
    sampleTime: str
    seq: int
    values: dict[str, float]
    scenario: dict[str, bool]
    features: Optional[dict[str, float]] = None


class ResourceAlarmNotify(_M):                       # NR-09
    alarmId: str
    resourceId: str
    severity: Literal["CRITICAL", "MAJOR", "MINOR", "WARNING", "CLEARED"]
    type: Literal["CAPACITY_EXHAUSTED", "CELL_DOWN", "THRESHOLD_CROSSED", "NODE_OVERLOAD"]
    details: dict[str, Any] = Field(default_factory=dict)
    affectedIntents: list[str] = Field(default_factory=list)
    raisedAt: str


class ToolInvokeRequest(_M):                         # NR-10
    invocationId: str
    toolId: str
    operation: str = "generate"
    arguments: dict[str, Any] = Field(default_factory=dict)
    timeoutMs: int = 15000
    caller: dict[str, Any] = Field(default_factory=dict)


class ComputeJobRequest(_M):                         # NR-11
    jobId: str
    kind: Literal["INFERENCE", "TRAINING", "AGENT_RUNTIME", "MODEL_LOAD"]
    model: Optional[str] = None
    inputRef: Optional[str] = None
    resources: dict[str, float] = Field(default_factory=lambda: {"cpu": 1, "gpu": 0})
    deadlineMs: int = 500
    node: Optional[str] = None
    cancel: bool = False


class ComputeJobStatusNotify(_M):                    # NR-12
    jobId: str
    state: Literal["QUEUED", "RUNNING", "DONE", "FAILED", "CANCELLED"]
    node: Optional[str] = None
    outputRef: Optional[str] = None
    reason: Optional[str] = None


class StorageObjectRequest(_M):                      # NR-13 (metadata; object goes in HTTP body)
    bucket: str
    key: str
    contentType: str = "application/octet-stream"
    retentionHours: Optional[float] = None
    intentId: Optional[str] = None


NR_CATALOGUE: dict[str, tuple[str, str, str, str]] = {
    "NR-01": ("ResourceInventoryQuery", "N->R", "ResourceController -> all resources", "GET /nr/v1/resources?type="),
    "NR-02": ("ResourceReserveRequest", "N->R", "ResourceController -> Communication, Computing, Storage", "POST /nr/v1/reservations"),
    "NR-03": ("ResourceReleaseRequest", "N->R", "ResourceController -> same", "DELETE /nr/v1/reservations/{reservationId}"),
    "NR-04": ("ConfigValidateRequest", "N->R", "PreValidator -> Communication (twin copy)", "POST /nr/v1/configs:validate"),
    "NR-05": ("ConfigApplyRequest", "N->R", "ActionExecutor -> Communication, UE/Edge", "POST /nr/v1/configs"),
    "NR-06": ("ConfigRollbackRequest", "N->R", "ActionExecutor -> same", "POST /nr/v1/configs/{configId}:rollback"),
    "NR-07": ("TelemetrySubscribeRequest", "N->R", "DataController -> Communication, UE/Edge, Computing", "POST /nr/v1/telemetry/subscriptions"),
    "NR-08": ("TelemetryNotify", "R->N", "resources -> PerformanceMonitor, ModelExecutor", "nr.notify.TelemetryNotify"),
    "NR-09": ("ResourceAlarmNotify", "R->N", "resources -> PerformanceMonitor, ConflictManager", "nr.notify.ResourceAlarmNotify"),
    "NR-10": ("ToolInvokeRequest", "N->R", "ToolRegistry -> Tool Resources", "POST /nr/v1/tools/{toolId}:invoke"),
    "NR-11": ("ComputeJobRequest", "N->R", "ModelExecutor -> Computing Resources", "POST /nr/v1/compute/jobs"),
    "NR-12": ("ComputeJobStatusNotify", "R->N", "Computing Resources -> ModelExecutor", "nr.notify.ComputeJobStatusNotify"),
    "NR-13": ("StorageObjectRequest", "N->R", "Repository -> Storage Resources", "PUT/GET /nr/v1/storage/{bucket}/{key}"),
}

NR_BODY_MODELS: dict[str, type[BaseModel]] = {
    "ResourceInventoryQuery": ResourceInventoryQuery,
    "ResourceReserveRequest": ResourceReserveRequest,
    "ResourceReleaseRequest": ResourceReleaseRequest,
    "ConfigValidateRequest": ConfigRequest,
    "ConfigApplyRequest": ConfigRequest,
    "ConfigRollbackRequest": ConfigRollbackRequest,
    "TelemetrySubscribeRequest": TelemetrySubscribeRequest,
    "TelemetryNotify": TelemetryNotify,
    "ResourceAlarmNotify": ResourceAlarmNotify,
    "ToolInvokeRequest": ToolInvokeRequest,
    "ComputeJobRequest": ComputeJobRequest,
    "ComputeJobStatusNotify": ComputeJobStatusNotify,
    "StorageObjectRequest": StorageObjectRequest,
}
NR_MSG_TYPES = set(NR_BODY_MODELS)
