"""SN reference point (Service Layer <-> Network Layer): catalogue and body models SN-01..SN-14."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

KPI = Literal["latency_ms", "reliability_percent", "bandwidth_mbps", "prb_utilization", "active_ues",
              "jitter_ms", "packet_loss_percent", "availability_percent"]
LOWER_IS_BETTER = {"latency_ms", "jitter_ms", "packet_loss_percent", "prb_utilization"}

IntentState = Literal["RECEIVED", "VALIDATING", "PLANNING", "DEPLOYING", "FULFILLED", "DEGRADED",
                      "TERMINATED", "REJECTED", "FAILED"]
INTENT_TRANSITIONS: dict[str, set[str]] = {
    "RECEIVED": {"VALIDATING", "REJECTED", "FAILED", "TERMINATED"},
    "VALIDATING": {"PLANNING", "REJECTED", "FAILED", "TERMINATED"},
    "PLANNING": {"DEPLOYING", "REJECTED", "FAILED", "TERMINATED"},
    "DEPLOYING": {"FULFILLED", "DEGRADED", "PLANNING", "REJECTED", "FAILED", "TERMINATED"},
    "FULFILLED": {"DEGRADED", "PLANNING", "TERMINATED"},
    "DEGRADED": {"FULFILLED", "PLANNING", "DEPLOYING", "FAILED", "TERMINATED"},
    "TERMINATED": set(), "REJECTED": set(), "FAILED": set(),
}


class _M(BaseModel):
    model_config = ConfigDict(extra="allow")


# ---- intent object (SN-01, SN-05) ----------------------------------------------
class Expectation(_M):
    kpi: str
    op: Literal["<=", ">=", "<", ">", "=="]
    value: float
    percentile: Optional[float] = None

    def met_by(self, actual: Optional[float]) -> bool:
        if actual is None:
            return False
        return {"<=": actual <= self.value, ">=": actual >= self.value, "<": actual < self.value,
                ">": actual > self.value, "==": abs(actual - self.value) < 1e-9}[self.op]


class Classifier(_M):
    model: str
    version: str = "1.0"
    confidence: float = 0.0


class Semantic(_M):
    serviceCategory: Literal["eMBB", "URLLC", "mMTC"]
    classifier: Optional[Classifier] = None


class Area(_M):
    type: Literal["cellList", "polygon", "trackingArea"] = "cellList"
    cells: list[str] = Field(default_factory=list)


class TimeWindow(_M):
    start: Optional[str] = None
    end: Optional[str] = None


class IntentContext(_M):
    area: Area = Field(default_factory=Area)
    ueGroup: Optional[str] = None
    dnn: Optional[str] = None
    timeWindow: Optional[TimeWindow] = None


class Constraints(_M):
    maxCostPerHour: Optional[float] = None
    preferredProviders: list[str] = Field(default_factory=list)
    isolation: Literal["dedicated", "shared"] = "shared"


class ReportPolicy(_M):
    periodSec: float = 10
    onDrift: bool = True


class Intent(_M):
    intentId: str
    tenantId: str
    utterance: str = ""
    semantic: Semantic
    expectations: list[Expectation]
    context: IntentContext = Field(default_factory=IntentContext)
    constraints: Constraints = Field(default_factory=Constraints)
    reportPolicy: ReportPolicy = Field(default_factory=ReportPolicy)


# ---- bodies ----------------------------------------------------------------------
class IntentSubmitRequest(_M):                       # SN-01
    intent: Intent


class IntentModifyRequest(_M):                       # SN-02
    intentId: str
    expectations: Optional[list[Expectation]] = None
    context: Optional[IntentContext] = None
    constraints: Optional[Constraints] = None
    reportPolicy: Optional[ReportPolicy] = None
    priority: Optional[int] = None


class IntentCancelRequest(_M):                       # SN-03
    intentId: str
    reason: Optional[str] = None


class TaskInfo(_M):
    taskId: str
    agentId: Optional[str] = None
    type: str
    state: str


class IntentStatusNotify(_M):                        # SN-04
    intentId: str
    state: IntentState
    previousState: Optional[str] = None
    serviceInstanceId: Optional[str] = None
    tasks: list[TaskInfo] = Field(default_factory=list)
    fulfilment: dict[str, Any] = Field(default_factory=dict)
    explanation: Optional[str] = None
    reason: Optional[str] = None


class IntentFeasibilityQuery(_M):                    # SN-05
    intent: Intent


class MonitoringSubscribeRequest(_M):                # SN-06
    subscriptionId: str
    target: dict[str, Any]
    kpis: list[str] = Field(default_factory=lambda: ["latency_ms", "reliability_percent", "bandwidth_mbps"])
    periodSec: float = 10
    thresholdEvents: bool = True
    includeExplanation: bool = True


class PerformanceReportNotify(_M):                   # SN-07
    subscriptionId: Optional[str] = None
    intentId: str
    window: dict[str, str]
    kpis: dict[str, dict[str, float]]
    compliance: dict[str, Any]
    forecast: Optional[dict[str, Any]] = None
    explanation: Optional[str] = None


class AgentDescriptor(_M):
    agentId: str
    role: str
    version: str = "1.0.0"
    skills: list[str]
    tools: list[str] = Field(default_factory=list)
    policy: dict[str, Any] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=lambda: {"cpu": 1, "memoryMb": 512})
    permissions: list[str] = Field(default_factory=list)


class AgentDeployRequest(_M):                        # SN-08
    agentDescriptor: AgentDescriptor
    action: Literal["INSTANTIATE", "SCALE", "TERMINATE"] = "INSTANTIATE"


class AgentStateNotify(_M):                          # SN-09
    agentId: str
    state: Literal["REGISTERED", "ACTIVE", "BUSY", "DEGRADED", "TERMINATED"]
    endpoint: Optional[str] = None
    load: float = 0.0


class ModelSpec(_M):
    name: str
    version: str
    artifactUri: str
    framework: str = "numpy"
    inputSpec: dict[str, Any] = Field(default_factory=dict)
    outputSpec: dict[str, Any] = Field(default_factory=dict)


class DeployStrategy(_M):
    type: Literal["CANARY", "BLUE_GREEN", "ALL_AT_ONCE"] = "CANARY"
    trafficShare: float = 0.2
    rollbackOn: Optional[dict[str, Any]] = None


class ModelDeployRequest(_M):                        # SN-10
    deploymentId: str
    model: ModelSpec
    strategy: DeployStrategy = Field(default_factory=DeployStrategy)


class ModelPerformanceNotify(_M):                    # SN-11
    deploymentId: str
    metrics: dict[str, float]
    drift: bool = False
    state: Literal["DEPLOYING", "ACTIVE", "ROLLED_BACK", "PROMOTED"]


class DataPolicyRequest(_M):                         # SN-12
    policyId: str
    sources: list[str] = Field(default_factory=lambda: ["cell", "slice"])
    kpis: list[str] = Field(default_factory=list)
    periodSec: float = 5
    retentionHours: float = 24
    anonymize: bool = True


class ResourcePolicyRequest(_M):                     # SN-13
    tenantId: str
    quota: dict[str, float] = Field(default_factory=dict)
    overbooking: float = 1.0
    preemption: Literal["PRIORITY", "NONE"] = "PRIORITY"


class ConflictNotify(_M):                            # SN-14
    conflictId: str
    intents: list[str]
    resource: str
    resolution: Literal["PREEMPTED_LOWER_PRIORITY", "NEEDS_DECISION", "RESOLVED"]
    options: list[dict[str, Any]] = Field(default_factory=list)


# ---- catalogue -------------------------------------------------------------------
# id: (msgType, direction, sender -> receiver, binding)
SN_CATALOGUE: dict[str, tuple[str, str, str, str]] = {
    "SN-01": ("IntentSubmitRequest", "S->N", "ServiceProvisioner -> IntentMediator", "POST /sn/v1/intents"),
    "SN-02": ("IntentModifyRequest", "S->N", "ServiceProvisioner -> IntentMediator", "PATCH /sn/v1/intents/{intentId}"),
    "SN-03": ("IntentCancelRequest", "S->N", "ServiceProvisioner -> IntentMediator", "DELETE /sn/v1/intents/{intentId}"),
    "SN-04": ("IntentStatusNotify", "N->S", "TaskTracker -> ServiceMonitor", "sn.notify.IntentStatusNotify"),
    "SN-05": ("IntentFeasibilityQuery", "S->N", "IntentProcessor -> PreValidator", "POST /sn/v1/intents:feasibility"),
    "SN-06": ("MonitoringSubscribeRequest", "S->N", "ServiceMonitor -> PerformanceMonitor", "POST /sn/v1/subscriptions"),
    "SN-07": ("PerformanceReportNotify", "N->S", "ReportGenerator -> ServiceMonitor", "sn.notify.PerformanceReportNotify"),
    "SN-08": ("AgentDeployRequest", "S->N", "AgentLifecycleManager -> AgentRegistry", "POST /sn/v1/agents"),
    "SN-09": ("AgentStateNotify", "N->S", "AgentRegistry -> AgentLifecycleManager", "sn.notify.AgentStateNotify"),
    "SN-10": ("ModelDeployRequest", "S->N", "ModelLifecycleManager -> ModelController", "POST /sn/v1/models/deployments"),
    "SN-11": ("ModelPerformanceNotify", "N->S", "ModelController -> ModelLifecycleManager", "sn.notify.ModelPerformanceNotify"),
    "SN-12": ("DataPolicyRequest", "S->N", "DataManager -> DataController", "PUT /sn/v1/data-policies/{policyId}"),
    "SN-13": ("ResourcePolicyRequest", "S->N", "ResourceManager -> ResourceController", "PUT /sn/v1/resource-policies/{tenantId}"),
    "SN-14": ("ConflictNotify", "N->S", "ConflictManager -> ServiceOrchestratorAgent", "sn.notify.ConflictNotify"),
}

SN_BODY_MODELS: dict[str, type[BaseModel]] = {
    "IntentSubmitRequest": IntentSubmitRequest,
    "IntentModifyRequest": IntentModifyRequest,
    "IntentCancelRequest": IntentCancelRequest,
    "IntentStatusNotify": IntentStatusNotify,
    "IntentFeasibilityQuery": IntentFeasibilityQuery,
    "MonitoringSubscribeRequest": MonitoringSubscribeRequest,
    "PerformanceReportNotify": PerformanceReportNotify,
    "AgentDeployRequest": AgentDeployRequest,
    "AgentStateNotify": AgentStateNotify,
    "ModelDeployRequest": ModelDeployRequest,
    "ModelPerformanceNotify": ModelPerformanceNotify,
    "DataPolicyRequest": DataPolicyRequest,
    "ResourcePolicyRequest": ResourcePolicyRequest,
    "ConflictNotify": ConflictNotify,
}
SN_MSG_TYPES = set(SN_BODY_MODELS)
