"""Contract tests: every example of the design document round-trips through the Pydantic models,
plus envelope rules (refPoint, version, ttl, response naming) and token checks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from common import security  # noqa: E402
from common.envelope import Envelope, check, ep, make, respond, response_type  # noqa: E402
from common.nr_messages import NR_BODY_MODELS, NR_CATALOGUE  # noqa: E402
from common.sn_messages import SN_BODY_MODELS, SN_CATALOGUE, Expectation  # noqa: E402

INTENT = {
    "intentId": "intent-0001", "tenantId": "tenant-a",
    "utterance": "Remote surgery link between Hospital A and the city edge, under 5 ms",
    "semantic": {"serviceCategory": "URLLC",
                 "classifier": {"model": "distilbert-slice-intent", "version": "1.0", "confidence": 0.93}},
    "expectations": [{"kpi": "latency_ms", "op": "<=", "value": 5, "percentile": 99},
                     {"kpi": "reliability_percent", "op": ">=", "value": 99.999},
                     {"kpi": "bandwidth_mbps", "op": ">=", "value": 50}],
    "context": {"area": {"type": "cellList", "cells": ["gNB-012-1", "gNB-012-2"]}, "ueGroup": "hospital-a-devices",
                "timeWindow": {"start": "2026-11-17T08:00:00Z", "end": "2026-11-17T12:00:00Z"}},
    "constraints": {"maxCostPerHour": 250, "preferredProviders": [], "isolation": "dedicated"},
    "reportPolicy": {"periodSec": 10, "onDrift": True},
}

EXAMPLES = {
    "IntentSubmitRequest": {"intent": INTENT},
    "IntentFeasibilityQuery": {"intent": INTENT},
    "IntentStatusNotify": {"intentId": "intent-0001", "state": "FULFILLED", "previousState": "DEPLOYING",
                           "serviceInstanceId": "slice-7f3a91c2",
                           "tasks": [{"taskId": "task-11", "agentId": "agent-ran-01", "type": "ALLOCATE_SLICE", "state": "COMPLETED"}],
                           "fulfilment": {"latency_ms": 3.8, "met": True},
                           "explanation": "Dedicated URLLC slice on 2 cells, 20% PRB share; DQN action 14."},
    "MonitoringSubscribeRequest": {"subscriptionId": "sub-0001", "target": {"intentId": "intent-0001"},
                                   "kpis": ["latency_ms"], "periodSec": 10, "thresholdEvents": True, "includeExplanation": True},
    "PerformanceReportNotify": {"subscriptionId": "sub-0001", "intentId": "intent-0001",
                                "window": {"start": "2026-11-17T08:10:00Z", "end": "2026-11-17T08:10:10Z"},
                                "kpis": {"latency_ms": {"avg": 3.9, "p99": 4.7}}, "compliance": {"met": True, "violations": []},
                                "forecast": {"model": "lstm_single", "horizonSec": 60, "latency_ms_p99": 4.9},
                                "explanation": "Stable; forecast stays within 5 ms."},
    "AgentDeployRequest": {"agentDescriptor": {"agentId": "agent-ran-01", "role": "RAN_SLICE_ALLOCATOR", "version": "1.2.0",
                                               "skills": ["ALLOCATE_SLICE", "REBALANCE_PRB"],
                                               "tools": ["nr.config.apply", "nr.resource.reserve", "model.dqn.infer"],
                                               "policy": {"llm": "gemini-2.0-flash", "model": "dqn_slice_alloc:1.0"},
                                               "resources": {"cpu": 1, "memoryMb": 512},
                                               "permissions": ["nr:config:write", "nr:telemetry:read"]}},
    "AgentStateNotify": {"agentId": "agent-ran-01", "state": "ACTIVE", "endpoint": "bus://agents/agent-ran-01", "load": 0.35},
    "ModelDeployRequest": {"deploymentId": "dep-0003",
                           "model": {"name": "lstm_single", "version": "2.0", "artifactUri": "repo://models/lstm_single/2.0",
                                     "framework": "tensorflow-2.15", "inputSpec": {"shape": [10, 11]}, "outputSpec": {"shape": [3]}},
                           "strategy": {"type": "CANARY", "trafficShare": 0.2,
                                        "rollbackOn": {"metric": "mae", "op": ">", "value": 0.05}}},
    "ModelPerformanceNotify": {"deploymentId": "dep-0003", "metrics": {"mae": 0.027, "inferenceMsP95": 12}, "drift": False,
                               "state": "ACTIVE"},
    "DataPolicyRequest": {"policyId": "p1", "sources": ["cell", "slice", "ue"], "kpis": [], "periodSec": 5,
                          "retentionHours": 24, "anonymize": True},
    "ResourcePolicyRequest": {"tenantId": "tenant-a", "quota": {"prbSharePercent": 40, "computeCpu": 8, "storageGb": 50},
                              "overbooking": 1.2, "preemption": "PRIORITY"},
    "ConflictNotify": {"conflictId": "c1", "intents": ["intent-0001", "intent-0007"], "resource": "gNB-012-1/prb",
                       "resolution": "PREEMPTED_LOWER_PRIORITY", "options": []},
    # NR
    "ResourceReserveRequest": {"reservationId": "rsv-0042", "intentId": "intent-0001",
                               "items": [{"resourceId": "comm:gNB-012-1", "kind": "PRB_SHARE", "amount": 20, "unit": "percent"},
                                         {"resourceId": "comp:edge-1", "kind": "CPU", "amount": 2, "unit": "core"}],
                               "window": {"start": "2026-11-17T08:00:00Z", "end": "2026-11-17T12:00:00Z"},
                               "preemptible": False, "priority": 2},
    "ConfigValidateRequest": {"configId": "cfg-0101", "reservationId": "rsv-0042", "intentId": "intent-0001",
                              "target": "comm:gNB-012-1",
                              "config": {"slice": {"sNssai": {"sst": 2, "sd": "000101"}, "type": "URLLC"},
                                         "allocation": {"eMBB": 0.4, "URLLC": 0.2, "mMTC": 0.4},
                                         "qos": {"5qi": 82, "latencyBudgetMs": 5, "gbrMbps": 50},
                                         "scheduling": {"policy": "PRIORITY", "preemption": True}},
                              "origin": {"agentId": "agent-ran-01", "decision": {"model": "dqn_slice_alloc:1.0", "action": 14}},
                              "simulateSec": 60},
    "TelemetrySubscribeRequest": {"subscriptionId": "tsub-0007", "resources": ["comm:gNB-012-1"],
                                  "scope": {"sliceType": "URLLC", "intentId": "intent-0001"},
                                  "kpis": ["latency_ms"], "periodSec": 1, "format": "SAMPLE"},
    "TelemetryNotify": {"subscriptionId": "tsub-0007", "resourceId": "comm:gNB-012-1", "intentId": "intent-0001",
                        "sampleTime": "2026-11-17T08:10:03Z", "seq": 183,
                        "values": {"latency_ms": 4.1, "reliability_percent": 99.9995, "bandwidth_mbps": 59.8,
                                   "prb_utilization": 0.78, "active_ues": 42},
                        "scenario": {"is_emergency": False, "is_special_event": False, "is_iot_surge": False}},
    "ResourceAlarmNotify": {"alarmId": "alm-0009", "resourceId": "comm:gNB-012-1", "severity": "MAJOR",
                            "type": "CAPACITY_EXHAUSTED", "details": {"kpi": "prb_utilization", "value": 0.97, "threshold": 0.9},
                            "affectedIntents": ["intent-0001"], "raisedAt": "2026-11-17T08:42:10Z"},
    "ToolInvokeRequest": {"invocationId": "inv-0310", "toolId": "tool:gemini", "operation": "generate",
                          "arguments": {"model": "gemini-2.0-flash", "prompt": "Explain", "maxTokens": 512},
                          "timeoutMs": 15000, "caller": {"agentId": "agent-assure-01", "taskId": "task-12"}},
    "ComputeJobRequest": {"jobId": "j1", "kind": "INFERENCE", "model": "lstm_single:2.0",
                          "inputRef": "stor:bucket-1/windows/intent-0001/183", "resources": {"cpu": 1, "gpu": 0}, "deadlineMs": 500},
}


@pytest.mark.parametrize("msg_type,body", EXAMPLES.items())
def test_examples_roundtrip(msg_type, body):
    model = SN_BODY_MODELS.get(msg_type) or NR_BODY_MODELS[msg_type]
    obj = model.model_validate(body)
    again = model.model_validate(obj.model_dump(mode="json"))
    assert again == obj
    ref = "SN" if msg_type in SN_BODY_MODELS else "NR"
    env = make(msg_type, ref, "intent-0001", body, ep("service", "A"), ep("network", "B"))
    assert Envelope.model_validate_json(env.model_dump_json()).body == body


def test_catalogues_complete():
    assert len(SN_CATALOGUE) == 14 and len(NR_CATALOGUE) == 13
    for _id, (msg, *_rest) in {**SN_CATALOGUE, **NR_CATALOGUE}.items():
        assert msg in SN_BODY_MODELS or msg in NR_BODY_MODELS, msg


def test_envelope_rules():
    env = make("IntentSubmitRequest", "SN", "intent-1", {}, ep("service", "A"), ep("network", "B"), ttl_ms=1)
    assert check(env, "NR").status == "INVALID"
    env.header.version = "2.0"
    assert check(env, "SN").status == "INVALID"
    env.header.version = "1.3"
    env.header.timestamp = "2020-01-01T00:00:00.000Z"
    assert check(env, "SN").status == "EXPIRED"
    r = respond(env, "ACCEPTED", {"intentId": "intent-1"})
    assert r.header.msgType == "IntentSubmitResponse" and r.header.correlationId == "intent-1"
    assert r.header.source.entity == "B" and r.header.trace.traceId == env.header.trace.traceId
    assert response_type("ResourceInventoryQuery") == "ResourceInventoryResponse"


def test_tokens_and_perms():
    tok = security.issue("agent-x", "agent", None, ["nr:config:*"])
    c = security.verify(tok)
    assert c and security.has_perm(c, "nr:config:write") and not security.has_perm(c, "nr:resource:write")
    assert security.verify(tok[:-2] + "xx") is None


def test_expectation():
    assert Expectation(kpi="latency_ms", op="<=", value=5).met_by(4.9)
    assert not Expectation(kpi="bandwidth_mbps", op=">=", value=50).met_by(49)


def test_tracebook_keeps_one_trace_per_flow():
    from common.envelope import TraceBook, Trace, make, ep
    tb = TraceBook(maxlen=2)
    t = tb.for_flow("intent-1")
    tb.remember("intent-1", Trace())              # a later trace never replaces the flow's first one
    assert tb.get("intent-1") is t
    a = make("IntentSubmitRequest", "SN", "intent-1", {}, ep("service", "A"), ep("network", "B"), trace=t)
    b = make("ConfigApplyRequest", "NR", "intent-1", {}, ep("network", "B"), ep("resource", "C"), trace=tb.get("intent-1"))
    assert a.header.trace.traceId == b.header.trace.traceId == t.traceId
    assert a.header.trace.spanId != b.header.trace.spanId
    tb.for_flow("x"); tb.for_flow("y")            # bounded: oldest flow evicted
    assert tb.get("intent-1") is None
