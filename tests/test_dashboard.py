"""Dashboard gateway: event hub ring, stage classification and incident correlation (no processes needed)."""
import logging
import time

from common.events import EventHub, HubLogHandler, brief
from dashboard.app import QUIET_SEC, Gateway


def _ev(seq, **kw):
    return {"seq": seq, "at": None, **kw}


def test_hub_counts_noisy_and_streams_rest():
    hub = EventHub("t")
    hub.publish({"kind": "msg", "msgType": "TelemetryNotify"})
    hub.publish({"kind": "msg", "msgType": "ConfigApplyRequest"})
    assert hub.counts["TelemetryNotify"] == 1 and hub.seq == 1
    assert [e["msgType"] for e in hub.after(0)] == ["ConfigApplyRequest"]
    lg = logging.getLogger("t.dash")
    lg.addHandler(HubLogHandler(hub))
    lg.warning("drift", extra={"intentId": "intent-9", "stage": "DETECT"})
    last = hub.after(1)[-1]
    assert last["kind"] == "log" and last["intentId"] == "intent-9" and last["stage"] == "DETECT"


def test_brief_keeps_operator_fields():
    b = brief("ConflictNotify", {"conflictId": "c1", "resource": "comm:x/prb", "resolution": "NEEDS_DECISION",
                                 "options": [{"action": "ACCEPT_PARTIAL"}], "huge": list(range(1000))})
    assert b["resource"] == "comm:x/prb" and b["options"] == ["ACCEPT_PARTIAL"] and "huge" not in b


def test_incident_lifecycle_detect_to_resolve():
    g = Gateway()
    feed = [
        ("resource", _ev(1, kind="msg", dir="out", msgType="ResourceAlarmNotify", refPoint="NR", correlationId="intent-1",
                         brief={"type": "CELL_DOWN", "severity": "CRITICAL", "resourceId": "comm:c1", "affectedIntents": ["intent-1"]})),
        ("network", _ev(1, kind="msg", dir="out", msgType="IntentStatusNotify", refPoint="SN", correlationId="intent-1", traceId="T1",
                        brief={"intentId": "intent-1", "state": "DEGRADED", "previousState": "FULFILLED"})),
        ("network", _ev(2, kind="log", level="INFO", logger="perf_ctrl", msg="assurance task task-1 -> ASSURE_KPI",
                        intentId="intent-1", stage="EXECUTE")),
        ("network", _ev(3, kind="msg", dir="out", msgType="ConfigApplyRequest", refPoint="NR", correlationId="intent-1",
                        brief={"intentId": "intent-1", "configId": "cfg-1"})),
        ("network", _ev(4, kind="msg", dir="in", msgType="ConfigApplyResponse", refPoint="NR", correlationId="intent-1")),
        ("network", _ev(5, kind="msg", dir="out", msgType="IntentStatusNotify", refPoint="SN", correlationId="intent-1",
                        brief={"intentId": "intent-1", "state": "FULFILLED", "previousState": "DEGRADED"})),
    ]
    for layer, d in feed:
        g.on_layer_event(layer, "event", d)
    assert len(g.events) == 5                       # the "in" copy is dropped: every message is shown once
    stages = [e["stage"] for e in g.events]
    assert stages == ["DETECT", "REPORT", "EXECUTE", "EXECUTE", "RESOLVE"]
    (inc,) = g.incidents.values()
    v = inc.view()
    assert v["state"] == "RESOLVED" and v["outcome"] == "recovered" and v["actions"] == ["NR-05 apply cfg-1"]
    assert v["mtta"] is not None and v["mttr"] is not None and v["traceId"] == "T1"
    assert g.kpis()["autonomy"] == 1.0


def test_alarm_without_sla_impact_closes_quietly():
    g = Gateway()
    g.on_layer_event("resource", "event", _ev(1, kind="msg", dir="out", msgType="ResourceAlarmNotify", refPoint="NR",
                                              brief={"type": "CAPACITY_EXHAUSTED", "severity": "MAJOR",
                                                     "affectedIntents": ["intent-2"]}))
    (inc,) = g.incidents.values()
    inc.active_at = time.time() - QUIET_SEC - 1
    g.expire([{"intentId": "intent-2", "state": "FULFILLED"}])
    assert inc.state == "CLOSED" and inc.outcome == "no SLA impact" and not g.open_by_intent
