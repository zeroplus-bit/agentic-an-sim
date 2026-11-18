"""Digital-twin unit tests: mobility load balancing (steering), handover load on cell down, rollback."""
from __future__ import annotations

import pytest

from common import config
from resource_layer.comm_twin import RanCoreTwin
from resource_layer.ue_edge_sim import UeEdgeSimulator

ALLOC = {"eMBB": 0.9, "URLLC": 0.05, "mMTC": 0.05}


@pytest.fixture()
def twin():
    return RanCoreTwin(lambda cells: UeEdgeSimulator(config.CLIENT_CSV, cells), config.BS_CSV)


def _cfg(share, offload=None):
    c = {"slice": {"type": "eMBB"}, "intentShare": share, "isolation": "dedicated", "demandMbps": 200,
         "allocation": ALLOC, "qos": {}}
    if offload is not None:
        c["mobility"] = {"offload": offload}
    return c


def _apply(tw, cid, cfg, cell="gNB-013-1"):
    req = {"configId": cid, "intentId": "i1", "target": f"comm:{cell}", "config": cfg}
    assert tw.check(req) is None
    tw.apply(req)


def test_neighbours_are_same_site_and_adjacent_sites(twin):
    assert twin.neighbours["gNB-013-1"] == ["gNB-013-2", "gNB-012-1", "gNB-012-2", "gNB-014-1", "gNB-014-2"]
    assert "gNB-010-1" not in twin.neighbours["gNB-013-1"]


def test_steering_relieves_the_source_and_rollback_restores(twin):
    _apply(twin, "c1", _cfg(80))
    twin.ue.scenario.is_special_event = True
    twin.step()
    before = twin.cells["gNB-013-1"].last["intents"]["i1"]
    _apply(twin, "c2", _cfg(80, {"gNB-014-1": {"fraction": 0.3, "share": 36, "allocation": {"eMBB": 0.6, "URLLC": 0.25, "mMTC": 0.15}}}))
    twin.step()
    after = twin.cells["gNB-013-1"].last["intents"]["i1"]
    assert after["load"] < before["load"] - 0.2               # own part of the load drops by ~30%
    assert after["latency_ms"] < before["latency_ms"]          # merged KPI = worse of own and steered part
    assert "steeredLoad" in after
    assert "i1@gNB-013-1" in twin.cells["gNB-014-1"].bindings
    assert twin.intent_share("gNB-014-1", "i1") == pytest.approx(0.36)
    _apply(twin, "c3", _cfg(80))                               # a resize without mobility keeps the steering
    assert twin.cells["gNB-013-1"].bindings["i1"].offload == {"gNB-014-1": 0.3}
    assert twin.rollback("c3") and twin.rollback("c2")
    assert "i1@gNB-013-1" not in twin.cells["gNB-014-1"].bindings
    assert twin.cells["gNB-013-1"].bindings["i1"].offload == {}


def test_steering_checks(twin):
    _apply(twin, "c1", _cfg(80))
    bad = {"configId": "x", "intentId": "i1", "target": "comm:gNB-013-1"}
    assert "not a neighbour" in twin.check({**bad, "config": _cfg(80, {"gNB-010-1": {"fraction": 0.2, "share": 20}})})
    assert "fraction" in twin.check({**bad, "config": _cfg(80, {"gNB-014-1": {"fraction": 0.6, "share": 20}})})
    twin.ue.scenario.cell_down = ["gNB-014-1"]
    twin.step()
    assert "down" in twin.check({**bad, "config": _cfg(80, {"gNB-014-1": {"fraction": 0.2, "share": 20}})})


def test_handover_load_moves_to_surviving_cell(twin):
    for cell in ("gNB-012-1", "gNB-012-2"):
        req = {"configId": f"u-{cell}", "intentId": "u1", "target": f"comm:{cell}",
               "config": {"slice": {"type": "URLLC"}, "intentShare": 20, "isolation": "dedicated", "demandMbps": 50,
                          "allocation": {"eMBB": 0.4, "URLLC": 0.4, "mMTC": 0.2}, "qos": {}}}
        twin.apply(req)
    twin.step()
    base = twin.cells["gNB-012-2"].last["intents"]["u1"]["load"]
    twin.ue.scenario.cell_down = ["gNB-012-1"]
    twin.step()
    assert twin.cells["gNB-012-2"].last["intents"]["u1"]["load"] == pytest.approx(2 * base, rel=0.15)
