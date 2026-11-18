"""Communication Resources - RAN/Core digital twin (RanCoreTwin).

Extended from 5G-Marketplace/src/ndt/simulator.py::NetworkDigitalTwin with the defects fixed:
* ONE long-lived instance per Resource Layer process (validation runs on a deep copy)
* consistent KPI names everywhere: latency_ms, bandwidth_mbps, reliability_percent
* violation counter is labelled (slice, kpi) instead of an unlabelled Gauge
Cells and initial PRB splits come from 5G-Network-Slicing/network_data/base_station_data.csv.

Model (per cell, capacity C Mbps, PRB split a = [eMBB, URLLC, mMTC]):
  background demand_s = traffic_s * C * 0.5            (traffic from UeEdgeSimulator)
  dedicated intent share p_i is carved out of its slice's fraction a_s
  load u = demand / capacity ; latency = L0 * (1 + 0.5 * rho / (1 - rho)) ; reliability drops when u > 0.85
"""
from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .ue_edge_sim import SLICES, UeEdgeSimulator, cell_id

IDX = {s: i for i, s in enumerate(SLICES)}
L0 = {"eMBB": 12.0, "URLLC": 1.2, "mMTC": 30.0}            # ms at zero load
R0 = {"eMBB": 99.95, "URLLC": 99.9997, "mMTC": 99.9}        # % at low load
RK = {"eMBB": 0.5, "URLLC": 0.008, "mMTC": 1.0}             # reliability loss per unit overload
BASE_TRAFFIC = np.array([0.4, 0.3, 0.2])
# mobility load balancing: steered UEs sit at the edge of the neighbour cell -> worse radio
EDGE_EFFICIENCY = 0.85      # bits per PRB relative to the serving cell (steered demand needs 1/0.85 the PRBs)
EDGE_LATENCY = 1.10         # latency multiplier for steered UEs
MAX_OFFLOAD = 0.4           # at most this fraction of an intent's UEs in a cell are at a steerable edge


@dataclass
class Binding:
    intent_id: str
    slice: str
    share: float                     # fraction of cell PRBs dedicated to the intent (0..1)
    demand_mbps: float
    isolation: str = "dedicated"
    priority_sched: bool = False
    ue_group: Optional[str] = None
    config_id: Optional[str] = None
    offload: dict[str, float] = field(default_factory=dict)   # neighbour cell -> fraction of this binding's UEs
    steered_from: Optional[tuple[str, str]] = None             # (intentId, source cell) for a steered binding


@dataclass
class Cell:
    cell_id: str
    capacity_mbps: float
    allocation: np.ndarray
    bindings: dict[str, Binding] = field(default_factory=dict)
    traffic: np.ndarray = field(default_factory=lambda: BASE_TRAFFIC.copy())
    up: bool = True
    last: dict[str, Any] = field(default_factory=dict)       # last computed KPIs

    def dedicated(self, slice_type: str) -> float:
        return sum(b.share for b in self.bindings.values() if b.slice == slice_type and b.isolation == "dedicated")


class RanCoreTwin:
    def __init__(self, ue_sim_factory, bs_csv: Optional[Path] = None, capacity_mbps: float = 400.0, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.cells: dict[str, Cell] = {}
        self._load_cells(bs_csv, capacity_mbps)
        self.ue: UeEdgeSimulator = ue_sim_factory(list(self.cells))
        self.config_history: dict[str, dict[str, Any]] = {}
        self.violations: dict[tuple[str, str], int] = {}          # labelled counter (slice, kpi)
        self.t = 0
        self.step()

    # ---- setup ------------------------------------------------------------------
    def _load_cells(self, path: Optional[Path], cap: float) -> None:
        first: dict[int, dict] = {}
        if path and Path(path).exists():
            with open(path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    first.setdefault(int(r["base_station_id"]), r)
        if not first:
            first = {b: {"eMBB_allocation": .4, "URLLC_allocation": .4, "mMTC_allocation": .2} for b in range(6)}
        for bs, r in sorted(first.items()):
            a = np.array([float(r["eMBB_allocation"]), float(r["URLLC_allocation"]), float(r["mMTC_allocation"])])
            a = np.clip(a / a.sum(), 0.1, 0.8)
            a = a / a.sum()
            for sector in (1, 2):
                cid = cell_id(bs, sector)
                self.cells[cid] = Cell(cid, cap, a.copy())
        self.neighbours: dict[str, list[str]] = {}
        for bs in sorted(first):   # sites on a line: other sector of the site + both sectors of adjacent sites
            for sector in (1, 2):
                n = [cell_id(bs, 3 - sector)] + [cell_id(b, k) for b in (bs - 1, bs + 1) if b in first for k in (1, 2)]
                self.neighbours[cell_id(bs, sector)] = n

    # ---- dynamics ---------------------------------------------------------------
    def step(self) -> None:
        self.t += 1
        self.ue.step()
        for c in self.cells.values():
            c.up = c.cell_id not in self.ue.scenario.cell_down
            c.traffic = self.ue.traffic(c.cell_id, rng=self.rng)
        self._recompute()

    def _recompute(self) -> None:
        for c in self.cells.values():
            c.last = self._compute(c)
        for c in self.cells.values():
            self._merge_steered(c)

    def _merge_steered(self, c: Cell) -> None:
        """An intent's KPIs in a cell = its own part and its steered part: the worse of the two."""
        for iid, b in c.bindings.items():
            if not b.offload or iid not in c.last.get("intents", {}):
                continue
            own = dict(c.last["intents"][iid])
            for t in b.offload:
                part = self.cells[t].last.get("intents", {}).get(f"{iid}@{c.cell_id}") if t in self.cells else None
                if not part:
                    continue
                own["latency_ms"] = max(own["latency_ms"], part["latency_ms"])
                own["reliability_percent"] = min(own["reliability_percent"], part["reliability_percent"])
                own["bandwidth_mbps"] = own["bandwidth_mbps"] + part["bandwidth_mbps"]
                own["steeredLoad"] = part["load"]
            c.last["intents"][iid] = own

    def _lat(self, slice_type: str, u: float, prio: bool) -> float:
        rho = min(max(u, 0.0), 0.97)
        lat = L0[slice_type] * (1 + 0.5 * rho / (1 - rho))
        if u > 1.0:
            lat *= 1 + 3 * (u - 1)
        if prio:
            lat *= 0.8
        return float(lat * self.rng.uniform(0.95, 1.08))

    def _rel(self, slice_type: str, u: float) -> float:
        return float(max(0.0, R0[slice_type] - max(0.0, u - 0.85) * RK[slice_type] - self.rng.uniform(0, RK[slice_type] * 0.01)))

    def _intent_demand(self, c: Cell, b: Binding) -> float:
        if b.steered_from:                       # the part of the source binding's UEs steered to this cell
            iid, src = b.steered_from
            sb = self.cells[src].bindings.get(iid) if src in self.cells else None
            if sb is None or not self.cells[src].up:
                return 0.0
            return self._raw_demand(self.cells[src], sb) * sb.offload.get(c.cell_id, 0.0) / EDGE_EFFICIENCY
        return self._raw_demand(c, b) * (1.0 - sum(b.offload.values()))

    def _raw_demand(self, c: Cell, b: Binding) -> float:
        sc = self.ue.scenario
        factor = sc.load_factor
        if sc.is_emergency and b.slice == "URLLC":
            factor *= 1.4
        if sc.is_special_event and b.slice == "eMBB":
            factor *= 1.3
        if sc.is_iot_surge and b.slice == "mMTC":
            factor *= 1.4
        # handover: when some of the intent's cells are down, their UEs move to the intent's cells still up
        bound = [x for x in self.cells.values() if b.intent_id in x.bindings]
        up = [x for x in bound if x.up]
        if up and len(up) < len(bound):
            factor *= len(bound) / len(up)
        return b.demand_mbps * factor * float(self.rng.uniform(0.96, 1.04))

    def _compute(self, c: Cell) -> dict[str, Any]:
        C = c.capacity_mbps
        out: dict[str, Any] = {"slices": {}, "intents": {}}
        if not c.up:
            for s in SLICES:
                out["slices"][s] = {"latency_ms": 999.0, "reliability_percent": 0.0, "bandwidth_mbps": 0.0,
                                    "utilization": 0.0, "load": 0.0}
            for iid in c.bindings:
                out["intents"][iid] = {"latency_ms": 999.0, "reliability_percent": 0.0, "bandwidth_mbps": 0.0, "load": 0.0}
            out["prb_utilization"] = 0.0
            return out
        busy = 0.0
        for s in SLICES:
            i = IDX[s]
            shared_cap = max(1e-3, c.allocation[i] - c.dedicated(s)) * C
            demand = c.traffic[i] * C * 0.5
            shared_intents = [b for b in c.bindings.values() if b.slice == s and b.isolation != "dedicated"]
            demand += sum(self._intent_demand(c, b) for b in shared_intents)
            u = demand / shared_cap
            prio = any(b.priority_sched for b in shared_intents)
            out["slices"][s] = {"latency_ms": self._lat(s, u, prio), "reliability_percent": self._rel(s, u),
                                "bandwidth_mbps": float(min(demand, shared_cap * 0.95)), "load": float(u),
                                "utilization": float(c.traffic[i] / max(c.allocation[i], 1e-6))}
            busy += min(1.0, u) * max(0.0, c.allocation[i] - c.dedicated(s))
            for b in shared_intents:
                d = self._intent_demand(c, b)
                out["intents"][b.intent_id] = {"latency_ms": out["slices"][s]["latency_ms"],
                                               "reliability_percent": out["slices"][s]["reliability_percent"],
                                               "bandwidth_mbps": float(d / max(1.0, u)), "load": float(u)}
        for b in c.bindings.values():
            if b.isolation != "dedicated":
                continue
            cap = max(1e-3, b.share * C)
            d = self._intent_demand(c, b)
            u = d / cap
            busy += min(1.0, u) * b.share
            edge = EDGE_LATENCY if b.steered_from else 1.0
            out["intents"][b.intent_id] = {"latency_ms": self._lat(b.slice, u, b.priority_sched) * edge,
                                           "reliability_percent": self._rel(b.slice, u),
                                           "bandwidth_mbps": float(min(d, cap * 0.95)), "load": float(u)}
        out["prb_utilization"] = float(min(0.999, busy))
        return out

    # ---- telemetry --------------------------------------------------------------
    def sample(self, cell: str, intent_id: Optional[str] = None, slice_type: Optional[str] = None) -> dict[str, float]:
        c = self.cells[cell]
        k = c.last
        if intent_id and intent_id in k.get("intents", {}):
            v = dict(k["intents"][intent_id])
            b = c.bindings[intent_id]
            v["active_ues"] = self.ue.active_ues(cell, group=b.ue_group) if b.ue_group else self.ue.active_ues(cell, b.slice)
        elif slice_type:
            v = dict(k["slices"][slice_type])
            v["active_ues"] = self.ue.active_ues(cell, slice_type)
        else:
            v = {f"{s}_{m}": k["slices"][s][m] for s in SLICES for m in ("latency_ms", "bandwidth_mbps")}
            v["active_ues"] = self.ue.active_ues(cell)
        v["prb_utilization"] = k.get("prb_utilization", 0.0)
        return {key: round(float(val), 6) for key, val in v.items()}

    def features(self, cell: str) -> dict[str, float]:
        """The 11 LSTM input features (ml_model_inputs.md)."""
        c = self.cells[cell]
        now = self.ue.clock
        util = c.traffic / np.maximum(c.allocation, 1e-6)
        up = sum(1 for x in self.cells.values() if x.up) / len(self.cells)
        return {"traffic_load": float(c.traffic.sum() / 3.0), "time_of_day": now.hour / 24.0,
                "day_of_week": now.weekday() / 6.0, "embb_alloc": float(c.allocation[0]),
                "urllc_alloc": float(c.allocation[1]), "mmtc_alloc": float(c.allocation[2]),
                "embb_util": float(util[0]), "urllc_util": float(util[1]), "mmtc_util": float(util[2]),
                "client_count": self.ue.client_count_norm(), "bs_count": up}

    def count_violation(self, slice_type: str, kpi: str) -> None:
        self.violations[(slice_type, kpi)] = self.violations.get((slice_type, kpi), 0) + 1

    # ---- configuration (NR-04/05/06) ----------------------------------------------
    @staticmethod
    def owned_by(key: str, intent_id: Optional[str]) -> bool:
        """A binding key belongs to an intent: its own binding or one it steered here ("<intent>@<source cell>")."""
        return bool(intent_id) and (key == intent_id or key.startswith(f"{intent_id}@"))

    def intent_share(self, cell: str, intent_id: Optional[str]) -> float:
        return sum(b.share for k, b in self.cells[cell].bindings.items() if self.owned_by(k, intent_id))

    def free_share(self, cell: str, exclude_intent: Optional[str] = None) -> float:
        c = self.cells[cell]
        used = sum(b.share for i, b in c.bindings.items() if b.isolation == "dedicated" and not self.owned_by(i, exclude_intent))
        return max(0.0, 0.9 - used)

    def check(self, req: dict) -> Optional[str]:
        cell = req["target"].split(":", 1)[-1]
        if cell not in self.cells:
            return f"unknown target {req['target']}"
        cfg = req["config"]
        alloc = cfg.get("allocation") or {}
        a = self._alloc_vec(alloc, self.cells[cell].allocation)
        share = cfg.get("intentShare")
        st = (cfg.get("slice") or {}).get("type")
        for t, o in ((cfg.get("mobility") or {}).get("offload") or {}).items():
            if t not in self.neighbours.get(cell, []):
                return f"{t} is not a neighbour of {cell}"
            if not self.cells[t].up:
                return f"offload target {t} is down"
            if not 0 < float(o["fraction"]) <= MAX_OFFLOAD + 1e-9:
                return f"offload fraction {o['fraction']} outside (0, {MAX_OFFLOAD}]"
            ta = self._alloc_vec(o.get("allocation") or {}, self.cells[t].allocation)
            key = f"{req.get('intentId')}@{cell}"
            ded = sum(b.share for i, b in self.cells[t].bindings.items()
                      if b.slice == st and b.isolation == "dedicated" and i != key)
            if float(o["share"]) / 100.0 + ded > ta[IDX[st]] + 1e-9:
                return f"steered share {o['share']}% + other dedicated {ded * 100:.1f}% exceeds {st} allocation on {t}"
        if share is not None and st:
            ded_other = sum(b.share for i, b in self.cells[cell].bindings.items()
                            if b.slice == st and b.isolation == "dedicated" and i != req.get("intentId"))
            if (share / 100.0) + ded_other > a[IDX[st]] + 1e-9:
                return (f"intentShare {share}% + other dedicated {ded_other*100:.1f}% exceeds {st} allocation "
                        f"{a[IDX[st]]*100:.1f}% on {cell}")
        return None

    @staticmethod
    def _alloc_vec(alloc: dict, current: np.ndarray) -> np.ndarray:
        if not alloc:
            return current.copy()
        a = np.array([float(alloc.get(s, current[IDX[s]])) for s in SLICES])
        a = np.clip(a, 0.05, 0.9)
        return a / a.sum()

    def apply(self, req: dict) -> dict[str, Any]:
        cell = req["target"].split(":", 1)[-1]
        c = self.cells[cell]
        iid = req.get("intentId")
        cfg = req["config"]
        prev = {"allocation": c.allocation.copy(), "binding": copy.deepcopy(c.bindings.get(iid)) if iid else None,
                "targets": {}}
        old_offload = dict(c.bindings[iid].offload) if iid and iid in c.bindings else {}
        c.allocation = self._alloc_vec(cfg.get("allocation") or {}, c.allocation)
        st = (cfg.get("slice") or {}).get("type")
        if iid and st and cfg.get("intentShare") is not None:
            qos = cfg.get("qos") or {}
            sched = cfg.get("scheduling") or {}
            c.bindings[iid] = Binding(
                intent_id=iid, slice=st, share=float(cfg["intentShare"]) / 100.0,
                demand_mbps=float(cfg.get("demandMbps") or qos.get("gbrMbps") or 20.0) * 1.15,
                isolation=cfg.get("isolation", "dedicated"),
                priority_sched=sched.get("policy") == "PRIORITY", ue_group=cfg.get("ueGroup"),
                config_id=req["configId"])
            if cfg.get("ueGroup"):
                self.ue.ensure_group(cfg["ueGroup"], [cell], st)
            mob = cfg.get("mobility")
            new_off = (mob.get("offload") or {}) if mob is not None else None
            key = f"{iid}@{cell}"
            for t in set(old_offload) | set(new_off or {}):
                tc = self.cells[t]
                prev["targets"][t] = {"allocation": tc.allocation.copy(), "binding": copy.deepcopy(tc.bindings.get(key))}
            if new_off is None:
                c.bindings[iid].offload = old_offload             # a resize keeps the steering in place
            else:
                c.bindings[iid].offload = {t: float(o["fraction"]) for t, o in new_off.items()}
                for t in old_offload:
                    if t not in new_off:
                        self.cells[t].bindings.pop(key, None)
                for t, o in new_off.items():
                    tc = self.cells[t]
                    tc.allocation = self._alloc_vec(o.get("allocation") or {}, tc.allocation)
                    tc.bindings[key] = Binding(intent_id=key, slice=st, share=float(o["share"]) / 100.0, demand_mbps=0.0,
                                               isolation="dedicated", priority_sched=c.bindings[iid].priority_sched,
                                               steered_from=(iid, cell))
        self.config_history[req["configId"]] = {"cell": cell, "intentId": iid, "prev": prev}
        self._recompute()
        return {"configId": req["configId"], "rollbackToken": f"rb-{req['configId']}"}

    def rollback(self, config_id: str) -> bool:
        h = self.config_history.pop(config_id, None)
        if not h:
            return False
        c = self.cells[h["cell"]]
        c.allocation = h["prev"]["allocation"]
        iid = h["intentId"]
        if iid:
            if h["prev"]["binding"] is None:
                c.bindings.pop(iid, None)
            else:
                c.bindings[iid] = h["prev"]["binding"]
        for t, tp in h["prev"].get("targets", {}).items():
            tc = self.cells[t]
            tc.allocation = tp["allocation"]
            if tp["binding"] is None:
                tc.bindings.pop(f"{iid}@{h['cell']}", None)
            else:
                tc.bindings[f"{iid}@{h['cell']}"] = tp["binding"]
        self._recompute()
        return True

    def remove_intent(self, intent_id: str) -> None:
        for c in self.cells.values():
            c.bindings.pop(intent_id, None)
            for k in [k for k in c.bindings if k.startswith(f"{intent_id}@")]:
                c.bindings.pop(k)

    def validate(self, req: dict, simulate_sec: int = 60) -> dict[str, Any]:
        """Dry-run on twin copies: the candidate config vs. an unchanged baseline."""
        err = self.check(req)
        if err:
            return {"ok": False, "reason": err}
        simulate_sec = int(max(5, min(simulate_sec, 300)))
        cand = copy.deepcopy(self)
        base = copy.deepcopy(self)
        cand.apply(req)
        cell = req["target"].split(":", 1)[-1]
        iid = req.get("intentId")
        st = (req["config"].get("slice") or {}).get("type")
        lat, rel, bw, prb = [], [], [], []
        side = {s: [] for s in SLICES}
        for _ in range(simulate_sec):
            cand.step()
            base.step()
            v = cand.sample(cell, intent_id=iid if iid in cand.cells[cell].bindings else None, slice_type=st)
            lat.append(v.get("latency_ms", 0.0))
            rel.append(v.get("reliability_percent", 0.0))
            bw.append(v.get("bandwidth_mbps", 0.0))
            prb.append(v.get("prb_utilization", 0.0))
            for s in SLICES:
                side[s].append(cand.cells[cell].last["slices"][s]["bandwidth_mbps"]
                               - base.cells[cell].last["slices"][s]["bandwidth_mbps"])
        side_effects = [{"slice": s, "bandwidth_mbps_delta": round(float(np.mean(d)), 2),
                         "latency_ms_after": round(float(cand.cells[cell].last["slices"][s]["latency_ms"]), 3)}
                        for s, d in side.items() if s != st and abs(np.mean(d)) > 0.5]
        return {"ok": True, "predicted": {
            "latency_ms": round(float(np.mean(lat)), 3),
            "latency_ms_p99": round(float(np.percentile(lat, 99)), 3),
            "reliability_percent": round(float(np.mean(rel)), 5),
            "bandwidth_mbps": round(float(np.mean(bw)), 2),
            "prb_utilization": round(float(np.mean(prb)), 3)}, "sideEffects": side_effects}

    # ---- inventory ---------------------------------------------------------------
    def inventory(self) -> list[dict[str, Any]]:
        return [{"resourceId": f"comm:{c.cell_id}", "type": "comm", "up": c.up, "capacityMbps": c.capacity_mbps,
                 "allocation": {s: round(float(c.allocation[IDX[s]]), 4) for s in SLICES},
                 "freeSharePercent": round(self.free_share(c.cell_id) * 100, 1),
                 "prbUtilization": round(c.last.get("prb_utilization", 0.0), 3),
                 "neighbours": self.neighbours.get(c.cell_id, []),
                 "intents": {i: {"slice": b.slice, "sharePercent": round(b.share * 100, 1), "isolation": b.isolation,
                                 **({"steeredFrom": b.steered_from[1]} if b.steered_from else {}),
                                 **({"offload": dict(b.offload)} if b.offload else {})}
                             for i, b in c.bindings.items()}}
                for c in self.cells.values()]
