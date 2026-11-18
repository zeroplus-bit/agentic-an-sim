"""UE/Edge Resources - UE population, mobility and traffic generation.

Starts from 5G-Network-Slicing/network_data/client_data.csv (each CSV client becomes a cohort
of UEs) and the traffic model of ml_orchestrator_demo.py::MLOrchestrator.generate_traffic,
including the emergency / special-event / IoT-surge scenarios of orchestrator_demo.py.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

SLICES = ("eMBB", "URLLC", "mMTC")


def cell_id(bs: int, sector: int) -> str:
    return f"gNB-{10 + bs:03d}-{sector}"


@dataclass
class UE:
    ue_id: str
    cell: str
    slice: str
    app_type: str
    x: float
    y: float
    active: bool = False
    group: Optional[str] = None
    pinned: bool = False               # intent-bound devices stay attached


@dataclass
class Scenario:
    is_emergency: bool = False
    is_special_event: bool = False
    is_iot_surge: bool = False
    cell_down: list[str] = field(default_factory=list)
    load_factor: float = 1.0           # global multiplier, handy for stress tests

    def flags(self) -> dict[str, bool]:
        return {"is_emergency": self.is_emergency, "is_special_event": self.is_special_event,
                "is_iot_surge": self.is_iot_surge}


class UeEdgeSimulator:
    COHORT = 10

    def __init__(self, client_csv: Optional[Path], cells: list[str], seed: int = 7):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.cells = cells
        self.scenario = Scenario()
        self.ues: dict[str, UE] = {}
        self.groups: dict[str, list[str]] = {}
        self.clock = datetime.now()
        self._counts: dict[tuple[str, str], int] = {}
        self._load(client_csv)
        self._recount()

    def _recount(self) -> None:
        c: dict[tuple[str, str], int] = {}
        for ue in self.ues.values():
            if ue.active:
                c[(ue.cell, ue.slice)] = c.get((ue.cell, ue.slice), 0) + 1
        self._counts = c

    # ---- population -----------------------------------------------------------
    def _load(self, path: Optional[Path]) -> None:
        rows = []
        if path and Path(path).exists():
            with open(path, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r.get("step") == "0"]
        if not rows:  # synthesize if the CSV is missing
            apps = {"eMBB": "video streaming", "URLLC": "remote surgery", "mMTC": "smart meters"}
            for i in range(40):
                s = SLICES[i % 3]
                rows.append({"client_id": i, "base_station_id": i % 6, "slice": s, "app_type": apps[s],
                             "x": self.rng.uniform(0, 1000), "y": self.rng.uniform(0, 1000)})
        n = 0
        for r in rows:
            bs = int(r["base_station_id"])
            for k in range(self.COHORT):
                sector = 1 + (k % 2)
                cid = cell_id(bs, sector)
                if cid not in self.cells:
                    cid = self.cells[n % len(self.cells)]
                uid = f"ue-{n:05d}"
                self.ues[uid] = UE(uid, cid, r["slice"], r["app_type"],
                                   float(r["x"]) + self.rng.gauss(0, 15), float(r["y"]) + self.rng.gauss(0, 15),
                                   active=self.rng.random() < 0.5)
                g = "ue:group/" + r["app_type"].lower().replace(" ", "-").replace("/", "")
                self.ues[uid].group = g
                self.groups.setdefault(g, []).append(uid)
                n += 1

    def ensure_group(self, group: str, cells: list[str], slice_type: str, size: int = 20) -> list[str]:
        """Create a named UE group (e.g. hospital-a-devices) attached to the given cells."""
        gid = group if group.startswith("ue:") else f"ue:group/{group}"
        if gid in self.groups:
            have = {self.ues[u].cell for u in self.groups[gid]}
            if all(c in have for c in cells):
                return self.groups[gid]
        members = list(self.groups.get(gid, []))
        start = len(members)
        cells = [c for c in cells if c not in {self.ues[u].cell for u in members}] or cells
        for i in range(start, start + size):
            uid = f"ue-{gid.split('/')[-1]}-{i:03d}"
            cell = cells[i % len(cells)] if cells else self.cells[0]
            self.ues[uid] = UE(uid, cell, slice_type, gid.split("/")[-1], self.rng.uniform(0, 1000),
                               self.rng.uniform(0, 1000), active=True, group=gid, pinned=True)
            members.append(uid)
        self.groups[gid] = members
        self._recount()
        return members

    # ---- dynamics -------------------------------------------------------------
    def step(self) -> None:
        """One second of mobility and activity."""
        p_on = 0.55 + (0.2 if self.scenario.is_special_event else 0) + (0.2 if self.scenario.is_iot_surge else 0)
        for ue in self.ues.values():
            if ue.pinned:
                ue.active = True
            elif self.rng.random() < 0.1:
                ue.active = self.rng.random() < p_on
            ue.x += self.rng.gauss(0, 2)
            ue.y += self.rng.gauss(0, 2)
            if self.rng.random() < 0.002:            # occasional handover to the sibling sector
                bs, sec = ue.cell.rsplit("-", 1)
                sib = f"{bs}-{2 if sec == '1' else 1}"
                if sib in self.cells:
                    ue.cell = sib
        self._recount()

    def active_ues(self, cell: str, slice_type: Optional[str] = None, group: Optional[str] = None) -> int:
        if group is None:
            if slice_type is not None:
                return self._counts.get((cell, slice_type), 0)
            return sum(self._counts.get((cell, s), 0) for s in SLICES)
        n = 0
        for ue in self.ues.values():
            if ue.active and ue.cell == cell and (slice_type is None or ue.slice == slice_type) \
                    and (group is None or ue.group == group or ue.group == f"ue:group/{group}"):
                n += 1
        return n

    def traffic(self, cell: str, t_offset_sec: float = 0.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Normalized per-slice demand [eMBB, URLLC, mMTC] for a cell (MLOrchestrator.generate_traffic)."""
        rng = rng or self.np_rng
        if cell in self.scenario.cell_down:
            return np.zeros(3)
        now = self.clock
        hour = ((now.hour * 3600 + now.minute * 60 + now.second + t_offset_sec) / 3600.0 % 24) / 24.0
        dow = now.weekday() / 6.0
        base = np.array([0.4, 0.3, 0.2])
        traffic = base * (1 + 0.3 * math.sin(2 * math.pi * hour) + 0.2 * (1 - dow))
        traffic = traffic + rng.normal(0, 0.05, 3)
        # UE load on this cell vs. average
        per_slice = np.array([self.active_ues(cell, s) for s in SLICES], dtype=float)
        avg = max(1.0, len(self.ues) / max(1, len(self.cells)) / 3 * 0.55)
        traffic *= 0.6 + 0.4 * np.clip(per_slice / avg, 0.2, 2.5)
        sc = self.scenario
        if sc.is_emergency:
            traffic *= np.array([0.8, 2.0, 0.9])
        if sc.is_special_event:
            traffic *= np.array([1.5, 0.8, 1.0])
        if sc.is_iot_surge:
            traffic *= np.array([0.9, 0.9, 1.8])
        traffic *= sc.load_factor
        return np.clip(traffic, 0.05, 2.0)

    def client_count_norm(self) -> float:
        return sum(1 for u in self.ues.values() if u.active) / max(1, len(self.ues))

    def inventory(self) -> list[dict]:
        out = []
        for g, members in self.groups.items():
            active = sum(1 for m in members if self.ues[m].active)
            out.append({"resourceId": g, "type": "ue", "members": len(members), "active": active})
        return out
