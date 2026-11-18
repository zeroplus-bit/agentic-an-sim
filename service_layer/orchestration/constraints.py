"""Constraint validation of a parsed intent against declarative shapes (datasets/intent_nlu/constraints.json).

Only values the user actually STATED are checked (explicit slots), so per-slice defaults never cause a conflict.
Result: {ok, blocking, violations: [{rule, severity, kpi, value, message, basis, suggest?}]}.
"""
from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Any

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
BLOCKING = ("error", "conflict")


class ConstraintValidator:
    def __init__(self, path: Path):
        spec = json.loads(path.read_text(encoding="utf-8"))
        self.rules = spec["rules"]
        self.slice_types = spec.get("sliceTypes", {})
        self.version = f"{path.name}:{len(self.rules)} rules"

    def check(self, slice_type: str, explicit: dict[str, Any]) -> dict[str, Any]:
        out = []
        for r in self.rules:
            if r["slice"] not in ("*", slice_type) or explicit.get(r["kpi"]) is None:
                continue
            v = float(explicit[r["kpi"]])
            hit = (not (r["range"][0] <= v <= r["range"][1])) if r["test"] == "outside" else OPS[r["test"]](v, r["value"])
            if hit:
                out.append({"rule": r["id"], "severity": r["severity"], "kpi": r["kpi"], "value": v,
                            "message": r["message"], "basis": r["basis"], **({"suggest": r["suggest"]} if "suggest" in r else {})})
        blocking = [x for x in out if x["severity"] in BLOCKING]
        return {"ok": not blocking, "blocking": bool(blocking), "violations": out,
                "sst": self.slice_types.get(slice_type, {}).get("sst"), "ruleset": self.version}
