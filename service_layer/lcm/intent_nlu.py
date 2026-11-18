"""Model LCM for the intent NLU (Flow 4 applied to the Service Layer's own model).

Versions are configurations over the shipped artifacts (keyword rules, SBERT kNN memory, TF joint model):
the primary serves SN-01; an optional SHADOW version runs on every submitted utterance without affecting it.
Operators confirm the correct interpretation (API / SN MCP `confirm_intent_interpretation`); confirmed rows are
appended to datasets/intent_nlu/feedback.jsonl (lineage for the next calibration/training round).

Decision rule (objective, paired): after >= MIN_SAMPLES confirmed intents seen by both versions, an exact
McNemar test on per-intent correctness. Shadow significantly better (p < ALPHA) -> PROMOTE_RECOMMENDED (promoted
automatically when autoPromote); significantly worse -> REJECTED (shadow stopped); otherwise COLLECTING.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Optional

from common import config

MIN_SAMPLES = 20
ALPHA = 0.05
VERSIONS: dict[str, dict[str, Any]] = {
    "1.0": {"intent": "keyword", "slots": "regex", "desc": "keyword rules + regex KPIs (original)"},
    "1.1": {"intent": "knn", "slots": "regex", "desc": "P1: SBERT kNN + temperature + conformal, regex KPIs"},
    "2.0": {"intent": "knn", "slots": "joint", "desc": "P2: SBERT kNN intent + TF joint slot tagger"},
    "2.1": {"intent": "ensemble", "slots": "joint+regex", "desc": "kNN+TF ensemble intent, tagger with regex fill-in"},
    "2.2": {"intent": "joint", "slots": "joint", "desc": "TF joint model for intent and slots"},
}


def mcnemar_exact(a_ok: list[bool], b_ok: list[bool]) -> dict[str, Any]:
    b = sum(x and not y for x, y in zip(a_ok, b_ok))
    c = sum(y and not x for x, y in zip(a_ok, b_ok))
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return {"onlyPrimaryRight": b, "onlyShadowRight": c, "p": round(p, 5)}


class IntentNluLifecycle:
    def __init__(self, processor, data_dir: Optional[Path] = None):
        self.ip = processor
        self.dir = data_dir or config.INTENT_DATA_DIR
        self.records: dict[str, dict[str, Any]] = {}
        self.shadow: Optional[dict[str, Any]] = None
        self.history: list[dict[str, Any]] = []
        J = self._joint()
        self.primary = "1.0" if J is None else next(
            (v for v, c in VERSIONS.items() if c["intent"] == J.mode and c["slots"] == J.slot_mode), "custom")
        self.previous: Optional[str] = None

    def _joint(self):
        J = self.ip.semantic
        return J if (J is not None and hasattr(J, "slots")) else None

    def available(self) -> list[str]:
        return [v for v, c in VERSIONS.items() if c["intent"] == "keyword" or self._joint() is not None]

    # ---- running versions -----------------------------------------------------------------------------
    def run(self, versions: list[str], text: str) -> dict[str, dict[str, Any]]:
        J = self._joint()
        r = J.raw([text]) if J is not None else None
        out = {}
        for v in versions:
            c = VERSIONS[v]
            if c["intent"] == "keyword":
                label = self.ip.keyword_classify(text)[0]
            else:
                p = J.probs(r, c["intent"])[0]
                label = ["URLLC", "eMBB", "mMTC"][int(p.argmax())]
            slots = self.ip._regex_slots(text) if (c["slots"] == "regex" or r is None) else J.slots(r, c["slots"])[0]
            out[v] = {"label": label, "slots": {k: float(x) for k, x in slots.items()}}
        return out

    def observe(self, intent_id: str, text: str, intent: dict[str, Any]) -> None:
        sem = intent.get("semantic", {})
        prim = {"label": sem.get("serviceCategory"),
                "slots": {k: float(x) for k, x in (sem.get("extractedSlots") or self.ip._regex_slots(text)).items()}}
        rec = {"intentId": intent_id, "text": text, "at": time.time(), "pred": {self.primary: prim}, "gold": None}
        if self.shadow and self.shadow["state"] == "SHADOW":
            rec["pred"].update(self.run([self.shadow["version"]], text))
        self.records[intent_id] = rec
        if len(self.records) > 5000:
            self.records.pop(next(iter(self.records)))

    # ---- feedback -------------------------------------------------------------------------------------
    def feedback(self, intent_id: str, slice_type: str, expectations: Optional[dict[str, float]] = None,
                 devices: Optional[int] = None, operator: str = "operator") -> dict[str, Any]:
        rec = self.records.get(intent_id)
        if rec is None:
            return {"ok": False, "reason": f"no parsed utterance recorded for {intent_id}"}
        if slice_type not in ("URLLC", "eMBB", "mMTC"):
            return {"ok": False, "reason": "slice must be URLLC, eMBB or mMTC"}
        rec["gold"] = {"slice": slice_type, "expectations": expectations, "devices": devices, "by": operator, "at": time.time()}
        row = {"id": f"fb-{intent_id}", "text": rec["text"], "slice": slice_type, "expectations": expectations or {},
               "devices": devices, "source": "operator-feedback", "slotsConfirmed": expectations is not None,
               "predictions": rec["pred"]}
        with (self.dir / "feedback.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        decision = self.evaluate()
        return {"ok": True, "correct": {v: p["label"] == slice_type for v, p in rec["pred"].items()}, "decision": decision}

    # ---- metrics / decision ---------------------------------------------------------------------------
    @staticmethod
    def _slot_ok(pred: dict[str, Any], gold: dict[str, Any]) -> Optional[bool]:
        if gold.get("expectations") is None:
            return None
        g = {k: float(v) for k, v in gold["expectations"].items()}
        if gold.get("devices") is not None:
            g["devices"] = float(gold["devices"])
        return pred["slots"] == g

    def metrics(self, version: str) -> dict[str, Any]:
        rows = [r for r in self.records.values() if r["gold"] and version in r["pred"]]
        sl = [self._slot_ok(r["pred"][version], r["gold"]) for r in rows]
        sl = [x for x in sl if x is not None]
        return {"labelled": len(rows),
                "accuracy": round(sum(r["pred"][version]["label"] == r["gold"]["slice"] for r in rows) / len(rows), 4) if rows else None,
                "slotExactMatch": round(sum(sl) / len(sl), 4) if sl else None, "slotLabelled": len(sl)}

    def evaluate(self) -> Optional[dict[str, Any]]:
        if not self.shadow:
            return None
        if self.shadow["state"] != "SHADOW":
            return self.shadow.get("decision")         # already decided (REJECTED / PROMOTED)
        v = self.shadow["version"]
        rows = [r for r in self.records.values() if r["gold"] and v in r["pred"] and self.primary in r["pred"]]
        a = [r["pred"][self.primary]["label"] == r["gold"]["slice"] for r in rows]
        b = [r["pred"][v]["label"] == r["gold"]["slice"] for r in rows]
        # a version that ties on intent but extracts KPIs better counts as better on that intent
        for i, r in enumerate(rows):
            sa, sb = self._slot_ok(r["pred"][self.primary], r["gold"]), self._slot_ok(r["pred"][v], r["gold"])
            if sa is not None and a[i] == b[i] and a[i]:
                a[i], b[i] = bool(sa), bool(sb)
        test = mcnemar_exact(a, b)
        d = {"paired": len(rows), "test": test, "minSamples": MIN_SAMPLES, "alpha": ALPHA}
        if len(rows) < MIN_SAMPLES:
            d["state"] = "COLLECTING"
        elif test["p"] < ALPHA and test["onlyShadowRight"] > test["onlyPrimaryRight"]:
            d["state"] = "PROMOTE_RECOMMENDED"
        elif test["p"] < ALPHA:
            d["state"] = "REJECTED"
        else:
            d["state"] = "COLLECTING"
        self.shadow["decision"] = d
        if d["state"] == "REJECTED":
            self.shadow["state"] = "REJECTED"
            self.history.append({"at": time.time(), "event": "SHADOW_REJECTED", "version": v, "decision": d})
        elif d["state"] == "PROMOTE_RECOMMENDED" and self.shadow.get("autoPromote"):
            self.promote(v, reason=f"auto: McNemar p={test['p']}")
        return d

    # ---- control --------------------------------------------------------------------------------------
    def start_shadow(self, version: str, auto_promote: bool = False) -> dict[str, Any]:
        if version not in self.available():
            return {"ok": False, "reason": f"unknown or unavailable version {version}; available {self.available()}"}
        if version == self.primary:
            return {"ok": False, "reason": f"{version} is already primary"}
        self.shadow = {"version": version, "state": "SHADOW", "autoPromote": auto_promote, "since": time.time()}
        self.history.append({"at": time.time(), "event": "SHADOW_STARTED", "version": version})
        return {"ok": True, **self.shadow}

    def _apply(self, version: str) -> None:
        c = VERSIONS[version]
        J = self._joint()
        if c["intent"] == "keyword":
            self.ip.force_keyword = True
        else:
            self.ip.force_keyword = False
            J.mode, J.slot_mode = c["intent"], c["slots"]
            self.ip.model_info.update({"intentMode": J.mode, "slotMode": J.slot_mode})
        self.ip.model_info["lcmVersion"] = version

    def promote(self, version: str, reason: str = "operator") -> dict[str, Any]:
        if version not in self.available():
            return {"ok": False, "reason": f"unknown or unavailable version {version}"}
        self.previous, self.primary = self.primary, version
        self._apply(version)
        if self.shadow and self.shadow["version"] == version:
            self.shadow["state"] = "PROMOTED"
        self.history.append({"at": time.time(), "event": "PROMOTED", "version": version, "from": self.previous, "reason": reason})
        return {"ok": True, "primary": version, "previous": self.previous}

    def rollback(self) -> dict[str, Any]:
        if not self.previous:
            return {"ok": False, "reason": "nothing to roll back to"}
        v = self.previous
        self.previous, self.primary = None, v
        self._apply(v)
        self.history.append({"at": time.time(), "event": "ROLLED_BACK", "version": v})
        return {"ok": True, "primary": v}

    def view(self) -> dict[str, Any]:
        vs = {self.primary} | ({self.shadow["version"]} if self.shadow else set())
        return {"primary": self.primary, "versions": {v: {**c, "available": v in self.available()} for v, c in VERSIONS.items()},
                "shadow": self.shadow, "metrics": {v: self.metrics(v) for v in vs},
                "recorded": len(self.records), "labelled": sum(1 for r in self.records.values() if r["gold"]),
                "history": self.history[-20:], "rule": {"minSamples": MIN_SAMPLES, "alpha": ALPHA, "test": "exact McNemar"}}
