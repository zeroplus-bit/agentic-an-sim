"""Intent-NLU evaluation harness (P0), calibration fitter (P1) and joint TF model evaluation (P2).

    python tools/eval_intent.py                       # evaluate all systems on test and test_v2
    python tools/eval_intent.py --write-calibration   # also store temperatures, conformal q-hat, chosen modes

Protocol: train.jsonl (+ train_slots.jsonl for the TF model) are training data, calib.jsonl fits temperatures,
conformal threshold and the deployed mode, test.jsonl (seen during P1 development) and test_v2.jsonl (written
blind before the P2 model was trained) are reported only. Outputs datasets/intent_nlu/eval_report.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import config  # noqa: E402
from service_layer.orchestration.intent_processor import IntentProcessor  # noqa: E402
from service_layer.orchestration.joint_nlu import JointIntentModel, JointNluNumpy  # noqa: E402
from service_layer.orchestration.semantic import (CLASSES, CharNgramEncoder, SbertOnnxEncoder,  # noqa: E402
                                                  SemanticIntentClassifier, conformal_qhat, conformal_sets,
                                                  load_jsonl, load_references)

D = config.INTENT_DATA_DIR
JOINT = ROOT / "models/joint_nlu/joint_nlu.npz"
KPIS = ["latency_ms", "bandwidth_mbps", "reliability_percent", "devices"]
TESTS = {"test": "test.jsonl", "test_v2": "test_v2.jsonl"}


# ---- metrics -------------------------------------------------------------------------------------------
def f1_report(y: list[str], p: list[str]) -> dict:
    per = {}
    for c in CLASSES:
        tp = sum(a == c and b == c for a, b in zip(y, p))
        fp = sum(a != c and b == c for a, b in zip(y, p))
        fn = sum(a == c and b != c for a, b in zip(y, p))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0}
    return {"accuracy": round(float(np.mean([a == b for a, b in zip(y, p)])), 4),
            "macroF1": round(float(np.mean([v["f1"] for v in per.values()])), 4), "perClass": per,
            "confusion": {a: {b: sum(1 for u, v in zip(y, p) if u == a and v == b) for b in CLASSES} for a in CLASSES}}


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return round(float(e), 4)


def mcnemar_exact(a_ok: list[bool], b_ok: list[bool]) -> dict:
    """Exact (binomial) McNemar test on paired correctness."""
    b = sum(x and not y for x, y in zip(a_ok, b_ok))
    c = sum(y and not x for x, y in zip(a_ok, b_ok))
    n = b + c
    if n == 0:
        return {"onlyA": 0, "onlyB": 0, "p": 1.0}
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return {"onlyA": b, "onlyB": c, "p": round(p, 5)}


def bootstrap_ci(y: list[str], pa: list[str], pb: list[str], n: int = 2000, seed: int = 0) -> list[float]:
    """95% paired bootstrap CI of macroF1(B) - macroF1(A)."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        s = rng.choice(len(y), len(y))
        yy = [y[i] for i in s]
        d.append(f1_report(yy, [pb[i] for i in s])["macroF1"] - f1_report(yy, [pa[i] for i in s])["macroF1"])
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def gold_slots(r: dict) -> dict[str, float]:
    g = {k: float(v) for k, v in (r.get("expectations") or {}).items()}
    if r.get("devices") is not None:
        g["devices"] = float(r["devices"])
    return g


def regex_slots(text: str) -> dict:
    d = dict(IntentProcessor.extract_explicit(text))
    dev = IntentProcessor.devices(text)
    if dev is not None:
        d["devices"] = dev
    return d


def slot_report(rows: list[dict], preds: list[dict]) -> dict:
    out = {}
    for k in KPIS:
        tp = fp = fn = 0
        for r, p in zip(rows, preds):
            g, v = gold_slots(r).get(k), p.get(k)
            if v is not None and g is not None and abs(float(v) - g) < 1e-6:
                tp += 1
            else:
                fp += v is not None
                fn += g is not None
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[k] = {"precision": round(pr, 3), "recall": round(rc, 3),
                  "f1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0, "support": tp + fn}
    ok = [{k: float(v) for k, v in p.items()} == gold_slots(r) for r, p in zip(rows, preds)]
    out["exactMatch"] = round(float(np.mean(ok)), 4)
    out["errors"] = [r["id"] for r, o in zip(rows, ok) if not o]
    return out


# ---- per-system summary --------------------------------------------------------------------------------
def summarize(name: str, labels: list[str], conf: np.ndarray, test: list[dict], probs=None, sets=None) -> dict:
    y = [r["slice"] for r in test]
    rep = f1_report(y, labels)
    ok = np.array([a == b for a, b in zip(y, labels)], dtype=float)
    rep["accuracyAltAware"] = round(float(np.mean([p == r["slice"] or p in r.get("alt", []) for p, r in zip(labels, test)])), 4)
    rep["byLang"] = {lg: round(float(np.mean([o for o, r in zip(ok, test) if r["lang"] == lg])), 4) for lg in ("en", "ko")}
    rep["ece"] = ece(conf, ok)
    if probs is not None:
        rep["nll"] = round(float(-np.mean([math.log(max(p[CLASSES.index(r["slice"])], 1e-12)) for p, r in zip(probs, test)])), 4)
    if sets is not None:
        cov = [r["slice"] in s for s, r in zip(sets, test)]
        single = [len(s) == 1 for s in sets]
        rep["conformal"] = {"coverage": round(float(np.mean(cov)), 4), "avgSetSize": round(float(np.mean([len(s) for s in sets])), 3),
                            "singletonRate": round(float(np.mean(single)), 4),
                            "singletonAccuracy": round(float(np.mean([c for c, s in zip(cov, single) if s])), 4) if any(single) else None,
                            "clarifyIds": [r["id"] for s, r in zip(sets, test) if len(s) > 1],
                            "clarifyCatchesError": sum(1 for s, l, r in zip(sets, labels, test) if len(s) > 1 and l != r["slice"])}
    rep["priorityAccuracy"] = round(float(np.mean([IntentProcessor.priority(r["text"], p) == r["priority"] for p, r in zip(labels, test)])), 4)
    rep["errors"] = [{"id": r["id"], "gold": r["slice"], "pred": p, "text": r["text"]} for p, r in zip(labels, test) if p != r["slice"]]
    rep["system"] = name
    return rep


def from_probs(name, p: np.ndarray, qhat, test) -> dict:
    labels = [CLASSES[i] for i in p.argmax(1)]
    return summarize(name, labels, p.max(1), test, p, conformal_sets(p, qhat))


# ---- report --------------------------------------------------------------------------------------------
def to_md(report: dict) -> str:
    L = ["# Intent NLU evaluation", "",
         f"Splits: train {report['n']['train']} (template, kNN memory) + train_slots {report['n']['train_slots']} "
         f"(template, TF joint model) · calib {report['n']['calib']} (hand-written: temperatures, conformal q-hat, "
         f"mode selection) · test {report['n']['test']} (hand-written, seen during P1) · test_v2 {report['n']['test_v2']} "
         f"(hand-written blind, before P2 training). alpha = {report['alpha']}.", ""]
    if report.get("deployed"):
        dep = report["deployed"]
        L += [f"Deployed (selected on calib): intent mode **{dep['mode']}**, slot mode **{dep['slotMode']}**. "
              f"Calib NLL {dep['selection']['calibNLL']}, calib slot exact match {dep['selection']['calibSlotExactMatch']}.", ""]
    for split, block in report["splits"].items():
        L += [f"## {split}", "",
              "| system | acc | macro-F1 | acc (alt ok) | en | ko | ECE | NLL | coverage | avg set | clarify→error | priority acc |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in block["systems"]:
            c = s.get("conformal") or {}
            cl = f"{c['clarifyCatchesError']}/{len(c['clarifyIds'])}" if c else "–"
            L.append(f"| {s['system']} | {s['accuracy']} | {s['macroF1']} | {s['accuracyAltAware']} | {s['byLang']['en']} | "
                     f"{s['byLang']['ko']} | {s['ece']} | {s.get('nll', '–')} | {c.get('coverage', '–')} | "
                     f"{c.get('avgSetSize', '–')} | {cl} | {s['priorityAccuracy']} |")
        L += ["", "Paired significance (McNemar exact, Δ macro-F1 95% bootstrap CI):", ""]
        for k, v in block["comparisons"].items():
            L.append(f"- {k}: p = {v['mcnemar']['p']} (only A right {v['mcnemar']['onlyA']}, only B right "
                     f"{v['mcnemar']['onlyB']}); Δ macro-F1 CI {v['deltaMacroF1CI']}")
        L += ["", "KPI slot extraction:", "", "| extractor | latency F1 | bandwidth F1 | reliability F1 | devices F1 (R) | exact match |",
              "|---|---|---|---|---|---|"]
        for name, sl in block["slots"].items():
            L.append(f"| {name} | {sl['latency_ms']['f1']} | {sl['bandwidth_mbps']['f1']} | {sl['reliability_percent']['f1']} | "
                     f"{sl['devices']['f1']} ({sl['devices']['recall']}) | {sl['exactMatch']} |")
        L.append("")
        for s in block["systems"]:
            if s["errors"]:
                L += [f"<details><summary>{split} errors: {s['system']} ({len(s['errors'])})</summary>", ""]
                L += [f"- {e['id']} gold {e['gold']} → {e['pred']}: {e['text']}" for e in s["errors"]] + ["", "</details>", ""]
        for name, sl in block["slots"].items():
            L.append(f"- slot errors {name}: {', '.join(sl['errors']) or 'none'}")
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--write-calibration", action="store_true")
    a = ap.parse_args()
    train, calib = load_jsonl(D / "train.jsonl"), load_jsonl(D / "calib.jsonl")
    tests = {k: load_jsonl(D / f) for k, f in TESTS.items()}
    refs = load_references(D / "references.json")
    yc = np.array([CLASSES.index(r["slice"]) for r in calib])
    cal: dict = {}

    knns = {"char-ngram kNN": SemanticIntentClassifier(CharNgramEncoder()).fit(train, refs)}
    sbert = None
    if (config.SBERT_PATH / "onnx" / "model.onnx").exists():
        sbert = SemanticIntentClassifier(SbertOnnxEncoder(config.SBERT_PATH)).fit(train, refs)
        knns["SBERT kNN"] = sbert
    for clf in knns.values():
        clf.calibrate(calib, a.alpha)
        cal[clf.enc.id] = clf.state()

    J = None
    if sbert is not None and JOINT.exists():
        J = JointIntentModel(sbert, JointNluNumpy(JOINT), regex_slots)
        J.calibrate(calib, a.alpha)
        cal["joint"] = J.state()
        rc = J.raw([r["text"] for r in calib])
        q_mode = {m: conformal_qhat(J.probs(rc, m), yc, a.alpha) for m in J.MODES}

    report = {"alpha": a.alpha, "n": {"train": len(train), "train_slots": len(load_jsonl(D / "train_slots.jsonl")),
                                      "calib": len(calib), **{k: len(v) for k, v in tests.items()}},
              "deployed": J.state() if J else None, "splits": {}}
    for split, test in tests.items():
        texts = [r["text"] for r in test]
        y = [r["slice"] for r in test]
        kw = [IntentProcessor.keyword_classify(t) for t in texts]
        systems = [summarize("keyword (baseline)", [k[0] for k in kw], np.array([k[1] for k in kw]), test)]
        labels = {"keyword (baseline)": [k[0] for k in kw]}
        for name, clf in knns.items():
            p = clf.proba_from_scores(clf.scores(clf.enc.encode(texts)))
            systems.append(from_probs(name, p, clf.qhat, test)); labels[name] = systems[-1]["_labels"] = [CLASSES[i] for i in p.argmax(1)]
        slots = {"regex": slot_report(test, [regex_slots(t) for t in texts])}
        if J:
            r = J.raw(texts)
            for m, name in (("joint", "TF joint"), ("ensemble", "ensemble kNN+TF")):
                p = J.probs(r, m)
                systems.append(from_probs(name, p, q_mode[m], test)); labels[name] = [CLASSES[i] for i in p.argmax(1)]
            for sm in ("joint", "joint+regex"):
                slots[f"TF tagger ({sm})"] = slot_report(test, J.slots(r, sm))
        for s in systems:
            s.pop("_labels", None)
        comps = {}
        base_ok = [p == t for p, t in zip(labels["keyword (baseline)"], y)]
        for name, lab in labels.items():
            if name == "keyword (baseline)":
                continue
            comps[f"keyword vs {name}"] = {"mcnemar": mcnemar_exact(base_ok, [p == t for p, t in zip(lab, y)]),
                                           "deltaMacroF1CI": bootstrap_ci(y, labels["keyword (baseline)"], lab)}
        if J:
            k_ok = [p == t for p, t in zip(labels["SBERT kNN"], y)]
            for name in ("TF joint", "ensemble kNN+TF"):
                comps[f"SBERT kNN vs {name}"] = {"mcnemar": mcnemar_exact(k_ok, [p == t for p, t in zip(labels[name], y)]),
                                                 "deltaMacroF1CI": bootstrap_ci(y, labels["SBERT kNN"], labels[name])}
        report["splits"][split] = {"systems": systems, "comparisons": comps, "slots": slots}

    (D / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (D / "eval_report.md").write_text(to_md(report), encoding="utf-8")
    if a.write_calibration:
        (D / "calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(to_md(report))


if __name__ == "__main__":
    main()
