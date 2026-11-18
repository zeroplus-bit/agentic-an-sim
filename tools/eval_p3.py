"""P3 evaluation: clarification gate (conformal set, NLI zero-shot disagreement, constraint violations).

    python tools/eval_p3.py      -> datasets/intent_nlu/eval_p3.{json,md}

Selective-prediction view: an intent is either auto-accepted or sent back for clarification. For every policy we
report clarify rate, accuracy on auto-accepted intents (selective accuracy) and how many classifier errors the
gate caught. Adoption rule for the NLI gate, fixed before looking at test data: adopt it only if, on calib, at
least half of the clarifications it ADDS on top of the conformal gate are real errors (extra-flag precision >= 0.5).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import config  # noqa: E402
from service_layer.orchestration.constraints import ConstraintValidator  # noqa: E402
from service_layer.orchestration.intent_processor import IntentProcessor  # noqa: E402
from service_layer.orchestration.nli_check import NliZeroShot  # noqa: E402
from service_layer.orchestration.semantic import load_jsonl  # noqa: E402

D = config.INTENT_DATA_DIR
SPLITS = ["calib", "test", "test_v2"]
NLI_ADOPT_MIN_PRECISION = 0.5


def gate_report(rows, preds, nli, val) -> dict:
    y = [r["slice"] for r in rows]
    lab = [p["label"] for p in preds]
    err = np.array([a != b for a, b in zip(lab, y)])
    flags = {"C": np.array([len(p["predictionSet"]) > 1 for p in preds]),
             "N": np.array([n["label"] != l for n, l in zip(nli, lab)]),
             "V": np.array([v["blocking"] for v in val])}
    pol = {"conformal": flags["C"], "conformal+NLI": flags["C"] | flags["N"],
           "conformal+constraints": flags["C"] | flags["V"], "conformal+NLI+constraints": flags["C"] | flags["N"] | flags["V"]}
    out = {"n": len(rows), "errors": int(err.sum()), "nliAccuracy": round(float(np.mean([n["label"] == t for n, t in zip(nli, y)])), 4),
           "agreement": round(float(np.mean(~flags["N"])), 4),
           "accuracyWhenAgree": round(float(np.mean(~err[~flags["N"]])), 4) if (~flags["N"]).any() else None,
           "accuracyWhenDisagree": round(float(np.mean(~err[flags["N"]])), 4) if flags["N"].any() else None,
           "policies": {}}
    for name, f in pol.items():
        acc = ~f
        out["policies"][name] = {"clarifyRate": round(float(f.mean()), 4),
                                 "selectiveAccuracy": round(float(np.mean(~err[acc])), 4) if acc.any() else None,
                                 "errorsCaught": int((f & err).sum()), "flagPrecision": round(float((f & err).sum() / f.sum()), 4) if f.any() else None}
    extra = flags["N"] & ~flags["C"]
    out["nliExtraFlags"] = int(extra.sum())
    out["nliExtraFlagPrecision"] = round(float((extra & err).sum() / extra.sum()), 4) if extra.any() else None
    out["constraintFlags"] = [r["id"] for r, v in zip(rows, val) if v["blocking"]]
    return out


def main() -> None:
    ip = IntentProcessor()
    J = ip.semantic
    assert hasattr(J, "slots"), "joint model must be calibrated first (tools/eval_intent.py --write-calibration)"
    nli = NliZeroShot(config.NLI_PATH)
    val = ConstraintValidator(config.INTENT_CONSTRAINTS)
    report = {"model": ip.model_info, "nli": nli.id, "ruleset": val.version, "splits": {}}
    for split in SPLITS:
        rows = load_jsonl(D / f"{split}.jsonl")
        texts = [r["text"] for r in rows]
        preds = J.predict_batch(texts)
        t0 = time.time()
        nl = nli.predict_batch(texts)
        ms = (time.time() - t0) * 1000 / len(texts)
        vals = [val.check(p["label"], p["slots"]) for p in preds]
        rep = gate_report(rows, preds, nl, vals)
        rep["nliMsPerUtterance"] = round(ms, 1)
        # false alarms of the rule set itself: gold slice + gold KPIs must never block
        gold = [val.check(r["slice"], {**r["expectations"], **({"devices": r["devices"]} if r.get("devices") else {})}) for r in rows]
        rep["goldBlockingFalseAlarms"] = [r["id"] for r, g in zip(rows, gold) if g["blocking"]]
        report["splits"][split] = rep
    c = report["splits"]["calib"]
    report["nliAdopted"] = bool(c["nliExtraFlagPrecision"] is not None and c["nliExtraFlagPrecision"] >= NLI_ADOPT_MIN_PRECISION)

    cv = load_jsonl(D / "constraints_eval.jsonl")
    cp = J.predict_batch([r["text"] for r in cv])
    det = [val.check(p["label"], p["slots"]) for p in cp]
    tp = sum(d["blocking"] and r["blocking"] for d, r in zip(det, cv))
    fp = sum(d["blocking"] and not r["blocking"] for d, r in zip(det, cv))
    fn = sum(not d["blocking"] and r["blocking"] for d, r in zip(det, cv))
    report["constraintSet"] = {"n": len(cv), "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                               "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                               "rows": [{"id": r["id"], "gold": r["blocking"], "pred": d["blocking"], "slice": p["label"],
                                         "slots": p["slots"], "rules": [v["rule"] for v in d["violations"]], "text": r["text"]}
                                        for r, d, p in zip(cv, det, cp)]}
    (D / "eval_p3.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    L = ["# P3 evaluation: clarification gate", "", f"Model {report['model']['model']} · NLI {report['nli']} · {report['ruleset']}", "",
         f"NLI gate adoption rule (fixed beforehand): calib extra-flag precision ≥ {NLI_ADOPT_MIN_PRECISION} → "
         f"calib value {c['nliExtraFlagPrecision']} → **{'adopted' if report['nliAdopted'] else 'not adopted'}**", ""]
    for split, r in report["splits"].items():
        L += [f"## {split} (n={r['n']}, classifier errors {r['errors']})", "",
              f"NLI zero-shot accuracy {r['nliAccuracy']}, agreement with classifier {r['agreement']} "
              f"(classifier accuracy when agree {r['accuracyWhenAgree']}, when disagree {r['accuracyWhenDisagree']}); "
              f"NLI adds {r['nliExtraFlags']} flags, precision {r['nliExtraFlagPrecision']}; {r['nliMsPerUtterance']} ms/utterance. "
              f"Rule false alarms on gold labels: {r['goldBlockingFalseAlarms'] or 'none'}.", "",
              "| policy | clarify rate | selective accuracy | errors caught | flag precision |", "|---|---|---|---|---|"]
        for k, v in r["policies"].items():
            L.append(f"| {k} | {v['clarifyRate']} | {v['selectiveAccuracy']} | {v['errorsCaught']}/{r['errors']} | {v['flagPrecision']} |")
        L.append("")
    cs = report["constraintSet"]
    L += [f"## Constraint set (n={cs['n']}): blocking precision {cs['precision']}, recall {cs['recall']}", "",
          "| id | gold | pred | slice | slots | rules | text |", "|---|---|---|---|---|---|---|"]
    L += [f"| {x['id']} | {x['gold']} | {x['pred']} | {x['slice']} | {x['slots']} | {', '.join(x['rules'])} | {x['text']} |" for x in cs["rows"]]
    (D / "eval_p3.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
