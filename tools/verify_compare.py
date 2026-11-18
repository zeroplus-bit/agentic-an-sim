"""Before/after comparison of two verification summaries (python tools/verify_compare.py [before] [after])."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = json.loads((ROOT / (sys.argv[1] if len(sys.argv) > 1 else "logs/verify/summary_before.json")).read_text(encoding="utf-8"))
A = json.loads((ROOT / (sys.argv[2] if len(sys.argv) > 2 else "logs/verify/summary.json")).read_text(encoding="utf-8"))
FOCUS = {"emergency": "urllc", "emergency_noloop": "urllc", "special_event": "embb", "iot_surge": "mmtc",
         "stress_1.6": "urllc", "cell_down": "urllc", "contention": "urllc2"}


def fmt(i):
    if not i:
        return "-"
    a = i["avgShare"]
    return (f"{i['compliance']['fault']} / {i['violationSecondsDuringFault']} s / {i['timeToRecoverSec']} s / "
            f"{a['steady']}→{a['faultMax']}→{a['endOfRecovery']}")


print("| scenario (intent) | before: fault compliance / violation / recover / share | after | after: escalation, reclaim, human |")
print("|---|---|---|---|")
for sc, it in FOCUS.items():
    b, a = B.get(sc, {}).get("intents", {}).get(it), A.get(sc, {}).get("intents", {}).get(it)
    s = A.get(sc, {})
    esc = ", ".join(f"{e['intent']} +{e['afterFaultSec']}s (need {e['needed']}% > {e['max']}%)" for e in s.get("escalations", [])) or "none"
    rec = len(s.get("reclaims", []))
    h = s.get("human", {})
    print(f"| {sc} ({it}) | {fmt(b)} | {fmt(a)} | esc: {esc}; reclaims {rec}; humanReview {h.get('humanReviewDecisions')}; "
          f"not fulfilled {h.get('endedNotFulfilled')} |")
print()
for sc in A:
    for name, i in A[sc]["intents"].items():
        bi = B.get(sc, {}).get("intents", {}).get(name)
        if bi and (bi["finalState"] != i["finalState"]):
            print(f"{sc}/{name}: final {bi['finalState']} -> {i['finalState']}")
    print(sc, "decisions:", [(d.get("action"), d.get("intentId"), d.get("humanReview")) for d in A[sc]["human"]["autoDecisions"]])
