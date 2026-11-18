"""Render logs/verify/summary.json as markdown tables (python tools/verify_table.py)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "logs/verify/summary.json").read_text(encoding="utf-8"))
L = ["| scenario | intent | compliance steady/fault/recover | viol s (fault) | detect s | recover s | share steady→max→end | scaled back | final |",
     "|---|---|---|---|---|---|---|---|---|"]
for sc, s in S.items():
    for name, i in s["intents"].items():
        c, a = i["compliance"], i["avgShare"]
        L.append(f"| {sc} | {name} | {c['steady']}/{c['fault']}/{c['recover']} | {i['violationSecondsDuringFault']} | "
                 f"{i['timeToDetectSec']} | {i['timeToRecoverSec']} | {a['steady']}→{a['faultMax']}→{a['endOfRecovery']} | "
                 f"{i['scaledBackAfterFault']} | {i['finalState']} |")
L += ["", "| scenario | PRB util steady/fault/recover | dedicated % steady/fault/recover | NEEDS_DECISION | auto decisions | not fulfilled at end | applied/rolled back |",
      "|---|---|---|---|---|---|---|"]
for sc, s in S.items():
    ce, h = s["cells"], s["human"]
    dec = "; ".join(f"{d.get('action') or d.get('observed')} {d.get('intentId')} {d.get('newCells') or d.get('error') or d.get('reason') or ''}"
                    for d in h["autoDecisions"]) or "-"
    L.append(f"| {sc} | {ce['steady']['avgPrbUtilization']}/{ce['fault']['avgPrbUtilization']}/{ce['recover']['avgPrbUtilization']} | "
             f"{ce['steady']['avgDedicatedSharePercent']}/{ce['fault']['avgDedicatedSharePercent']}/{ce['recover']['avgDedicatedSharePercent']} | "
             f"{h['needsDecisionConflicts']} | {dec} | {', '.join(h['endedNotFulfilled']) or '-'} | "
             f"{s['actions']['applied']}/{s['actions']['rolledBack']} |")
print("\n".join(L))
if "-v" in sys.argv:
    for sc, s in S.items():
        for name, i in s["intents"].items():
            if i["finalState"] != "FULFILLED" or (i["compliance"]["fault"] or 1) < 0.9:
                print(f"\n{sc}/{name}: {i['finalState']} :: {i['finalExplanation']}")
