"""Autonomy verification harness: does the system allocate bandwidth efficiently, resolve faults by itself and
keep humans out of the loop?

Every scenario runs on a FRESH system (three layer processes started here, own agent-memory DB), so scenarios do
not influence each other. Timeline per scenario:
    provision 3 intents (URLLC hospital, eMBB stadium, mMTC port) -> steady -> fault ON -> fault OFF (recovery)
sampled at 1 Hz from the Network Layer (intent state, PRB shares, KPI compliance, per-intent load) and the twin.

    python tools/verify_autonomy.py                 # all scenarios -> logs/verify/<scenario>.json + summary.json
    python tools/verify_autonomy.py emergency cell_down
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import config  # noqa: E402

OUT = ROOT / "logs" / "verify"
SVC, NET, RES = config.SERVICE_URL, config.NETWORK_URL, config.RESOURCE_URL
LAYERS = [("resource", "resource_layer.app:app", config.RESOURCE_PORT, RES),
          ("network", "network_layer.app:app", config.NETWORK_PORT, NET),
          ("service", "service_layer.app:app", config.SERVICE_PORT, SVC)]
STEADY, FAULT, RECOVER = 20, 60, 60

BASE_INTENTS = [
    {"name": "urllc", "utterance": "Remote surgery link at Hospital A under 5 ms, 50 Mbps, 99.999%", "priority": 2,
     "cells": ["gNB-012-1", "gNB-012-2"]},
    {"name": "embb", "utterance": "Stadium 4K live streaming for fans, 200 Mbps, latency under 50 ms", "priority": 3,
     "cells": ["gNB-013-1", "gNB-013-2"]},
    {"name": "mmtc", "utterance": "Smart meters at the port, 5000 devices, 5 Mbps, latency up to 200 ms", "priority": 3,
     "cells": ["gNB-015-1", "gNB-015-2"]},
]
EXTRA_URLLC = [{"name": f"urllc{k}", "utterance": f"Robot arm motion control line {k} at Hospital A, under 10 ms, 60 Mbps",
                "priority": 3, "cells": ["gNB-012-1", "gNB-012-2"]} for k in (2, 3, 4)]

SCENARIOS: dict[str, dict[str, Any]] = {
    "emergency":        {"fault": {"is_emergency": True}, "clear": {"is_emergency": False}, "target": "urllc"},
    "emergency_noloop": {"fault": {"is_emergency": True}, "clear": {"is_emergency": False}, "target": "urllc",
                         "disable_agent": "agent-assure-01"},
    "special_event":    {"fault": {"is_special_event": True}, "clear": {"is_special_event": False}, "target": "embb"},
    "iot_surge":        {"fault": {"is_iot_surge": True}, "clear": {"is_iot_surge": False}, "target": "mmtc"},
    "stress_1.6":       {"fault": {"load_factor": 1.6}, "clear": {"load_factor": 1.0}, "target": None},
    "cell_down":        {"fault": {"cell_down": ["gNB-012-1"]}, "clear": {"cell_down": []}, "target": "urllc"},
    "contention":       {"fault": {"is_emergency": True}, "clear": {"is_emergency": False}, "target": "urllc",
                         "extra": EXTRA_URLLC},
}


# ---- process control ----------------------------------------------------------------------------------
def start_system(tag: str) -> list[subprocess.Popen]:
    tmp = Path(tempfile.mkdtemp(prefix=f"aans-{tag}-"))
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "AANS_MEMORY_DB": str(tmp / "mem.sqlite")}
    procs = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name, target, port, _ in LAYERS:
        log = open(OUT / f"{tag}-{name}.log", "w", encoding="utf-8")
        procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", target, "--host", config.HOST, "--port", str(port),
                                       "--log-level", "warning"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            if all(httpx.get(f"{u}/health", timeout=1).status_code == 200 for *_, u in LAYERS):
                ag = httpx.get(f"{SVC}/api/agents", timeout=5).json()
                if len(json.dumps(ag)) and json.dumps(ag).count("ACTIVE") >= 3:
                    return procs
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(1)
    stop_system(procs)
    raise RuntimeError("system did not come up")


def stop_system(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()


# ---- one scenario --------------------------------------------------------------------------------------
async def run(name: str, sc: dict[str, Any]) -> dict[str, Any]:
    procs = start_system(name)
    try:
        async with httpx.AsyncClient(timeout=30) as h:
            ids: dict[str, str] = {}
            subs = {}
            for it in BASE_INTENTS + sc.get("extra", []):
                r = (await h.post(f"{SVC}/api/intents", json={"utterance": it["utterance"], "priority": it["priority"],
                                                              "cells": it["cells"]})).json()
                subs[it["name"]] = {"submitted": r.get("submitted"), "status": r.get("status"),
                                    "slice": r["intent"]["semantic"]["serviceCategory"]}
                if r.get("submitted"):
                    ids[it["name"]] = r["intent"]["intentId"]
                await asyncio.sleep(0.3)
            timeline: list[dict[str, Any]] = []
            t0 = time.time()
            phase_at = {"provision": 0.0}

            async def sample(phase: str) -> None:
                net = (await h.get(f"{NET}/status/intents")).json()
                sim = (await h.get(f"{RES}/sim/state")).json()
                row = {"t": round(time.time() - t0, 2), "phase": phase, "intents": {}, "cells": {}}
                for r in net:
                    name_ = next((k for k, v in ids.items() if v == r["intentId"]), r["intentId"])
                    f = r.get("fulfilment") or {}
                    row["intents"][name_] = {"state": r["state"], "shares": r["shares"], "met": f.get("met"),
                                             "latency_ms": f.get("latency_ms"), "bandwidth_mbps": f.get("bandwidth_mbps"),
                                             "cells": r["cells"], "explanation": (r.get("explanation") or "")[:200]}
                for c in sim["cells"]:
                    row["cells"][c["resourceId"].split(":")[1]] = {"prb": c["prbUtilization"], "up": c["up"],
                                                                   "free": c["freeSharePercent"], "alloc": c["allocation"],
                                                                   "intents": c["intents"]}
                timeline.append(row)

            # wait for provisioning
            for _ in range(40):
                await sample("provision")
                if all(timeline[-1]["intents"].get(k, {}).get("state") in ("FULFILLED", "REJECTED", "FAILED") for k in ids):
                    break
                await asyncio.sleep(1)
            phase_at["steady"] = time.time() - t0
            for _ in range(STEADY):
                await sample("steady"); await asyncio.sleep(1)
            if sc.get("disable_agent"):
                await h.delete(f"{SVC}/api/agents/{sc['disable_agent']}")
            await h.post(f"{RES}/sim/scenario", json=sc["fault"])
            phase_at["fault"] = time.time() - t0
            for _ in range(FAULT):
                await sample("fault"); await asyncio.sleep(1)
            await h.post(f"{RES}/sim/scenario", json=sc["clear"])
            phase_at["recover"] = time.time() - t0
            for _ in range(RECOVER):
                await sample("recover"); await asyncio.sleep(1)
            conflicts = (await h.get(f"{SVC}/api/conflicts")).json()
            metrics = (await h.get(f"{NET}/metrics")).json()
            svc_intents = (await h.get(f"{SVC}/api/intents")).json()
        res = {"scenario": name, "t0": t0, "config": {k: v for k, v in sc.items() if k != "extra"}, "submissions": subs, "ids": ids,
               "phaseAt": phase_at, "timeline": timeline, "conflicts": conflicts, "metrics": metrics,
               "serviceIntents": [{k: x.get(k) for k in ("intentId", "state", "priority")} for x in svc_intents]}
        res["summary"] = summarize(res)
        return res
    finally:
        stop_system(procs)


# ---- analysis -----------------------------------------------------------------------------------------
def summarize(res: dict[str, Any]) -> dict[str, Any]:
    tl, pa = res["timeline"], res["phaseAt"]
    out: dict[str, Any] = {"intents": {}}
    for name in res["ids"]:
        rows = [(r["t"], r["phase"], r["intents"].get(name)) for r in tl if r["intents"].get(name)]
        def comp(phase):
            xs = [x["met"] for _, p, x in rows if p == phase and x["met"] is not None]
            return round(sum(xs) / len(xs), 3) if xs else None
        def own(x):      # mean share on the intent's own cells (steered shares on neighbours counted apart)
            v = [x["shares"][c] for c in (x.get("cells") or []) if c in x["shares"]]
            return sum(v) / len(v) if v else None
        def steered(x):
            return sum(v for c, v in x["shares"].items() if c not in (x.get("cells") or []))
        def share(phase, fn=max):
            xs = [own(x) for _, p, x in rows if p == phase and own(x) is not None]
            return round(fn(xs), 1) if xs else None
        def last_share(phase):
            xs = [own(x) for _, p, x in rows if p == phase and own(x) is not None]
            return round(xs[-1], 1) if xs else None
        fault_rows = [(t, x) for t, p, x in rows if p in ("fault", "recover")]
        first_deg = next((t for t, x in fault_rows if x["state"] == "DEGRADED"), None)
        first_bad = next((t for t, x in fault_rows if x["met"] is False), None)
        rec_after = None
        if first_deg is not None:
            rec_after = next((t for t, x in fault_rows if t > first_deg and x["state"] == "FULFILLED"), None)
        fault_viol_sec = sum(1 for t, p, x in rows if p == "fault" and x["met"] is False)
        end = rows[-1][2] if rows else {}
        out["intents"][name] = {
            "compliance": {p: comp(p) for p in ("steady", "fault", "recover")},
            "violationSecondsDuringFault": fault_viol_sec,
            "timeToDetectSec": round(first_deg - pa["fault"], 1) if first_deg is not None else None,
            "firstViolationAfterFaultSec": round(first_bad - pa["fault"], 1) if first_bad is not None else None,
            "timeToRecoverSec": round(rec_after - first_deg, 1) if (first_deg is not None and rec_after is not None) else None,
            "avgShare": {"steady": share("steady", statistics.mean), "faultMax": share("fault"),
                         "endOfRecovery": last_share("recover")},
            "steeredShareMax": round(max([steered(x) for _, p, x in rows if p == "fault"] or [0]), 1),
            "steeredShareEnd": round(steered(rows[-1][2]), 1) if rows else None,
            "scaledBackAfterFault": (last_share("recover") is not None and share("steady", statistics.mean) is not None
                                     and last_share("recover") <= share("steady", statistics.mean) + 1),
            "finalState": end.get("state"), "finalExplanation": end.get("explanation"),
            "finalCells": end.get("cells"),
        }
    # efficiency: per-cell PRB utilisation and dedicated-but-idle share on cells hosting intents
    cells_used = {c for r in tl for x in r["intents"].values() for c in (x.get("cells") or [])}
    for phase in ("steady", "fault", "recover"):
        prb = [r["cells"][c]["prb"] for r in tl if r["phase"] == phase for c in cells_used if c in r["cells"] and r["cells"][c]["up"]]
        ded = [sum(i["sharePercent"] for i in r["cells"][c]["intents"].values()) for r in tl if r["phase"] == phase
               for c in cells_used if c in r["cells"]]
        out.setdefault("cells", {})[phase] = {"avgPrbUtilization": round(statistics.mean(prb), 3) if prb else None,
                                              "avgDedicatedSharePercent": round(statistics.mean(ded), 1) if ded else None}
    cf = res["conflicts"]
    out["human"] = {
        "needsDecisionConflicts": sum(1 for c in cf.get("conflicts", []) if c.get("resolution") == "NEEDS_DECISION"),
        "autoDecisions": [{k: d.get(k) for k in ("action", "intentId", "ok", "newCells", "how", "error", "observed", "reason", "humanReview")}
                          for d in cf.get("decisions", [])],
        "rejectedAtSubmit": [k for k, v in res["submissions"].items() if not v["submitted"]],
        "endedNotFulfilled": [k for k, v in out["intents"].items() if v["finalState"] != "FULFILLED"],
    }
    asr = res["metrics"].get("assurance") or {}
    fault_at = res.get("t0", 0) + pa["fault"]
    out["human"]["humanReviewDecisions"] = sum(1 for d in cf.get("decisions", []) if d.get("humanReview"))
    out["escalations"] = [{"intent": e.get("intentId"), "afterFaultSec": round(e["at"] - fault_at, 1), "cell": e.get("cell"),
                           "needed": e.get("needed"), "max": e.get("max"), "reason": e.get("reason")}
                          for e in asr.get("escalations", [])]
    out["reclaims"] = [{"intent": r.get("intentId"), "afterFaultSec": round(r["at"] - fault_at, 1), "what": r.get("explanation")}
                       for r in asr.get("reclaims", [])]
    out["actions"] = res["metrics"].get("actions")
    out["conflictStats"] = {k: v for k, v in res["metrics"].get("conflicts", {}).items() if k != "recent"}
    return out


async def main(names: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    sp = OUT / "summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text(encoding="utf-8"))
    for n in names:
        print(f"=== {n}", flush=True)
        res = await run(n, SCENARIOS[n])
        (OUT / f"{n}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        summary[n] = res["summary"]
        sp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(res["summary"], ensure_ascii=False)[:1500], flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or list(SCENARIOS)))
