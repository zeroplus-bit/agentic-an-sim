"""Layer-interaction scenario: one intent through its whole life, then the three layers' message logs are
checked against the SN/NR catalogue of the design.

    python tools/layer_scenario.py      -> logs/layer_scenario.md, logs/layer_scenario.mmd, logs/layer_scenario.json

Phases: A feasibility + provisioning (SN-05, SN-01 / Flow 1) · B emergency surge -> closed loop (Flow 3)
        · C surge over -> reclaim · D cancel (SN-03)
Checks:  C1 reference-point discipline (S<->N on SN, N<->R on NR, never S<->R)
         C2 catalogue conformance (message type, direction and sending/receiving entity as the catalogue defines)
         C3 every request answered by its response type
         C4 one correlationId carries the flow through all three layers
         C5 message order per phase is the defined flow; intent states follow the defined transitions
         C6 agentic roles: which agent did which task, through which MCP tools; every NR change came from an agent
         C7 validate (NR-04) before every apply (NR-05)
         C8 one traceId for the whole flow (W3C trace-context style: same traceId, a span per message)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import config  # noqa: E402
from common.envelope import response_type  # noqa: E402
from common.nr_messages import NR_CATALOGUE  # noqa: E402
from common.sn_messages import INTENT_TRANSITIONS, SN_CATALOGUE  # noqa: E402
from tools.verify_autonomy import start_system, stop_system  # noqa: E402

SVC, NET, RES = config.SERVICE_URL, config.NETWORK_URL, config.RESOURCE_URL
LAYER = {"S": "service", "N": "network", "R": "resource"}
CAT = {}
for rp, cat in (("SN", SN_CATALOGUE), ("NR", NR_CATALOGUE)):
    for mid, (mt, d, ents, _) in cat.items():
        CAT[mt] = {"id": mid, "rp": rp, "from": LAYER[d[0]], "to": LAYER[d[-1]],
                   "fromEntity": ents.split("->")[0].strip(), "toEntity": ents.split("->")[1].strip()}
RESP = {response_type(mt): mt for mt in CAT}
EXPECTED = {   # ordered subsequences of catalogue messages per phase (the design's flows)
    "A": ["IntentFeasibilityQuery", "IntentSubmitRequest", "ResourceReserveRequest", "ConfigValidateRequest",
          "ConfigApplyRequest", "TelemetrySubscribeRequest", "IntentStatusNotify"],
    "B": ["TelemetryNotify", "IntentStatusNotify", "ConfigValidateRequest", "ConfigApplyRequest", "IntentStatusNotify"],
    "C": ["ConfigValidateRequest", "ConfigApplyRequest"],
    "D": ["IntentCancelRequest", "ResourceReleaseRequest", "IntentStatusNotify"],
}


RESOURCE_BLOCKS = {"Communication": "CommunicationResources", "UE/Edge": "UeEdgeResources", "Computing": "ComputingResources",
                   "Storage": "StorageResources", "Tool": "ToolResources"}


def entity_ok(actual: str, defined: str, layer: str) -> bool:
    """Catalogue entity text vs. the envelope's entity: a named block ("ServiceProvisioner"), a list of resource
    blocks ("Communication, Computing, Storage"), or a generic receiver ("all resources", "same", "resources")."""
    d = defined.replace("(twin copy)", "").strip()
    if d in ("all resources", "same", "resources") or d.startswith("all"):
        return layer == "resource" or actual == "ResourceLayer"
    names = [x.strip() for x in d.split(",")]
    allowed = {RESOURCE_BLOCKS.get(n.replace(" Resources", "").strip(), n.replace(" ", "")) for n in names}
    allowed |= {n.replace(" ", "") for n in names}
    return actual in allowed or actual.replace(" ", "") in allowed


def subsequence(seq: list[str], want: list[str]) -> tuple[bool, list[str]]:
    it, missing = iter(seq), []
    for w in want:
        if not any(x == w for x in it):
            missing.append(w)
            break
    return not missing, missing


async def run() -> dict[str, Any]:
    procs = start_system("layers")
    try:
        async with httpx.AsyncClient(timeout=30) as h:
            marks = {}
            utter = "Remote surgery link at Hospital A under 5 ms, 50 Mbps, 99.999%"
            marks["A"] = time.time()
            r = (await h.post(f"{SVC}/api/intents", json={"utterance": utter, "checkFeasibility": True})).json()
            iid = r["intent"]["intentId"]
            for _ in range(30):
                st = (await h.get(f"{NET}/status/intents/{iid}")).json()
                if st["state"] in ("FULFILLED", "REJECTED", "FAILED"):
                    break
                await asyncio.sleep(1)
            await asyncio.sleep(10)
            marks["B"] = time.time()
            await h.post(f"{RES}/sim/scenario", json={"is_emergency": True})
            await asyncio.sleep(40)
            marks["C"] = time.time()
            await h.post(f"{RES}/sim/scenario", json={"is_emergency": False})
            await asyncio.sleep(45)
            marks["D"] = time.time()
            await h.delete(f"{SVC}/api/intents/{iid}", params={"reason": "scenario done"})
            await asyncio.sleep(4)
            marks["end"] = time.time()
            rows = (await h.get(f"{SVC}/api/trace/{iid}")).json()
            mermaid = (await h.get(f"{SVC}/api/trace/{iid}", params={"format": "mermaid"})).text
            net = (await h.get(f"{NET}/status/intents/{iid}")).json()
            svc = (await h.get(f"{SVC}/api/intents/{iid}", params={"network": False})).json()
        return {"intentId": iid, "submit": r, "marks": marks, "rows": rows, "mermaid": mermaid, "net": net, "svc": svc}
    finally:
        stop_system(procs)


def ts(row: dict[str, Any]) -> float:
    from datetime import datetime
    return datetime.fromisoformat(row["at"].replace("Z", "+00:00")).timestamp()


def analyse(d: dict[str, Any]) -> dict[str, Any]:
    rows, iid, marks = d["rows"], d["intentId"], d["marks"]
    msgs = [r for r in rows if r.get("dir") in ("out", "in") and r.get("refPoint")]
    # one row per message: the sender's 'out' (bus notifies are logged by sender + receiver)
    out = {r["msgId"]: r for r in msgs if r["dir"] == "out"}
    inb = {r["msgId"]: r for r in msgs if r["dir"] == "in"}
    every = list({**inb, **out}.values())
    every.sort(key=ts)
    res: dict[str, Any] = {"intentId": iid, "messages": len(every)}

    # C1 reference points
    bad_rp = []
    for r in every:
        a, b = r["from"].split("/")[0], r["to"].split("/")[0]
        ok = {a, b} == ({"service", "network"} if r["refPoint"] == "SN" else {"network", "resource"})
        if not ok:
            bad_rp.append(f"{r['msgType']} {r['from']}->{r['to']} on {r['refPoint']}")
    res["C1"] = {"ok": not bad_rp, "violations": bad_rp,
                 "pairs": dict(Counter(f"{r['from'].split('/')[0]}->{r['to'].split('/')[0]} ({r['refPoint']})" for r in every))}

    # C2 catalogue conformance (requests / notifies)
    bad_cat, seen = [], Counter()
    for r in every:
        mt = r["msgType"]
        if mt in RESP:
            continue
        c = CAT.get(mt)
        if not c:
            bad_cat.append(f"{mt}: not in catalogue")
            continue
        seen[f"{c['id']} {mt}"] += 1
        fl, fe = r["from"].split("/")
        tl, te = r["to"].split("/")
        if c["rp"] != r["refPoint"] or c["from"] != fl or c["to"] != tl:
            bad_cat.append(f"{c['id']} {mt}: {fl}->{tl} on {r['refPoint']} (defined {c['from']}->{c['to']} on {c['rp']})")
        elif not entity_ok(fe, c["fromEntity"], fl):
            bad_cat.append(f"{c['id']} {mt}: sent by {fe}, defined sender {c['fromEntity']}")
        elif not entity_ok(te, c["toEntity"], tl):
            bad_cat.append(f"{c['id']} {mt}: received by {te}, defined receiver {c['toEntity']}")
    res["C2"] = {"ok": not bad_cat, "violations": bad_cat[:20], "messagesByCatalogueId": dict(sorted(seen.items()))}

    # C3 request/response pairing
    req = Counter(r["msgType"] for r in every if r["msgType"] in CAT and not r["msgType"].endswith("Notify"))
    rsp = Counter(RESP[r["msgType"]] for r in every if r["msgType"] in RESP)
    unanswered = {k: v - rsp.get(k, 0) for k, v in req.items() if v != rsp.get(k, 0)}
    statuses = Counter(f"{r['msgType']}:{r.get('status')}" for r in every if r["msgType"] in RESP)
    res["C3"] = {"ok": not unanswered, "requests": dict(req), "unanswered": unanswered, "responseStatus": dict(statuses)}

    # C4 correlation through the layers
    layers = Counter(r["layer"] for r in rows)
    traces = {r.get("traceId") for r in every if r.get("traceId")}
    res["C4"] = {"ok": set(layers) >= {"service", "network", "resource"}, "rowsPerLayer": dict(layers),
                 "distinctTraceIds": len(traces)}

    # C5 order per phase + state transitions
    phase_of = lambda t: "A" if t < marks["B"] else "B" if t < marks["C"] else "C" if t < marks["D"] else "D"  # noqa: E731
    seqs = defaultdict(list)
    for r in every:
        if r["msgType"] in CAT:
            seqs[phase_of(ts(r))].append(r["msgType"])
    order = {}
    for p, want in EXPECTED.items():
        ok, missing = subsequence(seqs[p], want)
        order[p] = {"ok": ok, "missing": missing, "observed": compress(seqs[p])}
    hist = d["net"].get("history") or []
    states = [hist[0]["from"]] + [x["to"] for x in hist] if hist else []
    illegal = [f"{a}->{b}" for a, b in zip(states, states[1:]) if b not in INTENT_TRANSITIONS.get(a, set())]
    res["C5"] = {"ok": all(v["ok"] for v in order.values()) and not illegal, "phases": order, "states": states,
                 "illegalTransitions": illegal}

    # C6 agentic roles
    internal = [r for r in rows if r.get("dir") == "internal"]
    tools = defaultdict(Counter)
    for r in internal:
        if r["msgType"].startswith("ToolCall"):
            tools[r["from"].split("/")[1]][r["msgType"].split(" ", 1)[1]] += 1
    tasks = [{"type": t.get("type"), "agent": t.get("agentId"), "state": t.get("state"),
              "mode": (t.get("input") or {}).get("mode"),
              "explanation": ((t.get("output") or {}).get("explanation") or (t.get("output") or {}).get("error") or "")[:160]}
             for t in d["net"].get("tasks", [])]
    applies = sum(1 for r in every if r["msgType"] == "ConfigApplyRequest")
    tool_applies = sum(v.get("nr_config_apply", 0) for v in tools.values())
    res["C6"] = {"ok": applies > 0 and tool_applies >= applies and all(t["agent"] for t in tasks),
                 "toolsByAgent": {a: dict(c) for a, c in tools.items()}, "tasks": tasks,
                 "nr05": applies, "nr05ViaAgentTools": tool_applies,
                 "agentBus": sum(1 for r in internal if r["msgType"].startswith("Agent"))}

    # C7 validate before apply
    unval, validated, applied = 0, 0, 0      # each apply needs a validation before it (several cells: n then n)
    for r in every:
        if r["msgType"] == "ConfigValidateRequest":
            validated += 1
        elif r["msgType"] == "ConfigApplyRequest":
            applied += 1
            if applied > validated:
                unval += 1
                validated = applied
    res["C7"] = {"ok": unval == 0, "appliesWithoutPriorValidate": unval, "applies": applies}
    by_trace = Counter(r.get("traceId") for r in every)
    res["C8"] = {"ok": len(by_trace) == 1, "distinctTraceIds": len(by_trace),
                 "messagesPerTraceId": [n for _, n in by_trace.most_common(5)],
                 "outliers": sorted({f"{r['msgType']} ({r['layer']})" for r in every
                                     if r.get("traceId") != by_trace.most_common(1)[0][0]})[:15]}
    res["svcFulfilment"] = d["svc"].get("fulfilmentRatio")
    return res


def compress(seq: list[str]) -> list[str]:
    out = []
    for x in seq:
        if out and out[-1][0] == x:
            out[-1][1] += 1
        else:
            out.append([x, 1])
    return [f"{x}×{n}" if n > 1 else x for x, n in out]


def to_md(a: dict[str, Any]) -> str:
    L = [f"# Layer interaction scenario ({a['intentId']}, {a['messages']} SN/NR messages)", ""]
    for k, title in (("C1", "reference points"), ("C2", "catalogue"), ("C3", "request/response"), ("C4", "correlation"),
                     ("C5", "flow order + states"), ("C6", "agentic roles"), ("C7", "validate before apply"),
                     ("C8", "one traceId per flow")):
        L.append(f"- **{k} {title}: {'PASS' if a[k]['ok'] else 'FAIL'}**")
    L += ["", "```json", json.dumps({k: a[k] for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")}, ensure_ascii=False, indent=1),
          "```"]
    return "\n".join(L)


def main() -> None:
    d = asyncio.run(run())
    a = analyse(d)
    (ROOT / "logs/layer_scenario.json").write_text(json.dumps({"analysis": a, "raw": d}, ensure_ascii=False, indent=1,
                                                              default=str), encoding="utf-8")
    (ROOT / "logs/layer_scenario.mmd").write_text(d["mermaid"], encoding="utf-8")
    (ROOT / "logs/layer_scenario.md").write_text(to_md(a), encoding="utf-8")
    print(to_md(a))


if __name__ == "__main__":
    main()
