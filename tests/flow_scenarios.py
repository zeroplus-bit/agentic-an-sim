"""End-to-end flow checks against running layers (start them first with `python run_all.py`).

    python tests/flow_scenarios.py

Drives everything through the SN MCP server (stdio), i.e. exactly as an LLM agent would, and one external
agent through the Network Layer MCP tool endpoint.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402
from mcp.client.streamable_http import streamablehttp_client  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


async def call(s: ClientSession, tool: str, **args):
    r = await s.call_tool(tool, args)
    if r.isError:
        raise RuntimeError(f"{tool}: {r.content}")
    sc = r.structuredContent
    if sc is not None:
        return sc["result"] if set(sc) == {"result"} else sc
    return json.loads(r.content[0].text) if r.content else None


async def wait_state(s, iid: str, states: set[str], timeout: float = 30) -> str:
    t0 = time.time()
    st = None
    while time.time() - t0 < timeout:
        st = (await call(s, "get_intent", intentId=iid))["state"]
        if st in states:
            return st
        await asyncio.sleep(0.5)
    return st


async def main() -> int:
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_servers.sn_mcp_server"], cwd=str(ROOT))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            print(f"SN MCP server: {len(tools)} tools")
            check("submit_intent" in tools and "set_scenario" in tools, "SN MCP tool list")

            print("Flow 1 - intent-based provisioning")
            await call(s, "set_scenario", is_emergency=False, is_special_event=False, is_iot_surge=False, load_factor=1.0)
            p = await call(s, "parse_intent", utterance="원격수술 Hospital A, 지연 5ms 이하, 50Mbps, 신뢰도 99.999%")
            check(p["intent"]["semantic"]["serviceCategory"] == "URLLC" and p["priority"] == 2, "Korean utterance parsed as URLLC p2")
            r = await call(s, "submit_intent", utterance="Remote surgery link at Hospital A under 5 ms, 50 Mbps, 99.999%",
                           checkFeasibility=True)
            iid = r["intent"]["intentId"]
            check(r["feasibility"]["feasible"] and r["response"]["status"] == "ACCEPTED", f"{iid} feasible + ACCEPTED")
            check(await wait_state(s, iid, {"FULFILLED", "REJECTED", "FAILED"}) == "FULFILLED", f"{iid} FULFILLED")

            print("Conflict mgmt - preemption of a lower-priority intent (SN-14)")
            r = await call(s, "submit_intent", utterance="best effort 4K video streaming in the stadium, 150 Mbps")
            low = r["intent"]["intentId"]
            check(r["priority"] == 4, "best-effort parsed as priority 4")
            check(await wait_state(s, low, {"FULFILLED", "REJECTED", "FAILED"}) == "FULFILLED", f"{low} FULFILLED")
            r = await call(s, "submit_intent", utterance="emergency public safety drone control at the stadium, 100 Mbps under 10 ms")
            hi = r["intent"]["intentId"]
            st = await wait_state(s, hi, {"FULFILLED", "REJECTED", "FAILED"})
            check(st == "FULFILLED", f"{hi} (priority 1) FULFILLED after preemption [{st}]")
            conf = await call(s, "list_conflicts")
            pre = [c for c in conf["conflicts"] if c["resolution"] == "PREEMPTED_LOWER_PRIORITY"]
            check(bool(pre) and low in pre[-1]["intents"], f"SN-14 PREEMPTED_LOWER_PRIORITY names {low}")
            lv = await call(s, "get_intent", intentId=low)
            check(max(lv["networkView"]["shares"].values()) < 54, f"{low} share reduced: {lv['networkView']['shares']}")

            print("SN-03 cancel")
            await call(s, "cancel_intent", intentId=low, reason="test")
            check(await wait_state(s, low, {"TERMINATED"}) == "TERMINATED", f"{low} TERMINATED")
            state = await call(s, "get_network_state")
            check(all(low not in c["intents"] for c in state["cells"]), "twin bindings removed")

            print("Flow 4 - model canary")
            d = await call(s, "deploy_model", version="2.0", trafficShare=0.3, rollbackMae=0.9, autoPromote=True)
            check(d["status"] == "ACCEPTED", "SN-10 ACCEPTED")
            t0, promoted = time.time(), False
            while time.time() - t0 < 60 and not promoted:
                m = await call(s, "list_models")
                promoted = m["networkView"]["primary"]["lstm_single"] == "2.0"
                await asyncio.sleep(2)
            check(promoted, f"canary promoted to primary via second SN-10 ({m['networkView']['metrics']})")
            d = await call(s, "deploy_model", version="1.0", trafficShare=0.5, rollbackMae=0.0001)
            await asyncio.sleep(12)
            m = await call(s, "list_models")
            states = [v["state"] for v in m["deployments"].values()]
            check("ROLLED_BACK" in states and not m["networkView"]["canary"], f"rollback rule fired: {states}")

            print("External LLM agent onboarding (Flow 2) + Network MCP tools")
            ext = await call(s, "onboard_external_agent", agentId="agent-claude-01", skills=["ASSURE_KPI"])
            await asyncio.sleep(1.5)
            async with streamablehttp_client(ext["mcpUrl"], headers={"Authorization": f"Bearer {ext['agentToken']}"}) as (r2, w2, _):
                async with ClientSession(r2, w2) as ns:
                    await ns.initialize()
                    nt = [t.name for t in (await ns.list_tools()).tools]
                    check(len(nt) >= 20, f"Network MCP exposes {len(nt)} tools")
                    il = await call(ns, "intent_list")
                    check(any(i["intentId"] == iid for i in il["intents"]), "external agent reads intents")
                    fc = await call(ns, "model_lstm_forecast", cell="gNB-012-1")
                    check(fc.get("allocation") is not None, f"LSTM via MCP: {fc.get('allocation')}")
                    inv = await call(ns, "nr_resource_inventory", type="comp")
                    check(inv["status"] == "SUCCESS", "NR-01 via MCP")
            async with streamablehttp_client(ext["mcpUrl"], headers={"Authorization": "Bearer bogus"}) as (r3, w3, _):
                async with ClientSession(r3, w3) as bad:
                    await bad.initialize()
                    res = await bad.call_tool("intent_list", {})
                    check(res.isError, "unauthenticated MCP call denied by AuthGuard")

            print("Metrics and trace")
            mt = await call(s, "get_metrics")
            row = next(x for x in mt["intents"] if x["intentId"] == iid)
            check(row["timeToFulfilSec"] is not None and row["signalling"].get("NR", {}).get("count", 0) > 0,
                  f"metrics: ttf={row['timeToFulfilSec']}s signalling={row['signalling']}")
            tr = await call(s, "get_trace", correlationId=iid)
            check("IntentSubmitRequest" in tr and "ConfigApplyRequest" in tr, "Mermaid trace has SN-01 .. NR-05")
            (ROOT / "logs" / f"trace-{iid}.mmd").write_text(tr, encoding="utf-8")

            print("Intent NLU P3 - constraint gate and model lifecycle")
            r = await call(s, "submit_intent", utterance="Smart water meters, 20,000 devices, latency under 2 ms.")
            check(r.get("status") == "CLARIFICATION_REQUIRED" and not r.get("submitted"),
                  f"contradictory KPIs blocked: {r.get('reason')}")
            st = await call(s, "intent_model_status")
            check(st["primary"] == "2.0" and "2.1" in st["versions"], f"intent NLU primary {st['primary']}")
            sh = await call(s, "manage_intent_model", action="shadow", version="1.0")
            check(sh.get("ok"), "shadow version 1.0 (keyword) started")
            import httpx
            rows = [json.loads(l) for l in (ROOT / "datasets/intent_nlu/test_v2.jsonl").read_text(encoding="utf-8").splitlines()][:30]
            fb = None
            async with httpx.AsyncClient(base_url="http://127.0.0.1:8100", timeout=60) as h:
                for k, row in enumerate(rows):
                    await h.post("/api/intents:parse", json={"utterance": row["text"], "intentId": f"lcm-e2e-{k}"})
                    fb = await call(s, "confirm_intent_interpretation", intentId=f"lcm-e2e-{k}", slice=row["slice"])
            d = fb["decision"]
            check(d["paired"] >= 20 and d["state"] in ("REJECTED", "COLLECTING") and
                  d["test"]["onlyPrimaryRight"] > d["test"]["onlyShadowRight"],
                  f"McNemar shadow decision: {d['state']} {d['test']}")
            st = await call(s, "intent_model_status")
            check(st["metrics"]["2.0"]["labelled"] == 30, f"online accuracy 2.0={st['metrics']['2.0']['accuracy']} "
                  f"1.0={st['metrics']['1.0']['accuracy']}")
            p = await call(s, "manage_intent_model", action="promote", version="2.1")
            rb = await call(s, "manage_intent_model", action="rollback")
            check(p.get("ok") and rb.get("primary") == "2.0", "promote 2.1 then rollback to 2.0")
            # the confirmations above are test_v2 rows: keep them out of the feedback lineage
            fbf = ROOT / "datasets/intent_nlu/feedback.jsonl"
            if fbf.exists():
                keep = [l for l in fbf.read_text(encoding="utf-8").splitlines() if '"fb-lcm-e2e-' not in l]
                fbf.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8") if keep else fbf.unlink()
    print(f"\n{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
