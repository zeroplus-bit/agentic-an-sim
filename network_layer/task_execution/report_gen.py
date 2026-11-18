"""Report Generation (ReportGenerator): KPI and explanation reports for SN (SN-06 subscriptions -> SN-07).
Explanations come from an LLM invoked as a Tool Resource over NR-10 (starts from gemini_integration.py)."""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from common.envelope import Envelope, iso, respond
from common.refpoint import new_id

from ..task_control.pre_validator import compare


class ReportGenerator:
    def __init__(self, nl):
        self.nl = nl
        self.subs: dict[str, dict[str, Any]] = {}
        self.sent = 0

    def ensure_auto(self, rec) -> None:
        rp = rec.intent.get("reportPolicy") or {}
        sid = f"sub-auto-{rec.intent_id}"
        if sid not in self.subs:
            self.subs[sid] = {"subscriptionId": sid, "intentId": rec.intent_id,
                              "kpis": [e["kpi"] for e in rec.expectations] or ["latency_ms", "bandwidth_mbps"],
                              "periodSec": float(rp.get("periodSec", 10)), "includeExplanation": True,
                              "thresholdEvents": bool(rp.get("onDrift", True)), "last": time.time(), "auto": True}

    async def subscribe(self, env: Envelope) -> Envelope:
        _, err = self.nl.auth.verify_sn(env, "sn:monitor:read")
        if err:
            return respond(env, "UNAUTHORIZED", reason=err)
        b = env.body
        iid = (b.get("target") or {}).get("intentId")
        if iid not in self.nl.intents:
            return respond(env, "REJECTED", {"subscriptionId": b["subscriptionId"]}, reason="unknown intent")
        self.subs[b["subscriptionId"]] = {**b, "intentId": iid, "last": time.time(), "periodSec": float(b.get("periodSec", 10))}
        return respond(env, "SUCCESS", {"subscriptionId": b["subscriptionId"]})

    def drop_intent(self, intent_id: str) -> None:
        for k in [k for k, s in self.subs.items() if s["intentId"] == intent_id]:
            self.subs.pop(k)

    async def loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            now = time.time()
            for sub in list(self.subs.values()):
                if now - sub["last"] >= sub["periodSec"]:
                    try:
                        await self.report(sub, now)
                    finally:
                        sub["last"] = now

    async def report(self, sub: dict[str, Any], now: float) -> None:
        rec = self.nl.intents.get(sub["intentId"])
        if not rec or rec.state not in ("FULFILLED", "DEGRADED", "DEPLOYING"):
            return
        samples = [s for d in rec.samples.values() for s in d if s["t"] > sub["last"]]
        if not samples:
            return
        kpis = sub.get("kpis") or ["latency_ms", "reliability_percent", "bandwidth_mbps"]
        agg = self.nl.perf_mon.aggregate(samples, kpis)
        summary = {}
        for e in rec.expectations:
            a = agg.get(e["kpi"])
            if a:
                summary[e["kpi"]] = a.get("p99") if (e["kpi"] == "latency_ms" and e.get("percentile")) else a["avg"]
        if "latency_ms" in agg:
            summary["latency_ms_p99"] = agg["latency_ms"]["p99"]
        viol = compare(rec.expectations, summary)
        met = not viol
        if sub.get("auto"):
            rec.windows_total += 1
            rec.windows_met += int(met)
        forecast = self.nl.model_exec.kpi_forecast(rec)
        expl = None
        if sub.get("includeExplanation", True):
            facts = {"state": rec.state, "kpis": summary, "violations": viol,
                     "forecast": f"latency p99 {forecast.get('latency_ms_p99')} ms in 60 s" if "latency_ms_p99" in forecast else None,
                     "action": rec.explanation or None}
            prompt = (f"Explain in one sentence the KPI status of intent {rec.intent_id} "
                      f"({rec.slice_type}, expectations {rec.expectations}): measured {summary}, violations {viol}, "
                      f"forecast {forecast}, last action: {rec.explanation}")
            r = await self.llm(prompt, facts=facts, kind="explain", tool_id="tool:gemini",
                               caller={"entity": "ReportGenerator"}, corr=rec.intent_id, max_tokens=128)
            expl = r.get("text")
        body = {"subscriptionId": sub["subscriptionId"], "intentId": rec.intent_id,
                "window": {"start": iso_ts(sub["last"]), "end": iso_ts(now)}, "kpis": agg,
                "compliance": {"met": met, "violations": viol, "samples": len(samples)}, "forecast": forecast,
                "explanation": expl, "state": rec.state}
        self.sent += 1
        await self.nl.sn_notify("PerformanceReportNotify", rec.intent_id, body, "ReportGenerator", "ServiceMonitor",
                                priority=rec.priority, trace=rec.trace)

    async def llm(self, prompt: str, facts: Optional[dict] = None, kind: str = "explain", tool_id: str = "tool:gemini",
                  caller: Optional[dict] = None, corr: str = "llm", max_tokens: int = 512) -> dict[str, Any]:
        inv = new_id("inv")
        r = await self.nl.nr_call("ToolInvokeRequest", "POST", f"/nr/v1/tools/{tool_id}:invoke",
                                  {"invocationId": inv, "toolId": tool_id, "operation": "generate",
                                   "arguments": {"prompt": prompt, "facts": facts or {}, "kind": kind, "maxTokens": max_tokens},
                                   "timeoutMs": 15000, "caller": caller or {}}, corr, "ToolRegistry")
        out = r.body.get("output") or {}
        return {"status": r.status, "text": out.get("text"), "servedBy": out.get("servedBy"), "usage": r.body.get("usage"),
                "reason": r.result.reason if r.result else None}


def iso_ts(t: float) -> str:
    from datetime import datetime, timezone
    return iso(datetime.fromtimestamp(t, tz=timezone.utc))
