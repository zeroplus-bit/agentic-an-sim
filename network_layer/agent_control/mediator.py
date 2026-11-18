"""Intent Mediation (IntentMediator): SN-01/02/03 handling, decomposition of an intent into tasks and
assignment of tasks to agents found via discovery (Flow 1)."""
from __future__ import annotations

import logging
import secrets
from typing import Any

from common.envelope import Envelope, respond
from common.sn_messages import Intent, IntentModifyRequest

from ..core import IntentRecord

log = logging.getLogger("mediator")
TERMINAL = {"TERMINATED", "REJECTED", "FAILED"}


class IntentMediator:
    def __init__(self, nl):
        self.nl = nl

    # ---- SN-01 ------------------------------------------------------------------------
    async def submit(self, env: Envelope) -> Envelope:
        claims, err = self.nl.auth.verify_sn(env, "sn:intent:write")
        if err:
            return respond(env, "UNAUTHORIZED", reason=err)
        intent = Intent.model_validate(env.body["intent"]).model_dump(mode="json", exclude_none=True)
        iid = intent["intentId"]
        old = self.nl.intents.get(iid)
        if old and old.state not in TERMINAL:
            return respond(env, "REJECTED", {"intentId": iid, "state": old.state}, reason="DUPLICATE_INTENT")
        if not (intent.get("context") or {}).get("area", {}).get("cells"):
            intent.setdefault("context", {}).setdefault("area", {"type": "cellList"})["cells"] = await self._pick_cells(intent)
        unknown = [c for c in intent["context"]["area"]["cells"] if c not in await self.nl.resource_ctrl.cell_ids()]
        if unknown:
            return respond(env, "REJECTED", {"intentId": iid}, reason=f"UNKNOWN_CELLS:{','.join(unknown)}")
        reason = self.nl.conflict.check_new(intent, env.header.priority)
        if reason:
            return respond(env, "REJECTED", {"intentId": iid, "state": "REJECTED"}, reason=reason)
        rec = IntentRecord(intent=intent, priority=env.header.priority,
                           tenant=intent.get("tenantId") or claims.get("tenantId") or "unknown", trace=env.header.trace)
        rec.service_instance_id = f"slice-{secrets.token_hex(4)}"
        self.nl.intents[iid] = rec
        self.nl.spawn(self.provision(rec))
        return respond(env, "ACCEPTED", {"intentId": iid, "state": "RECEIVED",
                                         "cells": intent["context"]["area"]["cells"]})

    async def _pick_cells(self, intent: dict[str, Any], n: int = 2) -> list[str]:
        inv = await self.nl.resource_ctrl.inventory("comm", corr=intent["intentId"])
        cells = sorted([r for r in inv.get("resources", []) if r.get("up")], key=lambda r: -r["freeSharePercent"])
        return [r["resourceId"].split(":", 1)[1] for r in cells[:n]]

    # ---- Flow 1 pipeline ------------------------------------------------------------------
    async def provision(self, rec: IntentRecord) -> None:
        nl, iid = self.nl, rec.intent_id
        await nl.tracker.set_state(rec, "VALIDATING")
        quota_err = nl.resource_ctrl.quota_check(rec.tenant, nl.translator.estimate_share(rec.intent, 0.72))
        if quota_err:
            await nl.tracker.set_state(rec, "REJECTED", reason=quota_err)
            return
        await nl.tracker.set_state(rec, "PLANNING")
        plan = await self.decompose(rec)
        rec.explanation = f"plan: {', '.join(p['type'] for p in plan)}"
        alloc = nl.tracker.new_task(iid, "ALLOCATE_SLICE", {"intentId": iid, "cells": rec.cells})
        out = await nl.agent_bus.delegate(alloc, "ALLOCATE_SLICE")
        if rec.state in TERMINAL:      # cancelled meanwhile
            return
        if not out.get("ok"):
            await self._release(rec)
            err = out.get("error", "allocation failed")
            final = "FAILED" if err.startswith(("NO_AGENT", "Tool", "MCP")) or "Error" in err else "REJECTED"
            await nl.tracker.set_state(rec, final, reason=err, explanation=out.get("explanation"),
                                       fulfilment={"predicted": out.get("predicted"), "met": False})
            return
        await self._after_deploy(rec, out)

    async def _after_deploy(self, rec: IntentRecord, out: dict[str, Any]) -> None:
        nl, iid = self.nl, rec.intent_id
        if not rec.telemetry_sub:
            r = await nl.data_ctrl.subscribe([f"comm:{c}" for c in rec.cells], {"intentId": iid, "sliceType": rec.slice_type},
                                             1.0, iid)
            rec.telemetry_sub = r.body.get("subscriptionId")
        if not any(nl.tasks[t].type == "ASSURE_KPI" and nl.tasks[t].state == "RUNNING" for t in rec.tasks):
            watch = nl.tracker.new_task(iid, "ASSURE_KPI", {"intentId": iid, "mode": "watch"})
            agents = nl.registry.discover("ASSURE_KPI")
            nl.tracker.update_task(watch, "RUNNING" if agents else "PENDING", agent_id=agents[0]["agentId"] if agents else None)
        nl.reports.ensure_auto(rec)
        if rec.state == "PLANNING":
            await nl.tracker.set_state(rec, "DEPLOYING")
        pred = out.get("predicted") or {}
        rec.recent_ok.clear()
        await nl.tracker.set_state(rec, "FULFILLED", explanation=out.get("explanation"),
                                   fulfilment={**{k: v for k, v in pred.items()}, "met": True, "source": "validated"})

    async def decompose(self, rec: IntentRecord) -> list[dict[str, Any]]:
        """Ask a planner agent (skill DECOMPOSE_INTENT) if one is registered, else the rule-based plan."""
        if self.nl.registry.discover("DECOMPOSE_INTENT"):
            t = self.nl.tracker.new_task(rec.intent_id, "DECOMPOSE_INTENT", {"intentId": rec.intent_id})
            out = await self.nl.agent_bus.delegate(t, "DECOMPOSE_INTENT")
            if out.get("ok") and out.get("plan"):
                return out["plan"]
        return self.rule_plan(rec)

    @staticmethod
    def rule_plan(rec: IntentRecord) -> list[dict[str, Any]]:
        plan = [{"type": "ALLOCATE_SLICE", "skill": "ALLOCATE_SLICE", "cells": rec.cells},
                {"type": "ASSURE_KPI", "skill": "ASSURE_KPI", "kpis": [e["kpi"] for e in rec.expectations]}]
        if (rec.intent.get("reportPolicy") or {}).get("periodSec"):
            plan.append({"type": "REPORT", "skill": "REPORT", "periodSec": rec.intent["reportPolicy"]["periodSec"]})
        return plan

    # ---- SN-02 --------------------------------------------------------------------------------
    async def modify(self, env: Envelope, intent_id: str) -> Envelope:
        _, err = self.nl.auth.verify_sn(env, "sn:intent:write")
        if err:
            return respond(env, "UNAUTHORIZED", reason=err)
        rec = self.nl.intents.get(intent_id)
        if not rec or rec.state in TERMINAL:
            return respond(env, "REJECTED", {"intentId": intent_id}, reason="UNKNOWN_OR_TERMINATED_INTENT")
        if rec.state not in ("FULFILLED", "DEGRADED"):
            return respond(env, "REJECTED", {"intentId": intent_id, "state": rec.state}, reason="INTENT_BUSY")
        m = IntentModifyRequest.model_validate({"intentId": intent_id, **env.body}).model_dump(mode="json", exclude_none=True)
        for k in ("expectations", "constraints", "reportPolicy"):
            if k in m:
                rec.intent[k] = m[k]
        if "context" in m:
            rec.intent["context"] = {**rec.intent.get("context", {}), **m["context"]}
        if m.get("priority"):
            rec.priority = int(m["priority"])
        self.nl.spawn(self._replan(rec, "modify"))
        return respond(env, "ACCEPTED", {"intentId": intent_id, "state": "PLANNING"})

    async def _replan(self, rec: IntentRecord, why: str) -> None:
        await self.nl.tracker.set_state(rec, "PLANNING", reason=why)
        t = self.nl.tracker.new_task(rec.intent_id, "REBALANCE_PRB", {"intentId": rec.intent_id, "why": why})
        out = await self.nl.agent_bus.delegate(t, "REBALANCE_PRB")
        if out.get("ok"):
            await self._after_deploy(rec, out)
        else:
            if rec.state == "PLANNING":
                await self.nl.tracker.set_state(rec, "DEPLOYING")
            await self.nl.tracker.set_state(rec, "DEGRADED", reason=out.get("error"), explanation=out.get("explanation"))

    # ---- SN-03 --------------------------------------------------------------------------------
    async def cancel(self, env: Envelope, intent_id: str) -> Envelope:
        _, err = self.nl.auth.verify_sn(env, "sn:intent:write")
        if err:
            return respond(env, "UNAUTHORIZED", reason=err)
        rec = self.nl.intents.get(intent_id)
        if not rec or rec.state in TERMINAL:
            return respond(env, "REJECTED", {"intentId": intent_id}, reason="UNKNOWN_OR_TERMINATED_INTENT")
        self.nl.spawn(self.terminate(rec, env.body.get("reason") or "cancelled by service layer"))
        return respond(env, "ACCEPTED", {"intentId": intent_id, "state": rec.state})

    async def terminate(self, rec: IntentRecord, reason: str) -> None:
        nl, iid = self.nl, rec.intent_id
        for cfg in reversed(rec.configs):
            await nl.actions.rollback(cfg["configId"], iid, "intent terminated", record=False)
        rec.configs.clear()
        rec.shares.clear()
        await self._release(rec)
        if rec.telemetry_sub:
            await nl.data_ctrl.cancel(rec.telemetry_sub, iid)
        nl.reports.drop_intent(iid)
        for tid in rec.tasks:
            t = nl.tasks.get(tid)
            if t and t.state in ("PENDING", "ASSIGNED", "RUNNING"):
                nl.tracker.update_task(t, "CANCELLED")
        await nl.tracker.set_state(rec, "TERMINATED", reason=reason)

    async def _release(self, rec: IntentRecord) -> None:
        if rec.reservation_id:
            await self.nl.resource_ctrl.release(rec.reservation_id, rec.intent_id)
            rec.reservation_id = None
