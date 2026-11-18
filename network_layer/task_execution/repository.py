"""Repository: store task inputs/outputs, configs and model artefacts on Storage Resources (NR-13)."""
from __future__ import annotations

import json
from typing import Any, Optional

from common.envelope import ep, make, respond


class Repository:
    def __init__(self, nl):
        self.nl = nl
        self.retention_hours: Optional[float] = None

    def _env(self, method: str, bucket: str, key: str, corr: str):
        return make("StorageObjectRequest", "NR", corr, {"bucket": bucket, "key": key, "method": method},
                    ep("network", "Repository"), ep("resource", "StorageResources"), trace=self.nl.flow_trace(corr))

    @staticmethod
    def _received(env, r, status: str, reason: Optional[str] = None):
        """The response as received: same msgId the Resource Layer logged (X-Msg-Id / envelope header)."""
        resp = respond(env, status, reason=reason)
        mid = None
        if r is not None:
            mid = r.headers.get("X-Msg-Id")
            if not mid and r.headers.get("content-type", "").startswith("application/json"):
                try:
                    mid = (r.json().get("header") or {}).get("msgId")
                except ValueError:
                    mid = None
        if mid:
            resp.header.msgId = mid
        return resp

    async def put(self, bucket: str, key: str, data: bytes, content_type: str = "application/json",
                  corr: str = "repo", intent_id: Optional[str] = None) -> dict[str, Any]:
        env = self._env("PUT", bucket, key, corr)
        self.nl.tracer.record("out", env, transport="http PUT storage")
        headers = {"X-Msg-Id": env.header.msgId, "X-Correlation-Id": corr, "Content-Type": content_type,
                   "traceparent": f"00-{env.header.trace.traceId}-{env.header.trace.spanId}-01"}
        if intent_id:
            headers["X-Intent-Id"] = intent_id
        if self.retention_hours:
            headers["X-Retention-Hours"] = str(self.retention_hours)
        r = None
        try:
            r = await self.nl.nr.raw("PUT", f"/nr/v1/storage/{bucket}/{key}", content=data, headers=headers)
            ok = r.status_code < 300
            meta = r.json().get("body", {}).get("object") if ok else None
        except Exception as e:
            ok, meta = False, {"error": str(e)}
        self.nl.tracer.record("in", self._received(env, r, "SUCCESS" if ok else "FAILED"), transport="http PUT storage")
        return {"ok": ok, "object": meta}

    async def get(self, bucket: str, key: str, corr: str = "repo") -> tuple[Optional[bytes], dict[str, Any]]:
        env = self._env("GET", bucket, key, corr)
        self.nl.tracer.record("out", env, transport="http GET storage")
        try:
            r = await self.nl.nr.raw("GET", f"/nr/v1/storage/{bucket}/{key}",
                                     headers={"X-Msg-Id": env.header.msgId, "X-Correlation-Id": corr,
                                              "traceparent": f"00-{env.header.trace.traceId}-{env.header.trace.spanId}-01"})
        except Exception as e:
            self.nl.tracer.record("in", respond(env, "FAILED", reason=str(e)), transport="http GET storage")
            return None, {"error": str(e)}
        ok = r.status_code == 200
        self.nl.tracer.record("in", self._received(env, r, "SUCCESS" if ok else "REJECTED"), transport="http GET storage")
        if not ok:
            return None, {"status": r.status_code}
        return r.content, {"contentType": r.headers.get("content-type"), "size": len(r.content)}

    async def get_artifact(self, uri: str, corr: str) -> tuple[Optional[bytes], dict[str, Any]]:
        """repo://models/lstm_single/2.0  ->  bucket 'models', key 'lstm_single/2.0'"""
        path = uri.split("://", 1)[-1]
        bucket, _, key = path.partition("/")
        return await self.get(bucket, key, corr)

    async def save_task(self, task) -> None:
        doc = {"taskId": task.task_id, "intentId": task.intent_id, "type": task.type, "agentId": task.agent_id,
               "state": task.state, "input": task.input, "output": task.output, "created": task.created,
               "updated": task.updated}
        await self.put("repository", f"tasks/{task.task_id}.json", json.dumps(doc, default=str).encode(),
                       corr=task.intent_id or task.task_id, intent_id=task.intent_id)
