"""Message log for every envelope crossing SN/NR (and MCP tool calls), keyed by correlationId.

Lets the simulator rebuild any intent as a sequence diagram and compute signalling overhead.
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

from . import config
from .envelope import Envelope, iso
from .events import brief, get_hub


class MessageLog:
    def __init__(self, layer: str, maxlen: int = 20000, to_file: bool = True):
        self.layer = layer
        self._rows: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._file: Optional[Path] = config.LOG_DIR / f"trace-{layer}.jsonl" if to_file else None
        self.totals: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "bytes": 0})
        self.hub = get_hub(layer)

    def record(self, direction: str, env: Envelope, transport: str = "http") -> None:
        raw = env.model_dump_json(exclude_none=True)
        h = env.header
        row = {
            "at": iso(), "layer": self.layer, "dir": direction, "transport": transport,
            "msgId": h.msgId, "correlationId": h.correlationId, "msgType": h.msgType,
            "refPoint": h.refPoint, "from": f"{h.source.layer}/{h.source.entity}",
            "to": f"{h.destination.layer}/{h.destination.entity}", "traceId": h.trace.traceId,
            "status": (env.body.get("result") or {}).get("status"), "bytes": len(raw),
        }
        with self._lock:
            self._rows.append(row)
            if direction == "out":
                t = self.totals[h.refPoint]
                t["count"] += 1
                t["bytes"] += len(raw)
            if self._file:
                with self._file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.hub.publish({**row, "kind": "msg", "brief": brief(h.msgType, env.body)})

    def record_raw(self, row: dict[str, Any]) -> None:
        row = {"at": iso(), "layer": self.layer, **row}
        with self._lock:
            self._rows.append(row)
        ev = {k: v for k, v in row.items() if k != "body"}
        if isinstance(row.get("body"), dict):
            ev["brief"] = brief(row.get("msgType", ""), row["body"])
        self.hub.publish({**ev, "kind": "internal"})

    def query(self, correlation_id: Optional[str] = None, trace_id: Optional[str] = None,
              limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._rows)
        if correlation_id:
            rows = [r for r in rows if r.get("correlationId") == correlation_id]
        if trace_id:
            rows = [r for r in rows if r.get("traceId") == trace_id]
        return rows[-limit:]

    def overhead(self, correlation_id: str) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "bytes": 0})
        for r in self.query(correlation_id, limit=100000):
            if r.get("dir") == "out" and "refPoint" in r:
                out[r["refPoint"]]["count"] += 1
                out[r["refPoint"]]["bytes"] += r.get("bytes", 0)
        return dict(out)


def to_mermaid(rows: list[dict[str, Any]]) -> str:
    """Sequence diagram (Mermaid) from 'out' rows of one or more layer logs."""
    seen = set()
    lines = ["sequenceDiagram"]
    for r in sorted(rows, key=lambda x: x.get("at", "")):
        key = r.get("msgId") or f'{r.get("at")}{r.get("msgType")}{r.get("to")}'
        if r.get("dir") not in ("out", "internal") or key in seen:
            continue
        seen.add(key)
        a = r["from"].split("/")[-1]
        b = r["to"].split("/")[-1]
        arrow = "-->>" if r["msgType"].endswith("Response") else "->>"
        label = f'{r.get("refPoint", "agent")} {r["msgType"]}' + (f' [{r["status"]}]' if r.get("status") else "")
        lines.append(f"    {a}{arrow}{b}: {label}")
    return "\n".join(lines)
