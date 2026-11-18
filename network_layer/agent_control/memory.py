"""Memory (AgentMemory): short-term (per task) and long-term (episodes, outcomes) memory in SQLite."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Optional

from common import config


class AgentMemory:
    def __init__(self, nl, path=config.MEMORY_DB):
        self.nl = nl
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS episodes(
              id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, intent_id TEXT, agent_id TEXT, task_id TEXT,
              state TEXT, action INTEGER, reward REAL, next_state TEXT, outcome TEXT, details TEXT);
            CREATE TABLE IF NOT EXISTS short_term(
              task_id TEXT, key TEXT, value TEXT, at REAL, PRIMARY KEY(task_id, key));
        """)
        self.db.commit()
        self._short: dict[str, dict[str, Any]] = {}

    # short-term ------------------------------------------------------------------------
    def put(self, task_id: str, key: str, value: Any) -> None:
        self._short.setdefault(task_id, {})[key] = value
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO short_term VALUES(?,?,?,?)",
                            (task_id, key, json.dumps(value, default=str), time.time()))
            self.db.commit()

    def get(self, task_id: str, key: Optional[str] = None) -> Any:
        d = self._short.get(task_id, {})
        return d if key is None else d.get(key)

    # long-term ---------------------------------------------------------------------------
    def store_episode(self, intent_id: str, agent_id: str, task_id: str, state, action: int, reward: float,
                      next_state, outcome: str, details: Optional[dict] = None) -> int:
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO episodes(at,intent_id,agent_id,task_id,state,action,reward,next_state,outcome,details) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (time.time(), intent_id, agent_id, task_id, json.dumps(list(map(float, state))), int(action),
                 float(reward), json.dumps(list(map(float, next_state))), outcome, json.dumps(details or {}, default=str)))
            self.db.commit()
            return int(cur.lastrowid)

    def recall(self, intent_id: Optional[str] = None, agent_id: Optional[str] = None, limit: int = 20) -> list[dict]:
        q = "SELECT id,at,intent_id,agent_id,task_id,action,reward,outcome,details FROM episodes"
        cond, args = [], []
        if intent_id:
            cond.append("intent_id=?")
            args.append(intent_id)
        if agent_id:
            cond.append("agent_id=?")
            args.append(agent_id)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self.db.execute(q, args).fetchall()
        keys = ["id", "at", "intentId", "agentId", "taskId", "action", "reward", "outcome", "details"]
        out = []
        for r in rows:
            d = dict(zip(keys, r))
            d["details"] = json.loads(d["details"] or "{}")
            out.append(d)
        return out

    def stats(self) -> dict[str, Any]:
        with self._lock:
            n, avg = self.db.execute("SELECT COUNT(*), AVG(reward) FROM episodes").fetchone()
        return {"episodes": n, "avgReward": round(avg, 3) if avg is not None else None}
