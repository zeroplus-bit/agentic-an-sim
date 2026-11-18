"""Computing Resources (ComputePool) and Storage Resources (StoragePool)."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from common import config


@dataclass
class Node:
    node_id: str
    cpu: float
    gpu: float
    memory_mb: float
    used_cpu: float = 0.0
    used_gpu: float = 0.0
    used_mem: float = 0.0

    @property
    def load(self) -> float:
        return max(self.used_cpu / self.cpu if self.cpu else 0, self.used_gpu / self.gpu if self.gpu else 0)


@dataclass
class Job:
    job_id: str
    kind: str
    node: str
    cpu: float
    gpu: float
    model: Optional[str] = None
    input_ref: Optional[str] = None
    state: str = "QUEUED"
    output_ref: Optional[str] = None
    started: float = field(default_factory=time.time)
    reason: Optional[str] = None


JobListener = Callable[[Job], Awaitable[None]]

DURATION = {"INFERENCE": 0.2, "MODEL_LOAD": 0.5, "TRAINING": 3.0}   # AGENT_RUNTIME runs until cancelled


class ComputePool:
    def __init__(self):
        self.nodes: dict[str, Node] = {
            "edge-1": Node("edge-1", cpu=8, gpu=0, memory_mb=16384),
            "edge-2": Node("edge-2", cpu=8, gpu=0, memory_mb=16384),
            "node-1": Node("node-1", cpu=32, gpu=2, memory_mb=131072),
            "node-2": Node("node-2", cpu=32, gpu=2, memory_mb=131072),
            "node-3": Node("node-3", cpu=64, gpu=4, memory_mb=262144),
        }
        self.jobs: dict[str, Job] = {}
        self.listener: Optional[JobListener] = None

    # reservations ---------------------------------------------------------------
    def can_reserve(self, node: str, kind: str, amount: float) -> bool:
        n = self.nodes.get(node)
        if not n:
            return False
        if kind == "CPU":
            return n.used_cpu + amount <= n.cpu
        if kind == "GPU":
            return n.used_gpu + amount <= n.gpu
        if kind == "MEMORY_MB":
            return n.used_mem + amount <= n.memory_mb
        return False

    def reserve(self, node: str, kind: str, amount: float, sign: int = 1) -> None:
        n = self.nodes[node]
        if kind == "CPU":
            n.used_cpu = max(0.0, n.used_cpu + sign * amount)
        elif kind == "GPU":
            n.used_gpu = max(0.0, n.used_gpu + sign * amount)
        elif kind == "MEMORY_MB":
            n.used_mem = max(0.0, n.used_mem + sign * amount)

    def pick_node(self, cpu: float, gpu: float) -> Optional[str]:
        best = None
        for n in sorted(self.nodes.values(), key=lambda x: x.load):
            if n.used_cpu + cpu <= n.cpu and n.used_gpu + gpu <= n.gpu:
                best = n.node_id
                break
        return best

    # jobs ----------------------------------------------------------------------
    async def submit(self, body: dict[str, Any]) -> Job:
        res = body.get("resources") or {}
        cpu, gpu = float(res.get("cpu", 1)), float(res.get("gpu", 0))
        node = (body.get("node") or "").split(":")[-1] or self.pick_node(cpu, gpu)
        if not node or node not in self.nodes:
            job = Job(body["jobId"], body["kind"], node or "-", cpu, gpu, state="FAILED", reason="NO_CAPACITY")
            self.jobs[job.job_id] = job
            return job
        job = Job(body["jobId"], body["kind"], node, cpu, gpu, model=body.get("model"), input_ref=body.get("inputRef"))
        self.jobs[job.job_id] = job
        self.reserve(node, "CPU", cpu)
        self.reserve(node, "GPU", gpu)
        asyncio.create_task(self._run(job))
        return job

    async def _run(self, job: Job) -> None:
        await self._emit(job)
        await asyncio.sleep(0.05)
        job.state = "RUNNING"
        await self._emit(job)
        if job.kind == "AGENT_RUNTIME":
            return
        await asyncio.sleep(DURATION.get(job.kind, 0.2))
        if job.state == "RUNNING":
            job.state = "DONE"
            job.output_ref = f"stor:jobs/{job.job_id}/output"
            self._free(job)
            await self._emit(job)

    def _free(self, job: Job) -> None:
        if job.node in self.nodes:
            self.reserve(job.node, "CPU", job.cpu, -1)
            self.reserve(job.node, "GPU", job.gpu, -1)

    async def cancel(self, job_id: str) -> Optional[Job]:
        job = self.jobs.get(job_id)
        if job and job.state in ("QUEUED", "RUNNING"):
            job.state = "CANCELLED"
            self._free(job)
            await self._emit(job)
        return job

    async def _emit(self, job: Job) -> None:
        if self.listener:
            await self.listener(job)

    def inventory(self) -> list[dict[str, Any]]:
        return [{"resourceId": f"comp:{n.node_id}", "type": "comp", "cpu": n.cpu, "gpu": n.gpu,
                 "memoryMb": n.memory_mb, "usedCpu": n.used_cpu, "usedGpu": n.used_gpu,
                 "load": round(n.load, 3),
                 "runningJobs": sum(1 for j in self.jobs.values() if j.node == n.node_id and j.state == "RUNNING")}
                for n in self.nodes.values()]


class StoragePool:
    """Object store: buckets in memory, mirrored to data/storage/ for durability."""

    def __init__(self, root: Path = config.DATA_DIR / "storage", quota_gb: float = 100.0):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta: dict[tuple[str, str], dict[str, Any]] = {}
        self.quota_gb = quota_gb
        self.reserved_gb: dict[str, float] = {}
        self._seed()

    def _path(self, bucket: str, key: str) -> Path:
        safe = key.replace("..", "_")
        return self.root / bucket / safe

    def put(self, bucket: str, key: str, data: bytes, meta: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        p = self._path(bucket, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        m = {"bucket": bucket, "key": key, "size": len(data), "storedAt": time.time(), **(meta or {})}
        self.meta[(bucket, key)] = m
        return m

    def get(self, bucket: str, key: str) -> Optional[tuple[bytes, dict[str, Any]]]:
        p = self._path(bucket, key)
        if not p.exists():
            return None
        return p.read_bytes(), self.meta.get((bucket, key), {"bucket": bucket, "key": key, "size": p.stat().st_size})

    def link(self, bucket: str, key: str, source: Path, meta: dict[str, Any]) -> None:
        """Register an existing file (e.g. a repo model) without copying it."""
        if source.exists():
            self.meta[(bucket, key)] = {"bucket": bucket, "key": key, "size": source.stat().st_size,
                                        "path": str(source), **meta}

    def _seed(self) -> None:
        self.link("models", "lstm_single/1.0", config.LSTM_MODEL_PATH,
                  {"contentType": "application/x-hdf5", "framework": "keras-h5"})
        self.link("models", "lstm_single/2.0", config.LSTM_MODEL_V2_PATH,
                  {"contentType": "application/x-hdf5", "framework": "keras-h5"})

    def get_any(self, bucket: str, key: str) -> Optional[tuple[bytes, dict[str, Any]]]:
        got = self.get(bucket, key)
        if got:
            return got
        m = self.meta.get((bucket, key))
        if m and m.get("path"):
            return Path(m["path"]).read_bytes(), m
        return None

    def used_gb(self) -> float:
        return sum(m.get("size", 0) for m in self.meta.values()) / 1e9

    def inventory(self) -> list[dict[str, Any]]:
        buckets: dict[str, int] = {}
        for (b, _k) in self.meta:
            buckets[b] = buckets.get(b, 0) + 1
        return [{"resourceId": f"stor:{b}", "type": "stor", "objects": n} for b, n in buckets.items()] + \
               [{"resourceId": "stor:*", "type": "stor", "quotaGb": self.quota_gb, "usedGb": round(self.used_gb(), 4),
                 "reservedGb": sum(self.reserved_gb.values())}]
