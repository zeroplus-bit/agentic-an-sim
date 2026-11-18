"""Start the three layer processes (Resource :8300, Network :8200, Service :8100) and the operations dashboard
(:8400, follows the layers over SSE) and wait until healthy.

    python run_all.py            # Ctrl+C stops all three
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import config  # noqa: E402

LAYERS = [("resource", "resource_layer.app:app", config.RESOURCE_PORT, config.RESOURCE_URL),
          ("network", "network_layer.app:app", config.NETWORK_PORT, config.NETWORK_URL),
          ("service", "service_layer.app:app", config.SERVICE_PORT, config.SERVICE_URL),
          ("dashboard", "dashboard.app:app", config.DASHBOARD_PORT, config.DASHBOARD_URL)]


def main() -> int:
    procs = []
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    for name, target, port, _ in LAYERS:
        log = open(config.LOG_DIR / f"{name}.log", "w", encoding="utf-8")
        p = subprocess.Popen([sys.executable, "-m", "uvicorn", target, "--host", config.HOST, "--port", str(port),
                              "--log-level", "warning"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((name, p))
        print(f"started {name:8s} layer pid={p.pid} port={port}  (log: logs/{name}.log)")
    try:
        deadline = time.time() + 60
        pending = {name: url for name, _, _, url in LAYERS}
        while pending and time.time() < deadline:
            for name, url in list(pending.items()):
                try:
                    if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                        print(f"  {name} healthy at {url}")
                        pending.pop(name)
                except httpx.HTTPError:
                    pass
            time.sleep(0.5)
        if pending:
            print(f"not healthy: {list(pending)} - see logs/")
        print(f"\nDashboard   : {config.DASHBOARD_URL}/\nService API : {config.SERVICE_URL}/docs\nNetwork MCP : {config.NETWORK_URL}/tools/mcp"
              f"\nSN MCP      : python -m mcp_servers.sn_mcp_server   (stdio)\nCtrl+C to stop.")
        while all(p.poll() is None for _, p in procs):
            time.sleep(1)
        for name, p in procs:
            if p.poll() is not None:
                print(f"{name} exited with {p.returncode}; see logs/{name}.log")
    except KeyboardInterrupt:
        pass
    finally:
        for _, p in procs:
            if p.poll() is None:
                p.terminate()
        for _, p in procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
