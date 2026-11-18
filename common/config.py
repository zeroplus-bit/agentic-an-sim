"""Central configuration (all paths and endpoints come from here, never hard-coded).

Every value can be overridden with an AANS_* environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # agentic-an-sim/
REPO = Path(os.environ.get("AANS_REPO", ROOT.parent))  # Build-A-Thon-SRM-main/ (upstream repo, optional)
VENDOR = ROOT / "vendor"                            # copies of the upstream files the simulator needs (MIT)


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _upstream(rel: str) -> str:
    """An upstream repo file: the vendored copy when present (standalone checkout), else the upstream repo."""
    v = VENDOR / rel
    return str(v if v.exists() else REPO / rel)


# ---- processes ---------------------------------------------------------
SERVICE_PORT = int(_env("AANS_SERVICE_PORT", "8100"))
NETWORK_PORT = int(_env("AANS_NETWORK_PORT", "8200"))
RESOURCE_PORT = int(_env("AANS_RESOURCE_PORT", "8300"))
DASHBOARD_PORT = int(_env("AANS_DASHBOARD_PORT", "8400"))
HOST = _env("AANS_HOST", "127.0.0.1")

SERVICE_URL = _env("AANS_SERVICE_URL", f"http://{HOST}:{SERVICE_PORT}")
NETWORK_URL = _env("AANS_NETWORK_URL", f"http://{HOST}:{NETWORK_PORT}")
RESOURCE_URL = _env("AANS_RESOURCE_URL", f"http://{HOST}:{RESOURCE_PORT}")
DASHBOARD_URL = _env("AANS_DASHBOARD_URL", f"http://{HOST}:{DASHBOARD_PORT}")

# ---- security ----------------------------------------------------------
# SN: bearer tokens issued by IdentityManager (Service Layer), verified by AuthGuard.
SN_SECRET = _env("AANS_SN_SECRET", "dev-sn-secret-change-me")
# NR: mutual service token, only Network Layer components may call the Resource Layer.
NR_SERVICE_TOKEN = _env("AANS_NR_TOKEN", "dev-nr-service-token-change-me")
# Bus bridge token (cross-process notifications)
BUS_TOKEN = _env("AANS_BUS_TOKEN", "dev-bus-token-change-me")

# ---- transports ---------------------------------------------------------
BUS_KIND = _env("AANS_BUS", "http")          # inproc | http | nats
NATS_URL = _env("AANS_NATS_URL", "nats://127.0.0.1:4222")
TOOL_MODE = _env("AANS_TOOL_MODE", "mcp")      # mcp | direct  (how agents reach ToolRegistry)

# ---- models / data -------------------------------------------------------
LSTM_MODEL_PATH = Path(_env("AANS_LSTM_MODEL", _upstream("models/lstm_single/final_model.h5")))
LSTM_MODEL_V2_PATH = Path(_env("AANS_LSTM_MODEL_V2", _upstream("models/lstm_single/best_model.h5")))
LSTM_SCALER_PATH = Path(_env("AANS_LSTM_SCALER", _upstream("models/lstm_single/X_scaler.npy")))
DISTILBERT_PATH = Path(_env("AANS_DISTILBERT", str(REPO / "Context_Model/fine_tuned_slice_intent_model")))
# intent NLU: auto = semantic (SBERT ONNX if present, else char n-grams) | semantic | keyword
INTENT_BACKEND = _env("AANS_INTENT_BACKEND", "auto")
INTENT_DATA_DIR = Path(_env("AANS_INTENT_DATA", str(ROOT / "datasets/intent_nlu")))
SBERT_PATH = Path(_env("AANS_SBERT", str(ROOT / "models/sbert-multilingual-minilm")))
INTENT_NLI = _env("AANS_INTENT_NLI", "off")   # off | info (adds a non-gating NLI second opinion to SN-01)
NLI_PATH =Path(_env("AANS_NLI", str(ROOT / "models/nli-multilingual-minilm")))   # zero-shot cross-check
INTENT_CONSTRAINTS = Path(_env("AANS_INTENT_CONSTRAINTS", str(ROOT / "datasets/intent_nlu/constraints.json")))
JOINT_NLU_PATH =Path(_env("AANS_JOINT_NLU", str(ROOT / "models/joint_nlu/joint_nlu.npz")))  # TF joint model weights
BS_CSV = Path(_env("AANS_BS_CSV", _upstream("5G-Network-Slicing/network_data/base_station_data.csv")))
CLIENT_CSV = Path(_env("AANS_CLIENT_CSV", _upstream("5G-Network-Slicing/network_data/client_data.csv")))
VENDOR_OFFERINGS = Path(_env("AANS_OFFERINGS", _upstream("5G-Marketplace/data/vendor_registry/offerings.json")))
VENDORS = Path(_env("AANS_VENDORS", _upstream("5G-Marketplace/data/vendor_registry/vendors.json")))

DATA_DIR = Path(_env("AANS_DATA_DIR", str(ROOT / "data")))
LOG_DIR = Path(_env("AANS_LOG_DIR", str(ROOT / "logs")))
MEMORY_DB = Path(_env("AANS_MEMORY_DB", str(DATA_DIR / "agent_memory.sqlite")))

# ---- simulation ----------------------------------------------------------
TELEMETRY_PERIOD_SEC = float(_env("AANS_TELEMETRY_PERIOD", "1.0"))
RESERVATION_HOLD_SEC = float(_env("AANS_RESERVATION_HOLD", "300"))
ROLLBACK_WATCH_SEC = float(_env("AANS_ROLLBACK_WATCH", "30"))
MAX_PROPOSALS = int(_env("AANS_MAX_PROPOSALS", "3"))
RANDOM_SEED = int(_env("AANS_SEED", "7"))

# ---- LLM tools (only the Resource Layer ToolHost reads these) -------------
GEMINI_MODEL = _env("AANS_GEMINI_MODEL", "gemini-2.0-flash")
OLLAMA_URL = _env("AANS_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = _env("AANS_OLLAMA_MODEL", "llama3.1")

for _d in (DATA_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
