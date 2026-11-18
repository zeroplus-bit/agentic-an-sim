"""Fetch the pretrained encoders that are too large for git (pinned Hugging Face revisions, SHA-256 checked).

    .venv/Scripts/python tools/download_models.py          # SBERT (required for semantic intent NLU)
    .venv/Scripts/python tools/download_models.py --nli    # + NLI cross-check model (only for INTENT_NLI / tools/eval_p3.py)

Without the SBERT model the intent classifier falls back to char n-grams (it still runs, with lower accuracy).
The small trained heads (models/joint_nlu) and the calibration files are in the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "sbert": {"repo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
              "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
              "dir": "models/sbert-multilingual-minilm",
              "files": ["onnx/model.onnx", "tokenizer.json", "config.json", "1_Pooling/config.json"],
              "sha256": {"onnx/model.onnx": "10f7a088420252b26caf819236ca2c9d2987afd0fc06fec7553b542a5655a05a"}},
    "nli": {"repo": "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
            "revision": "0a71e92a985b6e1ad1828cf67ce9c459639c1dca",
            "dir": "models/nli-multilingual-minilm",
            "files": ["onnx/model.onnx", "tokenizer.json", "config.json"],
            "sha256": {"onnx/model.onnx": "79f8cda2b1230585a95ea0514a6f1bd21c5c986ba0529bb3261213a3e195fa6e"}},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str) -> bool:
    from huggingface_hub import hf_hub_download
    m = MODELS[name]
    out = ROOT / m["dir"]
    ok = True
    for f in m["files"]:
        target = out / f
        if not target.exists():
            print(f"[{name}] downloading {m['repo']}@{m['revision'][:8]}/{f}")
            hf_hub_download(m["repo"], f, revision=m["revision"], local_dir=str(out))
        want = m["sha256"].get(f)
        if want:
            got = sha256(target)
            if got != want:
                print(f"[{name}] SHA-256 mismatch for {f}: {got}")
                ok = False
                continue
        print(f"[{name}] ok  {target.relative_to(ROOT)}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", action="store_true", help="also fetch the NLI cross-check model (~410 MB)")
    a = ap.parse_args()
    names = ["sbert"] + (["nli"] if a.nli else [])
    return 0 if all(fetch(n) for n in names) else 1


if __name__ == "__main__":
    sys.exit(main())
