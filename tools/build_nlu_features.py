"""Precompute frozen SBERT token features + BIO tags for the TensorFlow joint model (runs in the main venv).

    python tools/build_nlu_features.py   -> datasets/intent_nlu/features/{train,calib,test,test_v2}.npz
Training (TensorFlow) then runs in the separate Python 3.12 venv on these arrays: tools/train_joint_nlu.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import config  # noqa: E402
from service_layer.orchestration.semantic import CLASSES, SbertOnnxEncoder, load_jsonl  # noqa: E402
from service_layer.orchestration.slots import TAGS, bio_tags, gold_spans  # noqa: E402

D = config.INTENT_DATA_DIR
MAX_LEN = 64
SPLITS = {"train": ["train_slots.jsonl", "train.jsonl"], "calib": ["calib.jsonl"], "test": ["test.jsonl"],
          "test_v2": ["test_v2.jsonl"]}


def main() -> None:
    enc = SbertOnnxEncoder(config.SBERT_PATH)
    out = D / "features"
    out.mkdir(exist_ok=True)
    for split, files in SPLITS.items():
        rows = [r for f in files for r in load_jsonl(D / f)]
        h, m, offs, s = enc.encode_tokens([r["text"] for r in rows], MAX_LEN)
        tags, miss, total = [], 0, 0
        for r, off in zip(rows, offs):
            spans = gold_spans(r)
            n_gold = len(r.get("expectations") or {}) + (r.get("devices") is not None)
            # 'five nines' has no number to tag; it is handled by a rule at decode time
            n_gold -= int("reliability_percent" in (r.get("expectations") or {}) and "nine" in r["text"].lower()
                          or "나인" in r["text"])
            total += n_gold
            miss += max(0, n_gold - len(spans))
            tags.append(bio_tags(off, spans))
        y = np.array([CLASSES.index(r["slice"]) for r in rows])
        np.savez_compressed(out / f"{split}.npz", h=h.astype(np.float16), mask=m.astype(np.uint8),
                            tags=np.array(tags, dtype=np.int8), sent=s, y=y,
                            ids=np.array([r["id"] for r in rows]))
        print(f"{split}: {len(rows)} rows, slot spans aligned {total - miss}/{total}, tags {len(TAGS)}")


if __name__ == "__main__":
    main()
