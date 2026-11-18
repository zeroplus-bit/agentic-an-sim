"""Independent second opinion on the slice type: NLI-based zero-shot classification (Yin, Hay & Roth, EMNLP 2019).

The utterance is the premise; each slice type is phrased as a hypothesis; the class whose hypothesis gets the
highest entailment logit wins (softmax over classes, as in the standard zero-shot pipeline). The NLI model
(`MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`, trained on MNLI + XNLI, ONNX) never sees our training data,
so disagreement with the primary classifier is evidence the utterance is ambiguous or out of distribution.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .semantic import CLASSES, softmax

# Hypothesis wording and template were selected on the calib split only (accuracy: L6 'concrete'+'about' 0.617,
# mDeBERTa-base best 0.617 at 6x the latency; the original abstract wording scored 0.50).
DESCRIPTIONS = {
    "URLLC": "machines, vehicles, robots or medical procedures controlled remotely that must react instantly",
    "eMBB": "people watching or sharing video and using fast internet",
    "mMTC": "many sensors, meters or trackers reporting small amounts of data",
}
TEMPLATES = {"needs": "This network request needs {}.", "about": "This is about {}."}


class NliZeroShot:
    def __init__(self, model_dir: Path, template: str = "about"):
        import onnxruntime as ort  # type: ignore
        from tokenizers import Tokenizer  # type: ignore
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(256)
        self.tok.enable_padding()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_dir / "onnx" / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        id2label = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["id2label"]
        self.ent = next(int(i) for i, l in id2label.items() if l.lower().startswith("entail"))
        self.template = template
        self.id = f"nli-zeroshot:{model_dir.name}"

    def entail_logits(self, texts: list[str], template: str | None = None) -> np.ndarray:
        """(n, C) entailment logits of each slice hypothesis."""
        hyps = [TEMPLATES[template or self.template].format(DESCRIPTIONS[c]) for c in CLASSES]
        pairs = [(t, h) for t in texts for h in hyps]
        out = []
        for i in range(0, len(pairs), 48):
            enc = self.tok.encode_batch(pairs[i:i + 48])
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            out.append(self.sess.run(None, feed)[0][:, self.ent])
        return np.concatenate(out).reshape(len(texts), len(CLASSES))

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        p = softmax(self.entail_logits(texts))
        return [{"label": CLASSES[int(r.argmax())], "probs": {c: round(float(v), 4) for c, v in zip(CLASSES, r)}}
                for r in p]

    def predict(self, text: str) -> dict[str, Any]:
        return self.predict_batch([text])[0]
