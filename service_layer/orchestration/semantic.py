"""Semantic intent classification: sentence embeddings + kNN, temperature scaling and split conformal prediction.

Methods (all published and widely used):
  * Sentence-BERT embeddings (Reimers & Gurevych, EMNLP 2019; multilingual distillation: Reimers & Gurevych 2020),
    model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, run as ONNX with mean pooling.
  * k-nearest-neighbour classification over a labelled memory (train split + standard reference use cases).
  * Temperature scaling for probability calibration (Guo et al., ICML 2017), fitted on a held-out calibration split.
  * Split conformal prediction with the LAC score s = 1 - p(y|x) (Vovk et al. 2005; Sadinle et al. 2019;
    Angelopoulos & Bates 2021). The prediction set contains the true class with probability >= 1 - alpha
    (marginal guarantee, exchangeability assumed). A set with more than one class -> ask for clarification.

Fallback encoder when the ONNX model is absent: hashed character n-grams (lexical, no semantics) so the pipeline
still runs; the report says which encoder was used.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

log = logging.getLogger("semantic")
CLASSES = ["URLLC", "eMBB", "mMTC"]


# ---- encoders ------------------------------------------------------------------------------------------
class SbertOnnxEncoder:
    def __init__(self, model_dir: Path, max_len: int = 128):
        import onnxruntime as ort  # type: ignore
        from tokenizers import Tokenizer  # type: ignore
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_dir / "onnx" / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.id = f"sbert-onnx:{model_dir.name}"

    def encode(self, texts: list[str], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[i:i + batch])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]                       # (b, t, d)
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)     # mean pooling
            out.append(v)
        x = np.concatenate(out).astype(np.float32)
        return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)

    def encode_tokens(self, texts: list[str], max_len: int = 64, batch: int = 32):
        """Token states for the joint slot model: (h[n,L,d], mask[n,L], offsets list, sentence emb[n,d])."""
        H, M, OFF = [], [], []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[i:i + batch])
            L = max(len(e.ids) for e in enc)
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]
            pad = np.zeros((len(enc), max_len, h.shape[2]), dtype=np.float32)
            pm = np.zeros((len(enc), max_len), dtype=np.float32)
            n = min(L, max_len)
            pad[:, :n] = h[:, :n]
            pm[:, :n] = mask[:, :n]
            H.append(pad); M.append(pm)
            OFF += [list(e.offsets[:max_len]) + [(0, 0)] * (max_len - min(len(e.offsets), max_len)) for e in enc]
        h, m = np.concatenate(H), np.concatenate(M)
        s = (h * m[..., None]).sum(1) / np.clip(m.sum(1, keepdims=True), 1e-9, None)
        s = s / np.clip(np.linalg.norm(s, axis=1, keepdims=True), 1e-9, None)
        return h, m, OFF, s.astype(np.float32)


class CharNgramEncoder:
    """Hashed char 2-4 gram TF vectors (lexical baseline / fallback)."""
    def __init__(self, dim: int = 2 ** 15):
        self.dim = dim
        self.id = "char-ngram-hash"

    def encode(self, texts: list[str], batch: int = 0) -> np.ndarray:
        x = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            s = f" {re.sub(r'\s+', ' ', t.lower())} "
            for n in (2, 3, 4):
                for i in range(len(s) - n + 1):
                    h = int.from_bytes(hashlib.md5(s[i:i + n].encode()).digest()[:4], "little") % self.dim
                    x[r, h] += 1.0
        return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


def make_encoder(model_dir: Optional[Path], kind: str = "auto"):
    if kind in ("auto", "sbert") and model_dir and (model_dir / "onnx" / "model.onnx").exists():
        try:
            return SbertOnnxEncoder(model_dir)
        except Exception as e:  # missing onnxruntime/tokenizers
            if kind == "sbert":
                raise
            log.warning("SBERT encoder unavailable (%s); using char n-grams", e)
    return CharNgramEncoder()


# ---- data ----------------------------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_references(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["references"]


# ---- classifier ----------------------------------------------------------------------------------------
class SemanticIntentClassifier:
    def __init__(self, encoder, k: int = 5):
        self.enc = encoder
        self.k = k
        self.T = 1.0
        self.qhat: Optional[float] = None
        self.alpha: Optional[float] = None
        self.calib_n = 0
        self.mem = np.zeros((0, 1), dtype=np.float32)
        self.mem_y: list[str] = []
        self.mem_meta: list[dict[str, Any]] = []

    # memory = train utterances + reference use-case descriptions
    def fit(self, rows: Iterable[dict[str, Any]], references: Iterable[dict[str, Any]] = ()) -> "SemanticIntentClassifier":
        texts, ys, meta = [], [], []
        for r in rows:
            texts.append(r["text"]); ys.append(r["slice"]); meta.append({"kind": "example", "id": r["id"], "text": r["text"]})
        for ref in references:
            for t in ref["text"]:
                texts.append(t); ys.append(ref["slice"])
                meta.append({"kind": "reference", "id": ref["id"], "text": t, "source": ref["source"],
                             "verified": ref.get("verified", False)})
        self.mem = self.enc.encode(texts)
        self.mem_y, self.mem_meta = ys, meta
        self._y_idx = np.array([CLASSES.index(y) for y in ys])
        return self

    def scores(self, emb: np.ndarray) -> np.ndarray:
        """Per-class score = mean cosine similarity of the k nearest memory items of that class. (n, C)"""
        sims = emb @ self.mem.T
        out = np.zeros((len(emb), len(CLASSES)), dtype=np.float64)
        for c in range(len(CLASSES)):
            s = np.sort(sims[:, self._y_idx == c], axis=1)[:, ::-1][:, :self.k]
            out[:, c] = s.mean(1)
        return out

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def proba_from_scores(self, sc: np.ndarray) -> np.ndarray:
        return self._softmax(sc / self.T)

    # ---- calibration -------------------------------------------------------------------------------------
    def calibrate(self, rows: list[dict[str, Any]], alpha: float = 0.1) -> dict[str, Any]:
        sc = self.scores(self.enc.encode([r["text"] for r in rows]))
        y = np.array([CLASSES.index(r["slice"]) for r in rows])
        grid = np.exp(np.linspace(math.log(0.005), math.log(2.0), 300))
        nll = [float(-np.log(self._softmax(sc / t)[np.arange(len(y)), y] + 1e-12).mean()) for t in grid]
        self.T = float(grid[int(np.argmin(nll))])
        p = self.proba_from_scores(sc)
        s = 1.0 - p[np.arange(len(y)), y]                           # LAC non-conformity score
        n = len(s)
        level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
        self.qhat = float(np.quantile(s, level, method="higher"))
        self.alpha, self.calib_n = alpha, n
        return self.state()

    def state(self) -> dict[str, Any]:
        return {"encoder": self.enc.id, "k": self.k, "temperature": self.T, "qhat": self.qhat, "alpha": self.alpha,
                "calibN": self.calib_n, "memory": len(self.mem_y)}

    def load_state(self, st: dict[str, Any]) -> bool:
        if st.get("encoder") != self.enc.id or st.get("k") != self.k:
            log.warning("calibration was made for %s/k=%s, encoder is %s/k=%s -> ignored",
                        st.get("encoder"), st.get("k"), self.enc.id, self.k)
            return False
        self.T, self.qhat, self.alpha, self.calib_n = st["temperature"], st["qhat"], st["alpha"], st["calibN"]
        return True

    # ---- inference -----------------------------------------------------------------------------------
    def prediction_sets(self, p: np.ndarray) -> list[list[str]]:
        if self.qhat is None:
            return [[CLASSES[int(i)]] for i in p.argmax(1)]
        sets = []
        for row in p:
            s = [CLASSES[c] for c in np.argsort(-row) if 1.0 - row[c] <= self.qhat + 1e-12]
            sets.append(s or [CLASSES[int(row.argmax())]])
        return sets

    def evidence(self, emb: np.ndarray) -> list[dict[str, Any]]:
        """Nearest standard reference use case and nearest labelled example for each embedding."""
        sims = emb @ self.mem.T
        ex = [j for j, m in enumerate(self.mem_meta) if m["kind"] == "example"]
        rf = [j for j, m in enumerate(self.mem_meta) if m["kind"] == "reference"]
        out = []
        for i in range(len(emb)):
            ev: dict[str, Any] = {}
            if rf:
                j = max(rf, key=lambda j: sims[i, j]); m = self.mem_meta[j]
                ev["reference"] = {"id": m["id"], "slice": self.mem_y[j], "source": m["source"],
                                   "verified": m["verified"], "similarity": round(float(sims[i, j]), 3)}
            if ex:
                j = max(ex, key=lambda j: sims[i, j])
                ev["nearestExample"] = {"id": self.mem_meta[j]["id"], "text": self.mem_meta[j]["text"],
                                        "slice": self.mem_y[j], "similarity": round(float(sims[i, j]), 3)}
            out.append(ev)
        return out

    def predict_batch(self, texts: list[str], emb: Optional[np.ndarray] = None) -> list[dict[str, Any]]:
        emb = self.enc.encode(texts) if emb is None else emb
        p = self.proba_from_scores(self.scores(emb))
        return package(p, self.prediction_sets(p), self.evidence(emb))

    def predict(self, text: str) -> dict[str, Any]:
        return self.predict_batch([text])[0]


# ---- shared calibration helpers (also used by the joint TF model / ensemble) -----------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    grid = np.exp(np.linspace(math.log(0.005), math.log(20.0), 400))
    nll = [nll_of(softmax(logits / t), y) for t in grid]
    return float(grid[int(np.argmin(nll))])


def nll_of(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def conformal_qhat(p: np.ndarray, y: np.ndarray, alpha: float) -> float:
    s = 1.0 - p[np.arange(len(y)), y]
    n = len(s)
    return float(np.quantile(s, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def conformal_sets(p: np.ndarray, qhat: Optional[float]) -> list[list[str]]:
    out = []
    for row in p:
        s = [CLASSES[c] for c in np.argsort(-row) if qhat is not None and 1.0 - row[c] <= qhat + 1e-12]
        out.append(s or [CLASSES[int(row.argmax())]])
    return out


def package(p: np.ndarray, sets: list[list[str]], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i in range(len(p)):
        c = int(p[i].argmax())
        out.append({"label": CLASSES[c], "confidence": round(float(p[i, c]), 4),
                    "probs": {k: round(float(v), 4) for k, v in zip(CLASSES, p[i])},
                    "predictionSet": sets[i], "evidence": evidence[i]})
    return out
