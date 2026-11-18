"""NumPy runtime of the TensorFlow joint intent+slot model (tools/train_joint_nlu.py), so the simulator's
Python 3.14 venv needs no TensorFlow. Weights: models/joint_nlu/joint_nlu.npz (exported from the .keras model).
Inputs are the frozen SBERT token states from SbertOnnxEncoder.encode_tokens."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .semantic import CLASSES, conformal_qhat, conformal_sets, fit_temperature, nll_of, package, softmax
from .slots import decode, five_nines

MAX_LEN = 64


class JointNluNumpy:
    def __init__(self, path: Path):
        z = np.load(path)
        self.W = {k: z[k].astype(np.float32) for k in z.files}
        self.id = f"joint-nlu:{path.stem}"

    def forward(self, h: np.ndarray, m: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """h (n,L,d), m (n,L), s (n,ds) -> intent logits (n,3), slot logits (n,L,9)."""
        W = self.W
        x = h * m[..., None]
        k, b = W["conv/0"], W["conv/1"]                  # (3, d, 128), same padding
        xp = np.pad(x, ((0, 0), (1, 1), (0, 0)))
        c = sum(xp[:, i:i + x.shape[1]] @ k[i] for i in range(3)) + b
        c = np.maximum(c, 0.0)
        slot = np.concatenate([x, c], -1) @ W["slot/0"] + W["slot/1"]
        p = (c * m[..., None]).sum(1) / np.maximum(m.sum(1, keepdims=True), 1.0)
        z = np.maximum(np.concatenate([p, s], -1) @ W["int1/0"] + W["int1/1"], 0.0)
        return z @ W["int2/0"] + W["int2/1"], slot


class JointIntentModel:
    """kNN (P1) + TF joint model (P2) behind one interface.

    Intent mode  knn | joint | ensemble (mean of the two calibrated distributions), chosen by calib NLL.
    Slot mode    regex | joint | joint+regex (tagger first, regex fills KPIs the tagger left empty),
                 chosen by calib exact match. Conformal q-hat is fitted for the chosen intent mode.
    """
    MODES = ("knn", "joint", "ensemble")
    SLOT_MODES = ("regex", "joint", "joint+regex")

    def __init__(self, knn, joint: JointNluNumpy, regex_fn: Callable[[str], dict[str, Any]]):
        self.knn, self.joint, self.regex_fn = knn, joint, regex_fn
        self.enc = knn.enc
        self.T_joint = 1.0
        self.mode, self.slot_mode = "ensemble", "joint+regex"
        self.qhat: Optional[float] = None
        self.alpha: Optional[float] = None
        self.selection: dict[str, Any] = {}
        self.id = f"{knn.enc.id}+{joint.id}"

    # ---- raw outputs ----------------------------------------------------------------------------------
    def raw(self, texts: list[str]) -> dict[str, Any]:
        h, m, off, s = self.enc.encode_tokens(texts, MAX_LEN)
        il, sl = self.joint.forward(h, m, s)
        return {"texts": texts, "emb": s, "il": il, "tags": sl.argmax(-1), "off": off,
                "pk": self.knn.proba_from_scores(self.knn.scores(s))}

    def probs(self, r: dict[str, Any], mode: Optional[str] = None) -> np.ndarray:
        mode = mode or self.mode
        pj = softmax(r["il"] / self.T_joint)
        return {"knn": r["pk"], "joint": pj, "ensemble": (r["pk"] + pj) / 2}[mode]

    def slots(self, r: dict[str, Any], mode: Optional[str] = None) -> list[dict[str, Any]]:
        mode = mode or self.slot_mode
        out = []
        for text, off, tg in zip(r["texts"], r["off"], r["tags"]):
            rx = self.regex_fn(text)
            if mode == "regex":
                out.append(rx); continue
            d = decode(text, off, [int(t) for t in tg])
            if "reliability_percent" not in d and five_nines(text):
                d["reliability_percent"] = five_nines(text)
            if mode == "joint+regex":
                d = {**rx, **d}
            out.append(d)
        return out

    # ---- calibration / selection (calib split only) ---------------------------------------------------
    def calibrate(self, rows: list[dict[str, Any]], alpha: float = 0.1) -> dict[str, Any]:
        r = self.raw([x["text"] for x in rows])
        y = np.array([CLASSES.index(x["slice"]) for x in rows])
        self.T_joint = fit_temperature(r["il"], y)
        nll = {m: round(nll_of(self.probs(r, m), y), 4) for m in self.MODES}
        self.mode = min(nll, key=nll.get)
        self.alpha = alpha
        self.qhat = conformal_qhat(self.probs(r), y, alpha)

        def gold(x):
            g = {k: float(v) for k, v in (x.get("expectations") or {}).items()}
            if x.get("devices") is not None:
                g["devices"] = float(x["devices"])
            return g
        em = {}
        for sm in self.SLOT_MODES:
            pred = self.slots(r, sm)
            em[sm] = round(float(np.mean([{k: float(v) for k, v in p.items()} == gold(x) for p, x in zip(pred, rows)])), 4)
        self.slot_mode = max(self.SLOT_MODES, key=lambda k: (em[k], k != "regex"))
        self.selection = {"calibNLL": nll, "calibSlotExactMatch": em}
        return self.state()

    def state(self) -> dict[str, Any]:
        return {"model": self.id, "mode": self.mode, "slotMode": self.slot_mode, "Tjoint": self.T_joint,
                "Tknn": self.knn.T, "qhat": self.qhat, "alpha": self.alpha, "selection": self.selection}

    def load_state(self, st: dict[str, Any]) -> bool:
        if st.get("model") != self.id:
            return False
        self.mode, self.slot_mode, self.T_joint = st["mode"], st["slotMode"], st["Tjoint"]
        self.knn.T, self.qhat, self.alpha, self.selection = st["Tknn"], st["qhat"], st["alpha"], st.get("selection", {})
        return True

    # ---- inference ------------------------------------------------------------------------------------
    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        r = self.raw(texts)
        p = self.probs(r)
        out = package(p, conformal_sets(p, self.qhat), self.knn.evidence(r["emb"]))
        for o, sl in zip(out, self.slots(r)):
            o["slots"] = sl
        return out

    def predict(self, text: str) -> dict[str, Any]:
        return self.predict_batch([text])[0]
