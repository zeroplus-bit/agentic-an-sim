"""TensorFlow/Keras joint intent classification + KPI slot tagging (JointBERT-style, Chen et al. 2019) on frozen
multilingual SBERT token states. Runs in the Python 3.12 training venv (TensorFlow has no stable 3.14 wheel):

    .venv-train/Scripts/python tools/train_joint_nlu.py
    -> models/joint_nlu/joint_nlu.keras (Keras model) + joint_nlu.npz (weights for the NumPy runtime) + train_log.json

Architecture (same in serving, see service_layer/orchestration/joint_nlu.py):
    x = h * mask                                   # (L, 384) frozen encoder states
    c = Dropout(ReLU(Conv1D_128,k=3,same)(x))      # local n-gram context around each token
    slot_logits = Dense_9([x ; c])                 # BIO tags: O, B/I-{LAT,BW,REL,DEV}
    p = masked_mean(c)
    intent_logits = Dense_3(ReLU(Dense_64(Dropout([p ; sentence_emb]))))
Loss = CE(intent) + CE(slot tokens, masked; non-O tokens weighted x3).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import keras  # noqa: E402
from keras import layers, ops  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "datasets/intent_nlu/features"
OUT = ROOT / "models/joint_nlu"
SEED, EPOCHS, BATCH, LR, PATIENCE = 13, 60, 64, 1e-3, 8
N_TAGS, N_CLS = 9, 3


def load(split):
    z = np.load(F / f"{split}.npz")
    return z["h"].astype(np.float32), z["mask"].astype(np.float32), z["tags"].astype(np.int32), z["sent"], z["y"]


def build(L: int, d: int, ds: int) -> keras.Model:
    h = keras.Input((L, d), name="h")
    m = keras.Input((L,), name="mask")
    s = keras.Input((ds,), name="sent")
    x = h * ops.expand_dims(m, -1)
    c = layers.Conv1D(128, 3, padding="same", activation="relu", name="conv")(x)
    c = layers.Dropout(0.2)(c)
    slot = layers.Dense(N_TAGS, name="slot")(layers.Concatenate()([x, c]))
    p = ops.sum(c * ops.expand_dims(m, -1), 1) / ops.maximum(ops.sum(m, 1, keepdims=True), 1.0)
    z = layers.Dropout(0.2)(layers.Concatenate()([p, s]))
    z = layers.Dense(64, activation="relu", name="int1")(z)
    intent = layers.Dense(N_CLS, name="int2")(z)
    return keras.Model([h, m, s], [intent, slot])


def main() -> None:
    keras.utils.set_random_seed(SEED)
    h, m, t, s, y = load("train")
    idx = np.random.default_rng(SEED).permutation(len(y))
    nv = len(y) // 10
    va, tr = idx[:nv], idx[nv:]
    model = build(h.shape[1], h.shape[2], s.shape[1])
    opt = keras.optimizers.Adam(LR)
    ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    def loss_fn(b, training):
        il, sl = model([h[b], m[b], s[b]], training=training)
        li = tf.reduce_mean(ce(y[b], il))
        w = m[b] * np.where(t[b] > 0, 3.0, 1.0).astype(np.float32)
        ls = tf.reduce_sum(ce(t[b], sl) * w) / tf.reduce_sum(w)
        return li + ls, li, ls

    best, wait, log = 1e9, 0, []
    best_w = None
    rng = np.random.default_rng(SEED)
    for ep in range(EPOCHS):
        perm = rng.permutation(tr)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            with tf.GradientTape() as tape:
                loss, _, _ = loss_fn(b, True)
            g = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(g, model.trainable_variables))
        vl, vi, vs = (float(v) for v in loss_fn(va, False))
        il, sl = model([h[va], m[va], s[va]], training=False)
        acc = float(np.mean(np.argmax(il, 1) == y[va]))
        tok = m[va] > 0
        tacc = float(np.mean((np.argmax(sl, -1) == t[va])[tok & (t[va] > 0)]))
        log.append({"epoch": ep + 1, "valLoss": round(vl, 4), "valIntentAcc": round(acc, 4), "valSlotTokAcc": round(tacc, 4)})
        print(log[-1])
        if vl < best - 1e-4:
            best, wait, best_w = vl, 0, model.get_weights()
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.set_weights(best_w)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save(OUT / "joint_nlu.keras")
    W = {f"{l.name}/{i}": w for l in model.layers if l.weights for i, w in enumerate(l.get_weights())}
    np.savez(OUT / "joint_nlu.npz", **W)
    (OUT / "train_log.json").write_text(json.dumps({"tensorflow": tf.__version__, "keras": keras.__version__,
                                                     "seed": SEED, "trainRows": int(len(tr)), "valRows": int(nv),
                                                     "bestValLoss": round(best, 4), "epochs": log}, indent=1))
    print("saved", sorted(W))


if __name__ == "__main__":
    main()
