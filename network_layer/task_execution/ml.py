"""Model runtimes used by Data/Model Execution (no TensorFlow needed).

* KerasLstmNumpy - pure-NumPy forward pass of the repo's models/lstm_single/*.h5
  (LSTM64 -> LSTM32 -> Dense3 softmax, input (10, 11)), with X_scaler.npy (MinMax).
  If TensorFlow is installed it is NOT required; weights are read with h5py.
* NumpyDQN - port of dqn_agent.py (state 14, 27 actions, ReplayBuffer, calculate_reward,
  action_to_allocation) as a small NumPy MLP trained online from the replay buffer.
  Until enough experience exists the policy is bootstrapped by a one-step reward model
  that uses the repo's reward function, so early decisions are sensible.
"""
from __future__ import annotations

import io
import random
from collections import deque
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

FEATURES = ["traffic_load", "time_of_day", "day_of_week", "embb_alloc", "urllc_alloc", "mmtc_alloc",
            "embb_util", "urllc_util", "mmtc_util", "client_count", "bs_count"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class KerasLstmNumpy:
    def __init__(self, source: Union[str, Path, bytes], scaler_path: Optional[Path] = None, name: str = "lstm_single",
                 version: str = "1.0"):
        import h5py
        self.name, self.version = name, version
        f = h5py.File(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else str(source), "r")
        w = f["model_weights"]
        self.layers = []
        for lname in ("lstm", "lstm_1"):
            g = w[lname][lname]["lstm_cell"]
            self.layers.append((np.array(g["kernel:0"]), np.array(g["recurrent_kernel:0"]), np.array(g["bias:0"])))
        self.dense_k = np.array(w["dense"]["dense"]["kernel:0"])
        self.dense_b = np.array(w["dense"]["dense"]["bias:0"])
        f.close()
        self.scale = self.min_ = None
        if scaler_path and Path(scaler_path).exists():
            s = np.load(scaler_path, allow_pickle=True)
            # rows: data_min_, data_range_, scale_, min_  (sklearn MinMaxScaler)
            self.scale, self.min_ = s[2].astype(float), s[3].astype(float)

    @staticmethod
    def _lstm(x: np.ndarray, k: np.ndarray, rk: np.ndarray, b: np.ndarray, return_sequences: bool) -> np.ndarray:
        units = rk.shape[0]
        h = np.zeros(units)
        c = np.zeros(units)
        outs = []
        for t in range(x.shape[0]):
            z = x[t] @ k + h @ rk + b
            i, f, g, o = z[:units], z[units:2 * units], z[2 * units:3 * units], z[3 * units:]
            i, f, o = _sigmoid(i), _sigmoid(f), _sigmoid(o)
            c = f * c + i * np.tanh(g)
            h = o * np.tanh(c)
            outs.append(h)
        return np.array(outs) if return_sequences else h

    def predict(self, window: np.ndarray) -> np.ndarray:
        """window: (10, 11) raw features -> allocation [eMBB, URLLC, mMTC] (sums to 1)."""
        x = np.asarray(window, dtype=float)
        if self.scale is not None:
            x = x * self.scale + self.min_
        h = self._lstm(x, *self.layers[0], return_sequences=True)
        h = self._lstm(h, *self.layers[1], return_sequences=False)
        z = h @ self.dense_k + self.dense_b
        e = np.exp(z - z.max())
        return e / e.sum()


class RuleAllocator:
    """Fallback when no LSTM is available (ml_orchestrator_demo.update_allocation_rule_based)."""
    name, version = "rule_based", "1.0"

    def predict(self, window: np.ndarray) -> np.ndarray:
        last = np.asarray(window)[-1]
        alloc = last[3:6]
        util = last[6:9]
        demand = alloc * util
        target = np.clip(demand / max(demand.sum(), 1e-6), 0.1, 0.8)
        return target / target.sum()


# ---------------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, size: int = 10000):
        self.buffer: deque = deque(maxlen=size)

    def store(self, s, a, r, s2, done) -> None:
        self.buffer.append((np.asarray(s, float), int(a), float(r), np.asarray(s2, float), bool(done)))

    def sample(self, n: int):
        batch = random.sample(self.buffer, min(len(self.buffer), n))
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self) -> int:
        return len(self.buffer)


def action_to_allocation(action: int) -> np.ndarray:
    """dqn_agent.py::action_to_allocation - 27 actions = {0.2,0.4,0.6}^3, normalized."""
    vals = [0.2, 0.4, 0.6]
    a = np.array([vals[action // 9], vals[(action % 9) // 3], vals[action % 3]])
    return a / a.sum()


def calculate_reward(state: dict, next_state: dict) -> float:
    """dqn_agent.py::calculate_reward (utilization in 0.7-0.9 is rewarded)."""
    r = 0.0
    for util in next_state["utilization"]:
        if 0.7 <= util <= 0.9:
            r += 2.0
        elif util > 1.0:
            r -= 5.0 * (util - 1.0)
        elif util < 0.5:
            r -= 1.0 * (0.5 - util)
    r -= 0.5 * float(np.sum(np.abs(np.asarray(next_state["allocation"]) - np.asarray(state["allocation"]))))
    if state.get("is_emergency"):
        r += 3.0 * next_state["allocation"][1]
    if state.get("is_special_event"):
        r += 3.0 * next_state["allocation"][0]
    if state.get("is_iot_surge"):
        r += 3.0 * next_state["allocation"][2]
    r -= 2.0 * len(next_state.get("violations", []))
    return r


def make_state(features: dict[str, float], scenario: dict[str, bool], violations: int = 0) -> np.ndarray:
    """14-d DQN state: 11 LSTM features + 3 scenario flags (violations folded into traffic load)."""
    v = [features.get(k, 0.0) for k in FEATURES]
    v[0] += 0.05 * violations
    return np.array(v + [float(scenario.get("is_emergency", False)), float(scenario.get("is_special_event", False)),
                         float(scenario.get("is_iot_surge", False))])


class NumpyDQN:
    name, version = "dqn_slice_alloc", "1.0"

    def __init__(self, state_dim: int = 14, action_dim: int = 27, hidden: tuple[int, int] = (64, 32), seed: int = 7):
        rng = np.random.default_rng(seed)
        dims = [state_dim, *hidden, action_dim]
        self.W = [rng.normal(0, np.sqrt(2 / dims[i]), (dims[i], dims[i + 1])) for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        self.target = [w.copy() for w in self.W], [b.copy() for b in self.b]
        self.gamma, self.lr = 0.95, 1e-3
        self.epsilon, self.epsilon_min, self.epsilon_decay = 0.2, 0.01, 0.995
        self.buffer = ReplayBuffer()
        self.train_steps = 0
        self.warmup = 64
        self.action_dim = action_dim

    def _forward(self, x: np.ndarray, params=None):
        W, b = params or (self.W, self.b)
        acts = [x]
        for i in range(len(W)):
            z = acts[-1] @ W[i] + b[i]
            acts.append(np.maximum(z, 0) if i < len(W) - 1 else z)
        return acts

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self._forward(np.atleast_2d(state))[-1][0]

    def one_step_scores(self, state: np.ndarray) -> np.ndarray:
        """Reward model used to bootstrap: utilization traffic/alloc under each candidate action."""
        alloc = state[3:6]
        traffic = alloc * state[6:9]
        flags = {"is_emergency": state[11] > 0.5, "is_special_event": state[12] > 0.5, "is_iot_surge": state[13] > 0.5}
        scores = np.zeros(self.action_dim)
        for a in range(self.action_dim):
            na = action_to_allocation(a)
            scores[a] = calculate_reward({"allocation": alloc, **flags},
                                         {"allocation": na, "utilization": traffic / np.maximum(na, 1e-6)})
        return scores

    def act(self, state: np.ndarray, explore: bool = False) -> tuple[int, str]:
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim), "explore"
        if self.train_steps < self.warmup:
            return int(np.argmax(self.one_step_scores(state))), "reward-model"
        q = self.q_values(state) + 0.1 * self.one_step_scores(state)
        return int(np.argmax(q)), "q-network"

    def remember(self, s, a, r, s2, done=False) -> None:
        self.buffer.store(s, a, r, s2, done)

    def train(self, batch_size: int = 32, iters: int = 1) -> Optional[float]:
        if len(self.buffer) < 8:
            return None
        loss = 0.0
        for _ in range(iters):
            s, a, r, s2, d = self.buffer.sample(batch_size)
            q_next = self._forward(s2, self.target)[-1].max(axis=1)
            y = r + self.gamma * q_next * (1 - d)
            acts = self._forward(s)
            q = acts[-1]
            grad = np.zeros_like(q)
            idx = np.arange(len(a))
            err = q[idx, a] - y
            grad[idx, a] = err / len(a)
            loss += float(np.mean(err ** 2))
            for i in reversed(range(len(self.W))):
                gw = acts[i].T @ grad
                gb = grad.sum(axis=0)
                if i > 0:
                    grad = (grad @ self.W[i].T) * (acts[i] > 0)
                self.W[i] -= self.lr * np.clip(gw, -1, 1)
                self.b[i] -= self.lr * np.clip(gb, -1, 1)
            self.train_steps += 1
            if self.train_steps % 10 == 0:
                self.target = [w.copy() for w in self.W], [b.copy() for b in self.b]
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss / iters

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "trainSteps": self.train_steps,
                "replaySize": len(self.buffer), "epsilon": round(self.epsilon, 4),
                "mode": "reward-model" if self.train_steps < self.warmup else "q-network"}
