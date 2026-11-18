"""Unit tests for the intent-NLU pieces that do not need the ONNX models."""
from __future__ import annotations

import numpy as np

from common import config
from service_layer.lcm.intent_nlu import mcnemar_exact
from service_layer.orchestration.constraints import ConstraintValidator
from service_layer.orchestration.semantic import conformal_qhat, conformal_sets
from service_layer.orchestration.slots import TAGS, bio_tags, decode, gold_spans


def _char_tokens(text):
    """one token per character: enough to exercise span logic without a tokenizer"""
    return [(i, i + 1) for i in range(len(text))]


def test_gold_spans_prefer_unit_and_skip_distractors():
    row = {"text": "for 5 hours, latency 5 ms, 1.5 Gbps, 2,000 sensors", "expectations": {"latency_ms": 5, "bandwidth_mbps": 1500},
           "devices": 2000}
    spans = {s: row["text"][a:b] for a, b, s in gold_spans(row)}
    assert spans == {"LAT": "5", "BW": "1.5", "DEV": "2,000"}
    assert row["text"].index("5 ms") == next(a for a, b, s in gold_spans(row) if s == "LAT")


def test_decode_normalises_units_and_korean_multiplier():
    text = "지연 2초, 3Gbps, 2만 대"
    off = _char_tokens(text)
    tags = [0] * len(text)
    tags[text.index("2초")] = TAGS.index("B-LAT")
    tags[text.index("3G")] = TAGS.index("B-BW")
    tags[text.index("2만")] = TAGS.index("B-DEV")
    assert decode(text, off, tags) == {"latency_ms": 2000.0, "bandwidth_mbps": 3000.0, "devices": 20000}


def test_decode_split_number_uses_unit_to_pick_slot():
    text = "reliability 150%"
    off = _char_tokens(text)
    tags = [0] * len(text)
    i = text.index("150")
    tags[i], tags[i + 1], tags[i + 2] = TAGS.index("B-BW"), TAGS.index("B-REL"), TAGS.index("I-REL")
    assert decode(text, off, tags) == {"reliability_percent": 150.0}


def test_bio_roundtrip():
    text = "under 5 ms"
    off = _char_tokens(text)
    assert decode(text, off, bio_tags(off, [(6, 7, "LAT")])) == {"latency_ms": 5.0}


def test_constraints():
    v = ConstraintValidator(config.INTENT_CONSTRAINTS)
    assert v.check("mMTC", {"latency_ms": 2})["blocking"]
    assert v.check("URLLC", {"latency_ms": 5, "reliability_percent": 99.999})["ok"]
    assert v.check("eMBB", {"reliability_percent": 101})["violations"][0]["severity"] == "error"
    w = v.check("eMBB", {"reliability_percent": 99.999})
    assert w["ok"] and w["violations"][0]["severity"] == "warning"


def test_conformal_and_mcnemar():
    p = np.array([[0.9, 0.05, 0.05], [0.5, 0.45, 0.05]])
    q = conformal_qhat(p, np.array([0, 1]), 0.5)
    assert conformal_sets(p, q)[1] == ["URLLC", "eMBB"]
    assert mcnemar_exact([True] * 10, [False] * 10)["p"] < 0.01
    assert mcnemar_exact([True, False], [False, True])["p"] == 1.0
