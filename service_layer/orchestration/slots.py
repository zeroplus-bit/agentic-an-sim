"""KPI slot spans for the joint intent+slot model (BIO tagging of number tokens) and value normalisation.

A slot span is the number itself ("15,000", "1.5"); the tagger decides WHICH slot the number fills (latency,
bandwidth, reliability, device count or none) from context, and the unit right after the number converts it
(Gbps -> x1000 Mbps, seconds -> x1000 ms). Spans for hand-written data are derived from the gold values.
"""
from __future__ import annotations

import re
from typing import Any, Optional

SLOTS = ["LAT", "BW", "REL", "DEV"]
SLOT_KPI = {"LAT": "latency_ms", "BW": "bandwidth_mbps", "REL": "reliability_percent", "DEV": "devices"}
KPI_SLOT = {v: k for k, v in SLOT_KPI.items()}
TAGS = ["O"] + [f"{p}-{s}" for s in SLOTS for p in ("B", "I")]
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

_BW_UNIT = re.compile(r"\s*(gbps|gbit|gb/s|g\b|기가|mbps|mbit|mb/s|메가|kbps|kbit)", re.I)
_LAT_UNIT = re.compile(r"\s*(ms\b|msec|millisecond|밀리초|밀리|s\b|sec|second|초)", re.I)


def number_spans(text: str) -> list[tuple[int, int, float]]:
    out = []
    for m in NUM_RE.finditer(text):
        s = m.group(0).rstrip(",")
        try:
            out.append((m.start(), m.start() + len(s), float(s.replace(",", ""))))
        except ValueError:
            pass
    return out


def normalise(slot: str, value: float, after: str) -> float:
    if slot == "BW":
        m = _BW_UNIT.match(after)
        u = m.group(1).lower() if m else "mbps"
        return value * (1000.0 if u[0] in "g기" else 0.001 if u[0] == "k" else 1.0)
    if slot == "LAT":
        m = _LAT_UNIT.match(after)
        u = m.group(1).lower() if m else "ms"
        return value * (1.0 if (u.startswith("m") or u.startswith("밀")) else 1000.0)
    return value


def gold_spans(row: dict[str, Any]) -> list[tuple[int, int, str]]:
    """Char spans for a row whose gold is given as values: match each gold value to one number in the text."""
    text, spans, used = row["text"], [], set()
    gold = dict(row.get("expectations") or {})
    if row.get("devices") is not None:
        gold["devices"] = row["devices"]
    nums = number_spans(text)
    for kpi, val in gold.items():
        slot = KPI_SLOT[kpi]
        for strict in (True, False):          # prefer a number followed by the slot's own unit
            hit = next(((a, b) for (a, b, v) in nums if (a, b) not in used
                        and (not strict or has_unit(slot, text[b:]))
                        and abs(normalise(slot, v, text[b:]) - float(val)) < 1e-6), None)
            if hit:
                spans.append((*hit, slot)); used.add(hit)
                break
    return spans


_OTHER_UNIT = re.compile(r"\s*(%|ms\b|mbps|gbps|gbit|mbit|km|gb\b|hours?|minutes?|시간|분|초|밀리|메가|기가)", re.I)


def has_unit(slot: str, after: str) -> bool:
    if slot == "BW":
        return bool(_BW_UNIT.match(after))
    if slot == "LAT":
        return bool(_LAT_UNIT.match(after))
    if slot == "REL":
        return after.lstrip().startswith("%")
    return not _OTHER_UNIT.match(after)


def bio_tags(offsets: list[tuple[int, int]], spans: list[tuple[int, int, str]]) -> list[int]:
    tags = [0] * len(offsets)
    for (a, b, slot) in spans:
        first = True
        for i, (s, e) in enumerate(offsets):
            if e > s and s < b and e > a:
                tags[i] = TAGS.index(f"{'B' if first else 'I'}-{slot}")
                first = False
    return tags


def decode(text: str, offsets: list[tuple[int, int]], tag_ids: list[int]) -> dict[str, float]:
    """Token tags -> {kpi: normalised value}, number by number.

    Every number in the text collects the slot tags of the tokens overlapping it. If its tokens disagree (a
    number split into sub-word pieces), the slot whose unit follows the number wins, else the majority.
    The first number per slot wins. Korean 만/천/억 multipliers are applied.
    """
    out: dict[str, float] = {}
    for m in NUM_RE.finditer(text):
        sv = m.group(0).rstrip(",")
        st, en = m.start(), m.start() + len(sv)
        votes = [TAGS[t][2:] for (o, t) in zip(offsets, tag_ids) if o[1] > o[0] and o[0] < en and o[1] > st and t > 0]
        if not votes:
            continue
        try:
            v = float(sv.replace(",", ""))
        except ValueError:
            continue
        if en < len(text) and text[en] in _KO_MULT:
            v *= _KO_MULT[text[en]]
            en += 1
        after = text[en:]
        cands = sorted(set(votes), key=lambda x: (-int(has_unit(x, after)), -votes.count(x), votes.index(x)))
        slot = cands[0]
        if SLOT_KPI[slot] in out:
            continue
        val = normalise(slot, v, after)
        out[SLOT_KPI[slot]] = int(val) if slot == "DEV" else val
    return out


_KO_MULT = {"만": 10000.0, "천": 1000.0, "억": 1e8}


def five_nines(text: str) -> Optional[float]:
    low = text.lower()
    return 99.999 if ("five nines" in low or "5 nines" in low or "파이브 나인" in low) else None
