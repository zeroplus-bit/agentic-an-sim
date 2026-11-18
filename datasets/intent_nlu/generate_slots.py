"""Template generator for the joint intent+slot TRAINING data (train_slots.jsonl, seeded).

Same use cases as generate_train.py, but with varied KPI phrasing, unit variants (ms / milliseconds / seconds,
Mbps / Gbps / Gbit/s / 메가), a generic list of device nouns and distractor numbers (hours, km, GB, line numbers,
minutes) that must be tagged O. Spans are derived from the gold values by service_layer/orchestration/slots.py.

    python datasets/intent_nlu/generate_slots.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_train import AREAS, KPI_RANGES, USE_CASES  # noqa: E402

SEED = 7
OUT = Path(__file__).with_name("train_slots.jsonl")
N_PER = 260  # rows per (slice, lang)

LAT = {"en": ["under {v} ms", "below {v} milliseconds", "at most {v} ms", "max {v} ms", "{v} ms or less",
              "round trip of {v} ms", "latency {v} ms", "a {s} second delay is acceptable", "within {v} msec"],
       "ko": ["지연 {v}ms 이하", "{v}밀리초 이내", "최대 {v}ms", "지연 시간 {v} ms", "왕복 {v}ms", "{s}초 지연 허용",
              "{v} 밀리초 이하"]}
BW = {"en": ["{v} Mbps", "{v} Mbit/s", "{g} Gbps", "{g} Gbit/s", "uplink {v} Mbps", "throughput of {v} Mbps",
             "{v} Mbps per user"],
      "ko": ["{v}Mbps", "대역폭 {v}Mbps", "{g}Gbps", "업링크 {v}Mbps", "{v}메가", "전송률 {v} Mbps"]}
REL = {"en": ["{v}% reliability", "{v}% availability", "{v}% delivery", "{v} % success rate"],
       "ko": ["신뢰도 {v}%", "가용성 {v}%", "전달률 {v}%", "성공률 {v}%"]}
DEV = {"en": ["{n} devices", "{n} sensors", "{n} meters", "{n} tags", "{n} trackers", "{n} nodes", "{n} units",
              "{n} modules", "{n} terminals", "{n} cameras", "{n} users", "{n} subscribers", "{n} vehicles",
              "{n} gateways", "{n} beacons", "{n} wearables", "{n} handsets", "about {n} endpoints"],
       "ko": ["단말 {n}대", "센서 {n}개", "{n}대", "{n}개 장치", "사용자 {n}명", "가입자 {n}명", "노드 {n}개",
              "약 {n}대의 기기"]}
DISTRACT = {"en": ["for {h} hours", "on line {l}", "over {k} km", "with {gb} GB files", "every {m} minutes",
                   "in building {l}", "from {t}:00"],
            "ko": ["{h}시간 동안", "{l}라인에서", "{k}km 구간", "{gb}GB 파일", "{m}분마다", "{l}동 건물", "{t}시부터"]}
TEMPL = {"en": ["{uc} at {area}, {kpis}.", "We need {uc} in {area}: {kpis}.", "Provide {uc} {kpis}.",
                "Please set up {uc} for {area} ({kpis}).", "{area} wants {uc}; {kpis}."],
         "ko": ["{area} {uc}, {kpis}.", "{area}에서 {uc}가 필요합니다: {kpis}.", "{uc} 제공 ({kpis}).",
                "{area}에 {uc}를 구성해 주세요, {kpis}."]}


def fmt_int(rng, n: int) -> str:
    return f"{n:,}" if n >= 1000 and rng.random() < 0.5 else str(n)


def row(rng, st: str, lang: str, i: int) -> dict:
    r = KPI_RANGES[st]
    parts, exp, dev = [], {}, None
    if rng.random() < 0.65:
        if rng.random() < 0.12:
            s = rng.choice([1, 2, 3, 5]); exp["latency_ms"] = s * 1000.0
            parts.append(rng.choice([p for p in LAT[lang] if "{s}" in p]).format(s=s))
        else:
            v = rng.choice(r["lat"] + [3, 4, 8, 15, 25, 40, 60, 150]); exp["latency_ms"] = float(v)
            parts.append(rng.choice([p for p in LAT[lang] if "{v}" in p]).format(v=v))
    if rng.random() < 0.55:
        p = rng.choice(BW[lang])
        if "{g}" in p:
            g = rng.choice([1, 1.5, 2, 2.5, 5]); exp["bandwidth_mbps"] = g * 1000.0; parts.append(p.format(g=g))
        else:
            v = rng.choice(r["bw"] + [15, 30, 80, 150, 250, 400, 600]); exp["bandwidth_mbps"] = float(v); parts.append(p.format(v=v))
    if rng.random() < 0.4:
        v = rng.choice(r["rel"] + [99.5, 99.95]); exp["reliability_percent"] = v
        parts.append(rng.choice(REL[lang]).format(v=v))
    if rng.random() < (0.7 if st == "mMTC" else 0.25):
        dev = rng.choice([20, 150, 300, 800, 1200, 2500, 5000, 8000, 12000, 30000, 50000])
        parts.append(rng.choice(DEV[lang]).format(n=fmt_int(rng, dev)))
    if rng.random() < 0.45:
        d = rng.choice(DISTRACT[lang]).format(h=rng.choice([2, 3, 6, 12]), l=rng.choice([2, 3, 7]),
                                             k=rng.choice([40, 120, 300]), gb=rng.choice([20, 80, 120]),
                                             m=rng.choice([5, 10, 15, 30]), t=rng.choice([9, 18, 22]))
        parts.insert(rng.randrange(len(parts) + 1), d)
    rng.shuffle(parts)
    if not parts:
        parts = ["as soon as possible" if lang == "en" else "가능한 빨리"]
    text = rng.choice(TEMPL[lang]).format(uc=rng.choice(USE_CASES[st][lang]), area=rng.choice(AREAS[lang]),
                                          kpis=", ".join(parts))
    return {"id": f"ts-{i:05d}", "text": text[0].upper() + text[1:], "lang": lang, "slice": st,
            "expectations": exp, "devices": dev, "source": "template-slots"}


def main() -> None:
    rng = random.Random(SEED)
    rows, i = [], 0
    for st in USE_CASES:
        for lang in ("en", "ko"):
            for _ in range(N_PER):
                i += 1
                rows.append(row(rng, st, lang, i))
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
