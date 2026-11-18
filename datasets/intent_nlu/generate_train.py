"""Template generator for the TRAIN split of the intent-NLU dataset (seeded, reproducible).

The train split is the kNN memory of the semantic classifier. CALIB and TEST are hand-written (calib.jsonl,
test.jsonl) and never produced from these templates, so evaluation is not inflated by template leakage.

    python datasets/intent_nlu/generate_train.py      -> datasets/intent_nlu/train.jsonl
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20261118
OUT = Path(__file__).with_name("train.jsonl")

USE_CASES = {
    "URLLC": {
        "en": ["remote surgery", "telesurgery robot", "robot arm motion control", "factory motion control",
               "autonomous vehicle V2X coordination", "drone flight control", "smart grid protection relays",
               "haptic teleoperation", "mission-critical push-to-talk for first responders", "AGV fleet control",
               "real-time industrial closed-loop control", "remote crane operation", "cooperative platooning"],
        "ko": ["원격수술", "수술 로봇", "로봇 팔 모션 제어", "공장 자동화 실시간 제어", "자율주행 차량 V2X 협력",
               "드론 비행 제어", "스마트그리드 보호 계전", "햅틱 원격 조작", "재난 대응 무전(MCPTT)",
               "AGV 무인운반차 제어", "원격 크레인 조종", "산업용 폐루프 제어"],
    },
    "eMBB": {
        "en": ["4K live streaming", "8K video broadcast", "fan Wi-Fi-like coverage at the concert", "cloud gaming",
               "VR experience zone", "AR glasses for visitors", "HD video conferencing", "360-degree video replay",
               "large file downloads", "mobile broadcasting vans", "high-quality video on demand", "spectator uploads"],
        "ko": ["4K 생중계", "8K 영상 방송", "공연장 관람객 대용량 데이터", "클라우드 게임", "VR 체험존",
               "관람객용 AR 글래스", "고화질 화상회의", "360도 영상 리플레이", "대용량 파일 다운로드",
               "중계차 방송 송출", "고화질 VOD 스트리밍", "관객 영상 업로드"],
    },
    "mMTC": {
        "en": ["smart meters", "environmental sensors", "asset tracking tags", "smart parking sensors",
               "soil moisture sensors", "water leak detectors", "smart street lights", "container tracking",
               "air quality monitors", "wearable telemetry", "pallet tags", "vibration sensors on pipelines"],
        "ko": ["스마트 미터 검침", "환경 센서", "자산 추적 태그", "스마트 주차 센서", "토양 수분 센서",
               "누수 감지 센서", "스마트 가로등", "컨테이너 위치 추적", "대기질 측정기", "웨어러블 원격 측정",
               "팔레트 태그", "배관 진동 센서"],
    },
}
AREAS = {"en": ["Hospital A", "the factory", "the port", "downtown", "the campus", "the stadium", "the city center"],
         "ko": ["병원", "공장", "항만", "도심", "캠퍼스", "경기장", "산업단지"]}
TEMPLATES = {
    "en": ["Create a slice for {uc} at {area}{k}.", "We need {uc} in {area}{k}.",
           "Provide connectivity for {uc}{k}.", "{uc} service at {area}{k}.", "Set up {uc} for {area}{k}.",
           "Please allocate network resources to {uc}{k}."],
    "ko": ["{area}에 {uc}용 슬라이스를 만들어 줘{k}.", "{area}에서 {uc} 서비스가 필요합니다{k}.",
           "{uc}를 위한 연결을 제공해 주세요{k}.", "{area} {uc} 서비스{k}.", "{uc} 용도로 네트워크 자원을 할당해 줘{k}."],
}
KPI_RANGES = {"URLLC": {"lat": [1, 2, 5, 10], "bw": [10, 20, 50, 100], "rel": [99.99, 99.999, 99.9999]},
              "eMBB": {"lat": [20, 30, 50, 100], "bw": [100, 200, 500, 1000], "rel": [99.0, 99.9]},
              "mMTC": {"lat": [100, 200, 500, 1000], "bw": [1, 2, 5], "rel": [99.0, 99.9], "dev": [500, 2000, 10000]}}


def kpi_phrase(rng: random.Random, st: str, lang: str) -> tuple[str, dict, int | None]:
    r = KPI_RANGES[st]
    parts, exp, dev = [], {}, None
    if rng.random() < 0.6:
        v = rng.choice(r["lat"]); exp["latency_ms"] = v
        parts.append(f"latency under {v} ms" if lang == "en" else f"지연 {v}ms 이하")
    if rng.random() < 0.5:
        v = rng.choice(r["bw"]); exp["bandwidth_mbps"] = v
        parts.append(f"{v} Mbps" if lang == "en" else f"{v}Mbps")
    if rng.random() < 0.4:
        v = rng.choice(r["rel"]); exp["reliability_percent"] = v
        parts.append(f"{v}% reliability" if lang == "en" else f"신뢰도 {v}%")
    if "dev" in r and rng.random() < 0.6:
        dev = rng.choice(r["dev"])
        parts.append(f"{dev} devices" if lang == "en" else f"단말 {dev}대")
    if not parts:
        return "", exp, dev
    return (", " if lang == "en" else ", ") + ", ".join(parts), exp, dev


def main() -> None:
    rng = random.Random(SEED)
    rows, n = [], 0
    for st, by_lang in USE_CASES.items():
        for lang, ucs in by_lang.items():
            for uc in ucs:
                for _ in range(4):
                    t = rng.choice(TEMPLATES[lang])
                    k, exp, dev = kpi_phrase(rng, st, lang)
                    text = t.format(uc=uc, area=rng.choice(AREAS[lang]), k=k)
                    text = text[0].upper() + text[1:]
                    n += 1
                    rows.append({"id": f"tr-{n:04d}", "text": text, "lang": lang, "slice": st,
                                 "expectations": exp, "devices": dev, "source": "template"})
    seen, uniq = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"]); uniq.append(r)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in uniq) + "\n", encoding="utf-8")
    print(f"wrote {len(uniq)} rows -> {OUT}")


if __name__ == "__main__":
    main()
