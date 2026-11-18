# P3 evaluation: clarification gate

Model joint-nlu/sbert-onnx:sbert-multilingual-minilm+joint-nlu:joint_nlu · NLI nli-zeroshot:nli-multilingual-minilm · constraints.json:12 rules

NLI gate adoption rule (fixed beforehand): calib extra-flag precision ≥ 0.5 → calib value 0.1053 → **not adopted**

## calib (n=60, classifier errors 6)

NLI zero-shot accuracy 0.6167, agreement with classifier 0.6 (classifier accuracy when agree 0.9444, when disagree 0.8333); NLI adds 19 flags, precision 0.1053; 19.1 ms/utterance. Rule false alarms on gold labels: none.

| policy | clarify rate | selective accuracy | errors caught | flag precision |
|---|---|---|---|---|
| conformal | 0.15 | 0.9216 | 2/6 | 0.2222 |
| conformal+NLI | 0.4667 | 0.9375 | 4/6 | 0.1429 |
| conformal+constraints | 0.15 | 0.9216 | 2/6 | 0.2222 |
| conformal+NLI+constraints | 0.4667 | 0.9375 | 4/6 | 0.1429 |

## test (n=90, classifier errors 12)

NLI zero-shot accuracy 0.5778, agreement with classifier 0.5333 (classifier accuracy when agree 0.9583, when disagree 0.7619); NLI adds 36 flags, precision 0.1111; 16.1 ms/utterance. Rule false alarms on gold labels: none.

| policy | clarify rate | selective accuracy | errors caught | flag precision |
|---|---|---|---|---|
| conformal | 0.0667 | 0.9286 | 6/12 | 1.0 |
| conformal+NLI | 0.4667 | 0.9583 | 10/12 | 0.2381 |
| conformal+constraints | 0.0667 | 0.9286 | 6/12 | 1.0 |
| conformal+NLI+constraints | 0.4667 | 0.9583 | 10/12 | 0.2381 |

## test_v2 (n=60, classifier errors 8)

NLI zero-shot accuracy 0.6, agreement with classifier 0.5 (classifier accuracy when agree 0.9667, when disagree 0.7667); NLI adds 25 flags, precision 0.12; 16.2 ms/utterance. Rule false alarms on gold labels: none.

| policy | clarify rate | selective accuracy | errors caught | flag precision |
|---|---|---|---|---|
| conformal | 0.1167 | 0.9434 | 5/8 | 0.7143 |
| conformal+NLI | 0.5333 | 1.0 | 8/8 | 0.25 |
| conformal+constraints | 0.1333 | 0.9615 | 6/8 | 0.75 |
| conformal+NLI+constraints | 0.5333 | 1.0 | 8/8 | 0.25 |

## Constraint set (n=24): blocking precision 1.0, recall 1.0

| id | gold | pred | slice | slots | rules | text |
|---|---|---|---|---|---|---|
| cv-01 | True | True | mMTC | {'devices': 20000, 'latency_ms': 2.0} | mmtc-low-latency | Smart water meters, 20,000 devices, latency under 2 ms. |
| cv-02 | True | True | mMTC | {'bandwidth_mbps': 200.0} | mmtc-high-bw | Soil sensors across the farm reporting hourly at 200 Mbps each. |
| cv-03 | True | True | URLLC | {'latency_ms': 200.0} | urllc-relaxed-latency | Remote surgery link at the hospital, latency up to 200 ms is fine. |
| cv-04 | True | True | URLLC | {'latency_ms': 500.0} | urllc-relaxed-latency | Robot arm motion control in the factory, 500 ms latency. |
| cv-05 | True | True | eMBB | {'latency_ms': 1.0} | embb-ultra-low-latency | Concert 4K streaming for fans with latency below 1 ms. |
| cv-06 | True | True | eMBB | {'reliability_percent': 101.0} | dom-reliability, embb-five-nines | Video streaming at the stadium, 101% reliability. |
| cv-07 | True | True | mMTC | {'devices': 20000, 'latency_ms': 1.0} | mmtc-low-latency | 스마트 미터 2만 대, 지연 1ms 이하로 검침해 주세요. |
| cv-08 | True | True | mMTC | {'bandwidth_mbps': 300.0} | mmtc-high-bw | 농장 토양 센서가 각각 300Mbps로 보고합니다. |
| cv-09 | True | True | URLLC | {'latency_ms': 300.0} | urllc-relaxed-latency | 병원 원격수술 연결, 지연은 300ms까지 괜찮습니다. |
| cv-10 | True | True | URLLC | {'latency_ms': 1000.0} | urllc-relaxed-latency | 공장 로봇 팔 모션 제어, 지연 1000ms. |
| cv-11 | True | True | mMTC | {'latency_ms': 1.0} | mmtc-low-latency | 공연장 4K 스트리밍, 지연 1ms 미만. |
| cv-12 | True | True | eMBB | {'reliability_percent': 150.0} | dom-reliability, embb-five-nines | 경기장 영상 중계, 신뢰도 150%. |
| cv-13 | False | False | mMTC | {'devices': 20000, 'latency_ms': 1000.0} |  | Smart water meters, 20,000 devices, latency up to 1000 ms. |
| cv-14 | False | False | URLLC | {'latency_ms': 5.0, 'bandwidth_mbps': 50.0, 'reliability_percent': 99.999} |  | Remote surgery link at the hospital under 5 ms, 50 Mbps, 99.999%. |
| cv-15 | False | False | eMBB | {'bandwidth_mbps': 800.0, 'latency_ms': 40.0} |  | Concert 4K streaming for fans, 800 Mbps, 40 ms. |
| cv-16 | False | False | URLLC | {'latency_ms': 2.0, 'reliability_percent': 99.9999} |  | Robot arm motion control in the factory, 2 ms, 99.9999%. |
| cv-17 | False | False | mMTC | {'bandwidth_mbps': 2.0, 'devices': 5000} |  | Soil sensors across the farm, 2 Mbps total, 5000 sensors. |
| cv-18 | False | False | eMBB | {'latency_ms': 20.0, 'bandwidth_mbps': 150.0} |  | Cloud gaming lounge downtown, 20 ms, 150 Mbps. |
| cv-19 | False | False | mMTC | {'latency_ms': 1000.0} |  | 스마트 미터 2만 대, 지연 1000ms 이하 검침. |
| cv-20 | False | False | URLLC | {'latency_ms': 5.0, 'bandwidth_mbps': 50.0, 'reliability_percent': 99.999} |  | 병원 원격수술 연결, 지연 5ms 이하, 50Mbps, 신뢰도 99.999%. |
| cv-21 | False | False | eMBB | {'bandwidth_mbps': 800.0, 'latency_ms': 40.0} |  | 공연장 4K 스트리밍, 800Mbps, 지연 40ms. |
| cv-22 | False | False | URLLC | {'latency_ms': 2.0, 'reliability_percent': 99.9999} |  | 공장 로봇 팔 모션 제어, 2ms, 신뢰도 99.9999%. |
| cv-23 | False | False | mMTC | {'devices': 5000, 'bandwidth_mbps': 2.0} |  | 농장 토양 센서 5000개, 합계 2Mbps. |
| cv-24 | False | False | eMBB | {'latency_ms': 20.0, 'bandwidth_mbps': 150.0} |  | 도심 클라우드 게임 라운지, 20ms, 150Mbps. |