# Intent NLU evaluation

Splits: train 292 (template, kNN memory) + train_slots 1560 (template, TF joint model) · calib 60 (hand-written: temperatures, conformal q-hat, mode selection) · test 90 (hand-written, seen during P1) · test_v2 60 (hand-written blind, before P2 training). alpha = 0.1.

Deployed (selected on calib): intent mode **knn**, slot mode **joint**. Calib NLL {'knn': 0.2753, 'joint': 0.3971, 'ensemble': 0.2895}, calib slot exact match {'regex': 0.95, 'joint': 0.9833, 'joint+regex': 0.9833}.

## test

| system | acc | macro-F1 | acc (alt ok) | en | ko | ECE | NLL | coverage | avg set | clarify→error | priority acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| keyword (baseline) | 0.6 | 0.5687 | 0.6222 | 0.6 | 0.6 | 0.1806 | – | – | – | – | 0.7889 |
| char-ngram kNN | 0.7333 | 0.7357 | 0.7889 | 0.7333 | 0.7333 | 0.0943 | 0.6762 | 0.9333 | 1.878 | 22/58 | 0.7 |
| SBERT kNN | 0.8667 | 0.866 | 0.8889 | 0.9111 | 0.8222 | 0.0515 | 0.256 | 0.9111 | 1.067 | 6/6 | 0.8333 |
| TF joint | 0.8667 | 0.866 | 0.9 | 0.8889 | 0.8444 | 0.0479 | 0.3213 | 0.9444 | 1.189 | 7/17 | 0.8444 |
| ensemble kNN+TF | 0.9111 | 0.9103 | 0.9222 | 0.9333 | 0.8889 | 0.0707 | 0.2574 | 0.9111 | 1.044 | 0/4 | 0.8667 |

Paired significance (McNemar exact, Δ macro-F1 95% bootstrap CI):

- keyword vs char-ngram kNN: p = 0.06525 (only A right 12, only B right 24); Δ macro-F1 CI [0.034, 0.297]
- keyword vs SBERT kNN: p = 4e-05 (only A right 5, only B right 29); Δ macro-F1 CI [0.1767, 0.416]
- keyword vs TF joint: p = 2e-05 (only A right 4, only B right 28); Δ macro-F1 CI [0.1817, 0.4113]
- keyword vs ensemble kNN+TF: p = 0.0 (only A right 4, only B right 32); Δ macro-F1 CI [0.222, 0.4576]
- SBERT kNN vs TF joint: p = 1.0 (only A right 4, only B right 4); Δ macro-F1 CI [-0.0656, 0.0627]
- SBERT kNN vs ensemble kNN+TF: p = 0.125 (only A right 0, only B right 4); Δ macro-F1 CI [0.0093, 0.092]

KPI slot extraction:

| extractor | latency F1 | bandwidth F1 | reliability F1 | devices F1 (R) | exact match |
|---|---|---|---|---|---|
| regex | 1.0 | 1.0 | 1.0 | 0.727 (0.571) | 0.9333 |
| TF tagger (joint) | 1.0 | 1.0 | 1.0 | 0.966 (1.0) | 0.9889 |
| TF tagger (joint+regex) | 1.0 | 1.0 | 1.0 | 0.966 (1.0) | 0.9889 |

<details><summary>test errors: keyword (baseline) (36)</summary>

- te-004 gold URLLC → eMBB: Police officers at the riot need push-to-talk that always gets through.
- te-005 gold URLLC → mMTC: Self-driving shuttles on campus share sensor data with roadside units, below 3 ms.
- te-009 gold URLLC → eMBB: Distribution automation switches isolate faults on the power grid within milliseconds.
- te-013 gold URLLC → eMBB: Platooning trucks on the highway brake together, 99.999 % reliability and 5 ms.
- te-014 gold URLLC → eMBB: Chemical plant safety interlocks are triggered over wireless; no delay is acceptable.
- te-015 gold URLLC → eMBB: The orchestra plays together from three studios and must hear each other without lag.
- te-019 gold URLLC → eMBB: 시위 현장 경찰관들의 무전이 반드시 연결되어야 합니다.
- te-022 gold URLLC → eMBB: 관제탑에서 안벽 크레인을 원격 운전합니다, 파이브 나인 신뢰도.
- te-024 gold URLLC → eMBB: 배전 자동화 개폐기가 수 밀리초 안에 고장 구간을 분리합니다.
- te-028 gold URLLC → eMBB: 고속도로 군집주행 트럭이 함께 제동합니다, 신뢰도 99.999%, 5ms.
- te-029 gold URLLC → eMBB: 화학공장 안전 인터록이 무선으로 동작합니다. 지연은 용납되지 않습니다.
- te-030 gold URLLC → eMBB: 세 스튜디오의 오케스트라가 서로 지연 없이 들으며 합주합니다.
- te-062 gold mMTC → eMBB: Beehives in the orchard carry weight scales that report daily.
- te-063 gold mMTC → eMBB: Each pallet in the logistics hub reports its position every few minutes.
- te-064 gold mMTC → eMBB: Gas leak detectors in old apartment blocks wake up only when something happens.
- te-065 gold mMTC → eMBB: Flood level gauges along the river send a value every 15 minutes, 300 units.
- te-066 gold mMTC → eMBB: Rental scooters report battery and GPS to the operator, around 8000 scooters.
- te-067 gold mMTC → eMBB: Greenhouse climate probes, 2 Mbps aggregate, delay up to 500 ms is fine.
- te-068 gold mMTC → eMBB: Patients wear patches that upload heart-rate summaries once an hour.
- te-069 gold mMTC → eMBB: Vending machines at the stadium report stock levels overnight.
- te-070 gold mMTC → eMBB: Noise monitors around the airport log decibel readings for regulators.
- te-071 gold mMTC → eMBB: Street lamps across downtown are dimmed and monitored remotely, 12,000 lamps.
- te-072 gold mMTC → eMBB: Soil probes on the golf course run for five years on a coin cell.
- te-074 gold mMTC → eMBB: Library books carry tags so the shelves can be inventoried automatically.
- te-077 gold mMTC → eMBB: 과수원 벌통에 달린 무게 저울이 하루 한 번 보고합니다.
- te-078 gold mMTC → eMBB: 물류센터 팔레트마다 몇 분 간격으로 위치를 보고합니다.
- te-079 gold mMTC → eMBB: 노후 아파트 가스 누출 감지기는 이상이 있을 때만 깨어납니다.
- te-080 gold mMTC → eMBB: 하천변 수위계 300개가 15분마다 값을 보냅니다.
- te-081 gold mMTC → eMBB: 공유 킥보드 약 8000대가 배터리와 GPS를 운영사에 보고합니다.
- te-082 gold mMTC → eMBB: 온실 기후 측정 장치, 합계 2Mbps, 지연 500ms까지 괜찮음.
- te-083 gold mMTC → eMBB: 환자가 붙인 패치가 한 시간에 한 번 심박 요약을 올립니다.
- te-084 gold mMTC → eMBB: 경기장 자판기들이 밤사이 재고 수준을 보고합니다.
- te-085 gold mMTC → eMBB: 공항 주변 소음 측정기가 규제기관용 데시벨 값을 기록합니다.
- te-086 gold mMTC → eMBB: 도심 가로등 12,000개를 원격으로 조광하고 감시합니다.
- te-087 gold mMTC → eMBB: 골프장 토양 측정기는 코인 전지 하나로 5년을 갑니다.
- te-089 gold mMTC → eMBB: 도서관 책마다 태그가 있어 서가 재고를 자동으로 조사합니다.

</details>

<details><summary>test errors: char-ngram kNN (24)</summary>

- te-005 gold URLLC → mMTC: Self-driving shuttles on campus share sensor data with roadside units, below 3 ms.
- te-012 gold URLLC → eMBB: Disaster response teams need a dedicated, ultra-reliable link at the collapsed building.
- te-014 gold URLLC → eMBB: Chemical plant safety interlocks are triggered over wireless; no delay is acceptable.
- te-015 gold URLLC → eMBB: The orchestra plays together from three studios and must hear each other without lag.
- te-020 gold URLLC → mMTC: 캠퍼스 자율주행 셔틀이 노변 장치와 센서 데이터를 공유합니다, 3ms 이하.
- te-030 gold URLLC → eMBB: 세 스튜디오의 오케스트라가 서로 지연 없이 들으며 합주합니다.
- te-033 gold eMBB → URLLC: News crews need to upload raw footage from the marathon route, 300 Mbps.
- te-036 gold eMBB → URLLC: Remote students take part in a hybrid classroom with two-way HD video.
- te-038 gold eMBB → URLLC: Telemedicine video consultations for outpatients, 10 Mbps per session.
- te-049 gold eMBB → URLLC: 공항 라운지 승객들이 탑승 전 영화를 볼 빠른 인터넷을 원합니다.
- te-050 gold eMBB → URLLC: 부동산 중개소 쇼룸에서 헤드셋으로 가상 집 둘러보기를 제공합니다.
- te-051 gold eMBB → URLLC: 원격 학생들이 양방향 HD 영상으로 하이브리드 수업에 참여합니다.
- te-052 gold eMBB → mMTC: 노트북 2,000대 소프트웨어 업데이트는 밤사이 백그라운드로 처리해도 됩니다.
- te-053 gold eMBB → URLLC: 외래 환자 원격진료 화상 상담, 세션당 10Mbps.
- te-062 gold mMTC → eMBB: Beehives in the orchard carry weight scales that report daily.
- te-063 gold mMTC → URLLC: Each pallet in the logistics hub reports its position every few minutes.
- te-070 gold mMTC → eMBB: Noise monitors around the airport log decibel readings for regulators.
- te-071 gold mMTC → URLLC: Street lamps across downtown are dimmed and monitored remotely, 12,000 lamps.
- te-073 gold mMTC → eMBB: Manhole covers have tilt sensors that alert when one is opened.
- te-077 gold mMTC → URLLC: 과수원 벌통에 달린 무게 저울이 하루 한 번 보고합니다.
- te-083 gold mMTC → URLLC: 환자가 붙인 패치가 한 시간에 한 번 심박 요약을 올립니다.
- te-084 gold mMTC → URLLC: 경기장 자판기들이 밤사이 재고 수준을 보고합니다.
- te-088 gold mMTC → URLLC: 맨홀 뚜껑 기울기 센서가 열리면 알립니다.
- te-089 gold mMTC → URLLC: 도서관 책마다 태그가 있어 서가 재고를 자동으로 조사합니다.

</details>

<details><summary>test errors: SBERT kNN (12)</summary>

- te-005 gold URLLC → mMTC: Self-driving shuttles on campus share sensor data with roadside units, below 3 ms.
- te-006 gold URLLC → mMTC: Earthquake early-warning messages must reach every train controller immediately.
- te-020 gold URLLC → mMTC: 캠퍼스 자율주행 셔틀이 노변 장치와 센서 데이터를 공유합니다, 3ms 이하.
- te-021 gold URLLC → mMTC: 지진 조기경보가 모든 열차 제어기에 즉시 도달해야 합니다.
- te-029 gold URLLC → mMTC: 화학공장 안전 인터록이 무선으로 동작합니다. 지연은 용납되지 않습니다.
- te-053 gold eMBB → URLLC: 외래 환자 원격진료 화상 상담, 세션당 10Mbps.
- te-068 gold mMTC → eMBB: Patients wear patches that upload heart-rate summaries once an hour.
- te-069 gold mMTC → URLLC: Vending machines at the stadium report stock levels overnight.
- te-080 gold mMTC → eMBB: 하천변 수위계 300개가 15분마다 값을 보냅니다.
- te-083 gold mMTC → URLLC: 환자가 붙인 패치가 한 시간에 한 번 심박 요약을 올립니다.
- te-084 gold mMTC → URLLC: 경기장 자판기들이 밤사이 재고 수준을 보고합니다.
- te-090 gold mMTC → URLLC: 산업단지 가스 계량기 5,000대 검침 업로드, 전달률 99.9%.

</details>

<details><summary>test errors: TF joint (12)</summary>

- te-005 gold URLLC → mMTC: Self-driving shuttles on campus share sensor data with roadside units, below 3 ms.
- te-006 gold URLLC → mMTC: Earthquake early-warning messages must reach every train controller immediately.
- te-015 gold URLLC → eMBB: The orchestra plays together from three studios and must hear each other without lag.
- te-021 gold URLLC → mMTC: 지진 조기경보가 모든 열차 제어기에 즉시 도달해야 합니다.
- te-030 gold URLLC → eMBB: 세 스튜디오의 오케스트라가 서로 지연 없이 들으며 합주합니다.
- te-053 gold eMBB → URLLC: 외래 환자 원격진료 화상 상담, 세션당 10Mbps.
- te-064 gold mMTC → eMBB: Gas leak detectors in old apartment blocks wake up only when something happens.
- te-068 gold mMTC → eMBB: Patients wear patches that upload heart-rate summaries once an hour.
- te-079 gold mMTC → eMBB: 노후 아파트 가스 누출 감지기는 이상이 있을 때만 깨어납니다.
- te-080 gold mMTC → eMBB: 하천변 수위계 300개가 15분마다 값을 보냅니다.
- te-084 gold mMTC → eMBB: 경기장 자판기들이 밤사이 재고 수준을 보고합니다.
- te-090 gold mMTC → URLLC: 산업단지 가스 계량기 5,000대 검침 업로드, 전달률 99.9%.

</details>

<details><summary>test errors: ensemble kNN+TF (8)</summary>

- te-005 gold URLLC → mMTC: Self-driving shuttles on campus share sensor data with roadside units, below 3 ms.
- te-006 gold URLLC → mMTC: Earthquake early-warning messages must reach every train controller immediately.
- te-021 gold URLLC → mMTC: 지진 조기경보가 모든 열차 제어기에 즉시 도달해야 합니다.
- te-053 gold eMBB → URLLC: 외래 환자 원격진료 화상 상담, 세션당 10Mbps.
- te-068 gold mMTC → eMBB: Patients wear patches that upload heart-rate summaries once an hour.
- te-080 gold mMTC → eMBB: 하천변 수위계 300개가 15분마다 값을 보냅니다.
- te-084 gold mMTC → eMBB: 경기장 자판기들이 밤사이 재고 수준을 보고합니다.
- te-090 gold mMTC → URLLC: 산업단지 가스 계량기 5,000대 검침 업로드, 전달률 99.9%.

</details>

- slot errors regex: te-031, te-037, te-061, te-065, te-066, te-071
- slot errors TF tagger (joint): te-087
- slot errors TF tagger (joint+regex): te-087

## test_v2

| system | acc | macro-F1 | acc (alt ok) | en | ko | ECE | NLL | coverage | avg set | clarify→error | priority acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| keyword (baseline) | 0.5833 | 0.5673 | 0.5833 | 0.6 | 0.5667 | 0.1855 | – | – | – | – | 0.7667 |
| char-ngram kNN | 0.6333 | 0.6237 | 0.6333 | 0.6 | 0.6667 | 0.1714 | 0.8263 | 0.8833 | 1.933 | 19/39 | 0.7 |
| SBERT kNN | 0.8667 | 0.8683 | 0.8667 | 0.9333 | 0.8 | 0.0511 | 0.3121 | 0.9333 | 1.117 | 5/7 | 0.7667 |
| TF joint | 0.8833 | 0.8853 | 0.8833 | 0.9 | 0.8667 | 0.0754 | 0.3256 | 0.95 | 1.267 | 5/14 | 0.8167 |
| ensemble kNN+TF | 0.8833 | 0.8853 | 0.8833 | 0.9667 | 0.8 | 0.0549 | 0.2524 | 0.9 | 1.017 | 1/1 | 0.7833 |

Paired significance (McNemar exact, Δ macro-F1 95% bootstrap CI):

- keyword vs char-ngram kNN: p = 0.64761 (only A right 8, only B right 11); Δ macro-F1 CI [-0.0823, 0.1933]
- keyword vs SBERT kNN: p = 0.00151 (only A right 5, only B right 22); Δ macro-F1 CI [0.1406, 0.4683]
- keyword vs TF joint: p = 0.00053 (only A right 4, only B right 22); Δ macro-F1 CI [0.166, 0.4747]
- keyword vs ensemble kNN+TF: p = 0.00028 (only A right 3, only B right 21); Δ macro-F1 CI [0.172, 0.4727]
- SBERT kNN vs TF joint: p = 1.0 (only A right 4, only B right 5); Δ macro-F1 CI [-0.0807, 0.1197]
- SBERT kNN vs ensemble kNN+TF: p = 1.0 (only A right 2, only B right 3); Δ macro-F1 CI [-0.052, 0.0917]

KPI slot extraction:

| extractor | latency F1 | bandwidth F1 | reliability F1 | devices F1 (R) | exact match |
|---|---|---|---|---|---|
| regex | 0.857 | 1.0 | 1.0 | 0.737 (0.583) | 0.8667 |
| TF tagger (joint) | 0.957 | 1.0 | 1.0 | 1.0 (1.0) | 0.9833 |
| TF tagger (joint+regex) | 1.0 | 1.0 | 1.0 | 1.0 (1.0) | 1.0 |

<details><summary>test_v2 errors: keyword (baseline) (25)</summary>

- t2-002 gold URLLC → eMBB: Printing presses must stop within one cycle when the operator hits the wireless e-stop.
- t2-003 gold URLLC → eMBB: Coast guard rescue boats need command links that cannot drop during the storm.
- t2-007 gold URLLC → eMBB: Microgrid inverters exchange phase measurements to keep frequency stable.
- t2-008 gold URLLC → eMBB: Pick-and-place arms in the electronics fab need a deterministic 1 ms cycle.
- t2-012 gold URLLC → eMBB: 작업자가 무선 비상정지를 누르면 인쇄기가 한 주기 안에 멈춰야 합니다.
- t2-013 gold URLLC → eMBB: 태풍 속 해경 구조선의 지휘 통신이 끊기면 안 됩니다.
- t2-015 gold URLLC → mMTC: 수확기들이 선도 기계를 따라 센티미터 단위로 협조하며 자율 주행합니다.
- t2-017 gold URLLC → eMBB: 마이크로그리드 인버터들이 주파수 안정을 위해 위상 측정값을 교환합니다.
- t2-019 gold URLLC → eMBB: 터널 붕괴 현장 구조대에 신뢰할 수 있는 전용망이 필요합니다.
- t2-020 gold URLLC → eMBB: 항만의 원격 조종 지게차는 사무실 운전자에게 즉시 반응해야 합니다.
- t2-029 gold eMBB → URLLC: Drone camera footage of the marathon is relayed to the TV studio in high definition.
- t2-042 gold mMTC → eMBB: Wildlife researchers attach tiny trackers to migrating birds.
- t2-043 gold mMTC → eMBB: Fire extinguishers in the office towers report pressure once a week.
- t2-045 gold mMTC → eMBB: Shopping carts carry beacons so the mall can study foot traffic.
- t2-046 gold mMTC → eMBB: Rooftop solar panels report generation data every 15 minutes, 2 second delay is fine.
- t2-048 gold mMTC → eMBB: Grain silos at the farm cooperative measure fill level and humidity.
- t2-049 gold mMTC → eMBB: Rental bikes report lock status and position, 6000 bicycles across the city.
- t2-050 gold mMTC → eMBB: Seismic nodes buried around the dam log ground motion for engineers, 99.5% delivery.
- t2-052 gold mMTC → eMBB: 야생동물 연구진이 철새에 초소형 추적기를 부착합니다.
- t2-053 gold mMTC → eMBB: 사무동 소화기가 일주일에 한 번 압력을 보고합니다.
- t2-055 gold mMTC → eMBB: 쇼핑 카트에 비콘을 달아 쇼핑몰 동선을 분석합니다.
- t2-056 gold mMTC → eMBB: 옥상 태양광 패널이 15분마다 발전량을 보고합니다, 2초 지연도 괜찮아요.
- t2-058 gold mMTC → eMBB: 농협 곡물 사일로가 적재량과 습도를 측정합니다.
- t2-059 gold mMTC → eMBB: 공공자전거 6000대가 잠금 상태와 위치를 보고합니다.
- t2-060 gold mMTC → eMBB: 댐 주변에 묻힌 지진계 노드가 지반 진동을 기록합니다, 전달률 99.5%.

</details>

<details><summary>test_v2 errors: char-ngram kNN (22)</summary>

- t2-002 gold URLLC → eMBB: Printing presses must stop within one cycle when the operator hits the wireless e-stop.
- t2-003 gold URLLC → eMBB: Coast guard rescue boats need command links that cannot drop during the storm.
- t2-007 gold URLLC → mMTC: Microgrid inverters exchange phase measurements to keep frequency stable.
- t2-024 gold eMBB → mMTC: Influencers at the food festival go live from their phones, around 5,000 creators.
- t2-025 gold eMBB → mMTC: The architecture firm reviews building models in VR with clients, 60 ms is acceptable.
- t2-028 gold eMBB → mMTC: The university lecture hall streams classes to 1,200 students at once, 500 Mbps.
- t2-030 gold eMBB → URLLC: Game patches of 80 GB roll out to the internet cafe machines, low priority.
- t2-032 gold eMBB → URLLC: 항구에 정박한 크루즈 승객들이 드라마를 몰아 보길 원합니다.
- t2-034 gold eMBB → mMTC: 음식 축제에서 인플루언서 약 5,000명이 휴대폰으로 라이브를 합니다.
- t2-037 gold eMBB → URLLC: 도심 버스 승객들이 출퇴근길에 스포츠 하이라이트를 봅니다.
- t2-040 gold eMBB → mMTC: PC방 컴퓨터에 80GB 게임 패치를 배포합니다, 저우선.
- t2-043 gold mMTC → eMBB: Fire extinguishers in the office towers report pressure once a week.
- t2-045 gold mMTC → eMBB: Shopping carts carry beacons so the mall can study foot traffic.
- t2-046 gold mMTC → URLLC: Rooftop solar panels report generation data every 15 minutes, 2 second delay is fine.
- t2-048 gold mMTC → URLLC: Grain silos at the farm cooperative measure fill level and humidity.
- t2-050 gold mMTC → URLLC: Seismic nodes buried around the dam log ground motion for engineers, 99.5% delivery.
- t2-052 gold mMTC → URLLC: 야생동물 연구진이 철새에 초소형 추적기를 부착합니다.
- t2-053 gold mMTC → URLLC: 사무동 소화기가 일주일에 한 번 압력을 보고합니다.
- t2-055 gold mMTC → URLLC: 쇼핑 카트에 비콘을 달아 쇼핑몰 동선을 분석합니다.
- t2-056 gold mMTC → eMBB: 옥상 태양광 패널이 15분마다 발전량을 보고합니다, 2초 지연도 괜찮아요.
- t2-058 gold mMTC → URLLC: 농협 곡물 사일로가 적재량과 습도를 측정합니다.
- t2-060 gold mMTC → URLLC: 댐 주변에 묻힌 지진계 노드가 지반 진동을 기록합니다, 전달률 99.5%.

</details>

<details><summary>test_v2 errors: SBERT kNN (8)</summary>

- t2-014 gold URLLC → eMBB: 트램 출입문과 견인 제어를 무선으로 합니다, 성공률 99.9999%, 3ms.
- t2-015 gold URLLC → mMTC: 수확기들이 선도 기계를 따라 센티미터 단위로 협조하며 자율 주행합니다.
- t2-017 gold URLLC → mMTC: 마이크로그리드 인버터들이 주파수 안정을 위해 위상 측정값을 교환합니다.
- t2-022 gold eMBB → URLLC: Cruise ship passengers want to binge series while docked at the port.
- t2-032 gold eMBB → URLLC: 항구에 정박한 크루즈 승객들이 드라마를 몰아 보길 원합니다.
- t2-039 gold eMBB → URLLC: 마라톤 드론 촬영 영상을 방송 스튜디오로 고화질 중계합니다.
- t2-046 gold mMTC → URLLC: Rooftop solar panels report generation data every 15 minutes, 2 second delay is fine.
- t2-051 gold mMTC → URLLC: 신축 아파트 단지 전력량계 15,000대가 매시간 검침값을 보냅니다.

</details>

<details><summary>test_v2 errors: TF joint (7)</summary>

- t2-009 gold URLLC → mMTC: Search-and-rescue teams in the tunnel collapse require a reliable dedicated network.
- t2-017 gold URLLC → mMTC: 마이크로그리드 인버터들이 주파수 안정을 위해 위상 측정값을 교환합니다.
- t2-022 gold eMBB → URLLC: Cruise ship passengers want to binge series while docked at the port.
- t2-026 gold eMBB → URLLC: Overnight replication of the photo library to the backup site, best effort.
- t2-032 gold eMBB → URLLC: 항구에 정박한 크루즈 승객들이 드라마를 몰아 보길 원합니다.
- t2-052 gold mMTC → URLLC: 야생동물 연구진이 철새에 초소형 추적기를 부착합니다.
- t2-053 gold mMTC → URLLC: 사무동 소화기가 일주일에 한 번 압력을 보고합니다.

</details>

<details><summary>test_v2 errors: ensemble kNN+TF (7)</summary>

- t2-015 gold URLLC → mMTC: 수확기들이 선도 기계를 따라 센티미터 단위로 협조하며 자율 주행합니다.
- t2-017 gold URLLC → mMTC: 마이크로그리드 인버터들이 주파수 안정을 위해 위상 측정값을 교환합니다.
- t2-022 gold eMBB → URLLC: Cruise ship passengers want to binge series while docked at the port.
- t2-032 gold eMBB → URLLC: 항구에 정박한 크루즈 승객들이 드라마를 몰아 보길 원합니다.
- t2-051 gold mMTC → URLLC: 신축 아파트 단지 전력량계 15,000대가 매시간 검침값을 보냅니다.
- t2-052 gold mMTC → URLLC: 야생동물 연구진이 철새에 초소형 추적기를 부착합니다.
- t2-053 gold mMTC → URLLC: 사무동 소화기가 일주일에 한 번 압력을 보고합니다.

</details>

- slot errors regex: t2-001, t2-024, t2-028, t2-041, t2-046, t2-047, t2-049, t2-056
- slot errors TF tagger (joint): t2-016
- slot errors TF tagger (joint+regex): none
