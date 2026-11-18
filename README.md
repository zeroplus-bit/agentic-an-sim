# Agentic AI Autonomous Network Simulator (agentic-an-sim)

설계서 *「Agentic AI Autonomous Network Simulator — Software Design」*의 구현입니다.
Service / Network / Resource 3계층을 독립 FastAPI 프로세스로 띄우고, 계층 간 통신은 **SN**·**NR** 참조점(REST + 버스 알림)만 사용합니다.
에이전트는 **MCP(Model Context Protocol)** 로 도구를 호출합니다.

```
Service Layer  :8100  ──SN /sn/v1 + sn.notify.*──▶  Network Layer :8200  ──NR /nr/v1 + nr.notify.*──▶  Resource Layer :8300
  (의도 처리·LCM)                                     (에이전트·Task Control/Exec)                       (디지털 트윈)
       ▲                                                   ▲
       │ MCP (stdio)  mcp_servers/sn_mcp_server.py         │ MCP (streamable HTTP)  /tools/mcp  = ToolRegistry
   외부 LLM(Claude 등) = Agent-based Orchestration          내부 에이전트 3종 + 온보딩된 외부 LLM 에이전트
```

## MCP가 들어가는 두 곳

| MCP 서버 | 위치 | 역할 | 클라이언트 |
|---|---|---|---|
| **Network Tool MCP** (`aan-network-tools`) | Network Layer `http://127.0.0.1:8200/tools/mcp` | 설계서의 *Tool/Function (ToolRegistry)* 블록. NR-01~13 연산, LSTM/DQN 추론, Decision Translation, Conflict check, Memory 등 22개 도구를 스키마와 권한으로 노출 | 내부 에이전트(`agent-ran-01`, `agent-assure-01`, `agent-planner-01`)와 SN-08로 온보딩된 외부 LLM 에이전트. 모든 호출은 `Authorization: Bearer <agent token>`을 AuthGuard가 검사 |
| **SN MCP** (`aan-service-layer`) | `mcp_servers/sn_mcp_server.py` (stdio/HTTP) | *Agent-based Orchestration* 역할을 LLM이 맡도록 Service Layer 기능(의도 제출·수정·취소, 실현성 질의, 충돌 결정, Agent/Model LCM, 정책, 지표, 트레이스, 시나리오 주입)을 22개 도구로 노출 | Claude Desktop / Claude Code 등 |

내부 에이전트도 같은 MCP 엔드포인트를 쓰므로(`AANS_TOOL_MODE=mcp`, 기본값) 내부·외부 에이전트가 같은 도구, 같은 권한 검사를 거칩니다. `AANS_TOOL_MODE=direct`로 두면 동일한 핸들러를 전송 계층 없이 직접 호출합니다.

## 실행

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python tools/download_models.py   # SBERT 인코더(약 450 MB, git 미포함). --nli 를 붙이면 NLI 모델도
.venv/Scripts/python run_all.py          # 3개 계층 + 대시보드 기동, 기본 에이전트 3종 온보딩(Flow 2)
```

이 폴더만으로 동작합니다. 원본 Build-A-Thon-SRM 저장소에서 쓰는 파일(LSTM 가중치, 기지국·단말 CSV, 벤더 카탈로그)은 `vendor/`에 복사해 두었고(MIT, `vendor/LICENSE-upstream`), 원본 저장소가 상위 폴더에 있으면 `vendor/`에 없는 파일만 그쪽에서 읽습니다(`AANS_REPO`로 위치 지정). DistilBERT 분류기는 선택 사항이라 원본 저장소가 있을 때만 쓰입니다.

- 운영 API(Swagger): http://127.0.0.1:8100/docs
- 의도 제출 예: `POST /api/intents {"utterance": "Remote surgery link at Hospital A under 5 ms, 50 Mbps, 99.999%", "checkFeasibility": true}`
- 한국어 발화도 지원: `"원격수술 Hospital A, 지연 5ms 이하, 50Mbps, 신뢰도 99.999%"`
- 시나리오 주입: `POST /api/sim/scenario {"is_emergency": true}` (트윈 제어용, SN/NR 밖)

### 운영 대시보드 (http://127.0.0.1:8400/)

`run_all.py`가 네 번째 프로세스로 대시보드 게이트웨이(`dashboard/app.py`, :8400)를 띄웁니다. 게이트웨이는 어떤 흐름에도 끼지 않고 세 계층을 구독만 합니다.

```
Service :8100 ─┐  GET /events (SSE)     ┌─ 이벤트 정규화: 단계 DETECT·REPORT·EXECUTE·COORDINATE·RESOLVE
Network :8200 ─┼──────────────────────▶ │  인시던트 상관: intentId 단위, 감지 → FULFILLED (MTTA·MTTR)
Resource:8300 ─┘  + 상태 API 2초 폴링    └─ GET /dash/stream (SSE) ─▶ 브라우저 (dashboard/static/index.html)
```

- 계층 쪽(`common/events.py`): `MessageLog`가 기록하는 모든 SN/NR 메시지, 에이전트 버스, MCP 도구 호출(거부 포함)과 프로세스 로그 레코드가 순번 링(5,000개)에 들어가 `GET /events?since=<seq>`로 나갑니다. NR-08 텔레메트리는 개수만 셉니다. `log.warning(..., extra={"intentId", "stage"})`로 남긴 로그는 인시던트에 묶입니다(드리프트 감지, 어슈어런스 작업, SN-14 에스컬레이션, 오케스트레이터 결정, 회수, 복구).
- 화면: KPI(SLA 준수율, 활성 알람, 진행 중 인시던트, MTTA, MTTR, 자율 처리율, 사람 검토 대기, PRB 이용률), 인시던트 파이프라인(5단계 시간), 셀 PRB 맵(장애·부하분산·알람 표시), 단계별 피드 4개, 에이전트·MCP 도구 표, 인텐트 표, 사람 검토 큐(결정·확인 버튼), 시나리오 제어, 통합 로그(계층·단계·종류·심각도·인텐트·traceId·검색 필터, 행 클릭 시 상세와 trace 시퀀스 다이어그램), 프로세스 원본 로그 탭.
- API: `/dash/api/snapshot`, `/dash/api/events?incident=&intentId=&traceId=&layer=&stage=&q=`, `/dash/api/incidents`, `/dash/api/trace/{traceId}/mermaid`, `/dash/api/logs/{layer}`, `POST /dash/api/scenario|intents|conflicts/{id}/decide|review/{key}/ack`.
- SN MCP: `get_dashboard_snapshot` 도구와 `aan://dashboard` 리소스로 외부 LLM도 같은 화면 정보를 읽습니다.
- 대시보드 프로세스가 없어도 계층 동작은 그대로입니다(링만 유지). 시퀀스 다이어그램은 Mermaid를 CDN에서 불러오며 오프라인이면 원문을 보여 줍니다.

Claude Code에서는 이 폴더의 `.mcp.json`이 SN MCP 서버를 등록합니다. Claude Desktop은 `claude_desktop_config.json`에 같은 `command/args`를 절대경로로 넣으면 됩니다.
외부 LLM을 Network Layer 에이전트로 참여시키려면 SN MCP의 `onboard_external_agent` → 반환된 `mcpUrl`과 `agentToken`으로 HTTP MCP 연결을 추가합니다.

## 테스트

```bash
.venv/Scripts/python -m pytest -q tests                 # 계약 테스트: 설계서의 모든 메시지 예제 round-trip (23개)
.venv/Scripts/python tests/flow_scenarios.py           # 기동된 시스템 대상 E2E (SN MCP stdio로 구동, 21개 검사)
```

E2E가 확인하는 것: Flow 1 프로비저닝(FULFILLED, ~0.9 s), 우선순위 선점과 SN-14, SN-03 취소와 트윈 바인딩 해제, Flow 4 카나리 승격(두 번째 SN-10) 및 롤백 규칙, 외부 에이전트 온보딩과 Network MCP 호출, 미인증 MCP 호출 거부, 지표·Mermaid 트레이스.

## 의도 해석(Intent NLU): 시맨틱 분류 + 보정 + 컨포멀 예측

`IntentProcessor`의 슬라이스 분류는 기본값(`AANS_INTENT_BACKEND=auto`)에서 **시맨틱 kNN**을 씁니다.

- **임베딩**: Sentence-BERT(`paraphrase-multilingual-MiniLM-L12-v2`, 한·영 지원)를 ONNX로 실행합니다. `models/sbert-multilingual-minilm/`에 모델이 없으면 문자 n-gram으로 대체하고, `keyword`로 두면 기존 키워드 분류기를 씁니다.
- **분류**: 학습 문장과 표준 참조 사례를 함께 메모리로 두고, 클래스별 상위 k개 코사인 유사도의 평균으로 점수를 매깁니다.
- **보정**: 온도 스케일링(Guo et al., 2017)과 split conformal(LAC 점수)을 씁니다. 두 값 모두 `calib` 분할로만 맞추고 `calibration.json`에 저장합니다.
- **SN-01 출력**: `semantic.classifier`에 `confidence`, `probs`, `predictionSet`을 넣습니다. `semantic.evidence`에는 가장 가까운 표준 참조 사례와 학습 예문이 들어가고, `semantic.needsClarification`은 예측 집합에 클래스가 2개 이상일 때 true가 됩니다.

데이터와 평가 파일은 `datasets/intent_nlu/`에 있습니다.

| 파일 | 내용 |
|---|---|
| `train.jsonl` (292) | `generate_train.py`로 템플릿 생성, kNN 메모리 |
| `calib.jsonl` (60) | 직접 작성, 온도·컨포멀 임계값 적합용 |
| `test.jsonl` (90) | 직접 작성, 보고 전용 (일부 문장에 허용 대체 라벨 `alt`) |
| `references.json` | 3GPP TS 22.261/22.104/22.186/22.179, GSMA NG.116 사용 사례 설명. KPI 값은 아직 검증 전(`verified:false`)이라 근거 표시에만 쓰고 기대값 채우기에는 쓰지 않음 |

```bash
# 모델 1회 다운로드 (약 470 MB)
.venv/Scripts/python -c "from huggingface_hub import hf_hub_download as d; [d('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', f, local_dir='models/sbert-multilingual-minilm') for f in ['onnx/model.onnx','tokenizer.json','config.json']]"
.venv/Scripts/python tools/eval_intent.py --write-calibration   # 평가 + 보정값 저장 → datasets/intent_nlu/eval_report.md
```

test 90문장 결과 (α = 0.1):

| 분류기 | 정확도 | macro-F1 | ECE | 컨포멀 coverage | 평균 집합 크기 | 우선순위 정확도 |
|---|---|---|---|---|---|---|
| 키워드 (기존) | 0.600 | 0.569 | 0.181 | – | – | 0.789 |
| 문자 n-gram kNN | 0.733 | 0.736 | 0.094 | 0.933 | 1.88 | 0.700 |
| **SBERT kNN** | **0.867** | **0.866** | **0.052** | 0.911 | 1.07 | 0.833 |

- 키워드 대비 SBERT의 차이: McNemar exact p = 4×10⁻⁵, Δ macro-F1 95% bootstrap CI [0.18, 0.42]
- SBERT가 확인 질문을 요청한 6건 중 5건은 실제로 틀린 예측이었습니다.
- KPI 수치 추출(정규식)은 지연·대역폭·신뢰도 모두 F1 = 1.0입니다. 단말 수는 재현율이 0.57로, "30,000 fans"나 "8000 scooters"처럼 단위 명사 목록에 없는 표현을 놓칩니다. 이 부분은 P2(TF 결합 슬롯 모델)에서 다룹니다.
- 한계: calib/test 문장과 라벨은 1인이 작성한 초안입니다. 2인 이상 독립 라벨링(Cohen's κ)을 거쳐야 객관성 주장의 근거가 됩니다. 참조 사례 KPI도 표준 원문과 대조해야 합니다(`verified`).

### P2: TensorFlow 결합 의도·슬롯 모델

JointBERT(Chen et al., 2019) 구조를 따르되, 인코더는 위의 다국어 SBERT를 **고정**해 그 토큰 상태 위에 Keras 헤드를 학습합니다.

- **모델 구조**: Conv1D(128, k=3) 뒤에 두 갈래가 붙습니다. 하나는 토큰별 BIO 태그 9종(O, B/I-LAT·BW·REL·DEV)을 내고, 다른 하나는 masked-mean과 문장 임베딩을 받아 Dense를 거쳐 의도 3종을 냅니다.
- **슬롯 범위**: 슬롯은 숫자 토큰만 태깅합니다. 단위 환산은 숫자 바로 뒤 단위로 결정론적으로 처리합니다(Gbps → ×1000, 초 → ×1000 ms).
- **학습 환경**: TensorFlow에는 Python 3.14용 안정판 휠이 없습니다(2.22.0rc1뿐). 그래서 학습은 Python 3.12 venv(`.venv-train`, TF 2.21.0 / Keras 3.15.1)에서 합니다. 서빙은 내보낸 가중치를 NumPy로 추론하는 `joint_nlu.py`가 맡습니다. Keras와의 출력 차이는 최대 4.8×10⁻⁶이고 argmax는 100% 일치합니다.
- **모드 선택**: 의도는 knn / joint / ensemble 중에서, 슬롯은 regex / joint / joint+regex 중에서 `calib` 분할만으로 고르고 `calibration.json`의 `joint` 항목에 저장합니다. 현재 선택은 의도 **knn**, 슬롯 **joint**입니다.

```bash
py -3.12 -m venv .venv-train && .venv-train/Scripts/python -m pip install "tensorflow>=2.16,<2.22"
.venv/Scripts/python datasets/intent_nlu/generate_slots.py      # 학습 문장 1560개 (다양한 단위 표현 + 방해 숫자)
.venv/Scripts/python tools/build_nlu_features.py                # 고정 SBERT 토큰 특징 + BIO 태그 → features/*.npz
.venv-train/Scripts/python tools/train_joint_nlu.py             # TF 학습 → models/joint_nlu/{joint_nlu.keras, joint_nlu.npz}
.venv/Scripts/python tools/eval_intent.py --write-calibration   # 평가 + 모드 선택·보정
```

평가 결과는 두 세트로 냅니다. `test`는 P1 개발 중에 이미 본 세트이고, `test_v2`(60문장)는 P2 학습 전에 새로 작성해 학습 결과를 보지 않은 블라인드 세트입니다.

| test_v2 | 정확도 | macro-F1 | ECE | coverage | 평균 집합 |
|---|---|---|---|---|---|
| 키워드 | 0.583 | 0.567 | 0.186 | – | – |
| SBERT kNN (**배포**) | 0.867 | 0.868 | 0.051 | 0.933 | 1.12 |
| TF joint | 0.883 | 0.885 | 0.075 | 0.950 | 1.27 |
| ensemble | 0.883 | 0.885 | 0.055 | 0.900 | 1.02 |

| KPI 추출 (test_v2) | 지연 F1 | 단말 수 재현율 | 완전일치 |
|---|---|---|---|
| 정규식 | 0.857 | 0.583 | 0.867 |
| **TF 태거 (배포)** | 0.957 | 1.000 | 0.983 |
| TF 태거 + 정규식 보완 | 1.000 | 1.000 | 1.000 |

- **의도 분류**: TF 모델의 이득은 통계적으로 유의하지 않습니다. 두 세트 모두 SBERT kNN 대비 McNemar p ≥ 0.125입니다. 의도 분류는 모델보다 데이터 쪽이 병목이라, calib NLL 기준으로 kNN을 유지합니다.
- **KPI 추출**: 이득이 분명합니다. 단말 수 재현율이 test에서 0.57 → 1.0, test_v2에서 0.58 → 1.0으로 올랐고, test_v2 완전일치는 0.867 → 0.983입니다. "2초 지연", "30,000 fans", "6000 bicycles"를 잡고 "3 hours", "300 km"는 무시합니다.
- **슬롯 모드 동률**: calib에서 joint와 joint+regex가 동률(0.983)이라 정해 둔 규칙대로 joint가 선택됐습니다. test_v2에서는 joint+regex가 1.0으로 더 낫지만, 이 결과를 보고 선택을 바꾸면 테스트 오염이 되므로 바꾸지 않았습니다.

### P3: 확인 질문 게이트, 제약 검증, 의도 모델 LCM

SN-01에서 의도를 자동으로 수용할지, 운영자에게 되물을지를 세 신호로 판단합니다.

| 신호 | 방법 | 배포 상태 |
|---|---|---|
| 컨포멀 예측 집합 | P1의 split conformal. 예측 집합에 클래스가 2개 이상이면 확인 질문 | 사용 (권고만, 제출을 막지는 않음) |
| KPI 제약 검증 | `datasets/intent_nlu/constraints.json`의 선언형 규칙 12개(SHACL 방식: 대상 슬라이스 + 속성 + 조건 + 심각도). 사용자가 **명시한** 값만 검사하고 기본값은 검사하지 않음 | 사용. `error`/`conflict`이면 제출이 `CLARIFICATION_REQUIRED`로 막히며 `force=true`로 강행 가능. `warning`은 표시만 |
| NLI zero-shot 교차검증 | Yin et al. (2019). `multilingual-MiniLMv2-L6-mnli-xnli` ONNX | **미채택**. `AANS_INTENT_NLI=info`로 켜면 게이트 없이 `secondOpinion`으로만 표시 |

- **NLI 미채택 근거:** 채택 규칙은 테스트 전에 정했습니다. "calib에서 NLI가 컨포멀 게이트에 더하는 확인 질문 중 실제 오류 비율이 0.5 이상이어야 채택"입니다.
  - 측정값은 0.105였습니다. NLI 단독 정확도는 0.58~0.62이고, 더 큰 mDeBERTa-xnli도 calib에서 0.62에 그쳤습니다.
  - URLLC/eMBB/mMTC 같은 기술 범주에는 범용 NLI의 zero-shot 전이가 약했습니다. test_v2에 붙이면 오류 8건을 모두 잡지만, 확인 질문 비율이 53%로 실용적이지 않습니다.
- **SST:** 3GPP TS 23.501의 표준 SST(1 eMBB, 2 URLLC, 3 MIoT)를 `validation.sst`로 함께 냅니다. 규칙의 임계값은 트윈 가정과 맞춘 공학적 가정(`basis: assumption`)이며, 표준 인용값이 아닙니다.

`python tools/eval_p3.py` → `datasets/intent_nlu/eval_p3.md`

| test_v2 (오류 8건) | 확인 질문 비율 | 자동 수용 정확도 | 잡은 오류 |
|---|---|---|---|
| 게이트 없음 | 0 | 0.867 | 0/8 |
| 컨포멀 | 0.117 | 0.943 | 5/8 |
| **컨포멀 + 제약 (배포)** | 0.133 | 0.962 | 6/8 |
| 컨포멀 + NLI + 제약 | 0.533 | 1.000 | 8/8 |

- **오탐:** 제약 규칙은 정답 라벨과 정답 KPI(calib, test, test_v2 전체)에 대해 오탐 0건입니다.
- **모순 발화 세트(24문장):** 차단 정밀도 1.0, 재현율 1.0입니다. 단, 이 세트로 디코더 버그(하위 토큰으로 쪼개진 숫자 "150%")를 찾아 고쳤으므로 낙관적인 값입니다. 그 버그 수정과 한국어 만·천 배수 처리는 test와 test_v2의 슬롯 결과를 바꾸지 않았습니다.

**의도 모델 LCM** (`service_layer/lcm/intent_nlu.py`)은 Flow 4를 Service Layer 자체 모델에 적용한 것입니다.

- **버전:** 1.0 키워드, 1.1 kNN+정규식, **2.0 kNN+TF 태거(현재)**, 2.1 앙상블+태거·정규식 보완, 2.2 TF 단독.
- **shadow 실행:** shadow 버전은 제출되는 모든 발화에 함께 돌지만 SN-01 결과에는 영향을 주지 않습니다.
- **운영자 확인:** 운영자가 정답을 확인하면(`POST /api/intents/{id}/feedback`, MCP `confirm_intent_interpretation`) `feedback.jsonl`에 기록되어 다음 보정·학습 데이터의 계보가 됩니다.
- **판정 규칙:** 짝지은 표본이 20개 이상이면 exact McNemar 검정을 합니다. p < 0.05에서 shadow가 낫다면 `PROMOTE_RECOMMENDED`(autoPromote이면 자동 승격), 못하다면 `REJECTED`입니다. 롤백도 가능합니다.
- **MCP 도구:** `intent_model_status`, `manage_intent_model(shadow|promote|rollback)`, `confirm_intent_interpretation`. 이 3개를 더해 SN MCP 도구는 25개입니다.

## 코드 구성 (설계서 블록 → 클래스)

| 계층 | 블록 → 클래스 (파일) |
|---|---|
| common | 봉투 `envelope.py`, SN-01~14 `sn_messages.py`, NR-01~13 `nr_messages.py`, 버스(InProc/HttpBridge/NATS) `bus.py`, 트레이스 `tracing.py`, 토큰 `security.py`, 설정 `config.py` |
| Service | IntentProcessor, ServiceProvisioner, ServiceMonitor, ServiceOrchestratorAgent, IdentityManager, ConnectivityManager (`orchestration/`) · DataManager, ResourceManager, AgentLifecycleManager, ModelLifecycleManager (`lcm/managers.py`) |
| Network – Agent Control | AgentRegistry, ConflictManager, **ToolRegistry(MCP)**, AgentBus, AgentMemory(SQLite), AuthGuard, IntentMediator, TaskTracker (`agent_control/`) |
| Network – Task Execution | ModelExecutor, ReportGenerator, Repository, ActionExecutor, `ml.py`(NumPy LSTM·DQN) (`task_execution/`) |
| Network – Task Control | ModelController, DecisionTranslator, DataController, PreValidator, ResourceController, PerformanceController, PerformanceMonitor (`task_control/`) |
| Network – Agents | RanAllocatorAgent, AssuranceAgent, PlannerAgent, ExternalAgent (`agents/`) |
| Resource | RanCoreTwin `comm_twin.py`, UeEdgeSimulator `ue_edge_sim.py`, ComputePool·StoragePool `pools.py`, ToolHost `tool_host.py`, NR 서버 `app.py` |

## 저장소 코드 재사용과 설계서의 "사전 수정" 항목

- **LSTM**: `models/lstm_single/*.h5` 가중치를 h5py로 읽어 NumPy로 추론(TensorFlow 불필요, Python 3.14에서도 동작). `X_scaler.npy` 적용.
- **DQN**: `dqn_agent.py`의 상태 14·행동 27·`action_to_allocation`·`calculate_reward`·ReplayBuffer를 NumPy MLP로 이식. 경험이 쌓이기 전에는 저장소의 보상 함수로 1-step 평가(`mode: reward-model`), 이후 Q-network로 전환. 회복 에피소드가 replay buffer로 들어가 온라인 학습.
- **디지털 트윈**: `ndt/simulator.py`를 확장. 프로세스당 **트윈 1개**(검증은 deep copy), **KPI 이름 통일**(`latency_ms`, `bandwidth_mbps`, `reliability_percent`), 위반 카운터 **라벨링**. 셀·초기 PRB 분할은 `base_station_data.csv`, UE는 `client_data.csv`.
- **모델·데이터 경로**는 모두 `common/config.py`(환경변수 `AANS_*`)에서 읽습니다.
- **Gemini 키**는 Resource Layer ToolHost만 환경변수 `GEMINI_API_KEY`로 읽고 메시지에는 절대 넣지 않습니다. 키가 없으면 결정론적 오프라인 템플릿으로 응답합니다. ⚠ 저장소의 `5G-Marketplace/.env`에 커밋된 키는 폐기(rotate)해야 합니다.
- **DistilBERT**: 저장소의 `model.safetensors`는 Git LFS 포인터(134 B)라 로드할 수 없어 키워드 분류기(저장소의 mock 로직 + 한국어 키워드)로 동작합니다. 실제 가중치를 넣고 `transformers`·`torch`를 설치하면 자동으로 DistilBERT를 씁니다.

## 설계서와 다르거나 단순화한 점

- 계층 간 알림 버스 기본값은 `HttpBridgeBus`(상대 계층의 `/bus/inbox`로 POST). 브로커 없이 3 프로세스가 통신하기 위함이며, `AANS_BUS=nats`로 NATS 사용 가능(미검증). Kafka는 미구현.
- Service LCM 4개 블록은 `lcm/managers.py` 한 파일에 클래스로 모았습니다.
- 카나리 MAE의 정답 레이블은 "다음 시점의 수요 비례 할당"으로 정의했습니다. 저장소 평가 지표(MAE 0.028)와 정의가 달라 값이 ~0.17 수준이므로, 기본 롤백 임계값을 0.2로 두었습니다(설계서 예시 0.05를 쓰면 즉시 롤백됨).
- 셀 용량(400 Mbps), 슬라이스별 기본 지연·신뢰도, 부하-지연 곡선은 트윈의 가정값입니다(`comm_twin.py` 상단).
