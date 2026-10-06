# FlyVL — 초파리 커넥톰을 비전 인코더로 쓸 수 있는가

측정된 초파리 커넥톰(MaleCNS v1.0, FlyWire v783)을 이미지·영상 인코더로 쓰고, 그 출력을 LLM/VLM에 연결할 수 있는지 사전 등록 실험으로 검증한 연구 기록이다(2026-09 ~ 2026-10).

> *Can a measured fly connectome serve as a vision encoder for LLMs? Pre-registered experiments with rewired-connectome controls and classical baselines (2026-09 – 2026-10). Short answer: not in any design we tested. A faithful replication of Liew et al. (NeurIPS 2025) shows the circuit does compute orientation, and that this needs the real wiring. But the computation does not become a useful natural-image representation.*

## 현재 결론 (2026-10-06)

시험한 범위 안의 결론이다. 범위의 한계는 맨 아래에 적었다.

1. **초파리를 고정 비전 인코더로 쓴 모든 설계에서, 실제 배선은 제대로 된 대조나 고전 기준선을 넘지 못했다.**
   - 사물 분류: 실제 배선이 재배선보다 낮거나 같았다(결과 1).
   - VLM 토큰 대체: 바닥 수준이었다(결과 16).
   - ViG Grapher 대체: 재배선, Grid, Bypass와 차이가 없었다(결과 18).
   - LLM 임베딩 해시: 실제 배선이 섞은 배선과 같았다(결과 20).
2. **이긴 것처럼 보인 두 경우는 대조군 허점이었다(D10-A, 결과 23).**
   - 저대비 움직임 방향(+31%p): 입력층(R7·R8·L1–L3) 기둥 안 국소 연결 때문이었다. 이 연결을 국소성만 보존해 섞으면 우위가 사라지고, 전체 뇌에서 빼도 사라진다.
   - looming(+2.6%p): 뉴런별 흥분/억제 입력 구성 때문이었다. 그 구성만 보존한 재배선이 실제 배선만큼 했다.
3. **회로가 실제로 계산하는 것은 있다(D10-B, 결과 24).**
   - Liew et al.(NeurIPS 2025)의 방향 지도를 저자 코드 수준으로 재현했다. Dm3 well-fit 97–99%, 방향 선택 유형 22개가 논문과 일치했다.
   - 논문에 없던 재배선 대조에서는 무너졌다(13–16%). 이 방향 선택성은 실제 배선에 의존한다.
4. **하지만 그 계산은 쓸모 있는 영상 표현이 되지 않았다.**
   - 자연 영상을 같은 입력 경로로 보여 주면 국소 어둠 지도가 나온다.
   - 방향 해독 R²는 0.01이다(Gabor 0.54).
5. **고전 방법이 늘 같거나 나았다.** Hassenstein-Reichardt 상관기, 28-파라미터 looming 검출기, Gabor 필터, GRU.

## 주요 결과 한눈에

| 실험 | 질문 | 결과 | 기록 |
|---|---|---|---|
| P5–P10 | 커넥톰 + 생리 학습을 인코더로 | 사물 분류 무효, 저대비 움직임만 우세. HR이 따라잡음 | FINDINGS 결과 1–8 |
| E·G 시리즈 | 폐루프, 강건성, 범용성 | 저대비 폐루프 1건 확정. 원천은 긴 시간 적분 | 결과 9–15 |
| D2 | 초파리 256마리 → InternVL3-1B 토큰 | POPE 0.58 = 픽셀 = 평균 토큰(ViT 0.88)으로 판별 불가. 층별로는 고전 V1 뱅크와 동률 | 결과 16 |
| D3 | 움직이며 보는 초파리 → VLM 읽기 | 모든 눈이 가만히 있는 수준. 판별 불가 | 결과 17 |
| D4·D5 | whole MaleCNS를 ViG Grapher로 | 중단 / smoke Case 1(Bypass 최선) | 결과 18 |
| D6–D8 | 학습 인코더 → 고정 뇌 → 판독 | 움직임 +31%p, looming +2.6%p, 추적·정지 무효 | 결과 19 |
| MB-1 | 버섯체 배선으로 LLM 임베딩 해시 | 실제 = 섞기 < 균등 무작위 < 밀집 | 결과 20 |
| D9 | 자연 영상, Dm3 방향 (우리 동역학) | 0단계 미재현 | 결과 21 |
| D10-A | 두 양성 효과의 기전 | 입력층 국소성 / E/I 구성 = 대조군 허점 | 결과 22–23 |
| D10-B | Liew et al. 정확 재현 | REPLICATED. 재배선 시 붕괴. 자연 영상은 어둠 지도 | 결과 24 |

판정 세부는 [`docs/FINDINGS.md`](docs/FINDINGS.md), 실험·태그 목록은 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)에 있다.

## 커넥톰 인코더를 만들려는 사람에게 (이 프로젝트의 교훈)

- **재배선 대조는 degree만 보존해서는 부족하다.** 뉴런별 흥분/억제 입력 구성, 입력층의 망막위상 국소성까지 보존해야 한다. 그러지 않으면 가짜 "topology 효과"가 나온다(결과 23).
- **학습되는 인코더나 판독을 붙이면, 계산은 그쪽이 한다.** 뇌는 비싼 고정 저수지가 된다. 고유 입출력, 고정 판독, Bypass, GRU를 대조로 둔다.
- **고전 기준선을 반드시 둔다.** HR 상관기, 손 검출기, Gabor가 대부분 이겼다.
- **seed 변동을 확인한다.** 같은 그래프라도 학습 인코더 하네스는 seed당 최대 ~7%p 흔들렸다(결과 22).
- 유행하는 "초파리 뇌로 운전·게임·LLM" 프로젝트의 실제 구현 조사는 [`docs/survey_mb_dopamine_2026-10.md`](docs/survey_mb_dopamine_2026-10.md)에 있다. 입력을 픽셀·센서값으로 꽂고, 역전파나 전역 보상으로 학습하며, 제대로 대조하면 배선 효과가 없었다.

## 다시 쓸 만한 코드

| 파일 | 내용 |
|---|---|
| `flyvl/connectome.py`, `flyvl/flygrapher.py` | MaleCNS 로더, 시각 발자국(footprint), 블록 재배선(`rewire_blocks`), rate 동역학 |
| `scripts/d10a_mechanism.py` | 회로 제한, 구조화 귀무 모델 11종(degree / 부호 / 단계 / 국소성 / 유형 / 방향 보존 MCMC / 부호 섞기 / 장거리 제거), 절제, 동역학 변형, 등록 판정 |
| `scripts/d10b_tsinghua.py` | Liew et al. / Shiu et al. 계열 LIF의 GPU 재구현(float64, Brian2 순서). Brian2와 3×10⁻¹³ mV까지 대조 검증. 저자 config 재생성, 저자 맞춤 함수, 유형 표, 관문 |
| `scripts/d10b_b3.py` | FlyWire 재배선(Rewired-FW), 자연 영상 OFF 입력, Gabor/Sobel 기준선, 지표, B4 판정 |
| `scripts/mb1_flyhash.py` | 버섯체 PN→KC 해시, degree 보존 이분 그래프 재배선 |
| `scripts/d6_*`–`d9_*` | 학습 인코더 → 고정 뇌 하네스(움직임, 추적, looming, 정지 영상) |

## 저장소 구조

```
PROTOCOL_*.md     실험별 사전 등록(결과 전에 커밋)과 부록(사후 수정·보충은 부록에 명시)
docs/             FINDINGS(판정), EXPERIMENTS(실험·태그 지도), 조사 문서, GIT_WORKFLOW
flyvl/            공용 패키지(커넥톰 로딩, 동역학, 대조 그래프, flyvis 연동)
scripts/          실험 스크립트(파일 이름 = 실험 ID)
results/          실험별 요약 결과(JSON/CSV/그림). 원시 시뮬레이션 출력은 git 밖
program.md        연구 루프 규칙과 현황
```

## 재현

**환경**
- Python 3.10. 서버 CUDA에 맞는 torch를 먼저 설치한 뒤 `pip install -r requirements.txt`.
- D10-B의 Brian2 검증만 별도 환경이 필요하다: brian2 2.5.4, numpy < 2, **Cython 0.29.x**. Cython 3에서는 brian2 2.5.4의 Cython 경로가 실패한다.

**데이터** (`FLYVL_DATA` 환경 변수, 기본 `D:\flyvl_data`)
- MaleCNS: `connectome/{weights.npz, brain.npz}`. PyPI `flybrain` 0.1.0의 prebuilt release와 같다(sha256 `c29919aa…`, `cc9bd1ec…`).
  - 주의: 이 가중치는 도파민 뉴런을 빠른 흥분성 시냅스로 둔다.
- FlyWire v783: Codex 공개 버킷 `storage.googleapis.com/flywire-data/codex/data/fafb/783/`.
  - `connections_no_threshold`, `connections`, `visual_neuron_types`, `column_assignment`, `classification`.

**D10-B 예시**
```bash
python scripts/d10b_tsinghua.py build_connectome
python scripts/d10b_tsinghua.py replica             # 저자 gen_config.py와 같은 산술로 자극 조건 생성
python scripts/d10b_tsinghua.py sim real 0          # 9,124 조건, A100에서 약 2–5 GPU시간
python scripts/d10b_tsinghua.py pack_extra && python scripts/d10b_tsinghua.py sim_extra real 0   # 저자 config 중 재현본에 없는 11개(저자 코드 실행 필요)
python scripts/d10b_tsinghua.py analyze real 0      # 관문 G1–G4, 유형 표, 구조 비교, 지도
```
- 저자 config 생성과 재현본 대조 절차는 `PROTOCOL_D10B_tsinghua.md` 부록 A·B에 있다.

## 범위의 한계

- 시험한 동역학은 rate 모델(고정 이득·시간상수)과 Shiu 계열 LIF다.
- 입력은 L1–L3(OFF) 또는 학습 patch 인코더였다.
- 판독은 선형, 또는 작은 하류 모델이었다.
- 세포 유형별 생리를 맞춘 모델(예: flyvis 계열), 다른 입력 부호화, 커넥톰 기반 가소성(도파민 학습)은 이 결론의 범위 밖이다.
- 모든 판정과 수치는 각 `PROTOCOL_*.md`와 `results/`로 추적할 수 있다.

## 브랜치·태그

- `master`가 공용 코드와 기록을 갖는다.
- 끝난 실험은 `archive/<이름>` 태그로 남긴다.
- 규칙은 [`docs/GIT_WORKFLOW.md`](docs/GIT_WORKFLOW.md)에 있다.
