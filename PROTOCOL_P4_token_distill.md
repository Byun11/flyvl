# P4: Fly swarm → aligner → InternVL3-1B visual tokens (exploratory, small data, trend-finding)

작성 2026-09-18. 탐색 실험이라 GO/STOP 판정 규칙은 두지 않고, 대조군과 함께 경향만 봄.
결과를 본 뒤 추가하는 변형은 이 문서 아래 "변형 로그"에 시간순으로 적음.

## Question
여러 마리의 동일한 frozen MaleCNS(영역별 patch 입력)가 만든 상태를, 작은 정렬기로 VLM(InternVL3-1B)의
visual token으로 바꿀 때 — 초파리가 없는 입력(픽셀, photoreceptor 입력, CNN) 대비 무엇을 더하거나 빼는가,
그리고 real 배선이 shuffle 배선과 다른가.

## Teacher / judge
- `OpenGVLab/InternVL3-1B-hf` (InternViT-300M + MLP projector + Qwen2.5-0.5B), bfloat16.
- CIFAR 32px → bicubic 448 → ImageNet 정규화 → projector 출력 (256, 896) = LLM이 받는 16×16 토큰.
- target: 4×4 average pooling (16 토큰). zero-shot 평가 시 4×4 → nearest upsample → 16×16.
- zero-shot: 고정 프롬프트 "What is the main object in this image? Answer with one word." + 10개 클래스 이름의
  답변 log-prob 합이 최대인 클래스.
- 파이프라인 확인 (CIFAR test 클래스당 20장 = 200장): 원본 256 토큰 79.5%, 4×4 pooling 71.0%, 3×3 63.5%,
  1×1 41.5%, 랜덤 토큰 10.5%, 0 토큰 10.0%.

## Data (고정, 작게)
train = CIFAR train 클래스당 첫 200장 (2,000) / val = 같은 클래스의 201~250번째 (500) / test = CIFAR test 클래스당 첫 100장 (1,000).

## Fly swarm (frozen, `flyvl/swarm.py`)
- 이미지 luminance, reflect pad 4 → 16×16 patch 16개 (stride 8, 4×4 토큰 격자 중심에 정렬).
- patch 하나 = 초파리 한 마리. v1 eye model(R1-6 retinotopy, 4방향 drift, 25 step) → FlyVL-v2 rate dynamics를
  공통 init(g=1)으로 고정. 16마리가 같은 connectome을 공유함.
- feature: 뉴런별 시간·방향 평균 evoked f(V)를 view별 고정 Gaussian projection(1024-d, seed 0)으로 요약.
  view = central_vnc (59,064), visual_projection (9,201), optic_lobe (graded 95,501),
  photoreceptor (R1-6 입력 자체 3,241, seed 1) — 마지막은 뇌를 거치지 않는 대조군.
- graph: `real`, `global_shuffle_s0`, `matched_shuffle_s0`(superclass×side 블록 보존, 생성되면 추가).

## Aligner (모든 rep 동일, `scripts/p4_align.py`)
- 토큰별 LayerNorm → Linear(d_in→256) → 학습 위치 임베딩 16 → Transformer encoder 2층 (d 256, 4 heads, ff 512,
  dropout 0.1, pre-norm) → LayerNorm → Linear(256→896).
- 입력은 train 통계로 차원별 표준화, target도 train 통계로 표준화.
- loss = MSE + (1 − cosine), AdamW lr 3e-4, wd 0.05, cosine schedule, 150 epoch, batch 64, val loss 최소 epoch 사용.
- seed 0, 1, 2.

## Representations
| rep | 의미 |
|---|---|
| `mean` | 모든 이미지에 train 평균 토큰 (하한) |
| `pixels` | 같은 16×16 RGB patch를 바로 정렬기에 (뇌·눈 없음) |
| `cnn` | 전체 이미지 → 작은 CNN(3 conv) → 4×4 → 같은 transformer (일반 작은 모델 기준선) |
| `real:photoreceptor` | 초파리 눈 입력만 (뇌 없음) |
| `GRAPH:central_vnc`, `GRAPH:visual_projection`, `GRAPH:optic_lobe` | 초파리 뇌 상태 |

## Metrics (test 1,000)
- zero-shot 정확도 (primary 관심), 토큰 cosine (raw, centered).
- 비교 기준: teacher 4×4 pooled 천장, `mean` 하한, `pixels`·`photoreceptor`(초파리 없음), `cnn`(일반 모델).

## 변형 로그
(결과를 보며 추가)

### 결과 1 — 2026-09-18 02시 (`results/p4/summary_main.json`)
천장: teacher 원본 256 토큰 83.3%, 4×4 pooled 73.5%. test 1,000장, 3 seed 평균 zero-shot %.

| rep | zero-shot | centered cos |
|---|---|---|
| cnn (일반 작은 CNN) | 36.3 | 0.567 |
| pixels (patch 픽셀) | 27.9 | 0.511 |
| real:photoreceptor (눈 입력만) | 24.6 | 0.493 |
| real: optic lobe / VPN / central (v2 g=1) | 23.2 / 23.3 / 23.9 | 0.496 / 0.491 / 0.492 |
| matched_shuffle: optic lobe / VPN / central | 23.0 / 23.6 / 24.1 | 0.491 / 0.488 / 0.489 |
| global_shuffle: optic lobe / VPN / central | 22.2 / 21.8 / 21.7 | 0.487 / 0.484 / 0.486 |
| real-v1 (LIF): optic lobe / VPN / central | 24.7 / 23.5 / 23.0 | 0.490 / 0.490 / 0.486 |
| mean | 10.0 | 0 |

paired bootstrap (seed 평균 정답 여부):
- real − global shuffle: optic lobe +1.0 [−0.1, +2.1], VPN **+1.6 [+0.2, +2.9]**, central **+2.3 [+1.0, +3.5]**
- real − matched shuffle: optic lobe +0.2 [−0.9, +1.3], VPN −0.3 [−1.5, +0.8], central −0.2 [−1.4, +0.9]
- real 뇌 view − photoreceptor: −0.7 ~ −1.4 (CI가 0을 포함하거나 경계), − pixels: −3.9 ~ −4.7 (CI < 0)

해석 (탐색적):
1. global shuffle 대비 real 우위는 superclass×side 블록을 보존한 shuffle 대비 사라짐 → 이점은 세부 배선이 아니라 영역 간 거시 연결 구조 수준.
2. 어떤 뇌 view도 눈 입력(photoreceptor)보다 높지 않음 → 뇌가 토큰 정렬에 쓸 정보를 더하지 않음.
3. v1 LIF(신호가 central까지 강함)와 v2 g=1(매우 약함)의 결과가 비슷 → 신호 세기는 병목이 아님.
4. 작은 CNN이 모든 초파리 조건보다 약 12%p 높음.

### 변형 로그
- 02시: concat(photoreceptor + 뇌 view, real vs global shuffle), v1 global shuffle 정렬기 진행.

### 결과 2 — 2026-09-18 04시
**(a) dynamics 모델에 따라 real vs global shuffle 방향이 뒤집힘** (3 seed, paired bootstrap):

| view | v2 (g=1) real − shuffle | v1 LIF real − shuffle |
|---|---|---|
| optic lobe | +1.0 [−0.1, +2.1] | +0.6 [−1.0, +2.0] |
| visual projection | +1.6 [+0.2, +2.9] | −1.1 [−2.6, +0.3] |
| central | +2.3 [+1.0, +3.5] | **−1.6 [−3.1, −0.2]** |

v1 LIF 절대값: real optic/VPN/central 24.7 / 23.5 / 23.0, shuffle 24.2 / 24.6 / 24.6.
→ block-matched shuffle에서 차이가 사라지는 것과 함께, real 배선 고유의 이점은 확인되지 않음.

**(b) concat: 눈 입력 + 뇌 (뇌를 대체물이 아니라 추가 정보로 사용)**

| rep | zero-shot | centered cos | val loss |
|---|---|---|---|
| real:photoreceptor | 24.60 | 0.4926 | 1.2569 |
| + real:visual_projection | 24.97 | 0.5020 | 1.2469 |
| + global_shuffle:visual_projection | 25.30 | 0.4975 | 1.2555 |
| **+ real:central_vnc** | **26.07** | 0.5009 | 1.2484 |
| + global_shuffle:central_vnc | 25.07 | 0.4990 | 1.2510 |

- real central concat − shuffle central concat: **+1.00 [−0.17, +2.17]**
- real central concat − photoreceptor only: **+1.47 [−0.03, +2.80]**
- 토큰 정렬 지표(val loss, centered cos)는 real 뇌를 붙일 때 일관되게 개선.
→ 뇌 상태를 입력의 **대체물**로 쓰면 손실이지만, **추가 채널**로 쓰면 약한 보완 효과가 있을 수 있음 (CI 경계).

**(c) 데이터 규모 (train 500 vs 2,000, central view)**

| rep | 500 | 2,000 |
|---|---|---|
| cnn | 29.10 | 36.27 |
| pixels | 20.63 | 27.87 |
| real:photoreceptor | 16.60 | 24.60 |
| real:central_vnc | 15.40 | 23.93 |
| matched_shuffle:central_vnc | 14.57 | 24.13 |
| global_shuffle:central_vnc | 14.03 | 21.67 |

순서(CNN > pixels > 눈 입력 ≥ real 뇌 > global shuffle)는 데이터 규모와 무관하게 유지.
val loss는 500에서도 real(1.376) < matched(1.395) < global(1.405).

### 변형 로그
- 04시: 초파리 마리 수 비교 (1마리 전체 이미지 / 4마리 2×2 / 16마리 4×4, 토큰은 4×4로 맞춤), train 1,000 조건 진행.

### 결과 3 — 2026-09-18 04시: 데이터 규모와 초파리 마리 수
모두 파일(`runs/p4/*.json`) 기준 3 seed 평균. 알림 스트림에 깨진 줄이 섞여 수치는 파일에서만 읽음.

**(a) 데이터 규모 (train 500 / 1,000 / 2,000, zero-shot %)**

| rep | 500 | 1,000 | 2,000 |
|---|---|---|---|
| cnn | 29.1 | 31.7 | 36.3 |
| pixels | 20.6 | 24.1 | 27.9 |
| real:photoreceptor | 16.6 | 19.0 | 24.6 |
| real:central_vnc | 15.4 | 16.9 | 23.9 |
| matched_shuffle:central_vnc | 14.6 | 18.7 | 24.1 |
| global_shuffle:central_vnc | 14.0 | 17.4 | 21.7 |

- 순서(cnn > pixels > 눈 입력 ≥ 뇌)는 규모와 무관하게 유지 → 데이터가 적을 때 초파리 구조가 유리해지지는 않음.
- real vs matched shuffle은 규모마다 방향이 뒤집힘(500 +0.8 / 1,000 −1.8 / 2,000 −0.2) → 사실상 동률.
- global shuffle만 모든 규모에서 최하위.

**(b) 초파리 마리 수 = 이미지 영역 분할 (zero-shot % / val loss)**

| rep | 1마리 (전체) | 4마리 (2×2) | 16마리 (4×4, 겹침) |
|---|---|---|---|
| pixels | 22.7 / 1.380 | 25.3 / 1.330 | 27.9 / 1.229 |
| real:photoreceptor | 18.4 / 1.342 | 19.6 / 1.321 | 24.6 / 1.257 |
| real:central_vnc | 17.4 / 1.326 | 17.4 / 1.315 | 23.9 / 1.260 |
| matched_shuffle:central_vnc | 16.4 / 1.353 | (진행) | 24.1 / 1.265 |

- 마리 수를 늘리면 네 조건 모두 정확도가 오르고 val loss가 줄어듦 (swarm 방향은 유효).
- 그러나 같은 크기의 이득이 pixels에서도 나타남 → 이득의 원인은 **영역 분할 구조**이고 초파리 뇌 특이적이지 않음.
- 16마리는 patch가 겹치므로(stride 8) 4마리·1마리보다 유효 해상도가 높다는 점을 감안해야 함.

### 결과 4 — 2026-09-18 04시: concat 전체 (뇌를 추가 채널로 사용)
눈 입력(photoreceptor) 토큰에 뇌 view 토큰을 이어붙이고 같은 정렬기를 학습. 3 seed, 파일 기준.

| 조건 | zero-shot | val loss | centered cos |
|---|---|---|---|
| eye only | 24.60 | 1.2569 | 0.4926 |
| eye + real optic_lobe | 25.00 | 1.2528 | 0.4986 |
| eye + real visual_projection | 24.97 | 1.2469 | 0.5020 |
| **eye + real central_vnc** | **26.07** | 1.2484 | 0.5009 |
| eye + matched_shuffle central_vnc | 24.53 | 1.2517 | 0.4993 |
| eye + global_shuffle central_vnc | 25.07 | 1.2510 | 0.4990 |
| eye + global_shuffle visual_projection | 25.30 | 1.2555 | 0.4975 |

paired bootstrap:
- **eye+real central − eye+matched_shuffle central: +1.53 [+0.23, +2.77]** ← block-matched 대조군을 통과한 유일한 real 우위
- eye+real central − eye+global_shuffle central: +1.00 [−0.17, +2.17]
- eye+real central − eye only: +1.47 [−0.03, +2.80]
- eye+matched_shuffle central − eye only: −0.07 [−1.43, +1.37]
- eye+real optic_lobe − eye only: +0.40 [−0.97, +1.83]

해석: 뇌 상태를 입력의 **대체물**로 쓰면 어느 view도 눈 입력을 넘지 못하지만, **추가 채널**로 쓰면 real 중추(central_vnc)만
+1.5%p를 주고 같은 구조의 shuffle은 0%p임. 즉 real 중추 표현에 눈 입력과 **상보적인** 성분이 있을 가능성.
주의: 2,000장 / seed 3개 규모, eye only 대비 CI 하한이 0에 걸침(−0.03). 확인에는 seed 확대가 필요.

### 결과 5 — 2026-09-18 05시: **결과 4의 정정** (seed 3 → 8)
결과 4의 "eye+real central − eye+matched_shuffle central = +1.53 [+0.23, +2.77]"은 seed 3개 기준이었음.
같은 조건을 seed 0–7 (8개)로 확장:

| 조건 (8 seed) | zero-shot | val loss |
|---|---|---|
| eye only | 24.40 | 1.2585 |
| eye + real central_vnc | 25.90 | 1.2464 |
| eye + matched_shuffle central_vnc | 25.25 | 1.2534 |

| 비교 | diff | CI95 | seed별 부호 |
|---|---|---|---|
| eye+real − eye only | **+1.50** | **[+0.49, +2.55]** | 8/8 양수 |
| eye+real − eye+matched_shuffle | +0.65 | [−0.19, +1.41] | 4/8 양수 |
| eye+matched_shuffle − eye only | +0.85 | [−0.21, +1.85] | 5/8 양수 |

**정정:** block-matched 대조군을 통과한 real 우위라는 결과 4의 해석은 **철회**함 (seed 3개에서의 요동이었음).
**남는 결과:** 뇌 상태를 추가 채널로 붙이면 8/8 seed에서 오르지만(+1.5%p), 그 이득은 real 배선 특이적이지 않음
(섞은 뇌도 +0.85%p, real과의 차이는 CI에 0 포함). → 고차원 비선형 채널 추가 효과로 해석하는 것이 타당.

**교훈(방법론):** 이 실험 규모에서 seed 3개는 ±1.5%p 수준의 요동을 만들며, 그 크기는 우리가 찾는 효과와 같은 크기임.
이후 real vs 대조군 비교는 최소 8 seed로 본다.

### 결과 6 — 2026-09-18 05시: central view도 8 seed로 재검증 (결과 1의 정정)
같은 3 조건(v2 g=1, central_vnc view)을 seed 0–7로 확장:

| 조건 (8 seed) | zero-shot | val loss | seed별 acc |
|---|---|---|---|
| real | **24.49** | 1.2619 | 23.1 23.9 24.8 25.8 24.2 25.5 24.0 24.6 |
| matched_shuffle | 23.38 | 1.2700 | 25.2 24.2 23.0 22.9 21.7 23.5 21.8 24.7 |
| global_shuffle | 21.59 | 1.2733 | 23.9 20.4 20.7 17.1 23.7 20.6 21.5 24.8 |

| 비교 | diff | CI95 | seed 부호 |
|---|---|---|---|
| real − matched_shuffle | **+1.11** | **[+0.27, +1.90]** | 5/8 |
| real − global_shuffle | **+2.90** | **[+1.96, +3.85]** | 6/8 |
| matched_shuffle − global_shuffle | +1.79 | [+0.89, +2.73] | 6/8 |

**정정:** 결과 1의 "real − matched_shuffle은 차이 없음(−0.2 [−1.4, +0.9], 3 seed)"은 seed 부족이었음.
8 seed에서는 **real > matched_shuffle > global_shuffle** 순서가 일관되고 세 비교 모두 CI가 0을 넘음.
val loss도 같은 순서(1.2619 < 1.2700 < 1.2733).

단, (1) 효과 크기는 1~3%p로 작고 seed 부호는 5~6/8, (2) real조차 눈 입력(24.40)·pixels(27.9)·CNN(36.3)을
넘지 못함. 해석: 배선을 덜 망가뜨릴수록 토큰 정렬이 낫지만, 초파리 뇌를 통과시키는 것 자체는 입력보다 손실.

### 결과 7 — 2026-09-18 05시: 세 view 전부 8 seed (최종)

| view | real | matched_shuffle | global_shuffle |
|---|---|---|---|
| optic_lobe | 24.41 (val 1.2559) | 23.81 (1.2643) | 22.68 (1.2731) |
| visual_projection | 23.45 (1.2606) | 22.96 (1.2706) | 21.12 (1.2781) |
| central_vnc | 24.49 (1.2619) | 23.38 (1.2700) | 21.59 (1.2733) |

| 비교 | optic_lobe | visual_projection | central_vnc |
|---|---|---|---|
| real − matched | +0.60 [−0.14, +1.33] 4/8 | +0.49 [−0.39, +1.29] 5/8 | **+1.11 [+0.27, +1.90]** 5/8 |
| real − global | **+1.74 [+0.82, +2.64]** 8/8 | **+2.32 [+1.40, +3.27]** 6/8 | **+2.90 [+1.96, +3.85]** 6/8 |
| matched − global | **+1.14 [+0.40, +1.89]** 6/8 | **+1.84 [+0.94, +2.77]** 6/8 | **+1.79 [+0.89, +2.73]** 6/8 |

확정 사항 (이 설정 한정):
1. real > global shuffle: 세 view 모두 CI>0.
2. matched > global: 세 view 모두 CI>0 → 우위의 상당 부분은 영역 간(superclass×side) 거시 연결 구조에서 옴.
3. real > matched: central에서만 CI>0 (+1.11), 나머지는 경계 → 거시 구조를 넘는 세부 배선의 기여는 작음.
4. val loss 순서도 real < matched < global로 일치.
5. 그러나 real도 eye input(24.40) · pixels(27.87) · cnn(36.27)을 넘지 못함.

### 결과 8 — 2026-09-18 06시: dynamics 모델 교차 검증 (central view)

| | real | matched_shuffle | global_shuffle |
|---|---|---|---|
| v2 (g=1), 8 seed | **24.49** (val 1.2619) | 23.38 (1.2700) | 21.59 (1.2733) |
| v1 LIF, 5 seed | 23.24 (1.2689) | 21.96 (1.2830) | **24.50** (1.2648) |

| 비교 | v2 (8 seed) | v1 LIF (5 seed) |
|---|---|---|
| real − matched | +1.11 [+0.27, +1.90] 5/8 | +1.28 [0.00, +2.60] 3/5 |
| real − global | **+2.90 [+1.96, +3.85]** 6/8 | **−1.26 [−2.50, +0.02]** 1/5 |
| matched − global | +1.79 [+0.89, +2.73] 6/8 | **−2.54 [−3.78, −1.24]** 0/5 |

해석:
- **real > matched shuffle은 두 dynamics에서 같은 방향** (+1.1 / +1.3%p). 이 부분만 모델에 비교적 견고함.
- **global shuffle의 위치는 dynamics에 따라 완전히 뒤집힘** (v2에서 최하위, v1에서 최상위).
  v2 g=1에서는 global shuffle이 신호를 central까지 가장 강하게 전달했고(1.5e-3 vs real 3.8e-5), v1에서는 반대(1.07 vs 1.57).
  즉 global shuffle 비교는 "배선 품질"보다 "그 dynamics에서 신호가 얼마나 전달되는지"를 반영할 수 있음.
- 따라서 **global shuffle 단독 대조군으로는 어떤 주장도 하기 어렵고, matched shuffle이 필수**임.
