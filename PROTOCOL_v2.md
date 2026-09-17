# FlyVL-v2 protocol (pre-registered, frozen before any CIFAR training run)

## Question
> With the same task-driven optimization of unknown physiological parameters, is the measured MaleCNS wiring
> a better computational substrate for image classification than degree-matched shuffled wiring?

v1(고정 dynamics)는 "학습 없이 즉시 좋은 표현"을 물었고 STOP이었음. v2는 동일한 학습 기회를 준 뒤의 비교임.
CLIP distillation(v2-B)은 이 결과가 나온 뒤 별도로 등록한다.

## 고정 (학습하지 않음)
MaleCNS topology, edge 위치, |w| (synapse count, 입력 정규화), E/I sign, effective graph 규칙
(nonvisual sensory 입력 차단, KC→KC 0 — v1과 같은 그래프 파일), photoreceptor 위치와 eye mapping, 4-way drift.

## 학습 (cell type 공유; 11,881 types, untyped 뉴런은 superclass별로 묶음)
- `g_pre[type]`, `g_post[type]` = softplus(·) > 0 → 유효 weight = w_ij · g_pre[type j] · g_post[type i] (부호 보존)
- `tau[type]` = dt + softplus(·) > dt
- `bias[type]`
- dynamics 파라미터 47,524개. real/shuffle에서 개수와 의미가 같음. type→type pair scale은 쓰지 않음
  (edge가 있는 type 쌍이 real 3.93M, shuffle 9.40M이라 파라미터 수가 달라짐).

## Dynamics (전 뉴런 graded, FlyVis식)
V ← V + (dt/τ)(−V + g_post·W(g_pre·f(V)) + bias + input),  f(V) = 0.5·tanh(2V),  dt = 20 ms.
- warm-up 50 step (no grad) 후 자극 25 step. blank branch를 같은 파라미터로 함께 돌려 evoked = image − blank.
- input = 4 × (luminance − adaptation), R1-6 3,241개. W는 상수 CSR, backward는 Wᵀ SpMM, step별 checkpointing.
- sigmoid f는 label 없는 점검에서 층마다 약 10배씩 감쇠해서(central evoked 약 1e-5) 기각함.

## 초기값 (모든 그래프 공통)
g_pre = g_post = 1, τ = 40 ms, bias = 0.
spectral radius가 real 1.00(ENS 2-뉴런 고립 루프, optic lobe feedback 모드 약 0.91), shuffle 0.29임.
균일 gain이 약 1.1을 넘으면 real만 rest가 불안정해져서(처음 고른 g_post=6이 그 영역이었음) 공통 안정 최대값 1을 씀.
init에서 real central evoked는 1e-5, shuffle은 1.4e-3로 약 100배 차이 → 아래 readout 표준화로 scale만 보정.
노이즈 이미지 + 랜덤 label 10 step 점검: 두 그래프 모두 finite, rest 안정, gradient 유한.

## Readout (모든 그래프 동일)
central_vnc 59,064 뉴런의 시간·방향 평균 evoked f(V) → 고정 Gaussian random projection (seed 0, 1024-d)
→ BatchNorm1d (affine 없음) → Linear(1024→10). 학습 가능한 readout 파라미터는 10,250개.

## 학습
- 데이터: P1-mini와 같음. train 5k 중 class별 50장을 val로 고정(split seed 0) → train 4.5k / val 0.5k, test 1k.
- Adam, lr: dynamics 3e-3, readout 1e-3. grad clip 1.0, batch 64, 20 epoch. test에는 val 최고 epoch를 씀(동률이면 이른 epoch).
- seed ∈ {0, 1, 2}. 같은 seed에서 real과 shuffle은 readout init과 batch 순서가 같음 (paired).
- 조건: `trained` (dynamics + readout 학습), `init` (dynamics를 init으로 고정, readout만 학습).
- GPU cuSPARSE라 bit-exact가 아님 → seed 반복으로 변동성을 보고.

## 판정 (P2-mini)
**Primary:** `trained` 조건에서 seed 평균 test acc, real − global_shuffle_s0.
- GO (P2-full + v2-B 명분): 차이 > 0 이고 seed 3쌍 모두 real > shuffle.
- STOP: 차이 ≤ 0.
- 그 외(평균은 양수인데 방향이 엇갈림): INCONCLUSIVE → seed를 늘리지 않고 그대로 보고.
보고용: init 조건 결과, 학습 이득(trained − init) real vs shuffle, test 이미지 paired bootstrap CI(seed 평균 정답 여부),
epoch별 gain/τ/bias 통계와 학습 후 rest 안정성.
