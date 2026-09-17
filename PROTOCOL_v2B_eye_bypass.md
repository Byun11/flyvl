# P2-B: learned sensory input / eye-bypass baseline (exploratory, pre-registered before any run)

v2-A(P2-mini) 판정과 섞지 않는 별도 탐색 실험. v2-A 실행·protocol·결과는 건드리지 않음.

## Question
> 성체 MaleCNS에서 virtual fly-eye를 제거하고 이미지→CNS 입력 자체를 학습시키면 CIFAR-10 성능이 얼마나 오르는가?
> (A) eye mapping이 병목인가, (B) 그 조건에서 real wiring이 shuffle보다 나은가.

Unlike BPU, which uses a larval connectome, this baseline uses the adult MaleCNS; the learned-input condition
intentionally removes the biological eye interface to isolate whether sensory alignment is the main bottleneck.

## Model (`flyvl/v2b.py`, `scripts/p2b_train.py`)
- 입력: CIFAR-10 RGB 32×32×3 = 3,072, [0,1]로 스케일 후 0.5를 빼서 중심화. 사용하지 않는 것: hex column sampling,
  ommatidia mapping, R1-6 입력, 4-way drift, photoreceptor temporal stimulus. 이미지는 25 step 동안 일정한 입력.
- Input projection: Linear(3072 → N_entry), bias 없음, PyTorch 기본 init, **학습함**.
- Entry population (neuron ID로 정의, 모든 graph에서 동일):
  - `visual_entry` (**primary**): lamina monopolar L1, L2, L3, L4, L5 = 8,884 neurons (R1-6의 직접 target,
    전부 ol_intrinsic, graded). photoreceptor 제외. 파라미터 27,291,648개.
  - `all_sensory` (diagnostic, seed 0만): superclass에 "sensory"가 들어간 전체 = 17,937 neurons
    (vnc 6,370 / ol 6,098 / cb 4,868 / 기타 601). 파라미터 55,102,464개. biologically realistic vision을 주장하지 않음.
- CNS: FlyVL-v2 dynamics를 **공통 init으로 고정함** (g_pre = g_post = 1, τ = 40 ms, bias = 0, f = 0.5·tanh(2V)).
  v2-A에서 학습한 파라미터는 재사용하지 않음. bias 0에 입력 0이면 V = 0이 정확한 rest라서 warm-up·blank branch 없음.
- Graph: v1 effective real graph와 `global_shuffle_s0` (같은 파일).
- Readout (v2-A와 동일): central_vnc 59,064 뉴런의 시간 평균 f(V) → 같은 고정 random projection(seed 0, 1024)
  → BatchNorm1d(affine 없음) → Linear(1024→10). 파라미터 10,250개.
- 학습 파라미터 합계: visual_entry 27,301,898 / all_sensory 55,112,714. real/shuffle 동일 (assert, meta.json 기록).

## 학습 (v2-A와 동일한 규칙)
- train 5k 중 class별 50장을 val로 고정(split seed 0) → 4.5k / 0.5k, test 1k (P1-mini, P2-mini와 같은 이미지).
- Adam lr 1e-3 (input projection과 readout 공통), grad clip 1.0, batch 64, 20 epoch, batch 순서 seed×1000+epoch.
- seed 0, 1, 2. 같은 seed에서 real과 shuffle은 input projection·classifier 초기값이 같음 (init 해시 기록, 스모크 테스트에서 일치 확인).
- test는 val 최고 epoch(동률이면 이른 epoch)로 1회 평가.
- 실행 순서: visual_entry seed 0 (real, shuffle) → seed 1 → seed 2 → all_sensory seed 0 (real, shuffle).
  v2-A와 GPU를 공유함 (각각 약 2.6 GB / 9.4 GB).

## 결과를 본 뒤 바꾸지 않는 것
input neuron set, projection width, epoch 수, normalization, gain, dynamics, readout 구조, lr.

## 스모크 테스트 (train 이미지 100장, 결과 확인 전)
- 두 모드, 두 graph 모두 finite. visual_entry 64장 × 8 step에서 loss 2.43→0.06 (real), 2.56→0.03 (shuffle),
  gradient 유한, 0.49 s/step, peak 2.5 GiB.
- label과 무관한 관찰: 같은 초기 projection에서 central |f| 평균이 real 9e-5 vs shuffle 7e-3
  (central 유닛 중 >1e-3 비율: real 2.3% vs shuffle 93%). all_sensory는 real 1.5e-3 vs shuffle 2.8e-3.
  readout BatchNorm이 scale 차이를 보정하지만, real에서 lamina 입력이 central까지 약하게 전달되는 상태로 실험함.
- 파라미터 수가 4.5k train 이미지에 비해 매우 많아 과적합 가능성이 큼 (설계대로 유지, best-val 선택만 사용).

## 보고 (판정 규칙 없음 — exploratory)
| Method | REAL | SHUFFLE |
|---|---|---|
| Fixed fly-eye + frozen CNS (P1-mini, grayscale linear probe, central_vnc K=1024) | 29.5 | 37.3 |
| Fixed fly-eye + trained dynamics (v2-A trained, seed 평균) | v2-A | v2-A |
| **Learned input + frozen CNS (P2-B visual_entry, seed 평균)** | new | new |

첫 줄은 readout(PCA + L2 logistic)과 입력(grayscale)이 달라서 참고용임.
해석: (A) learned-input real이 eye-fixed real보다 크게 높으면 sensory interface가 주요 병목.
(B) learned-input real vs shuffle: seed 3개 모두 real > shuffle → real wiring 이점 가능성, 비슷 → connectome 특이적
이점 없음, real < shuffle → generic classification에 불리. all_sensory는 표 아래 diagnostic으로만 보고.

## Results — 2026-09-17 (`results/p2b_mini/`)
visual_entry (primary), test acc %:

| seed | REAL | SHUFFLE | real − shuffle |
|---|---|---|---|
| 0 | 33.3 | 33.8 | −0.5 |
| 1 | 32.4 | 33.5 | −1.1 |
| 2 | 34.3 | 34.4 | −0.1 |
| mean | **33.3** | **33.9** | **−0.6 [95% CI −2.0, +0.9]** |

(CI: test 이미지별 정답 여부를 seed 평균한 뒤 paired bootstrap 1,000회.)
all_sensory (diagnostic, seed 0): REAL 32.2, SHUFFLE 31.6, 차이 +0.6 [−1.4, +2.5].
best epoch는 대부분 5~7 (train acc는 약 90%까지 오름, 강한 과적합).

해석 (사전 등록 기준):
- (A) learned input은 fixed eye 대비 real 기준 약 +6~7%p (v2-A trained real 26.7/28.4, P1-mini frozen 29.5와 비교;
  readout·입력 조건이 달라 대략적인 비교). sensory interface가 성능을 일부 제한하지만, 이 설정에서 BPU의
  약 58%에는 한참 못 미침.
- (B) real ≈ shuffle: seed 3개 모두 차이가 1.1%p 이하이고 CI가 0을 포함함 → connectome 특이적 이점은 관찰되지 않음.
- entry를 L1–L5에서 전체 sensory(2배)로 늘려도 이점이 없었음.
