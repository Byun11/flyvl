# FlyVL — A Biological Connectome as a Visual Encoder

> 실제 초파리 성체 MaleCNS connectome을 이미지 인코더로 사용하고, 그 출력 표현을 최종적으로 LLM/VLM에 연결할 수 있는지 검증한다.

마지막 갱신: 2026-09-17. 실험별 사전 등록과 원자료는 `PROTOCOL*.md`, `results/`, 관련 연구는
[`docs/survey_fly_connectome_2026-09.md`](survey_fly_connectome_2026-09.md).

---

## 1. 아이디어

기존 VLM:
```text
Image → ViT/CLIP → visual tokens → projector → LLM
```
FlyVL 최종 목표:
```text
Image → fly visual input → measured Drosophila MaleCNS → fly neural representation
      → adapter / Q-Former / resampler → LLM
```
**ViT 대신 실제 생물의 뇌 배선도를 visual computation substrate로 사용한다.**

- 사람이 만든 detector("적이 왼쪽", "looming")로 초파리 대신 이미지를 해석하지 않는다. 빛·밝기 패턴만 넣고, 무엇을 읽을지는 connectome 내부 계산에 맡긴다.
- 초파리가 인간 semantic label을 안다고 가정하지 않는다. 질문은 **실제 생물학적 wiring이 학습 가능한 visual representation architecture로 기능하는가**이다.
- 모든 비교는 같은 조건의 **shuffled connectome**(그리고 필요하면 뇌 없는 모델)과 함께 한다.

### 사용하는 구조
- MaleCNS v1.0: 166,700 neurons, 25,582,938 directed connections (optic lobe + central brain + VNC).
- 고정: topology, edge 위치, synapse-count 기반 |w|(뉴런별 입력 정규화), E/I sign.
- 측정되지 않은 생리(gain, τ, bias 등)는 모델링하거나 학습한다.

### 관련 연구와 위치
- **FlyVis**: optic lobe 배선 고정 + type별 dynamics를 motion task로 학습 → FlyVL은 adult whole CNS를 general visual encoder로 쓰려 함.
- **BPU**: 유충 connectome 고정, 입력·출력 projection 학습, CIFAR-10 약 58%. 이미지가 초파리 눈으로 들어가지 않고 shuffle 대조군 보고 없음.
- **게임·RL 데모**: 주로 sensory/task state → connectome → action. image → CNS representation → LLM은 아님.

---

## 2. 연구 질문 로드맵

| Q | 질문 | 실험 | 상태 |
|---|---|---|---|
| Q1 | 학습 없는 실제 MaleCNS가 visual representation으로 유용한가? | V1 | **NO** (STOP) |
| Q2 | 배선 고정 + dynamics를 task-optimize하면 real > shuffle인가? | v2-A | **INCONCLUSIVE** (real ≈ shuffle) |
| Q3a | 입력 alignment(eye bypass)가 병목인가? | P2-B | 부분적 (+6~7%p), real ≈ shuffle |
| Q3b | 앞뒤 번역기를 학습시키면 뇌가 기여하는가? | P2-C (seed 0) | 뇌 없음 ≥ real |
| Q3c | Direct retinotopic 입력(추가 파라미터 0) | 미실행 | P1-mini에서 eye 샘플링 손실은 약 2%p뿐 |
| Q4 | 여러 shared fly encoder(swarm)로 해상도를 보완할 수 있는가? | 미실행 | CIFAR 32px에서는 patch가 너무 작음 → 고해상도 데이터 필요 |
| Q5 | Fly encoder가 CLIP semantic space에 정렬되는가? | 미실행 (v2-B) | |
| Q6 | connectome을 visual backbone으로 하는 VLM이 가능한가? | 미실행 | |

---

## 3. 실험 결과 요약

모든 CIFAR-10 mini 실험: train 5k 중 4.5k 학습 / 0.5k val, test 1k (P1-mini는 5k train / 1k test). 수치는 test acc %.

| 방법 | REAL | SHUFFLE | 뇌 없음 | real − shuffle [95% CI] |
|---|---:|---:|---:|---|
| V1: fixed eye + frozen CNS (LIF, grayscale, PCA probe, central_vnc K=1024) | 29.5 | 37.3 | – | −7.8 [−10.5, −4.7] |
| v2-A init: fixed eye, dynamics 고정(g=1), readout만 | 25.5 | 25.1 | – | +0.4 [−1.3, +2.1] |
| **v2-A: fixed eye + trained dynamics** | **27.4** | **26.9** | – | **+0.5 [−1.2, +2.4]** |
| P2-B: learned input(3072→L1–L5) + frozen CNS | 33.3 | 33.9 | – | −0.6 [−2.0, +0.9] |
| P2-C: learned front + back MLP (seed 0만) | 33.5 | 38.0 | 36.0 | −4.5 [−7.5, −1.1] |

참고 baseline (P1-mini 선형 probe): pixels 30.5, random projection+ReLU 35.1, photoreceptor 입력만 28.4.
shuffle = degree 보존 global shuffle (`global_shuffle_s0`), 모든 실험에서 같은 파일.

### V1 — FlyVL-simple-v1 (PROTOCOL.md)
- 모델: optic lobe graded + central/VNC LIF, noise off, 비시각 감각 입력 차단, KC→KC 0, gain은 합성 자극으로만 선택 후 동결.
- P0: 신호 전파·looming 경로·central 활성 PASS, T4/T5 방향선택성 FAIL(known limitation). P0-2는 CUDA 비결정성으로 1회 STOP → CPU deterministic backend로 재실행 PASS.
- P1-mini **STOP**: 15개 view×K 조합 전부 shuffle > real. real ≈ 입력(pixels/stim_only), shuffle ≈ random projection 이상.

### v2-A — task-optimized connectome (PROTOCOL_v2.md)
- 전 뉴런 graded rate, f(V)=0.5·tanh(2V). 학습: cell type별 g_pre, g_post(>0), τ(>dt), bias (47,524개, real/shuffle 동일).
- 공통 init g=1: real W spectral radius 1.0(ENS 2-뉴런 고립 루프) / optic lobe feedback 모드 약 0.91, shuffle 0.29. 더 큰 균일 gain은 real만 불안정.
- readout: central_vnc → 고정 random projection 1024 → BatchNorm(affine 없음) → Linear.
- 결과 **INCONCLUSIVE**: seed별 −0.1 / +0.8 / +0.8. 학습 이득 real +1.9, shuffle +1.8. gain·τ는 거의 안 움직이고 변화는 주로 bias. best val epoch 2~4.

### P2-B — eye bypass (PROTOCOL_v2B_eye_bypass.md, exploratory)
- RGB 3072 → 학습 Linear → L1–L5 8,884개 (primary), CNS g=1 고정, readout은 v2-A와 같음. 학습 파라미터 27.3M, real/shuffle 동일.
- real ≈ shuffle (seed 3개 모두 차이 ≤ 1.1%p). 입력 학습으로 약 +6~7%p. train acc 약 90% vs val 약 38% (강한 과적합).
- all_sensory(17,937개, seed 0): real 32.2 vs shuffle 31.6.

### P2-C — front + back translators + no-brain control (PROTOCOL_v2C_translators.md, seed 0)
- 앞 번역기(P2-B와 같음) + 뒤 번역기 MLP(1024→1024→10). 세 조건 파라미터 28.35M 동일.
- real 33.5 / shuffle 38.0 / 뇌 없음 36.0. real − 뇌 없음 −2.5 [−5.4, +0.9]. val은 세 조건 39.8~41.6으로 비슷.

### 현재까지 결론
CIFAR-10 정적 이미지 분류에서, 이 dynamics 모델들로는 **실제 MaleCNS 배선이 degree 보존 shuffle보다도, 뇌 없는 번역기보다도 나은 점이 관찰되지 않았다.**
BPU식 번역기로 오른 성능은 번역기가 만든 것으로 보인다. 같은 패턴이 Firefly(MaleCNS 위성 분류: real ≈ rewiring, 픽셀 우위) 등에서도 보고됨.

### 주의할 해석
- 대조군 선택: 관련 연구에서 real 우위는 **배선 비용·공간 제약을 맞춘 randomization** 대비에서만 나왔음. global shuffle은 장거리 연결을 무작위로 만들어 real에 불리할 수 있음 → matched / 배선 길이 보존 shuffle 필요.
- 과제 불일치: 초파리 시각은 motion, looming, optic flow, target tracking에 특화. 정적 사물 분류는 부자연스러운 과제.
- 모델 한계: v1은 T4/T5 방향선택성 미재현, v2는 g=1 고정 init에서 real central 반응이 shuffle보다 약 100배 작음.
- 규모: 5k mini, seed 3개(P2-C는 1개).

---

## 4. 앞으로의 계획 (원안)

### 입력 방식
- **A. Biological fly eye** (현재 메인): image → virtual fly eye → photoreceptors → MaleCNS.
- **B. Direct retinotopic input**: 학습 파라미터 추가 없이 이미지 공간 정보를 optic-lobe column에 직접 대응.
- **BPU-style learned input**은 메인 architecture가 아니라 병목 확인용 diagnostic (P2-B/P2-C에서 수행).

### Swarm Fly Encoder
```text
IMAGE → patch decomposition → patch 1..N → same MaleCNS (shared weights) × N → state 1..N
      → aggregator → visual tokens → LLM
```
- 파라미터는 약 fly 1마리, 계산량은 N마리 (convolution과 비슷한 weight sharing).
- scaling: 1 → 4(2×2) → 16(4×4) → 64(8×8) → 100.
- 장점: 대규모 학습 입력층 없이 "같은 biological 연산의 공간 반복"으로 설명 가능.
- 주의: CIFAR 32px를 8×8로 나누면 patch가 4×4 px → STL-10 / ImageNet 등 고해상도에서 의미 있음.
  학습 비용은 patch 수에 비례 (v2-A 기준 64마리면 epoch당 약 17분 추정).
  aggregator는 학습하므로 single fly / swarm / shuffled swarm에 같은 aggregator를 붙여 비교해야 함.

### CLIP distillation (v2-B)
```text
teacher: image → CLIP → z_clip
student: image → Fly CNS (topology 고정, dynamics 학습) → projector → z_fly
loss: MSE(z_fly, z_clip) + cosine
```
질문: connectome-constrained biological encoder가 기존 semantic visual space에 정렬될 수 있는가?

### VLM 단계
```text
Image → Fly CNS (또는 Swarm) → Q-Former / Perceiver → visual tokens → frozen LLM
```

### 실행 순서 (원안)
1. v2-A 완료 및 판정 — **완료 (INCONCLUSIVE)**
2. v1/v2 기록 고정 — **완료**
3. Direct retinotopic input 검토
4. 4/16/64-fly patch swarm
5. 유망 architecture 선택 → CLIP distillation → Q-Former/Perceiver로 LLM 연결
6. STL / ImageNet subset 확장 → vision-language evaluation

결과를 반영한 추가 후보:
- **matched shuffle / 배선 비용 보존 shuffle** 대조군으로 v2-A 또는 P2-B 재비교.
- 초파리에게 자연스러운 시각 과제(looming, 움직임, 방향 전환)에서 real / shuffle / 뇌 없음 비교.

---

## 5. 성공 기준
- A: trained real MaleCNS > matched shuffled MaleCNS → biological topology가 유용한 inductive bias.
- B: single fly < multi-fly swarm → biological visual operator의 patch-wise 반복이라는 새 architecture.
- C: fly encoder → CLIP semantic space 정렬 가능.
- D: fly encoder → visual tokens → LLM → image understanding (진짜 FlyVL).

## 6. 주장하지 않는 것
- 실제 초파리의 주관적 시각 복원, simulator = 실제 뇌, connectome만으로 cognition 재현,
  Fly encoder가 ViT보다 효율적, 초파리가 object category를 인간처럼 이해함.

정확한 주장: **measured biological wiring을 계산 구조로 사용한 visual representation architecture를 구축하고 검증한다.**

> Biological brains are not simulated merely as agents; they are used as computational modules inside an artificial vision system.
