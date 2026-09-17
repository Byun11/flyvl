# P2-C: BPU-style front + back translators with a no-brain control (exploratory, small; registered before running)

## Question
앞(입력)·뒤(출력) 번역기를 모두 학습시켰을 때, 고정된 MaleCNS가 번역기만 있는 경우보다 무언가 기여하는가?
그리고 real이 shuffle보다 나은가?

## Conditions (`flyvl/v2c.py`, `scripts/p2c_train.py`)
- front (학습): RGB 3072 (−0.5 중심화) → Linear(no bias) → 8,884 (L1–L5 entry, P2-B와 같음)
- body:
  - `real`, `global_shuffle_s0`: FlyVL-v2 dynamics를 공통 init(g=1)으로 고정, 25 step, central_vnc 시간 평균 f(V) (P2-B와 같음)
  - `nobrain`: connectome 없이 f(front output), f = 0.5·tanh(2·)
- back (학습 MLP): 고정 Gaussian random projection → 1024 (seed 0) → BatchNorm(affine 없음) → Linear 1024→1024 → ReLU → Linear 1024→10
- 학습 파라미터: 모든 조건 28,351,498개 (front 27,291,648 + back 1,059,850). 같은 seed에서 init 해시 동일 (스모크 테스트 확인).

## Training (P2-B와 동일)
5k mini (train 4.5k / val 0.5k, split seed 0), test 1k, Adam lr 1e-3, clip 1.0, batch 64, 20 epoch, best val → test.
**seed 0만** (small run). 3 조건 순서: real → global_shuffle_s0 → nobrain.

## Report (판정 규칙 없음)
test acc 3개와 paired bootstrap CI (real − shuffle, real − nobrain). nobrain ≈ real이면 뇌가 기여하지 않는 것으로 해석.
seed 1개 소규모 실험이므로 결론은 방향성 확인에 한정.
