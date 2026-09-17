# P5: 초파리 도메인 시각 과제 (탐색적) — 2026-09-18

## Question
CIFAR 사물 분류에서는 frozen MaleCNS가 입력보다 나쁘다. **초파리가 실제로 처리하는 시각 과제**(움직임, looming)에서는 다른가?
사람이 만든 detector 없이 **망막 경로로만** 자극을 넣고, real / matched shuffle / 뇌 없음(눈 입력)을 비교한다.

## 방법 (`scripts/p5_flytask.py`)
- 자극은 합성이며 v1 eye model로 렌더링. trial마다 위치·크기·속도·대비·위상을 무작위화.
- dynamics: FlyVL-simple-v1 (P0-1에서 동결한 설정), 25 step, evoked = 자극 − blank의 시간 평균.
- readout: 뉴런 집합 → 고정 random projection(1024) → 표준화 → PCA(256) → multinomial logistic, seed 3개, 70/30 분할.
- 조건: real, matched_shuffle_s0, nobrain(= photoreceptor 입력 자체).

**중요:** feature가 시간 평균이므로, 방향처럼 시간 순서에만 담긴 정보는 눈 입력 단계에서 원리적으로 거의 읽을 수 없다.
따라서 "뇌 통과 후 성능 상승"은 재귀 회로가 시간 정보를 공간 패턴으로 변환했음을 뜻한다.

## 결과 (`results/p5/`)

### motion_dir — grating 이동 방향 4종 (chance 25%)

| 조건 | acc |
|---|---|
| nobrain (눈 입력) | 33.33 |
| real central_vnc | **99.44** |
| real visual_projection | 99.44 |
| real descending | 44.07 |
| matched_shuffle central_vnc | 96.11 |
| matched_shuffle visual_projection | 98.15 |
| matched_shuffle descending | 58.33 |

- **눈 입력 33% → 뇌 통과 96~99% (+63~66%p).** 시간 정보가 공간 패턴으로 변환됨.
- real − matched: central +3.3, VPN +1.3, **descending −14.3** (shuffle이 더 높음) → real 특이적이라고 할 수 없음.

### loom_vs_recede — 다가옴 vs 멀어짐 (chance 50%)

| 조건 | acc |
|---|---|
| nobrain (눈 입력) | 97.78 |
| real (모든 view) | 100.00 |
| matched_shuffle central/VPN | 100.00, descending 96.11 |

- 눈 모델의 시간 적응(x = lum − adapt) 때문에 밝아짐/어두워짐 부호만으로 풀린다 → **변별력 없는 과제**. 폐기.

### looming_side — 좌/우 판별 (chance 50%, n=120 예비)
모든 조건 100%, nobrain도 100% → 공간 위치는 시간 평균만으로 풀림. 폐기.

## 해석
1. **과제 의존성이 크다.** 같은 모델·같은 readout에서 사물 분류는 뇌 통과 시 손실(24.4→24.5, pixels 27.9보다 낮음),
   움직임 방향은 뇌 통과 시 대폭 이득(33→99).
2. **그 이득은 real 배선 특이적이지 않다.** block-matched shuffle도 96~98%. 재귀 네트워크의 일반적 성질로 보인다.
3. descending(1,314 뉴런)에서는 shuffle이 더 높다 → real의 깊은 경로가 방향 정보를 더 많이 버린다는 뜻일 수 있다
   (실제 초파리에서 DN은 방향 자체가 아니라 행동 명령을 싣는다는 점과는 일치하지만, 이 모델로 그것을 주장할 수는 없다).
4. 한계: seed 3개, n=600, 단일 shuffle 그래프, 합성 자극. real vs shuffle 차이(1~3%p)는 확정 불가.
