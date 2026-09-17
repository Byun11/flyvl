# 초파리 connectome 화제작·관련연구 서베이 (2026-09-17)

MaleCNS v1.0 공개(2026-09-03, Google Research + HHMI Janelia) 이후 뉴스·SNS에 뜬 프로젝트와 관련 연구가
**어떻게 학습하고 어떻게 사용했는지** 정리. FlyVL 결과(V1, v2-A, P2-B, P2-C)와의 연결점 포함.

## 요약

- 바이럴 데모 대부분은 **사람이 만든 입력 인코더 + 학습된 출력층**이 핵심이고, 섞은(shuffled) 뇌와 비교한 곳은 거의 없음.
- 대조군을 제대로 돌린 프로젝트는 대부분 **real ≈ rewired/random**, 그리고 **픽셀 baseline이 더 높음** → FlyVL 결과와 같은 패턴.
- real이 이긴 드문 결과(Correig-Fraga et al.)는 **배선 비용·공간 제약을 맞춘 randomization** 대비였음.
  FlyVL의 global shuffle은 그 제약을 맞추지 않은 대조군임.

## 1. 바이럴 게임·데모

출처: [404 Media](https://www.404media.co/a-digital-fly-brain-has-taken-over-the-internet/),
[Gizmodo](https://gizmodo.com/google-mapped-a-fruit-flys-brain-now-its-playing-doom-and-super-mario-64-2000808616),
[Know Your Meme](https://knowyourmeme.com/memes/fly-brain-simulations-fruit-fly-brain-mapped),
목록 [awesome-fly](https://github.com/cobanov/awesome-fly).
Minecraft(9/7 영상), Beat Saber(9/8 X 게시물), Doom, Super Mario 64, Rubik's cube, 트레이딩 등.

| 유형 | 예 | 입력 | 학습 | 출력 | 대조군 |
|---|---|---|---|---|---|
| 고정 뇌 + 손 매핑 | Minecraft, [Fly64](https://github.com/ornata/fly), desktop-fly, [fly.ai](https://github.com/alextitonis/fly.ai) | 게임 상태/화면 → 사람이 설계한 feature → 특정 뉴런(LC4, LPLC2, LC10a 등) 주입 | 없음 | descending neuron → 키 입력(손 매핑) | 없음 |
| 고정 뇌 + 학습 readout | [Fly Dino](https://github.com/cobanov/flyjump), [FLYT3](https://github.com/seanphan/flyt3), [fly-craftax](https://github.com/liuzihe02/fly-craftax), Fly Lab | 게임 feature, 보드 → photoreceptor 패치 | readout만 CEM / REINFORCE / PPO | 학습 선형층·MLP | 일부 |
| "도파민 학습" | Doom(Alex Wormuth), [FlyPong](https://github.com/jonatasperaza/FlyPong), Stonkfly | 프레임 → sensory 뉴런 | 피격 시 PPL1 도파민 뉴런 자극 | 활동 → 컨트롤 | 학습 효과 미입증 (FlyPong은 음성 결과로 명시) |
| LM 결합 | [FLM](https://github.com/nftechie/flm) | – | adapter만 | 사전학습 LM | "언어 능력은 LM에서 나옴"으로 명시 |
| 비교 프레임워크 | [FlyDoom](https://github.com/eganeganegan/flydoom) | fly-inspired feature(휘도, 대비, 움직임, looming) 또는 CNN → sensory 노드 | PPO, 3-factor, readout-only | descending 노드 → policy/value head | ER random, degree-preserving rewiring, MLP/GRU/LSTM (README에 수치 없음) |

사례: **Fly Dino** — 뉴런 80개 subset에 게임 feature 8개 입력, 16→12→3 readout 243개 파라미터를 CEM으로 학습,
held-out 99/100 코스 통과. 뇌 침묵 0/100, 랜덤 readout 0/100. **shuffle 뇌와 비교는 없음.**

## 2. 연구급 체현 모델

| 프로젝트 | 방식 | 핵심 |
|---|---|---|
| [Eon Systems](https://eon.systems/updates/embodied-brain-emulation) | FlyWire 중추뇌 LIF(약 14만 뉴런, 고정) + NeuroMechFly v2 / MuJoCo | 미각·촉각 입력 → DNa01/02(조향), oDN1(전진), MN9(섭식) → **imitation learning으로 학습한 운동 컨트롤러**가 관절 제어. 시각(Lappalainen motion pathway)은 연결돼 있지만 "행동에 실질적 영향 없음"으로 명시. looming 도주는 미구현 |
| [FlyGM, arXiv 2602.17997](https://arxiv.org/abs/2602.17997) | 성체 whole-brain connectome을 그래프 정책망으로, deep RL | 그래프/비그래프 baseline 대비 sample efficiency 우위 보고. 초록에는 shuffle 대조군 언급 없음 |
| [FlyVis, Nature 2024](https://www.nature.com/articles/s41586-024-07939-3) | optic lobe 배선 고정, cell-type 파라미터를 optic flow로 학습 | T4/T5 방향선택성 재현. FlyVL v2-A가 참고한 접근 |

## 3. 이미지 분류 — FlyVL과 가장 가까운 연구

| 연구 | 설정 | 결과 |
|---|---|---|
| [BPU, arXiv 2507.10951](https://arxiv.org/abs/2507.10951) | **유충** 뇌 약 3,000 뉴런(sensory 430 / internal 2,304 / output 218), 부호 있는 connectome weight, ReLU 재귀 dynamics 고정. 입력(3072→sensory)과 출력(output 뉴런→class) projection만 학습 | CIFAR-10 **58%**, MNIST 98%. size-matched MLP 52%. connectome 2배 확장 시 상승. **shuffle 대조군은 보고하지 않음** |
| [Firefly (PixelML)](https://github.com/PixelML/firefly) | **MaleCNS** 고정 LIF 48 tick, 위성 타일 99패치 → sensory 99개, 하류 뉴런 1,024개 spike → 작은 MLP | real AUC 0.622 ≈ degree-preserving rewiring 0.625 ≈ 억제 제거 0.642 ≈ nnz 맞춘 랜덤 희소행렬 0.632. **pooled pixels 0.711이 더 높음** |
| [FLYT3](https://github.com/seanphan/flyt3) | MaleCNS 고정, descending/VNC readout REINFORCE. 위성 피해 5클래스 | 60.8% vs **픽셀 MLP 85.7%** |
| [NeuroTerrarium](https://github.com/5p00kyy/neuroterrarium) | giant-fiber 회로 51 body | real과 rewiring 5종이 동일 반응 (음성 결과) |
| [Correig-Fraga, Guimerà, Sales-Pardo (train-your-fly / connectome)](https://github.com/eudald-seeslab/connectome) | FlyWire + 해부학적 겹눈 모델(약 8천 photoreceptor), 3-step message passing, **synapse별 gain(tanh) + Kenyon cell 선형 readout** 학습. 색·모양·개수 과제 | **배선 비용을 맞춘 조건에서 real이 가장 정확**. 공간 제약 없는 rewiring은 배선 예산을 늘리거나 장거리 연결을 늘려야만 real을 넘음. randomization 4종: out-degree만 보존, 예산까지 pruning, 연결 길이 분포 보존, 뉴런별 길이 분포 보존 |

## 4. FlyVL과의 연결

1. **FlyVL 결과는 분야에서 예외가 아님.** V1(real < global shuffle), P2-B(real ≈ shuffle), P2-C(뇌 없음 ≥ real)는
   Firefly(MaleCNS, rewiring ≈ real, 픽셀 우위)와 같은 패턴.
2. **BPU 58% vs FlyVL 33~38%:** 뇌(유충 3천 개 vs 성체 16.7만 개), dynamics(ReLU vs tanh graded, g=1 고정),
   데이터(50k vs 5k mini)가 다름. BPU는 shuffle과 비교하지 않아서 "connectome 덕분"이라는 근거는 없음.
3. **대조군 선택이 결론을 좌우할 수 있음.** real 우위는 공간·배선 비용 제약을 맞춘 randomization 대비에서만 나왔음.
   FlyVL global shuffle은 장거리 연결을 무작위로 만들어 spectral radius가 0.29(real 1.0 / optic lobe 모드 약 0.91)로
   낮고 central까지 신호가 더 잘 퍼짐 → real에 불리한 비교일 가능성.
   후보: `controls.py`의 `matched_shuffle`(superclass × side 블록 보존), 연결 길이·배선 예산 보존 shuffle.
4. 체현 데모(Eon 등)의 성공은 주로 **학습된 운동 컨트롤러 + 비시각 감각** 경로에서 나왔고,
   시각이 행동을 이끄는 경우는 대부분 사람이 만든 detector를 거침.

## 확인하지 못한 것
- FlyDoom, fly-craftax, FlyGM의 rewired/random 대조군 수치(README·초록에 없음).
- BPU의 학습 epoch 수, shuffle/random graph 결과(본문 추출 범위에서 확인 못함).
- Correig-Fraga et al. 논문 본문의 정확한 정확도 수치.

## 추가 (2026-09-18): 대조군 관련 논문

| 논문 | 설정 | 결과 |
|---|---|---|
| [Topological Sensitivity in Connectome-Constrained Neural Networks, arXiv 2604.04033](https://arxiv.org/html/2604.04033v1) | FlyVis 네트워크(45,669 노드, 1.5M edge, 파라미터 734), MovingEdge 방향 decoding. connectome vs self-loop 맞춘 naive random vs degree-preserving rewire | 체크포인트 init에서는 connectome이 naive random보다 loss 낮음(0.514 vs 0.698). **같은 random init을 쓰면 차이가 사라지고(0.513 vs 0.515), degree-preserving 대비 0.516 vs 0.516.** 결론: "apparent topology advantages ... do not robustly persist under degree-preserving controls" |
| [FlyGM, arXiv 2602.17997](https://arxiv.org/html/2602.17997) | 성체 whole-brain connectome 위상·weight 고정, 뉴런별 descriptor + encoder/gate/decoder/공유 MLP 학습, imitation + RL로 보행·비행 | 가장 어려운 조건 각도 오차: FlyGM 8.29°, **degree-preserving rewire 13.55°**, ER random 125.36°, MLP 13.90°. 학습 수렴도 FlyGM이 빠름 → **제어 과제에서는 degree 보존 대조군 대비 real 우위 보고** |
| [Reproducibility and model-selection stability in connectome-constrained circuit modeling, bioRxiv 2026.04.18](https://www.biorxiv.org/content/10.64898/2026.04.18.717873v1) | FlyVis 계열 모델 앙상블 재학습 | 실험 응답과의 대응이 재학습마다 얼마나 안정적인지 검토 (상세 미확인) |
| [Connectome analysis reveals brainwide visual processing in Drosophila, bioRxiv 2026.02.02](https://www.biorxiv.org/content/10.64898/2026.02.02.700492v1) | whole-brain connectome, optic lobe ↔ central brain 장거리 투사 분석 | 구조 분석 (상세 미확인) |

요약: 대조군을 엄격히 하면 **지각(분류·motion decoding)에서는 real 우위가 약하거나 사라지고**,
**체현 제어(FlyGM)에서는 degree 보존 대조군 대비 우위가 보고됨**. 정적 이미지 분류에서의 FlyVL 결과와 방향이 맞음.
