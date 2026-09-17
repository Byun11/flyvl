# MaleCNS 논문 기준: 공개 범위 → 가능/불가능 → 설계 원칙 (2026-09-17)

원칙: **논문과 공식 공개 데이터에 있는 것만 "사실"로 쓰고, 없는 것은 "가정"이라고 명시한다.**

출처
- [S1] MaleCNS 본 논문: *Sexual dimorphism in the complete connectome of the Drosophila male CNS* —
  [bioRxiv 2025.10.09.680999](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v1) / [Cell 2026](https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6)
- [S2] *The organization of visual pathways in the Drosophila brain* —
  [bioRxiv 2025.12.22.696097](https://www.biorxiv.org/content/10.64898/2025.12.22.696097v2.full) / [Cell 2026](https://www.cell.com/cell/fulltext/S0092-8674(26)00941-4)
- [S3] Nern et al., *Connectome-driven neural inventory of a complete visual system*, Nature 2025 (오른쪽 optic lobe) —
  [PMC11042306](https://pmc.ncbi.nlm.nih.gov/articles/PMC11042306/)
- [D] 공식 다운로드 목록 [male-cns.janelia.org/download](https://male-cns.janelia.org/download/)

---

## 1. 공개된 것 (사실로 사용 가능)

| 항목 | 내용 | 출처 |
|---|---|---|
| 범위 | 중추뇌 + 양쪽 optic lobe + VNC, 목 연결 포함. 8 nm isotropic, 160 teravoxel | S1 |
| 뉴런 | 166,691개 (sensory 축삭 포함), cell type 11,691개. optic lobe가 전체 세포의 약 64% | S1 |
| 시냅스 | presynapse 4,600만, PSD 3.12억. 연결 precision / recall 0.82 / 0.81 | S1 |
| proofreading | nucleus 연관 98.9%. neuropil 내 pre 94% / post 42% / 양쪽 모두 40.1% | S1 |
| 연결 가중치 | segment 간 synapse count (`connectome-weights-...-minconf-0.5.feather`, 1.1 GB) | D |
| 시냅스 위치 | pre/post 좌표·body ID·ROI (`syn-points` 12.7 GB, `syn-partners` 6.8 GB) | D |
| 신경전달물질 | 뉴런별 집계 예측 + **pre-synapse별 확률** (`tbar-neurotransmitters` 2.7 GB). 7종(ACh, Glu, GABA, His, DA, OA, 5-HT), VGG형 CNN, 학습에 없던 79 type에서 EASI-FISH와 높은 일치 | D, S3 |
| 주석 | class, type, side, superclass 등 | D |
| 형태 | skeleton(SWC 등), EM 영상, segmentation, neuropil ROI | D |
| retinotopy | medulla column 육각 좌표. "렌즈와 1:1 대응(가장자리 일부 제외)", eye equator 표시, lobula plate column은 T4→Mi1로 유도 | S3 (+ `optic-columns.xlsx`, supplement) |
| 시각 입력 기준점 | 분석 seed = **L1, L2, L3, R7, R8, R7d, R8d, HB eyelet** (R1-6은 lamina에서 여러 렌즈 신호가 합쳐지므로 L1–L3에서 시작) | S2 |
| 구조적 시각 도달 추정 | **VIC** = 시각 입력에서 목표 뉴런까지 모든 경로의 입력 정규화 시냅스 곱의 합. 좌우 대칭 type 점수 일관성으로 threshold | S2 |
| VIC 결과 | 오른쪽 optic lobe 뉴런 약 5.2만 개 / 약 700 type. VPN 약 350 type (약 4,500 cell). central brain **약 절반(>5천 type, 약 1.1만 뉴런)**이 오른쪽 optic lobe에서 상당한 시각 입력을 받음. 이들은 계층 3~5층. 연결의 **55%가 측면·되먹임** | S2, S3 |
| 검증 자료 | split-GAL4 577 line ↔ >300 EM type, optic lobe type 98%가 FlyWire-FAFB와 대응 | S3 |

## 2. 공개되지 않았거나 불완전한 것

| 항목 | 상태 | 출처 |
|---|---|---|
| **망막(eye)** | 볼륨 밖. photoreceptor는 세포체 없이 들어오는 축삭만 있음 | S1, S3 |
| **R1-6** | 볼륨 가장자리 artefact로 **일부 재구성 실패** ("some R1-6 photoreceptor neurons in the laminae") | S1 |
| **lamina** | 재구성이 "particularly difficult", 완료 지표가 "considerably worse" (optic lobe 논문 기준) | S3 |
| **전기 시냅스 (gap junction)** | 데이터·논문 모두 언급 없음 (EM 화학 시냅스만) | S1, S2, S3 |
| **시냅스 강도 / 생리** | synapse count만 있음. 실제 전달 강도, 시간상수, 막 특성, 발화 특성 없음 | S1, D |
| **활동 데이터** | 공개 데이터에 없음. S2는 외부 칼슘 이미징 연구를 검증에만 인용 | S1, S2 |
| **신경전달물질의 효과 부호** | 전달물질 종류는 예측이지만 수용체(흥분/억제)는 없음. S2는 분석 편의상 ACh = 흥분, GABA·Glu = 억제로 가정 | S2 |
| **neuromodulation, 가소성** | 없음 | – |
| **postsynaptic 완전성** | neuropil 내 42%. 연결 중 양쪽 모두 proofread된 것은 40.1% | S1 |
| 우리 사용 파일 주의 | flybrain prebuilt `weights.npz`는 His도 억제로 처리하고 뉴런별 입력 정규화를 함 (S2의 VIC 정규화와 같은 방향, 부호 규칙은 다름) | 코드 확인 |

## 3. 그래서 가능한 것 / 불가능한 것

### 가능 (데이터로 뒷받침됨)
1. **구조 기반 계산**: 경로, hop 수, VIC 같은 선형 전파 점수, 영역 간 흐름, 계층. S2가 같은 방법을 씀 → 우리 파이프라인을 **S2 수치 재현으로 검증** 가능 (VPN 약 350 type, VCBN 약 1.1만 뉴런).
2. **논문 기준 입력 경로**: L1–L3(+R7/R8)을 입력으로, medulla column 육각 좌표로 **retinotopic 매핑**. 파라미터 추가 없이 이미지 → column → 뉴런 대응 가능.
3. **논문 기준 readout 집합**: VIC threshold를 넘는 central brain 뉴런(VCBN)만 읽기.
4. **공간·배선 비용 보존 대조군**: `syn-points`(시냅스 좌표)와 skeleton으로 연결 거리를 계산 → 거리 분포·배선 예산 보존 rewiring 가능. superclass×side 블록 보존 rewiring도 가능.
5. **부호 불확실성 반영**: synapse별 전달물질 확률을 쓰거나, 부호 규칙을 바꿔 결과 안정성 확인.
6. **dynamics를 가정으로 두고 비교**: 같은 가정을 real/대조군에 동일 적용하는 **상대 비교**는 가능.

### 불가능 또는 주장하면 안 되는 것
1. **실제 초파리가 무엇을 "보는지", 신호가 얼마나 세게 가는지** — 활동·생리 데이터 없음.
2. **망막·lamina 단계의 정확한 시뮬레이션** — 망막 없음, R1-6 일부 누락, lamina 불완전, gap junction 없음.
3. **"시뮬레이터 = 실제 뇌"**, **절대 성능을 생물학적 능력으로 해석** — dynamics는 전부 가정.
4. **신경조절·학습(가소성) 기반 행동 재현** — 데이터 없음.
5. **synapse count = 시냅스 강도**라는 주장 — 근거 없음 (proofreading 불완전 + 생리 부재).

## 4. 설계 원칙 (FlyVL에 적용)

| # | 원칙 | 이전 실험과 차이 |
|---|---|---|
| P1 | **입력은 논문의 시각 입력 seed 사용**: L1–L3 (+R7/R8), R1-6 직접 구동은 쓰지 않음 | V1/v2-A는 R1-6 구동 |
| P2 | **입력 매핑은 논문 retinotopy**: 이미지 → 육각 column 좌표 → 같은 column의 L1–L3. 학습 파라미터 0 | P2-B는 2,700만 개 학습 projection |
| P3 | **readout은 VIC 기반 VCBN** (논문 방법 재현, 좌우 모두 계산) | central_vnc 5.9만 개 전체 |
| P4 | **dynamics 가정을 최소화한 기준 모델을 먼저**: VIC와 같은 입력 정규화 선형 전파(가정 = 선형성만) → 그 다음에 비선형·학습 dynamics | 처음부터 LIF/tanh + gain 가정 |
| P5 | **대조군 3종 필수**: degree 보존 global shuffle, superclass×side 블록 보존 shuffle, **거리·배선 예산 보존 shuffle**. 추가로 뇌 없음(retinotopic 입력 직결) | global shuffle만 (P2-C만 뇌 없음) |
| P6 | **파이프라인 검증을 먼저**: S2 수치(VPN type 수, VCBN 약 1.1만)를 재현하지 못하면 이후 실험 중단 | 검증 기준 없음 |
| P7 | **부호 규칙 민감도**: S2 규칙(ACh+, GABA/Glu−)과 flybrain 규칙(His− 포함) 둘 다 | flybrain 규칙만 |
| P8 | 결과 문장은 "이 가정 아래에서 real vs 대조군"으로만 기술 | – |

## 5. 설계 초안: P3 (paper-grounded)

```text
P3-0  데이터 감사
      - 공식 feather에서 effective graph 재구성 (flybrain 파일과 edge/부호 대조)
      - 필요 시 syn-points(12.7 GB) 다운로드 → 연결 거리 계산
P3-1  S2 재현 (gate)
      - L1–L3/R7/R8 seed에서 VIC 계산, threshold 방법 재현
      - VPN 약 350 type, VCBN 약 1.1만 뉴런 / >5천 type 수준인지 확인
P3-2  retinotopic 입력 (파라미터 0)
      - 이미지 → hex column → 같은 column의 L1, L2, L3
P3-3  encoder
      (a) 선형 VIC 전파 → VCBN 상태
      (b) (a)가 의미 있으면) 비선형·학습 dynamics
P3-4  비교 (같은 readout, 같은 데이터)
      real | global shuffle | block shuffle | 거리 보존 shuffle | 뇌 없음
      + readout: VCBN → 고정 projection → BN → Linear
P3-5  과제
      - CIFAR-10 mini (이전 결과와 연결)
      - (선택) 초파리 도메인 합성 과제: looming 방향, 움직임 방향, 물체 위치
```

## 6. 결정 필요
1. `syn-points` 12.7 GB 다운로드 (거리 보존 shuffle에 필요). skeleton 기반 soma/중심 거리로 대체할지.
2. P3-5 과제: CIFAR만 할지, 초파리 도메인 합성 과제를 primary로 둘지.
3. P3-3 (a) 선형 전파만 먼저 끝까지 볼지, (b) 비선형까지 한 번에 진행할지.
