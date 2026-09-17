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
