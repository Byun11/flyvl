# P9 — 병목 가설 검증 (2026-09-23 시작)

목표: real 배선의 사물 분류 결손이 눈→중추 병목 때문인지 인과적으로 검증 → 논문의 메커니즘 절.
프로토콜: `PROTOCOL_P9_bottleneck.md` (사전 등록). 러너: `scripts/p9_run_all.sh`. 로그: `D:/flyvl_data/runs/p9/`.

- [x] 환경: venv `~/venvs/flyvl` (torch 2.14+cu130, RTX 4090), 데이터 `D:/flyvl_data` (sha256 일치)
- [x] 기본 대조군 그래프 11개 (중추 재배선이 느려 약 3시간)
- [x] `controls.py`에 `frac` scope 추가, `p9_bottleneck.py` 작성, 프로토콜 등록
- [x] frac 그래프 8개 (f = 0.001/0.01/0.1/0.3 × seed 0,1)
- [x] 진단: P-B 통과(f=0.01에서 hop 격차 58% 감소). P-A 절반 — hop 순서는 CIFAR와 일치, 반응비는 matched(0.15) < real(0.22)로 불일치
- [x] 재현 확인: CIFAR 정확히 재현(32.10). 움직임 real 80.78 vs 78.04 — 원인 확정: 뇌를 안 거치는 눈 기준선도 42.44→40.00으로 바뀜 = 입력 잡음 실현값(새 torch RNG) 차이, 뇌 시뮬레이션 아님. real−matched_s0 = +14.6 (기존 범위 +14~19 안). 모든 P9 비교는 이 머신 안에서만.
- [x] 용량-반응 CIFAR → P-C: visual_projection **기각**(f=0.01 회복 21%, f=0.3에서도 45% — hop은 95% 회복), central_vnc **부분 지지**(38% / f=0.1에서 57%). 경로 길이는 결손의 절반 이하만 설명
- [x] 용량-반응 움직임 → P-D: central f=0.1 |d|/SD 2.8 (seed 2 추가 후) = 교환관계, 단 비대칭(CIFAR 격차 59% 회복 vs 움직임 격차 17% 손실)
- [x] 판정 P-A(절반) / P-B(통과) / P-C(VPN 기각, central 부분) / P-D(교환관계)
- [x] FINDINGS.md 결과 5 추가
- [ ] (다음) P8 재실험: 세포 타입 공유 파라미터

## 리뷰

- P9 완료 (2026-09-24 05:12). 핵심: 경로 길이는 CIFAR 결손의 절반만 설명, 움직임은 엣지 대부분의 구체적 배치에 의존.
- 다음: 트랙 2 (시각 기반 비행 RL) — flybody 설치 승인 대기.
