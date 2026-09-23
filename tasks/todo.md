# P9 — 병목 가설 검증 (2026-09-23 시작)

목표: real 배선의 사물 분류 결손이 눈→중추 병목 때문인지 인과적으로 검증 → 논문의 메커니즘 절.
프로토콜: `PROTOCOL_P9_bottleneck.md` (사전 등록). 러너: `scripts/p9_run_all.sh`. 로그: `D:/flyvl_data/runs/p9/`.

- [x] 환경: venv `~/venvs/flyvl` (torch 2.14+cu130, RTX 4090), 데이터 `D:/flyvl_data` (sha256 일치)
- [x] 기본 대조군 그래프 11개 (중추 재배선이 느려 약 3시간)
- [x] `controls.py`에 `frac` scope 추가, `p9_bottleneck.py` 작성, 프로토콜 등록
- [x] frac 그래프 8개 (f = 0.001/0.01/0.1/0.3 × seed 0,1)
- [x] 진단: P-B 통과(f=0.01에서 hop 격차 58% 감소). P-A 절반 — hop 순서는 CIFAR와 일치, 반응비는 matched(0.15) < real(0.22)로 불일치
- [x] 재현 확인: CIFAR 정확히 재현(32.10). 움직임 real 80.78 vs 78.04 — 원인 확정: 뇌를 안 거치는 눈 기준선도 42.44→40.00으로 바뀜 = 입력 잡음 실현값(새 torch RNG) 차이, 뇌 시뮬레이션 아님. real−matched_s0 = +14.6 (기존 범위 +14~19 안). 모든 P9 비교는 이 머신 안에서만.
- [ ] 용량-반응: CIFAR + 움직임
- [ ] 판정 P-A ~ P-D, `audit_claims.py`에 frac 추가
- [ ] 결과를 FINDINGS.md에 반영
- [ ] (다음) P8 재실험: 세포 타입 공유 파라미터

## 리뷰
