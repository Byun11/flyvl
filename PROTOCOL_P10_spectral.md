# P10: P5의 +16%p는 배선인가, 스펙트럼 반경인가

2026-09-26 사전 등록. 계기: 다른 프로젝트(fly-connectome-lab #74)에서 real>shuffle 우위가 스펙트럼 반경(ρ) 차이로
생긴 것으로 드러남. 등록 전에 본 것 (`scripts/p10_spectral.py`, 시각엽 graded→graded 블록 W_gg의 ρ):

| 그래프 | ρ(W_gg) | P5 움직임 VPN K1024 (이전 머신) |
|---|---|---|
| real | 0.915 | 78.04 |
| shuffle_central_s0 | 0.915 | 79.26 (n=3 평균) |
| matched_shuffle_s0/1/2 | 0.532 / 0.499 / 0.514 | 62.22 (평균) |
| shuffle_ol_s0 | 0.478 | 62.06 (n=2 평균) |
| global_shuffle_s0 | 0.268 | 52.43 (평균) |

ρ 순서가 움직임 성능 순서와 완전히 같다. **움직임 결과는 아직 보지 않았다** (아래 두 조작 모두).

## 설계 (움직임 과제는 P5 그대로: `p5_flytask.py motion_dir_hard 3000`, 같은 머신에서 real 재측정 포함)
- **올리기**: 대조군의 W_gg 블록만 상수배해 ρ를 real(0.915)에 맞춘다 (`<graph>_rho`). 다른 블록·연결 구조는 그대로.
  대상: matched_shuffle_s0/1/2, shuffle_ol_s0/s1, global_shuffle_s0/1/2.
- **내리기**: real의 W_gg 블록을 상수배해 ρ를 matched 평균(0.515)으로 낮춘다 (`real_rho0.515`).
- 판독: visual_projection, central_vnc (K1024). 비교는 같은 머신에서 잰 값끼리만.

## 판정 (visual_projection 기준, central_vnc는 보조)
- **배선 효과 유지**: real − matched_rho ≥ 2 SD (n=3) **그리고** 격차가 원래(같은 머신 real − matched)의 절반 이상.
- **ρ 인공물**: real − matched_rho < 2 SD. → P5의 "real 배선이 움직임에서 이긴다"를 철회하고
  "real 시각엽은 재배선보다 순환 이득(ρ)이 크다; 움직임 이득은 그 이득에서 온다"로 고쳐 쓴다.
- **내리기 확인**: real_rho0.515가 matched 수준(같은 머신 matched 평균 ± 2 SD)으로 떨어지면 ρ 설명을 양방향에서 지지.
- 그 사이는 "부분 인공물"로 기록하고 남은 격차를 보고한다.

## 하지 않는 것
결과를 본 뒤 목표 ρ, 스케일 블록, 판독 뷰를 바꾸지 않는다.
