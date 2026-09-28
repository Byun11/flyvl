# 실험 목록 (브랜치·태그 지도)

## 규칙 (2026-09-28부터, 명령어는 `docs/GIT_WORKFLOW.md`)
- `master` = 공용 코드와 기록(FINDINGS, results.tsv 요약). 항상 돌아가는 상태.
- 새 실험 = `master`에서 `exp/<이름>` 브랜치. 끝나면:
  - 유지(코드·기록이 쓸모 있음) → `master`로 병합(`git merge --no-ff`), 브랜치 삭제, `archive/<이름>` 태그.
  - 실패 → FINDINGS에 음성 결과 기록만 `master`에 반영, 실험 코드는 `archive/<이름>` 태그로 보존.
- 판정 세부는 `docs/FINDINGS.md`, 모든 시도의 수치는 `results.tsv`.

## 진행 중
| 브랜치 | 실험 | 상태 |
|---|---|---|
| `exp/e7-duration` | E7 폐루프 에피소드 1초 → 3·5초 (초파리가 "찰나"만 봤다는 문제) | 중단 (HR만 측정: 3초 최선 0.823, 초파리 미측정). GPU를 D2에 |
| `exp/d2-distill` | D2 초파리 256마리 → InternVL3-1B 토큰 대체, POPE·MME | 초파리 추출 중 |

## 끝난 실험 (`archive/*` 태그, 오래된 순)
2026-09-28 이전에는 브랜치를 앞 브랜치에서 이어 따서 한 줄로 이어져 있다. 각 태그는 그 실험이 끝난 시점이다.

| 태그 | 날짜 | 결과 한 줄 |
|---|---|---|
| `archive/p9-bottleneck` | 09-24 | 경로 길이가 사물 분류 열세의 약 절반을 설명, 움직임은 대부분 연결의 정확한 배치에 의존 |
| `archive/fly-flight` | 09-24 | 비행 RL 래퍼·PPO (미완, 렌더링 병목으로 보류) |
| `archive/fly-doc` | 09-24 | 문서 주변시 실패: 시각엽 35.2% ≈ 눈 입력 36.5% < 썸네일 37.5% |
| `archive/fly-video` | 09-26 | 넓은 조사: 배선이 이기는 곳은 폐루프 몸 제어·경로·강건성뿐 |
| `archive/p10-spectral` | 09-26 | 움직임 우위는 스펙트럼 반경 착시가 아님 (+16.0 → +29.2%p) |
| `archive/fly-select` | 09-26 | 선택성 실패: 초파리 방향 70.1% vs 흐린 HR 98.3% |
| `archive/fly-mix` | 09-27 | E1 flyvis 미세조정 실패 (붕괴/학습 안 됨) |
| `archive/fly-photon` | 09-27 | E2 광자 잡음: 초파리 이점 = 시간 적분, 평활 HR이 동률·우위 |
| `archive/fly-arena` | 09-27 | E5 폐루프: 평활 조정 HR이 초파리와 동률 |
| `archive/fly-general` | 09-27 | G1 철회: 13과제에서 HR 한 설정 0.891 > 초파리 0.778 |
| `archive/fly-nonstat` | 09-28 | G2 기각 + **E5g 확정**: 저대비(0.03·0.05) 폐루프에서 초파리 003 > 조정 HR, 대비 0.1에선 HR 승 |
| `archive/fly-tau` | 09-28 | E6: 이긴 이유는 긴 시간 적분 — 시간상수를 늘리면 모든 모델이 좋아짐 |
| `archive/fly-duration`, `archive/fly-distill` | 09-28 | 정리 시점 표시용 (진행 중 실험은 위 `exp/*`) |
