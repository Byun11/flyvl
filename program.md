# program.md — 자동 연구 루프 규칙 (autoresearch 방식)

에이전트(Claude Code)가 30분마다 이 문서를 따라 실험을 진행한다. 사람이 직접 해도 규칙은 같다:
**한 실험 = 한 커밋, 개선 시만 유지, 모든 시도는 `results.tsv`에 기록.**

## 최종 목적
초파리 커넥톰 시각엽을 인코더로 쓴 **시각 기반 비행 강화학습**. 비교 대상은 같은 조건의 CNN과 재배선한 시각엽이다.
그 전에 P9(병목 가설)를 사전 등록대로 끝낸다.

## 트랙과 규칙

### 트랙 1 — 가설 검증 (P9 등)
- 사전 등록 프로토콜(`PROTOCOL_*.md`)대로 **한 번** 돌린다. keep/discard 하지 않는다.
  결과가 나쁘다고 설계를 바꾸면 결과를 보고 고른 것이 된다. 음성 결과도 커밋한다.
- 브랜치: `autoresearch/p9-bottleneck`.

### 트랙 2 — RL 최적화 (keep/discard 적용)
- 브랜치: `autoresearch/fly-flight`. 한 아이디어 = 한 커밋.
- **지표: 검증 에피소드 평균 return** (학습에 쓰지 않은 seed의 환경). 테스트 seed는 마지막에 한 번만 본다.
- 개선되면 커밋 유지 + push. 개선이 없거나 크래시면 `git reset --hard HEAD~1`로 되돌린다.
- **공정성 규칙 (이 프로젝트의 핵심)**: 인코더와 무관한 변경(학습률, 보상, PPO 설정)은 CNN / real 시각엽 /
  재배선 시각엽 **세 팔 모두에 동일하게** 적용한다. 한 팔만 튜닝하는 것은 금지한다.
  "real이 이긴다" 주장은 재배선 그래프 2개 이상, |차이|/SD ≥ 2일 때만 한다 (`scripts/audit_claims.py` 규칙).

## results.tsv (git 미추적, 리셋해도 남도록)
탭으로 구분한 열: `time  track  commit  idea  metric  value  verdict(keep|discard|crash|registered)  note`

## 매 루프에서 할 일
1. 돌고 있는 작업이 살아 있는지 확인한다. 죽었으면 원인을 고치고 재개한다.
2. 끝난 실험은 판정하고, `results.tsv`에 한 줄 쓰고, keep이면 커밋·push한다.
3. GPU가 비어 있으면 다음 아이디어 하나를 커밋하고 실행한다.
4. `tasks/todo.md`를 갱신하고 사용자에게 한국어로 짧게 보고한다.
