# Git 관리법

## 구조
```
master                 공용 코드 + 기록 (FINDINGS, EXPERIMENTS, program.md). 항상 돌아가는 상태
exp/<이름>             진행 중인 실험 하나. 반드시 master에서 딴다
archive/<이름> (태그)  끝난 실험의 마지막 커밋. 코드를 지워도 여기서 되살릴 수 있다
```
- 실험 목록과 결과 한 줄은 `docs/EXPERIMENTS.md`, 판정 세부는 `docs/FINDINGS.md`,
  모든 시도의 수치는 `results.tsv`에 둔다.

## 실험 한 개의 흐름
```bash
# 1. 시작: master에서 딴다 (앞 실험 브랜치에서 따지 않는다)
git checkout master && git pull
git checkout -b exp/e8-looming
# docs/EXPERIMENTS.md의 "진행 중" 표에 한 줄 추가

# 2. 진행: 한 아이디어 = 한 커밋, 커밋마다 push
git commit -am "E8: ..."        # 제목에 실험 ID와 결과를 쓴다
git push -u origin exp/e8-looming

# 3. 끝: 판정을 FINDINGS / results.tsv / EXPERIMENTS에 쓰고
git checkout master
git merge --no-ff exp/e8-looming       # 유지할 코드·기록을 master로
git tag -a archive/e8-looming exp/e8-looming -m "E8 결과 한 줄"
git push origin master --tags
git branch -d exp/e8-looming && git push origin --delete exp/e8-looming
```
- **실패한 실험**도 FINDINGS에 음성 결과를 쓰고 병합한다. 쓸모없는 코드만 master에서 빼고 커밋한다.
  코드는 archive 태그에 남아 있다.
- 두 실험이 같은 파일(예: `scripts/e5_arena.py`)을 고쳤다면, 먼저 끝난 쪽을 master에 합친다.
  다른 쪽은 `git merge master`로 따라온다.

## 커밋 규칙
- 제목: `<실험ID>: <무엇을 했고 결과가 어땠는지>`
  (예: `E6c: longer time constants help every flyvis member tested`)
- 끝에 공동 작성자 줄을 붙인다.
- 결과 파일(`D:\flyvl_data\...`)은 git에 넣지 않는다. 수치는 results.tsv와 FINDINGS에 옮긴다.

## 옛 기록
- 2026-09-28 이전에는 `autoresearch/*` 브랜치를 앞 브랜치에서 이어 따서 한 줄로 이어져 있었다.
- 지금은 전부 `archive/*` 태그다. 목록은 `docs/EXPERIMENTS.md`.
