#!/usr/bin/env bash
# P2-B: visual_entry seeds 0,1,2 (real then shuffle per seed), then all_sensory seed 0. Stops on the first failure.
PY=/c/Users/9722j/AppData/Local/het_tunnel_poc/venv/Scripts/python.exe
LOG=/d/flyvl_data/runs/p2b_mini
mkdir -p $LOG
cd "$(dirname "$0")/.."
run() {
  local g=$1 seed=$2 mode=$3
  [ -f $LOG/${g}_s${seed}_${mode}/result.json ] && return 0
  echo "START $g seed=$seed $mode $(date +%H:%M)"
  if ! $PY scripts/p2b_train.py $g $seed $mode > $LOG/${g}_s${seed}_${mode}.log 2>&1; then
    echo "FAILED $g seed=$seed $mode"; exit 1
  fi
  grep -E "^RESULT" $LOG/${g}_s${seed}_${mode}.log
}
for seed in 0 1 2; do for g in real global_shuffle_s0; do run $g $seed visual_entry; done; done
for g in real global_shuffle_s0; do run $g 0 all_sensory; done
echo ALLDONE
