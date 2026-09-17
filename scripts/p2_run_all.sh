#!/usr/bin/env bash
# P2-mini: paired runs. init conditions first (fast), then trained, real/shuffle alternating per seed.
PY=/c/Users/9722j/AppData/Local/het_tunnel_poc/venv/Scripts/python.exe
LOG=/d/flyvl_data/runs/p2_mini
mkdir -p $LOG
cd "$(dirname "$0")/.."
for cond in init trained; do
  for seed in 0 1 2; do
    for g in real global_shuffle_s0; do
      if [ -f $LOG/${g}_s${seed}_${cond}/result.json ]; then continue; fi
      echo "START $g seed=$seed $cond $(date +%H:%M)"
      $PY scripts/p2_train.py $g $seed $cond > $LOG/${g}_s${seed}_${cond}.log 2>&1 || echo "FAILED $g seed=$seed $cond"
      grep RESULT $LOG/${g}_s${seed}_${cond}.log
    done
  done
done
echo ALLDONE
