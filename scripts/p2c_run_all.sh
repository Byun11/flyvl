#!/usr/bin/env bash
PY=/c/Users/9722j/AppData/Local/het_tunnel_poc/venv/Scripts/python.exe
LOG=/d/flyvl_data/runs/p2c_mini
mkdir -p $LOG
cd "$(dirname "$0")/.."
for g in real global_shuffle_s0 nobrain; do
  [ -f $LOG/${g}_s0/result.json ] && continue
  echo "START $g $(date +%H:%M)"
  $PY scripts/p2c_train.py $g 0 > $LOG/${g}_s0.log 2>&1 || { echo "FAILED $g"; exit 1; }
  grep -E "^RESULT" $LOG/${g}_s0.log
done
echo ALLDONE
