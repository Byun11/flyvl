#!/bin/bash
# D7 screen, first round (PROTOCOL_D7 §4): tasks 0 (extras), N and A, seeds 1-3, then the verdicts.
source ~/flyvl_data/env.sh
cd "$(dirname "$0")/.."
L=~/flyvl_data/d7/logs
run() { python scripts/d7_screen.py run "$@" || exit 1; }
( export CUDA_VISIBLE_DEVICES=0
  for m in real rewired_l; do for s in 1 2 3; do run A $m $s; done; done
  for m in bypass gru; do for s in 1 2 3; do run A $m $s; done; done
  run A engineered 1; cp ~/flyvl_data/d7/taskA_engineered_s1.json ~/flyvl_data/d7/taskA_engineered_s2.json
  cp ~/flyvl_data/d7/taskA_engineered_s1.json ~/flyvl_data/d7/taskA_engineered_s3.json
  echo "GPU0 D7 DONE" ) > $L/gpu0.log 2>&1 &
p0=$!
( export CUDA_VISIBLE_DEVICES=1
  for m in real rewired_l bypass gru; do for s in 1 2 3; do run N $m $s; done; done
  for m in bypass gru; do for s in 1 2 3; do run 0 $m $s; done; done
  echo "GPU1 D7 DONE" ) > $L/gpu1.log 2>&1 &
p1=$!
wait $p0 $p1
for t in 0 N A; do python scripts/d7_screen.py verdict $t > $L/verdict_$t.log 2>&1; done
echo "D7 ROUND1 DONE" >> $L/verdict_A.log
