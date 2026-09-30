#!/bin/bash
# D8 (PROTOCOL_D8_looming.md): all models, seeds 1-3, then the verdict.
source ~/flyvl_data/env.sh
cd "$(dirname "$0")/.."
L=~/flyvl_data/d8/logs
run() { python scripts/d8_looming.py run "$@" || exit 1; }
( export CUDA_VISIBLE_DEVICES=0
  for m in real_vpn rewired_l_vpn; do for s in 1 2 3; do run $m $s; done; done
  for m in bypass detector; do for s in 1 2 3; do run $m $s; done; done
  echo "GPU0 D8 DONE" ) > $L/gpu0.log 2>&1 &
p0=$!
( export CUDA_VISIBLE_DEVICES=1
  for m in real_dn rewired_l_dn; do for s in 1 2 3; do run $m $s; done; done
  for m in gru hr; do for s in 1 2 3; do run $m $s; done; done
  echo "GPU1 D8 DONE" ) > $L/gpu1.log 2>&1 &
p1=$!
wait $p0 $p1
python scripts/d8_looming.py verdict > $L/verdict.log 2>&1
echo "D8 ALL DONE" >> $L/verdict.log
