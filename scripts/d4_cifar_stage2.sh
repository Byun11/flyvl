#!/bin/bash
# D4 appendix G stage 2: chosen variant, seeds 1-3, 100 epochs; four workers (two per GPU), verdict conditions first.
cd "$(dirname "$0")/.."
V=$1
L=${FLYVL_DATA}/runs/d4/logs; mkdir -p "$L"
worker() {   # $1 worker index 0..3 -> GPU $1 % 2
  i=0
  for s in 1 2 3; do for c in real rewired random dense_small vig; do
    [ $((i % 4)) -eq "$1" ] && CUDA_VISIBLE_DEVICES=$(($1 % 2)) python scripts/d4_train.py cifar_main "$V" "$c" "$s" 100
    i=$((i + 1))
  done; done
}
for w in 0 1 2 3; do worker $w > "$L/s2_w$w.log" 2>&1 & done
wait
echo "D4 CIFAR STAGE2 DONE $(date +%H:%M)"
