#!/bin/bash
# D4 appendix H: V3W (type mixer widened to ViG's parameter count), seeds 1-3, 100 epochs; waits for stage 2.
cd "$(dirname "$0")/.."
L=${FLYVL_DATA}/runs/d4/logs; mkdir -p "$L"
until grep -q "STAGE2 DONE" "$L/cifar_stage2.log" 2>/dev/null; do sleep 60; done
worker() {
  i=0
  for s in 1 2 3; do for c in real rewired random; do
    [ $((i % 4)) -eq "$1" ] && CUDA_VISIBLE_DEVICES=$(($1 % 2)) python scripts/d4_train.py cifar_main V3W "$c" "$s" 100
    i=$((i + 1))
  done; done
}
for w in 0 1 2 3; do worker $w > "$L/w_w$w.log" 2>&1 & done
wait
echo "D4 CIFAR WIDE DONE $(date +%H:%M)"
