#!/bin/bash
# D4 registered grid for one task on one GPU (two workers, verdict conditions first):
#   d4_grid.sh GPU TASK CONTRAST
cd "$(dirname "$0")/.."
L=${FLYVL_DATA}/runs/d4/logs
mkdir -p "$L"
worker() {
  i=0
  for n in 2100 300; do for s in 0 1 2; do for c in real rewired random dense dense_small vig; do
    [ $((i % 2)) -eq "$1" ] && CUDA_VISIBLE_DEVICES=$GPU python scripts/d4_train.py video "$TASK" "$CONTRAST" "$c" "$s" "$n"
    i=$((i + 1))
  done; done; done
}
GPU=$1 TASK=$2 CONTRAST=$3
worker 0 > "$L/${TASK}_w0.log" 2>&1 &
worker 1 > "$L/${TASK}_w1.log" 2>&1 &
wait
echo "D4 GRID $TASK DONE $(date +%H:%M)"
