#!/bin/bash
# D4 appendix F stage 1: pick the variant (seed 0, 30 epochs), two workers per GPU.
cd "$(dirname "$0")/.."
L=${FLYVL_DATA}/runs/d4/logs; mkdir -p "$L"
run() { CUDA_VISIBLE_DEVICES=$1 python scripts/d4_train.py cifar_main "$2" "$3" 0 30; }
(run 0 V1 real; run 0 V2 real) > "$L/s1_g0a.log" 2>&1 &
(run 0 V1 rewired; run 0 V2 rewired) > "$L/s1_g0b.log" 2>&1 &
(run 1 V3 real; run 1 V3 dense_small) > "$L/s1_g1a.log" 2>&1 &
(run 1 V3 rewired; run 1 V3 vig) > "$L/s1_g1b.log" 2>&1 &
wait
echo "D4 CIFAR STAGE1 DONE $(date +%H:%M)"
