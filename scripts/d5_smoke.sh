#!/bin/bash
# D5 architecture smoke test (PROTOCOL_D5 §8): two GPU chains, then the summary.
source ~/flyvl_data/env.sh
cd "$(dirname "$0")/.."
L=~/flyvl_data/d5/smoke/logs; mkdir -p $L
( export CUDA_VISIBLE_DEVICES=0
  python scripts/d5_smoke.py prep && for n in rewired_l grid bypass none vig; do python scripts/d5_smoke.py train $n || exit 1; done
  echo "GPU0 CHAIN DONE" ) > $L/gpu0.log 2>&1 &
p0=$!
( export CUDA_VISIBLE_DEVICES=1
  for n in real real_shuffled; do python scripts/d5_smoke.py train $n || exit 1; done && python scripts/d5_smoke.py timing
  echo "GPU1 CHAIN DONE" ) > $L/gpu1.log 2>&1 &
p1=$!
wait $p0 $p1
python scripts/d5_smoke.py summary > $L/summary.log 2>&1
echo "D5 SMOKE DONE" >> $L/summary.log
