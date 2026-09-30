#!/bin/bash
# D6 (PROTOCOL_D6_translation.md): all cells, seeds 1-3, then the verdict. Rewired graphs are built separately
# (logs in ~/flyvl_data/d6/logs/graph_*.log); cells that need one wait for its "GRAPH" line (written after the save).
source ~/flyvl_data/env.sh
cd "$(dirname "$0")/.."
L=~/flyvl_data/d6/logs
wait_graph() { until grep -q "^GRAPH" $L/graph_$1_s$2.log 2>/dev/null; do sleep 10; done; }
( export CUDA_VISIBLE_DEVICES=0
  for s in 1 2 3; do python scripts/d6_translation.py run A $s || exit 1; done
  for s in 1 2 3; do python scripts/d6_translation.py hr $s || exit 1; done
  for s in 1 2 3; do python scripts/d6_translation.py run C $s || exit 1; done
  echo "GPU0 D6 DONE" ) > $L/gpu0.log 2>&1 &
p0=$!
( export CUDA_VISIBLE_DEVICES=1
  for s in 1 2 3; do wait_graph rewired_l $s; python scripts/d6_translation.py run B $s || exit 1; done
  for s in 1 2 3; do python scripts/d6_translation.py run D $s || exit 1; done
  for s in 1 2 3; do wait_graph rewired_m $s; python scripts/d6_translation.py run Bm $s || exit 1; done
  echo "GPU1 D6 DONE" ) > $L/gpu1.log 2>&1 &
p1=$!
wait $p0 $p1
python scripts/d6_translation.py verdict > $L/verdict.log 2>&1
echo "D6 ALL DONE" >> $L/verdict.log
