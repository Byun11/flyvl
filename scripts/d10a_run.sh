#!/bin/bash
# D10-A: 4 workers (2 per GPU) over the registered queue; each run is skipped if its result exists.
# usage: scripts/d10a_run.sh PART [PART ...]
source ~/flyvl_data/env.sh
cd "$(dirname "$0")/.."
LOG=$FLYVL_DATA/d10a/logs
mkdir -p $LOG
for slot in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((slot % 2)) nohup python scripts/d10a_mechanism.py worker $slot 4 "$@" > $LOG/worker_${slot}_$(echo "$@" | tr ' ' '_').log 2>&1 &
done
wait
