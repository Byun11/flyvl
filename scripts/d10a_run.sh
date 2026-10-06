#!/bin/bash
# D10-A: NW workers (default 4) spread over the GPUs in GPUS (default "0 1") over the registered queue; each run is
# skipped if its result exists.  usage: [GPUS="0"] [NW=2] scripts/d10a_run.sh PART [PART ...]
source ~/flyvl_data/env.sh
cd "$(dirname "$0")/.."
LOG=$FLYVL_DATA/d10a/logs
mkdir -p $LOG
GPUS=(${GPUS:-0 1})
NW=${NW:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
for ((slot = 0; slot < NW; slot++)); do
  CUDA_VISIBLE_DEVICES=${GPUS[$((slot % ${#GPUS[@]}))]} nohup python scripts/d10a_mechanism.py worker $slot $NW "$@" > $LOG/worker_${slot}_$(echo "$@" | tr ' ' '_').log 2>&1 &
done
wait
