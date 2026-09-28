#!/bin/bash
# D3 Stage 1 (PROTOCOL_D3_swarm.md): every arm x contrast {0.1, 0.05} x CEM seed {0, 1, 2} on two GPU queues
# of about equal length, then the VLM reads every saved fovea set. Finished settings are skipped on rerun.
cd "$(dirname "$0")/.."
L=${FLYVL_DATA}/runs/d3/logs
mkdir -p "$L"
run() { CUDA_VISIBLE_DEVICES=$1 python scripts/d3_pursuit.py run "${@:2}"; }
q0() {
  for c in 0.1 0.05; do for s in 0 1 2; do run 0 $c $s flyvis 000,001,002; done; done
}
q1() {
  for c in 0.1 0.05; do
    run 1 $c 0 base
    for s in 0 1 2; do run 1 $c $s hr; run 1 $c $s framediff; done
  done
  for c in 0.1 0.05; do for s in 0 1 2; do run 1 $c $s flyvis 003,004; done; done
}
q0 > "$L/q0.log" 2>&1 &
q1 > "$L/q1.log" 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 python scripts/d3_pursuit.py read > "$L/read.log" 2>&1
echo "D3 STAGE1 DONE $(date +%H:%M)"
