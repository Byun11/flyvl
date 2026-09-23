#!/usr/bin/env bash
# P9 pipeline: controls -> frac graphs -> diagnostics -> reproduction -> dose-response. Resumable: skips done graphs.
set -e
cd "$(dirname "$0")/.."
P=~/venvs/flyvl/Scripts/python.exe
G=/d/flyvl_data/graphs
LOG=/d/flyvl_data/runs/p9; mkdir -p $LOG
stage() { echo "=== $(date '+%F %T') $*" | tee -a $LOG/stages.log; }

stage "wait for base controls"
while [ ! -f $G/matched_shuffle_s2.npz ] || [ ! -f $G/shuffle_ol_s1.npz ] || [ ! -f $G/shuffle_central_s2.npz ]; do sleep 30; done
sleep 60   # let the writers finish

stage "frac graphs"
for f in 0.001 0.01 0.1 0.3; do
  [ -f $G/shuffle_frac${f}_s1.npz ] || $P scripts/make_controls.py frac$f 0 1 >> $LOG/make_frac.log 2>&1
done

FRAC=$(for f in 0.001 0.01 0.1 0.3; do for s in 0 1; do printf "shuffle_frac%s_s%s," $f $s; done; done)
BASE="real,matched_shuffle_s0,matched_shuffle_s1,matched_shuffle_s2,global_shuffle_s0,global_shuffle_s1,global_shuffle_s2"
ALL="$BASE,${FRAC%,}"

stage "diagnostics"
$P scripts/p9_bottleneck.py ${ALL//,/ } >> $LOG/bottleneck.log 2>&1

stage "reproduction: CIFAR real,matched_s0"
$P scripts/p7_drift_brain.py 300 1.2 --graphs=real,matched_shuffle_s0 >> $LOG/repro_cifar.log 2>&1
stage "reproduction: motion real,matched_s0"
$P scripts/p5_flytask.py motion_dir_hard 3000 real,matched_shuffle_s0 >> $LOG/repro_motion.log 2>&1
stage "REPRO DONE - check against 32.10 / 78.04 before trusting what follows"

stage "dose-response CIFAR"
$P scripts/p7_drift_brain.py 300 1.2 --graphs=${FRAC%,},global_shuffle_s0,global_shuffle_s1,global_shuffle_s2,matched_shuffle_s1,matched_shuffle_s2 >> $LOG/dose_cifar.log 2>&1
stage "dose-response motion"
$P scripts/p5_flytask.py motion_dir_hard 3000 ${FRAC%,},global_shuffle_s0,global_shuffle_s1,global_shuffle_s2,matched_shuffle_s1,matched_shuffle_s2 >> $LOG/dose_motion.log 2>&1
stage "ALL DONE"
