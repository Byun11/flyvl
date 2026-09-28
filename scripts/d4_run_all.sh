#!/bin/bash
# D4 (PROTOCOL_D4_flyvig.md): CIFAR smoke now (small), then after D3 frees the GPUs: Stage 0 calibration per task,
# smoke gate, and the registered grid (fg16 on GPU0, mix16 on GPU1, two workers per GPU, key conditions first).
cd "$(dirname "$0")/.."
R=${FLYVL_DATA}/runs/d4; L=$R/logs; mkdir -p "$L"
py() { CUDA_VISIBLE_DEVICES=$1 python scripts/d4_train.py "${@:2}"; }
for c in vig dense dense_small real rewired random; do py 1 cifar $c 0; done > "$L/cifar_smoke.log" 2>&1 &
SMOKE=$!
until ! pgrep -f "scripts/d3_pursuit.py" >/dev/null && ! pgrep -f "d3_run_all" >/dev/null; do sleep 60; done
echo "D3 finished $(date +%H:%M)"
py 0 calib fg16 > "$L/calib_fg16.log" 2>&1 &
py 1 calib mix16 > "$L/calib_mix16.log" 2>&1 &
wait
python - <<PY || { echo "D4 SMOKE FAIL"; exit 1; }
import json, glob, sys
acc = {json.load(open(f))["cond"]: json.load(open(f))["test_acc"] for f in glob.glob("$R/cifar100_*_s0.json")}
print("smoke", acc)
sys.exit(0 if len(acc) == 6 and min(acc.values()) >= 0.20 else 1)
PY
echo "D4 SMOKE PASS $(date +%H:%M)"
choice() { python -c "import json; print(json.load(open('$R/calib_$1.json'))['choice'])"; }
worker() {   # $1 gpu, $2 task, $3 contrast, $4 worker index (0/1)
  i=0
  for n in 2100 300; do for s in 0 1 2; do for c in real rewired random dense dense_small vig; do
    [ $((i % 2)) -eq "$4" ] && py "$1" video "$2" "$3" "$c" "$s" "$n"
    i=$((i + 1))
  done; done; done
}
CF=$(choice fg16); CM=$(choice mix16)
echo "contrast fg16 $CF mix16 $CM"
worker 0 fg16 "$CF" 0 > "$L/g0w0.log" 2>&1 &
worker 0 fg16 "$CF" 1 > "$L/g0w1.log" 2>&1 &
worker 1 mix16 "$CM" 0 > "$L/g1w0.log" 2>&1 &
worker 1 mix16 "$CM" 1 > "$L/g1w1.log" 2>&1 &
wait
echo "D4 GRID DONE $(date +%H:%M)"
