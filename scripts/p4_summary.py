"""Aggregate P4 results: mean over seeds, and paired bootstrap CIs on per-image zero-shot correctness
(seed-averaged) for selected comparisons. Writes results/p4/summary.json and prints a markdown table."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, probe  # noqa: E402

RUNS = connectome.DATA_ROOT / "runs" / "p4"
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
y = np.load(connectome.DATA_ROOT / "p4" / "test_labels.npy")

rows = defaultdict(list)
for f in sorted(RUNS.glob(f"{tag}__*__s*.json")):
    r = json.loads(f.read_text())
    r["correct"] = np.load(f.with_suffix(".preds.npy")) == y
    rows[r["rep"]].append(r)

table = {}
for rep, rs in rows.items():
    table[rep] = {"n_seeds": len(rs),
                  "zeroshot": [round(r["test_zeroshot_acc"] * 100, 1) for r in rs],
                  "zeroshot_mean": round(float(np.mean([r["test_zeroshot_acc"] for r in rs])) * 100, 2),
                  "centered_cos_mean": round(float(np.mean([r["test_centered_cos"] for r in rs])), 4),
                  "val_loss_mean": round(float(np.mean([r.get("val_loss", np.nan) for r in rs])), 4),
                  "_correct": np.mean([r["correct"] for r in rs], 0)}

pairs = []
views = ["optic_lobe", "visual_projection", "central_vnc"]
for model in ("", "-v1"):
    for v in views:
        real = f"real{model}:{v}"
        for ctrl in (f"global_shuffle_s0{model}:{v}", f"matched_shuffle_s0{model}:{v}"):
            pairs.append((real, ctrl))
for rep in list(table):
    if ":" in rep:
        pairs.append((rep, "real:photoreceptor"))
        pairs.append((rep, "pixels"))
comparisons = {}
for a, b in pairs:
    if a in table and b in table and table[a]["n_seeds"] == table[b]["n_seeds"]:
        bs = probe.paired_bootstrap(table[a]["_correct"], table[b]["_correct"])
        comparisons[f"{a} - {b}"] = {"diff_pp": round(bs["diff"] * 100, 2), "ci95_pp": [round(x * 100, 2) for x in bs["ci95"]]}

ceiling = json.loads((RUNS / "ceiling.json").read_text())
out = {"ceiling": ceiling, "table": {k: {kk: vv for kk, vv in v.items() if kk != "_correct"} for k, v in table.items()},
       "comparisons": comparisons}
dst = ROOT / "results" / "p4"
dst.mkdir(parents=True, exist_ok=True)
(dst / f"summary_{tag}.json").write_text(json.dumps(out, indent=1))

print(f"ceiling: teacher 4x4 {ceiling['teacher_pooled4x4'] * 100:.1f}%, full 256 {ceiling['teacher_full256'] * 100:.1f}%")
print("| rep | seeds | zero-shot % (seeds) | mean | centered cos | val loss |\n|---|---|---|---|---|---|")
for rep, v in sorted(table.items(), key=lambda kv: -kv[1]["zeroshot_mean"]):
    print(f"| {rep} | {v['n_seeds']} | {v['zeroshot']} | {v['zeroshot_mean']} | {v['centered_cos_mean']} | {v['val_loss_mean']} |")
print("\n| comparison | diff pp | 95% CI |\n|---|---|---|")
for k, v in comparisons.items():
    print(f"| {k} | {v['diff_pp']} | {v['ci95_pp']} |")
