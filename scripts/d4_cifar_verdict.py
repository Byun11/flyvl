"""D4 CIFAR-100 stage 2 verdict as registered (PROTOCOL_D4_flyvig.md appendices F and G).

Seeds 1-3, 100 epochs, the chosen variant. Paired bootstrap over the 10,000 test images (2,000 resamples, the same
resample for all seeds) of the seed-mean accuracy difference.
  topology effect : real - rewired and real - random positive in all 3 seeds, both CI lower bounds > 1.0 %p
  structure effect: real - dense_small (same active parameters) positive in all seeds, CI lower bound > 1.0 %p
Also reported: every condition's mean accuracy and active parameters, and the gap to the full ViG.
usage: d4_cifar_verdict.py [VARIANT]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome  # noqa: E402

OUT = connectome.DATA_ROOT / "runs" / "d4"
CONDS, SEEDS, DELTA = ("real", "rewired", "random", "dense_small", "vig"), (1, 2, 3), 0.01


def main(variant="V3"):
    runs = {c: [json.loads((OUT / f"cifar100_{variant}_{c}_s{s}_e100.json").read_text()) for s in SEEDS] for c in CONDS}
    ok = {c: [np.asarray(r["test_ok"]) for r in runs[c]] for c in CONDS}
    acc = {c: [float(v.mean()) for v in ok[c]] for c in CONDS}
    g = np.random.default_rng(0)
    n = len(ok["real"][0])
    boots = [g.integers(0, n, n) for _ in range(2000)]
    rep, lines = {"variant": variant, "acc": acc, "active_params": {c: runs[c][0]["active_params"] for c in CONDS}}, []
    lines.append(f"CIFAR-100, {variant}, seeds {SEEDS}, 100 epochs (test %, mean ± sd; active parameters)")
    for c in CONDS:
        lines.append(f"  {c:12s} {100 * np.mean(acc[c]):.2f} ± {100 * np.std(acc[c]):.2f}   {rep['active_params'][c]:,}")
    for other in ("rewired", "random", "dense_small", "vig"):
        per = [acc["real"][i] - acc[other][i] for i in range(3)]
        d = np.array([np.mean([ok["real"][i][b].mean() - ok[other][i][b].mean() for i in range(3)]) for b in boots])
        lo, hi = np.percentile(d, [2.5, 97.5])
        passed = all(x > 0 for x in per) and lo > DELTA
        rep[f"real-{other}"] = {"per_seed": per, "mean": float(np.mean(per)), "ci95": [float(lo), float(hi)], "pass": bool(passed)}
        lines.append(f"  real - {other:11s}: seeds {', '.join(f'{100 * x:+.2f}' for x in per)}  mean {100 * np.mean(per):+.2f}"
                     f"  95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}] %p  {'PASS' if passed else '-'}")
    rep["topology_effect"] = rep["real-rewired"]["pass"] and rep["real-random"]["pass"]
    rep["structure_effect"] = rep["real-dense_small"]["pass"]
    lines.append(f"TOPOLOGY EFFECT: {rep['topology_effect']}   STRUCTURE (vs same-size dense): {rep['structure_effect']}")
    (OUT / f"cifar_verdict_{variant}.json").write_text(json.dumps(rep, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
