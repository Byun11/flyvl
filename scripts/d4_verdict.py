"""D4 verdict exactly as registered (PROTOCOL_D4_flyvig.md).

Per task x training size: D_rw = acc(real) - acc(rewired), D_rd = acc(real) - acc(random), seed s paired with graph s.
95% CI of the seed-mean D from 2,000 bootstrap resamples of the 900 test videos (the same resample for all seeds).
  topology effect: D_rw and D_rd positive in all 3 seeds AND both CI lower bounds > delta = 1.0 %p
  no cell passes -> stop D4
usage: d4_verdict.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome  # noqa: E402

OUT = connectome.DATA_ROOT / "runs" / "d4"
CONDS = ("real", "rewired", "random", "dense", "dense_small", "vig")
SEEDS, SIZES, TASKS, DELTA = (0, 1, 2), (2100, 300), ("fg16", "mix16"), 0.01


def main():
    rep, lines, passed = {}, [], []
    for task in TASKS:
        contrast = json.loads((OUT / f"calib_{task}.json").read_text())["choice"]
        for n in SIZES:
            ok = {c: [] for c in CONDS}
            for c in CONDS:
                for s in SEEDS:
                    f = OUT / f"{task}_c{contrast}_n{n}_{c}_s{s}.json"
                    ok[c].append(np.asarray(json.loads(f.read_text())["test_ok"]) if f.exists() else None)
            if any(v is None for c in ("real", "rewired", "random") for v in ok[c]):
                lines.append(f"{task} n{n}: incomplete")
                continue
            acc = {c: [float(v.mean()) for v in ok[c] if v is not None] for c in CONDS}
            g = np.random.default_rng(0)
            n_te = len(ok["real"][0])
            boots = [g.integers(0, n_te, n_te) for _ in range(2000)]
            cell = {"contrast": contrast, "acc": acc}
            ok_all = True
            for other in ("rewired", "random"):
                per_seed = [acc["real"][i] - acc[other][i] for i in range(3)]
                d = np.array([np.mean([ok["real"][i][b].mean() - ok[other][i][b].mean() for i in range(3)]) for b in boots])
                lo, hi = np.percentile(d, [2.5, 97.5])
                cell[f"d_{other}"] = {"per_seed": per_seed, "mean": float(np.mean(per_seed)), "ci95": [float(lo), float(hi)]}
                ok_all &= all(x > 0 for x in per_seed) and lo > DELTA
            cell["topology_effect"] = bool(ok_all)
            passed.append(ok_all)
            rep[f"{task}_n{n}"] = cell
            lines.append(f"{task} (contrast {contrast}) n{n}: " + " | ".join(
                f"{c} {100 * np.mean(acc[c]):.1f}±{100 * np.std(acc[c]):.1f}" for c in CONDS if acc[c]))
            for other in ("rewired", "random"):
                x = cell[f"d_{other}"]
                lines.append(f"    real - {other}: seeds {', '.join(f'{100 * v:+.1f}' for v in x['per_seed'])}  mean {100 * x['mean']:+.2f}"
                             f"  95% CI [{100 * x['ci95'][0]:+.2f}, {100 * x['ci95'][1]:+.2f}] %p")
            lines.append(f"    topology effect: {ok_all}")
    rep["verdict"] = "topology effect" if any(passed) else ("stop D4" if len(passed) == 4 else "incomplete")
    lines.append(f"VERDICT: {rep['verdict']}")
    (OUT / "verdict.json").write_text(json.dumps(rep, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
