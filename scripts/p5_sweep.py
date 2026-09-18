"""P5 sweep: where does the real-wiring advantage on motion direction appear and disappear?

Same 4-way motion-direction task as p5_flytask, but contrast and photoreceptor noise are set
explicitly per sweep point, and every point reuses the SAME trial parameters (direction, spatial
frequency, speed, phase) so the points differ only in stimulus quality.

usage: p5_sweep.py AXIS [n_trials] [graphs] [trial_seed]
  AXIS = contrast : noise fixed at 0.06, contrast in {0.03, 0.05, 0.08, 0.15, 0.30}
  AXIS = noise    : contrast fixed at 0.35, noise in {0.0, 0.03, 0.06, 0.12, 0.25}
Readout: 5 probe seeds (p5_flytask uses 3), time-mean evoked activity, fixed random projection.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from flyvl import connectome, frozen, masks, probe  # noqa: E402

import p5_flytask as P5  # noqa: E402

AXIS = sys.argv[1] if len(sys.argv) > 1 else "contrast"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
GRAPHS = (sys.argv[3].split(",") if len(sys.argv) > 3
          else ["real", "matched_shuffle_s0", "global_shuffle_s0", "nobrain"])
TRIAL_SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
SEEDS = (0, 1, 2, 3, 4)
POINTS = {"contrast": [(c, 0.06) for c in (0.03, 0.05, 0.08, 0.15, 0.30)],
          "noise": [(0.35, n) for n in (0.0, 0.03, 0.06, 0.12, 0.25)]}[AXIS]

P5.TASK = "motion_dir"                      # render() dispatches on this
OUT = connectome.DATA_ROOT / "runs" / "p5"
OUT.mkdir(parents=True, exist_ok=True)


def view_seed(view: str) -> int:
    """Deterministic across processes: Python's str hash is randomised per interpreter (PYTHONHASHSEED),
    so hash(view) gave a different random projection on every run and absolute numbers drifted between
    runs (graph comparisons inside one run were still valid, since all graphs shared the projection)."""
    return int(hashlib.md5(view.encode()).hexdigest()[:8], 16) % 2**31


if __name__ == "__main__":
    c = connectome.load()
    cfg, _, _ = frozen.load()
    M = masks.build(c)
    views = {"descending": M["descending"], "central_vnc": M["central_vnc"],
             "visual_projection": M["visual_projection"]}
    y, p = P5.trial_params(np.random.default_rng(TRIAL_SEED), "motion_dir", N)
    n_tr = int(N * 0.7)
    res = {}
    path = OUT / f"sweep_{AXIS}_t{TRIAL_SEED}.json"
    for contrast, noise in POINTS:
        p["contrast"] = np.full(N, contrast)
        p["noise"] = np.full(N, noise)
        tag = f"c{contrast:g}_n{noise:g}"
        print(f"=== {tag} ===", flush=True)
        for graph in GRAPHS:
            t0 = time.time()
            F = P5.features(graph, y, p, c, cfg, views)
            for view, X in F.items():
                if graph == "nobrain" and not view.startswith("photoreceptor"):
                    continue
                gen = torch.Generator().manual_seed(view_seed(view))
                proj = (torch.randn(P5.PROJ, X.shape[1], generator=gen) / np.sqrt(P5.PROJ)).numpy().astype(np.float32)
                Z = X @ proj.T
                sc = probe.standardize_pca(Z[:n_tr], Z[n_tr:], (256,))
                for K, (Str, Ste, k) in sc.items():
                    accs = [float((probe.fit_probe(Str, y[:n_tr], Ste, s)["pred"] == y[n_tr:]).mean())
                            for s in SEEDS]
                    res[f"{tag}:{graph}:{view}"] = {"acc": accs, "mean": float(np.mean(accs)),
                                                    "contrast": contrast, "noise": noise}
                    print(f"{tag:14s} {graph:20s} {view:20s} mean {np.mean(accs)*100:.2f} "
                          f"{[round(a*100,1) for a in accs]}", flush=True)
            print(f"  ({graph} {time.time()-t0:.0f}s)", flush=True)
            path.write_text(json.dumps({"axis": AXIS, "n": N, "chance": 0.25, "seeds": list(SEEDS),
                                        "trial_seed": TRIAL_SEED, "res": res}, indent=1))
