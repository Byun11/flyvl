"""P5 diagnostic: is the real-vs-shuffle gap on the hard motion task a difference in COMPUTATION,
or just a difference in DYNAMIC RANGE?

A rewired graph could score lower simply because its evoked responses are tiny (signal lost in the
probe's regularisation) or collapsed onto few dimensions. For each graph we therefore report, on the
same trials and the same readout neurons:
  evoked_absmean   mean |image - blank| activity
  evoked_std       std across trials of the per-neuron evoked response (usable signal, not offset)
  part_ratio       participation ratio (sum(l)^2 / sum(l^2) of the feature covariance eigenvalues)
                   = effective number of dimensions the representation occupies
  snr_dir          between-direction variance / within-direction variance, averaged over neurons
                   (a readout-free measure of how much direction information is linearly present)
  acc              probe accuracy, for reference

If shuffle matches real on range and part_ratio but loses on snr_dir and acc, the gap is structural.
usage: p5_diag.py [n_trials] [graphs] [trial_seed]
"""
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

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
GRAPHS = (sys.argv[2].split(",") if len(sys.argv) > 2
          else ["real", "matched_shuffle_s0", "global_shuffle_s0", "nobrain"])
TRIAL_SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
P5.TASK = "motion_dir_hard"
OUT = connectome.DATA_ROOT / "runs" / "p5"


def part_ratio(X):
    Xc = X - X.mean(0, keepdims=True)
    ev = np.linalg.svd(Xc / np.sqrt(len(Xc)), compute_uv=False) ** 2
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def snr_dir(X, y):
    """between-class / within-class variance per neuron, averaged (Fisher-style, readout-free)."""
    mus = np.stack([X[y == k].mean(0) for k in np.unique(y)])
    between = mus.var(0)
    within = np.mean([X[y == k].var(0) for k in np.unique(y)], 0)
    return float(np.mean(between / (within + 1e-12)))


if __name__ == "__main__":
    c = connectome.load()
    cfg, _, _ = frozen.load()
    M = masks.build(c)
    views = {"central_vnc": M["central_vnc"], "visual_projection": M["visual_projection"]}
    y, p = P5.trial_params(np.random.default_rng(TRIAL_SEED), "motion_dir_hard", N)
    n_tr = int(N * 0.7)
    res = {}
    for graph in GRAPHS:
        t0 = time.time()
        F = P5.features(graph, y, p, c, cfg, views)
        for view, X in F.items():
            if graph == "nobrain" and view != "photoreceptor":
                continue
            X = X.astype(np.float64)
            gen = torch.Generator().manual_seed(hash(view) % 2**31)
            proj = (torch.randn(P5.PROJ, X.shape[1], generator=gen) / np.sqrt(P5.PROJ)).numpy().astype(np.float32)
            Z = X.astype(np.float32) @ proj.T
            sc = probe.standardize_pca(Z[:n_tr], Z[n_tr:], (256,))
            (Str, Ste, k), = sc.values()
            accs = [float((probe.fit_probe(Str, y[:n_tr], Ste, s)["pred"] == y[n_tr:]).mean())
                    for s in range(5)]
            r = {"evoked_absmean": float(np.abs(X).mean()), "evoked_std": float(X.std(0).mean()),
                 "part_ratio": part_ratio(X), "part_ratio_proj": part_ratio(Z),
                 "snr_dir": snr_dir(X, y), "acc": float(np.mean(accs)), "pca_k": k,
                 "n_neurons": int(X.shape[1])}
            res[f"{graph}:{view}"] = r
            print(f"{graph:20s} {view:18s} |ev| {r['evoked_absmean']:.4g}  std {r['evoked_std']:.4g}  "
                  f"PR {r['part_ratio']:7.1f}  snr {r['snr_dir']:.4f}  acc {r['acc']*100:5.2f}", flush=True)
        print(f"  ({graph} {time.time()-t0:.0f}s)", flush=True)
        (OUT / f"diag_hard_t{TRIAL_SEED}.json").write_text(json.dumps({"n": N, "res": res}, indent=1))
