"""D1b: peripheral vision over a WHOLE page. Can the real fly, looking at a full page once, tell what
is in each cell of a 4x4 grid (blank / text / table / figure)? This is exactly the job of choosing
which tiles a VLM should read at full resolution.

A page is 16 D1 tiles (48 px each -> 192 x 192). The eye has ~437 viewing directions over the image,
so each cell gets only ~5 x 5 samples. Readout: one linear probe per cell position (same features).
Baselines: full-resolution pixels, a 21 x 21 thumbnail (about the eye's sampling density), eye input.

usage: d1b_page.py [n_train] [n_test] [drift]
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_layout import OUT, brain_features, tiles  # noqa: E402
from flyvl import connectome, frozen, masks, probe  # noqa: E402
from flyvl.sim import Sim  # noqa: E402
from flyvl.stimulus import Retina  # noqa: E402

G, T, K = 4, 48, 1024


def pages(n, seed):
    x, y = tiles(n * G * G, seed, S=T)
    x = x.view(n, G, G, T, T).permute(0, 1, 3, 2, 4).reshape(n, 1, G * T, G * T)
    return x, y.view(n, G * G).cpu().numpy()


def score_cells(Ftr, Fte, ytr, yte):
    """Mean over the 16 cells of per-cell 4-class test accuracy (one PCA, 16 probes, probe seed 0)."""
    (Str, Ste, _), = probe.standardize_pca(Ftr, Fte, (K,)).values()
    acc = [float((probe.fit_probe(Str, ytr[:, j], Ste, 0)["pred"] == yte[:, j]).mean()) for j in range(G * G)]
    return float(np.mean(acc)), acc


if __name__ == "__main__":
    N_TR = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    N_TE = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    DRIFT = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
    t0 = time.time()
    xtr, ytr = pages(N_TR, 10)
    xte, yte = pages(N_TE, 11)
    res = {}
    base = {"pixels": lambda x: x.flatten(1),
            "thumb21": lambda x: F.adaptive_avg_pool2d(x, 21).flatten(1)}
    for name, f in base.items():
        res[name], res[name + "_cells"] = score_cells(f(xtr).cpu().numpy(), f(xte).cpu().numpy(), ytr, yte)
        print(f"{name:17s} {res[name]*100:.2f}", flush=True)
    c = connectome.load()
    cfg, eye_cfg, _ = frozen.load()
    M = masks.build(c)
    views = {"visual_projection": M["visual_projection"], "central_vnc": M["central_vnc"],
             "optic_lobe": np.flatnonzero(c.graded)}
    r16 = c.types(["R1-6"])
    driven = r16[c.column[r16, 0] >= 0]
    ec = dataclasses.replace(eye_cfg, drift_extent=DRIFT)
    sim, retina = Sim(c, cfg, driven=driven), Retina(c, driven, ec, cfg.dt)
    Ftr = brain_features(c, cfg, sim, retina, xtr, ec.drift_directions, views)
    Fte = brain_features(c, cfg, sim, retina, xte, ec.drift_directions, views)
    for v in Ftr:
        res[v], res[v + "_cells"] = score_cells(Ftr[v], Fte[v], ytr, yte)
        print(f"{v:17s} {res[v]*100:.2f}", flush=True)
    res.update({"n_train": N_TR, "n_test": N_TE, "drift": DRIFT, "chance": 0.25,
                "minutes": (time.time() - t0) / 60})
    (OUT / f"d1b_page_drift{DRIFT:g}.json").write_text(json.dumps(res, indent=1))
    print(f"done in {res['minutes']:.1f} min", flush=True)
