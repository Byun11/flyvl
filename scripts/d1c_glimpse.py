"""D1c: fix D1b's one failure cause - eye coverage. The eye's ~437 viewing directions cover a page
unevenly (D1b per-cell eye accuracy fell to ~40% in the bottom-left), so the real fly looked worse
than a 21x21 thumbnail before the brain even started.

Fix: 4 glimpses. The page is split into its 2x2 quadrants and each quadrant is shown to the whole eye
in turn (the simplest form of scanning a page). Features of the 4 glimpses are concatenated.
Everything else is D1b: same pages, same labels, same probes, same baselines.

usage: d1c_glimpse.py [n_train] [n_test] [drift]
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_layout import OUT, brain_features  # noqa: E402
from d1b_page import G, T, pages, score_cells  # noqa: E402
from flyvl import connectome, frozen, masks  # noqa: E402
from flyvl.sim import Sim  # noqa: E402
from flyvl.stimulus import Retina  # noqa: E402

H = G * T // 2   # quadrant size in pixels


def quadrants(x):
    return [x[..., i * H:(i + 1) * H, j * H:(j + 1) * H] for i in (0, 1) for j in (0, 1)]


if __name__ == "__main__":
    N_TR = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    N_TE = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    DRIFT = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
    t0 = time.time()
    xtr, ytr = pages(N_TR, 10)
    xte, yte = pages(N_TE, 11)
    res = {}
    # 4 glimpses see ~4x the samples of one look, so the matched thumbnail is 42x42 (D1b used 21x21)
    th = lambda x: torch.nn.functional.adaptive_avg_pool2d(x, 42).flatten(1).cpu().numpy()
    res["thumb42"], res["thumb42_cells"] = score_cells(th(xtr), th(xte), ytr, yte)
    print(f"thumb42           {res['thumb42']*100:.2f}", flush=True)
    c = connectome.load()
    cfg, eye_cfg, _ = frozen.load()
    M = masks.build(c)
    views = {"visual_projection": M["visual_projection"], "central_vnc": M["central_vnc"],
             "optic_lobe": np.flatnonzero(c.graded)}
    r16 = c.types(["R1-6"])
    driven = r16[c.column[r16, 0] >= 0]
    ec = dataclasses.replace(eye_cfg, drift_extent=DRIFT)
    sim, retina = Sim(c, cfg, driven=driven), Retina(c, driven, ec, cfg.dt)
    feats = {"tr": [], "te": []}
    for split, x in (("tr", xtr), ("te", xte)):
        for q in quadrants(x):
            feats[split].append(brain_features(c, cfg, sim, retina, q.contiguous(), ec.drift_directions, views))
            print(f"  glimpse done ({split}) {(time.time()-t0)/60:.1f} min", flush=True)
    for v in feats["tr"][0]:
        Ftr = np.concatenate([f[v] for f in feats["tr"]], 1)
        Fte = np.concatenate([f[v] for f in feats["te"]], 1)
        res[v], res[v + "_cells"] = score_cells(Ftr, Fte, ytr, yte)
        print(f"{v:17s} {res[v]*100:.2f}", flush=True)
    res.update({"n_train": N_TR, "n_test": N_TE, "drift": DRIFT, "glimpses": 4, "chance": 0.25,
                "minutes": (time.time() - t0) / 60})
    (OUT / f"d1c_glimpse_drift{DRIFT:g}.json").write_text(json.dumps(res, indent=1))
    print(f"done in {res['minutes']:.1f} min", flush=True)
