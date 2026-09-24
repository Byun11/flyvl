"""D1d: the job a peripheral vision actually does - point at the one cell worth reading.

Every page has exactly ONE table cell among 16 (the other 15 are blank / text / figure). The real
fly looks with D1c's 4 quadrant glimpses; a linear probe per cell gives P(table); the fly "points"
at the cell with the highest score. Metrics: top-1 and top-4 hit rate (chance 1/16 and 4/16).
Baselines: 42x42 thumbnail (same sample count as 4 glimpses) and the eye input.
Features are cached in runs/doc so D2 can reuse the fly's choices without re-simulating.

usage: d1d_find_table.py [n_train] [n_test] [drift]
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_layout import OUT, brain_features, tiles  # noqa: E402
from d1b_page import G, K, T  # noqa: E402
from d1c_glimpse import quadrants  # noqa: E402
from flyvl import connectome, frozen, masks, probe  # noqa: E402
from flyvl.sim import Sim  # noqa: E402
from flyvl.stimulus import Retina  # noqa: E402

TABLE = 2


def pages_one_table(n, seed):
    """n pages, each with exactly one table cell at a random position."""
    x, y = tiles(n * G * G * 3, seed, S=T)
    tab, rest = torch.nonzero(y == TABLE).ravel(), torch.nonzero(y != TABLE).ravel()
    g = torch.Generator(device="cuda").manual_seed(seed)
    pos = torch.randint(0, G * G, (n,), generator=g, device="cuda")
    idx = rest[torch.randperm(len(rest), generator=g, device="cuda")[:n * G * G]].view(n, G * G)
    idx[torch.arange(n, device="cuda"), pos] = tab[torch.randperm(len(tab), generator=g, device="cuda")[:n]]
    lab = y[idx]
    pg = x[idx.ravel()].view(n, G, G, T, T).permute(0, 1, 3, 2, 4).reshape(n, 1, G * T, G * T)
    return pg, lab.cpu().numpy(), pos.cpu().numpy()


def locate(Ftr, Fte, ytr, pos_te):
    """Per-cell P(table) from a linear probe (table vs not), then rank cells."""
    (Str, Ste, _), = probe.standardize_pca(Ftr, Fte, (K,)).values()
    cols = []
    for j in range(G * G):
        lg = probe.fit_probe(Str, (ytr[:, j] == TABLE).astype(np.int64), Ste, 0)["logits"]
        cols.append(lg[:, 1] - lg[:, 0])
    score = np.stack(cols, 1)
    rank = np.argsort(-score, 1)
    top1 = float((rank[:, 0] == pos_te).mean())
    top4 = float((rank[:, :4] == pos_te[:, None]).any(1).mean())
    return top1, top4, rank


if __name__ == "__main__":
    N_TR = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    N_TE = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    DRIFT = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
    t0 = time.time()
    xtr, ytr, _ = pages_one_table(N_TR, 20)
    xte, yte, pte = pages_one_table(N_TE, 21)
    res = {}
    th = lambda x: torch.nn.functional.adaptive_avg_pool2d(x, 42).flatten(1).cpu().numpy()
    res["thumb42"] = locate(th(xtr), th(xte), ytr, pte)[:2]
    print(f"thumb42           top1 {res['thumb42'][0]*100:.1f}  top4 {res['thumb42'][1]*100:.1f}", flush=True)
    c = connectome.load()
    cfg, eye_cfg, _ = frozen.load()
    M = masks.build(c)
    views = {"visual_projection": M["visual_projection"], "optic_lobe": np.flatnonzero(c.graded)}
    r16 = c.types(["R1-6"])
    driven = r16[c.column[r16, 0] >= 0]
    ec = dataclasses.replace(eye_cfg, drift_extent=DRIFT)
    sim, retina = Sim(c, cfg, driven=driven), Retina(c, driven, ec, cfg.dt)
    F = {}
    for split, x in (("tr", xtr), ("te", xte)):
        g = [brain_features(c, cfg, sim, retina, q.contiguous(), ec.drift_directions, views) for q in quadrants(x)]
        F[split] = {v: np.concatenate([f[v] for f in g], 1) for v in g[0]}
    ranks = {}
    for v in F["tr"]:
        t1, t4, ranks[v] = locate(F["tr"][v], F["te"][v], ytr, pte)
        res[v] = (t1, t4)
        print(f"{v:17s} top1 {t1*100:.1f}  top4 {t4*100:.1f}", flush=True)
    np.savez_compressed(OUT / "d1d_choices.npz", pos_te=pte, **{f"rank_{v}": r for v, r in ranks.items()})
    res.update({"n_train": N_TR, "n_test": N_TE, "drift": DRIFT, "chance_top1": 1 / 16, "chance_top4": 4 / 16,
                "minutes": (time.time() - t0) / 60})
    (OUT / "d1d_find_table.json").write_text(json.dumps(res, indent=1))
    print(f"done in {res['minutes']:.1f} min", flush=True)
