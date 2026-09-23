"""P9-A: label-free bottleneck diagnostics per graph (see PROTOCOL_P9_bottleneck.md).

For each graph:
  hops    : shortest feed-forward hop count from R1-6 to every neuron; summarised over the
            pre-registered readout masks (mean hop, fraction reached within 3 / 4 hops).
  evoked  : the P7 CIFAR pipeline (v1 Sim, drift 1.2, 4 directions, 25 steps) on 64 fixed test images;
            mean |image - blank| per region and the fraction of the region's neurons that respond at all.
            central_vnc / optic_lobe is the attenuation ratio P8 saw as ~1e-3 on real.

usage: p9_bottleneck.py GRAPH [GRAPH ...]     results merged into runs/p9/bottleneck.json
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, data, frozen, masks  # noqa: E402
from flyvl.extract import load_graph, to_luma  # noqa: E402
from flyvl.sim import Sim  # noqa: E402
from flyvl.stimulus import GRAY, Retina  # noqa: E402

STEPS, N_IMG, DRIFT = 25, 64, 1.2
OUT = connectome.DATA_ROOT / "runs" / "p9"
OUT.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def evoked(c, cfg, sim, retina, imgs, dirs):
    """(n,) mean over images and directions of |image - blank| summed over steps."""
    B, tot = imgs.shape[0], torch.zeros(c.n, device="cuda")
    for d in dirs:
        retina.reset(B + 1)
        st, acc = sim.zero_state(B + 1), 0
        for k in range(STEPS):
            lum = retina.sample_images(imgs, k * cfg.dt, d)
            lum = torch.cat([lum, torch.full((lum.shape[0], 1), GRAY, device="cuda")], 1)
            st = sim.step(st, retina.transduce(lum))
            full = torch.zeros(c.n, B + 1, device="cuda")
            full[torch.as_tensor(sim.gi, device="cuda")] = st.x
            full[torch.as_tensor(sim.li, device="cuda")] = st.s
            acc = acc + full
        tot += (acc[:, :-1] - acc[:, -1:]).abs().mean(1) / STEPS
    return (tot / len(dirs)).cpu().numpy()


if __name__ == "__main__":
    c = connectome.load()
    cfg, eye_cfg, _ = frozen.load()
    M = masks.build(c)
    regions = {"optic_lobe": np.flatnonzero(c.graded), "visual_projection": M["visual_projection"],
               "central_vnc": M["central_vnc"], "descending": M["descending"]}
    r16 = c.types(["R1-6"])
    driven = r16[c.column[r16, 0] >= 0]
    Xte, yte = data.cifar10(False)
    imgs = to_luma(Xte[data.first_per_class(yte, N_IMG // 10 + 1)[:N_IMG]]).cuda()
    ec = dataclasses.replace(eye_cfg, drift_extent=DRIFT)
    path = OUT / "bottleneck.json"
    for graph in sys.argv[1:]:
        t0 = time.time()
        W = load_graph(c, graph, cfg)
        hops = masks.hops_from(W, r16)
        ev = evoked(c, cfg, Sim(c, cfg, W=W, driven=driven), Retina(c, driven, ec, cfg.dt), imgs,
                    ec.drift_directions)
        res = {}
        for name, idx in regions.items():
            h = hops[idx]
            res[name] = {"mean_hop": float(h[h >= 0].mean()), "reach_le3": float(((h >= 0) & (h <= 3)).mean()),
                         "reach_le4": float(((h >= 0) & (h <= 4)).mean()),
                         "evoked_mean": float(ev[idx].mean()), "frac_responding": float((ev[idx] > 1e-6).mean())}
        res["attenuation_central_over_optic"] = res["central_vnc"]["evoked_mean"] / res["optic_lobe"]["evoked_mean"]
        merged = json.loads(path.read_text()) if path.exists() else {}
        merged[graph] = res
        path.write_text(json.dumps(merged, indent=1))
        print(f"{graph:24s} hop(central) {res['central_vnc']['mean_hop']:.2f} "
              f"reach<=3 {res['central_vnc']['reach_le3']:.3f} "
              f"atten {res['attenuation_central_over_optic']:.2e} "
              f"vpn_resp {res['visual_projection']['frac_responding']:.3f} ({time.time()-t0:.0f}s)", flush=True)
