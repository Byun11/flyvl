"""V3c: fix V3a's one failure cause - sample count. In the pixel pipeline, frame difference uses 4,096
pooled pixels while one look of the fly uses ~800 eye directions, and it won (95.78% vs 89.07%).
On EQUAL samples the fly's computation wins (V2b +9.3pp). Here the fly gets 4 quadrant glimpses
(~3,200 samples, about the pixel baseline's count); each glimpse shows one quadrant of the same video to
the whole eye. Readout: optic-lobe response energy, concatenated over glimpses. Same trials as V3a
(pixel_where16, n=3000, trial seed 0), so the pixel frame difference number carries over.

usage: v3c_glimpse.py [n]
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

if sys.argv[1:2] != ["pixel_where16"]:
    sys.argv = [sys.argv[0], "pixel_where16"] + sys.argv[1:]          # p5_flytask reads TASK from argv
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5_flytask as P  # noqa: E402
from flyvl import connectome, frozen, probe  # noqa: E402
from flyvl.sim import Sim  # noqa: E402
from flyvl.stimulus import GRAY, Retina  # noqa: E402

H = P.PIX // 2


@torch.no_grad()
def glimpse_energy(c, cfg, sim, retina, p, q, ol, batch=48, frames=None):
    """Optic-lobe response energy for quadrant q of every trial's video."""
    i, j = divmod(q, 2)
    frames = frames or P.pixel_frames
    out = []
    for s in range(0, len(p["cell"]), batch):
        pb = {k: v[s:s + batch] for k, v in p.items()}
        B = len(pb["cell"])
        retina.reset(B + 1)
        st, en = sim.zero_state(B + 1), 0
        for k in range(P.STEPS):
            fr = frames(pb, k * cfg.dt)[..., i * H:(i + 1) * H, j * H:(j + 1) * H].contiguous()
            lum = retina.sample_images(fr, 0.0, "LR")
            lum = torch.cat([lum, torch.full((lum.shape[0], 1), GRAY, device="cuda")], 1)
            st = sim.step(st, retina.transduce(lum))
            xo = st.x[ol]
            en = en + (xo[:, :-1] - xo[:, -1:]) ** 2
        out.append((en / P.STEPS).T.float().cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    t0 = time.time()
    c = connectome.load()
    cfg, eye_cfg, _ = frozen.load()
    eye_cfg = dataclasses.replace(eye_cfg, image_half_width=1.0, drift_extent=0.0)
    r16 = c.types(["R1-6"])
    driven = r16[c.column[r16, 0] >= 0]
    sim, retina = Sim(c, cfg, driven=driven), Retina(c, driven, eye_cfg, cfg.dt)
    ol = torch.as_tensor(np.arange(sim.Ng), device="cuda")        # graded units = optic lobe (local index)
    y, p = P.trial_params(np.random.default_rng(0), "pixel_where16", N)
    n_tr = int(N * 0.7)
    F = []
    for q in range(4):
        F.append(glimpse_energy(c, cfg, sim, retina, p, q, ol))
        print(f"  glimpse {q} done {(time.time()-t0)/60:.1f} min", flush=True)
    X = np.concatenate(F, 1)
    Pm = torch.randn(1024, X.shape[1], generator=torch.Generator().manual_seed(P.view_seed("optic_lobe_energy4")))
    Z = (torch.as_tensor(X, device="cuda") @ (Pm / np.sqrt(1024)).to("cuda").T).cpu().numpy()
    (Str, Ste, k), = probe.standardize_pca(Z[:n_tr], Z[n_tr:], (1024,)).values()
    accs = [float((probe.fit_probe(Str, y[:n_tr], Ste, s)["pred"] == y[n_tr:]).mean()) for s in (0, 1, 2)]
    res = {"optic_lobe_energy_4glimpse": float(np.mean(accs)), "accs": accs, "n": N,
           "pixel_framediff_v3a": 0.9578, "minutes": (time.time() - t0) / 60}
    (connectome.DATA_ROOT / "runs" / "video" / "v3c_glimpse.json").write_text(json.dumps(res, indent=1))
    print(f"optic_lobe_energy x4 glimpses  {np.mean(accs)*100:.2f}  (pixel framediff 95.78)", flush=True)
