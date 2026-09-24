"""D1: does the fly see document LAYOUT? (real connectome only; controls come later, program.md)

A document tile is one of 4 kinds: blank / text (lines of word blobs) / table (grid + cell text) /
figure (a large filled blob). Tiles are generated procedurally ON THE GPU with random offsets,
densities, stroke darkness and paper brightness, so no single global statistic gives the class away.
Pipeline = P7 (fly eye with drift -> v1 Sim -> evoked activity per readout view -> linear probe).
Baselines: raw pixels and the eye input alone. Question: does the brain add anything over the eye?

usage: d1_layout.py [n_train] [n_test] [drift] [size]
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
from flyvl import connectome, frozen, masks, probe  # noqa: E402
from flyvl.sim import Sim  # noqa: E402
from flyvl.stimulus import GRAY, Retina  # noqa: E402

SIZE = 96
STEPS, K = 25, 1024
CLASSES = ["blank", "text", "table", "figure"]
OUT = connectome.DATA_ROOT / "runs" / "doc"
OUT.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def tiles(n, seed, S=SIZE, dev="cuda"):
    """(n, 1, S, S) luminance in [0, 1] and labels. Paper = bright, ink = dark."""
    g = torch.Generator(device=dev).manual_seed(seed)
    u = lambda *s: torch.rand(*s, generator=g, device=dev)
    y = torch.randint(0, 4, (n,), generator=g, device=dev)
    paper = 0.75 + 0.2 * u(n, 1, 1)
    ink = paper - (0.3 + 0.4 * u(n, 1, 1))
    yy, xx = torch.meshgrid(torch.arange(S, device=dev), torch.arange(S, device=dev), indexing="ij")
    yy, xx = yy[None].float(), xx[None].float()
    img = paper.expand(n, S, S).clone()

    def text_mask(line_h, gap, x0, x1, y0, y1):
        # rows of "words": dark runs along each line, broken by random word gaps
        row = ((yy - y0) % (line_h + gap) < line_h) & (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
        word = torch.sin(xx * (0.35 + 0.3 * u(n, 1, 1)) + 6.28 * u(n, 1, 1) + (yy // (line_h + gap)) * 1.7) > -0.3
        return row & word

    margin = (S * (0.05 + 0.1 * u(n, 1, 1)))
    lh = 2 + (3 * u(n, 1, 1)).floor()
    m_text = text_mask(lh, lh + 1 + (3 * u(n, 1, 1)).floor(), margin, S - margin, margin, S - margin)
    cell = S / (3 + (3 * u(n, 1, 1)).floor())
    grid = (((xx - margin) % cell < 1.2) | ((yy - margin) % cell < 1.2)) & (xx >= margin) & (xx < S - margin) \
        & (yy >= margin) & (yy < S - margin)
    m_table = grid | (text_mask(lh, cell - lh, margin + 3, S - margin, margin + 3, S - margin) & ((xx - margin) % cell < cell * 0.6))
    cx, cy = S * (0.3 + 0.4 * u(n, 1, 1)), S * (0.3 + 0.4 * u(n, 1, 1))
    r = S * (0.2 + 0.15 * u(n, 1, 1))
    m_fig = ((xx - cx) ** 2 / (1 + u(n, 1, 1)) + (yy - cy) ** 2 * (1 + u(n, 1, 1)) < r ** 2)
    mask = torch.stack([torch.zeros_like(m_text), m_text, m_table, m_fig])[y, torch.arange(n, device=dev)]
    shade = torch.where(y[:, None, None] == 3, paper - (paper - ink) * (0.5 + 0.5 * torch.sin(xx / 5 + yy / 7) ** 2), ink)
    img = torch.where(mask, shade.expand_as(img), img)
    img = (img + 0.02 * torch.randn(img.shape, generator=g, device=dev)).clamp(0, 1)
    return img[:, None], y


@torch.no_grad()
def brain_features(c, cfg, sim, retina, imgs, dirs, views, batch=32):
    out = {k: [] for k in list(views) + ["photoreceptor"]}
    for s in range(0, len(imgs), batch):
        x = imgs[s:s + batch]
        B = x.shape[0]
        acc_v, acc_pr = 0, 0
        for d in dirs:
            retina.reset(B + 1)
            st, acc, pr = sim.zero_state(B + 1), 0, 0
            for k in range(STEPS):
                lum = retina.sample_images(x, k * cfg.dt, d)
                lum = torch.cat([lum, torch.full((lum.shape[0], 1), GRAY, device="cuda")], 1)
                eye = retina.transduce(lum)
                pr = pr + (eye[:, :-1] - eye[:, -1:])
                st = sim.step(st, eye)
                full = torch.zeros(c.n, B + 1, device="cuda")
                full[torch.as_tensor(sim.gi, device="cuda")] = st.x
                full[torch.as_tensor(sim.li, device="cuda")] = st.s
                acc = acc + full
            acc_v = acc_v + (acc[:, :-1] - acc[:, -1:]) / STEPS
            acc_pr = acc_pr + pr / STEPS
        for name, idx in views.items():
            out[name].append((acc_v[torch.as_tensor(idx, device="cuda")] / len(dirs)).T.float().cpu())
        out["photoreceptor"].append((acc_pr / len(dirs)).T.float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def score(Ftr, Fte, ytr, yte):
    (Str, Ste, k), = probe.standardize_pca(Ftr, Fte, (K,)).values()
    accs = [float((probe.fit_probe(Str, ytr, Ste, s)["pred"] == yte).mean()) for s in (0, 1, 2)]
    return float(np.mean(accs)), k


if __name__ == "__main__":
    N_TR = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    N_TE = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    DRIFT = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
    t0 = time.time()
    c = connectome.load()
    cfg, eye_cfg, _ = frozen.load()
    M = masks.build(c)
    views = {"visual_projection": M["visual_projection"], "central_vnc": M["central_vnc"],
             "optic_lobe": np.flatnonzero(c.graded)}
    r16 = c.types(["R1-6"])
    driven = r16[c.column[r16, 0] >= 0]
    xtr, ytr = tiles(N_TR, 0)
    xte, yte = tiles(N_TE, 1)
    ytr, yte = ytr.cpu().numpy(), yte.cpu().numpy()
    torch.save({"x": xte[:16].cpu(), "y": yte[:16]}, OUT / "d1_examples.pt")
    res = {"pixels": score(xtr.flatten(1).cpu().numpy(), xte.flatten(1).cpu().numpy(), ytr, yte)[0]}
    print(f"pixels            {res['pixels']*100:.2f}", flush=True)
    ec = dataclasses.replace(eye_cfg, drift_extent=DRIFT)
    sim = Sim(c, cfg, driven=driven)
    retina = Retina(c, driven, ec, cfg.dt)
    Ftr = brain_features(c, cfg, sim, retina, xtr, ec.drift_directions, views)
    Fte = brain_features(c, cfg, sim, retina, xte, ec.drift_directions, views)
    for v in Ftr:
        res[v] = score(Ftr[v], Fte[v], ytr, yte)[0]
        print(f"{v:17s} {res[v]*100:.2f}", flush=True)
    res.update({"n_train": N_TR, "n_test": N_TE, "drift": DRIFT, "size": SIZE, "chance": 0.25,
                "minutes": (time.time() - t0) / 60})
    (OUT / f"d1_layout_drift{DRIFT:g}.json").write_text(json.dumps(res, indent=1))
    print(f"done in {res['minutes']:.1f} min", flush=True)
