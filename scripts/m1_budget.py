"""M1: can wiring structure buy SAMPLE EFFICIENCY where hand-tuned detectors are not tuned to the task?

Task pixel_mix16 (letter video, 4x4 cells): one cell moves in a random direction (L/R/U/D), a second cell
only flickers (changes every frame, no net motion). Target = the moving cell. Selectors are trained on
n_train videos for n_train in {50, 100, 300, 1000, 2100} and scored on the same 900 test videos.

  fly        real MaleCNS, 4 quadrant glimpses, optic-lobe change energy -> linear probe
  hr_bank    Hassenstein-Reichardt opponent arrays on 2 px-blurred pixels, horizontal AND vertical,
             energy Rx^2 + Ry^2 (flicker cancels in opponent outputs) -> linear probe
  framediff  2 px-blurred frame-difference energy -> linear probe (fooled by flicker)
  cnn3d      small 3D CNN on the pooled video, trained from scratch on the same n_train videos
At the full budget InternVL3-1B reads the chosen cell's letter for fly / hr_bank / cnn3d.

usage: m1_budget.py [n]
"""
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("V3_TASK", "pixel_mix16")          # pixel_fg16 = M2 figure-ground
TASK = os.environ["V3_TASK"]
sys.argv = [sys.argv[0], TASK] + sys.argv[1:]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3b_vlm as V  # noqa: E402
from v3c_glimpse import glimpse_energy  # noqa: E402
from flyvl import connectome, frozen, probe, teacher  # noqa: E402
from flyvl.sim import Sim  # noqa: E402
from flyvl.stimulus import Retina, _gaussian_blur  # noqa: E402

P = V.P
BUDGETS = (50, 100, 300, 1000, 2100)
OUT = connectome.DATA_ROOT / "runs" / "mix"
OUT.mkdir(parents=True, exist_ok=True)


def video(p, N, cfg, blur=0, pool=2, chunk=500):
    """(N, T, H/pool, W/pool) float16 on the GPU."""
    frames = []
    for k in range(P.STEPS):
        fr = torch.cat([(_gaussian_blur(f, blur) if blur else f) for f in
                        (V.letter_frames({kk: v[s:s + chunk] for kk, v in p.items()}, k * cfg.dt) for s in range(0, N, chunk))])
        frames.append(F.avg_pool2d(fr, pool)[:, 0].half())
    return torch.stack(frames, 1)


class CNN3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(1, 16, (3, 5, 5), padding=(1, 2, 2)), torch.nn.ReLU(), torch.nn.MaxPool3d((2, 2, 2)),
            torch.nn.Conv3d(16, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool3d((2, 2, 2)),
            torch.nn.Conv3d(32, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d((1, 4, 4)), torch.nn.Flatten(), torch.nn.Linear(32 * 16, 16))

    def forward(self, x):
        return self.net(x[:, None].float())


def train_cnn(X, y, tr, te, epochs=60, seed=0):
    torch.manual_seed(seed)
    m = CNN3D().cuda()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    yt = torch.as_tensor(y, device="cuda")
    X = (X - X[tr].float().mean()) / X[tr].float().std()
    steps = max(epochs * len(tr) // 32, 300)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        b = torch.as_tensor(tr)[torch.randint(0, len(tr), (32,), generator=g)]
        loss = F.cross_entropy(m(X[b]), yt[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return torch.cat([m(X[te[s:s + 100]]).argmax(1) for s in range(0, len(te), 100)]).cpu().numpy()


READOUT = os.environ.get("READOUT", "linear")      # "mlp": the same small nonlinear readout for every feature arm


def mlp_select(Z, y, tr, te, seed=0):
    """Same 2-layer MLP for every arm. Needed when the answer is a nonlinear function of the features,
    e.g. |local direction - global direction| in figure-ground (a linear readout cannot flip signs)."""
    torch.manual_seed(seed)
    Z = torch.as_tensor(Z, device="cuda", dtype=torch.float32)
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6
    Z = (Z - mu) / sd
    m = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                            torch.nn.Linear(256, 16)).cuda()
    opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    yt = torch.as_tensor(y, device="cuda")
    g = torch.Generator().manual_seed(seed)
    for _ in range(max(60 * len(tr) // 32, 300)):
        b = torch.as_tensor(tr)[torch.randint(0, len(tr), (32,), generator=g)]
        loss = F.cross_entropy(m(Z[b]), yt[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return m(Z[torch.as_tensor(te)]).argmax(1).cpu().numpy()


def probe_select(feat, y, tr, te, name):
    Pm = torch.randn(1024, feat.shape[1], generator=torch.Generator().manual_seed(P.view_seed(name))) / np.sqrt(1024)
    Z = (torch.as_tensor(feat, device="cuda") @ Pm.to("cuda").T).cpu().numpy()
    if READOUT == "mlp":
        return mlp_select(Z, y, tr, te)
    (Str, Ste, _), = probe.standardize_pca(Z[tr], Z[te], (1024,)).values()
    return probe.fit_probe(Str, y[tr], Ste, 0)["pred"]


if __name__ == "__main__":
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    t0 = time.time()
    y, p = P.trial_params(np.random.default_rng(0), TASK, N)
    p["letters"] = np.random.default_rng(1).integers(0, 26, (N, 16))
    te = np.arange(2100, N)
    c = connectome.load()
    cfg, eye_cfg, _ = frozen.load()
    eye_cfg = dataclasses.replace(eye_cfg, image_half_width=1.0, drift_extent=0.0)
    r16 = c.types(["R1-6"])
    driven = r16[c.column[r16, 0] >= 0]
    sim, retina = Sim(c, cfg, driven=driven), Retina(c, driven, eye_cfg, cfg.dt)
    ol = torch.as_tensor(np.arange(sim.Ng), device="cuda")
    feats = {"fly": np.concatenate([glimpse_energy(c, cfg, sim, retina, p, q, ol, frames=V.letter_frames, change=True)
                                    for q in range(4)], 1)}
    # flyvis: the published connectome-constrained optic lobe with trained, direction-selective T4/T5
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flyvl.flyvis_encoder import FlyvisEncoder, features as fv_features
    fv = fv_features(FlyvisEncoder(), V.letter_frames, p, N, P.STEPS, cfg.dt)
    feats["flyvis_mean"], feats["flyvis_energy"] = fv["mean"], fv["energy"]
    feats["flyvis"] = np.concatenate([fv["mean"], fv["energy"]], 1)
    vb = video(p, N, cfg, blur=2)                                   # blurred, pooled video (N, T, 64, 64)
    d = (vb[:, 1:].float() - vb[:, :-1].float())
    feats["framediff"] = (d ** 2).mean(1).flatten(1).cpu().numpy()
    D = 1
    b0, b1 = vb[:, :-1].float(), vb[:, 1:].float()
    rx = (b0[..., :-D] * b1[..., D:] - b0[..., D:] * b1[..., :-D]).mean(1)          # horizontal opponent
    ry = (b0[..., :-D, :] * b1[..., D:, :] - b0[..., D:, :] * b1[..., :-D, :]).mean(1)  # vertical opponent
    feats["hr_bank"] = (F.pad(rx, (0, D)) ** 2 + F.pad(ry, (0, 0, 0, D)) ** 2).flatten(1).cpu().numpy()
    # signed opponent maps too: with them a linear readout can find "the cell whose direction differs"
    feats["hr_signed"] = torch.cat([F.pad(rx, (0, D)).flatten(1), F.pad(ry, (0, 0, 0, D)).flatten(1)], 1).cpu().numpy()
    del b0, b1, d, rx, ry
    vid = video(p, N, cfg)                                          # unblurred pooled video for the CNN
    print(f"  features ready {(time.time()-t0)/60:.1f} min", flush=True)

    res, picks = {}, {}
    for n in BUDGETS:
        tr = np.arange(n)
        for k, f in feats.items():
            picks[(k, n)] = probe_select(f, y, tr, te, f"m1_{k}")
        picks[("cnn3d", n)] = train_cnn(vid, y, tr, te)
        for k in list(feats) + ["cnn3d"]:
            res[f"{k}:n{n}"] = float((picks[(k, n)] == y[te]).mean())
        print(f"n={n:5d}  " + "  ".join(f"{k} {res[f'{k}:n{n}']*100:5.1f}" for k in list(feats) + ["cnn3d"]), flush=True)

    proc, model = teacher.load()
    pt = {k: v[te] for k, v in p.items()}
    last = V.letter_frames(pt, (P.STEPS - 1) * cfg.dt)
    answer = [V.LETTERS[i] for i in p["letters"][te, y[te]]]
    for k in ("fly", "flyvis", "hr_bank", "hr_signed", "cnn3d", "framediff"):
        pred = V.ask(proc, model, V.cell_crop(last, torch.as_tensor(picks[(k, 2100)], device="cuda")))
        res[f"vqa_{k}"] = float(np.mean([a == b for a, b in zip(pred, answer)]))
        print(f"VQA {k:10s} {res[f'vqa_{k}']*100:.2f}", flush=True)
    res.update({"n": N, "budgets": BUDGETS, "minutes": (time.time() - t0) / 60})
    (OUT / f"{TASK}_budget_{READOUT}.json").write_text(json.dumps(res, indent=1))
    print(f"done in {res['minutes']:.1f} min", flush=True)
