"""P5: fly-domain visual tasks through the retina (no hand-made detectors), real vs matched shuffle vs no brain.

Stimuli (synthetic, parameterised, rendered onto the same eye model as every other experiment):
  looming_side : a dark disc expands at azimuth +-a  -> label = left / right            (2 classes)
  motion_dir   : a grating drifts front-to-back / back-to-front / up / down             (4 classes)
  loom_speed   : disc expands slowly / quickly (side randomised)                        (2 classes)
Each trial has randomised nuisance parameters (position, size, speed, contrast, phase) so the label cannot be
read off a single global statistic.

Readout: time-mean evoked activity (image - blank) of a neuron set -> fixed random projection (1024) ->
standardize -> multinomial logistic regression (same probe code as P1). Conditions:
  real / matched_shuffle_s0 / nobrain (the photoreceptor input itself)
Views: descending (1,314), central_vnc, visual_projection.
usage: p5_flytask.py TASK [n_trials]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, frozen, masks, probe  # noqa: E402
from flyvl.extract import load_graph  # noqa: E402
from flyvl.sim import Sim  # noqa: E402
from flyvl.stimulus import GRAY, Retina  # noqa: E402

TASK = sys.argv[1] if len(sys.argv) > 1 else "looming_side"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
STEPS, PROJ = 25, 1024
OUT = connectome.DATA_ROOT / "runs" / "p5"
OUT.mkdir(parents=True, exist_ok=True)


def trial_params(rng, task, n):
    """Returns (labels, per-trial parameter dict of arrays)."""
    p = {}
    if task == "looming_side":
        y = rng.integers(0, 2, n)
        p["side"] = np.where(y == 0, -1.0, 1.0)
        p["az"] = p["side"] * rng.uniform(0.25, 0.65, n)
        p["el"] = rng.uniform(-0.4, 0.4, n)
        p["rate"] = rng.uniform(0.4, 0.9, n)
        p["contrast"] = rng.uniform(0.6, 1.0, n)
    elif task == "loom_speed":
        y = rng.integers(0, 2, n)
        p["side"] = rng.choice([-1.0, 1.0], n)
        p["az"] = p["side"] * rng.uniform(0.25, 0.65, n)
        p["el"] = rng.uniform(-0.4, 0.4, n)
        p["rate"] = np.where(y == 0, rng.uniform(0.25, 0.4, n), rng.uniform(0.8, 1.2, n))
        p["contrast"] = rng.uniform(0.6, 1.0, n)
    elif task == "loom_vs_recede":                       # same positions/sizes, only the temporal order differs
        y = rng.integers(0, 2, n)
        p["expand"] = np.where(y == 0, 1.0, -1.0)
        p["side"] = rng.choice([-1.0, 1.0], n)
        p["az"] = p["side"] * rng.uniform(0.25, 0.65, n)
        p["el"] = rng.uniform(-0.4, 0.4, n)
        p["rate"] = rng.uniform(0.5, 0.9, n)
        p["contrast"] = rng.uniform(0.6, 1.0, n)
    elif task in ("motion_dir", "motion_dir_hard"):
        hard = task.endswith("hard")
        y = rng.integers(0, 4, n)
        p["dir"] = y
        p["freq"] = rng.uniform(2.5, 5.0, n) if not hard else rng.uniform(2.0, 7.0, n)
        p["speed"] = rng.uniform(0.7, 1.3, n) if not hard else rng.uniform(0.5, 1.6, n)
        p["phase"] = rng.uniform(0, 2 * np.pi, n)
        p["contrast"] = rng.uniform(0.25, 0.45, n) if not hard else rng.uniform(0.04, 0.10, n)
        p["noise"] = np.full(n, 0.0 if not hard else 0.06)
    else:
        raise ValueError(TASK)
    return y, p


def render(retina, task, p, t, device="cuda"):
    """Luminance (n_driven, B) at time t for all trials."""
    phi, th = retina.phi[:, None], retina.theta[:, None]
    g = lambda k: torch.as_tensor(p[k], dtype=torch.float32, device=device)[None]
    if task.startswith("motion_dir"):
        pass
    if task in ("looming_side", "loom_speed", "loom_vs_recede"):
        if task == "loom_vs_recede":
            span = g("rate") * (STEPS - 1) * 0.02
            grow = 0.02 + g("rate") * t
            r = torch.where(g("expand") > 0, grow, 0.02 + span - g("rate") * t).clamp(0.02, 0.6)
        else:
            r = (0.02 + g("rate") * t).clamp(max=0.6)
        d = torch.sqrt((phi - g("az")) ** 2 + (th * 0.5) ** 2)
        dark = GRAY * (1 - g("contrast"))
        return torch.where(d < r, dark.expand_as(d), torch.full_like(d, GRAY))
    fb = (phi.abs() - 0.05) / 0.95    # motion tasks
    along = torch.stack([2 * fb - 1, 1 - 2 * fb, th, -th], 0)          # ftb, btf, up, down
    coord = along[torch.as_tensor(p["dir"], device=device), torch.arange(len(phi), device=device)[:, None],
                  torch.arange(len(p["dir"]), device=device)[None]] if False else None
    idx = torch.as_tensor(p["dir"], dtype=torch.long, device=device)
    coord = torch.stack([along[i, :, 0] for i in range(4)], 0)[idx].T   # (n_driven, B)
    lum = GRAY + g("contrast") * torch.sin(2 * np.pi * g("freq") * (coord - g("speed") * t) + g("phase"))
    if "noise" in p and float(p["noise"][0]) > 0:                       # sensor noise, redrawn every step
        lum = lum + torch.randn_like(lum) * g("noise")
    return lum


@torch.no_grad()
def features(graph, y, p, c, cfg, views, batch=64):
    if graph == "nobrain":
        sim = None
    r16 = c.types(["R1-6"])
    driven = r16[c.column[r16, 0] >= 0]
    eye_cfg = frozen.load()[1]
    retina = Retina(c, driven, eye_cfg, cfg.dt)
    out = {k: [] for k in list(views) + ["photoreceptor"]}
    sim = None if graph == "nobrain" else Sim(c, cfg, W=load_graph(c, graph, cfg), driven=driven)
    for s in range(0, len(y), batch):
        sl = slice(s, min(s + batch, len(y)))
        B = sl.stop - sl.start
        pb = {k: v[sl] for k, v in p.items()}
        retina.reset(B + 1)                                            # last column = blank
        if sim is not None:
            st = sim.zero_state(B + 1)
        acc = None
        pr = 0
        for k in range(STEPS):
            lum = torch.cat([render(retina, TASK, pb, k * cfg.dt), torch.full((len(driven), 1), GRAY, device="cuda")], 1)
            eye = retina.transduce(lum)
            pr = pr + (eye[:, :-1] - eye[:, -1:])
            if sim is not None:
                st = sim.step(st, eye)
                x = torch.zeros(c.n, B + 1, device="cuda")
                x[torch.as_tensor(sim.gi, device="cuda")] = st.x
                x[torch.as_tensor(sim.li, device="cuda")] = st.s
                acc = x if acc is None else acc + x
        out["photoreceptor"].append((pr / STEPS).T.float().cpu())
        if sim is not None:
            ev = (acc[:, :-1] - acc[:, -1:]) / STEPS
            for k, idx in views.items():
                out[k].append(ev[torch.as_tensor(idx, device="cuda")].T.float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items() if v}


if __name__ == "__main__":
    c = connectome.load()
    cfg, _, _ = frozen.load()
    M = masks.build(c)
    views = {"descending": M["descending"], "central_vnc": M["central_vnc"],
             "visual_projection": M["visual_projection"]}
    trial_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    rng = np.random.default_rng(trial_seed)
    y, p = trial_params(rng, TASK, N)
    n_tr = int(N * 0.7)
    print(f"task={TASK} n={N} classes={len(set(y.tolist()))} train={n_tr}", flush=True)
    res = {}
    graphs = sys.argv[3].split(",") if len(sys.argv) > 3 else ["real", "matched_shuffle_s0", "nobrain"]
    for graph in graphs:
        t0 = time.time()
        F = features(graph, y, p, c, cfg, views)
        gproj = torch.Generator().manual_seed(0)
        for view, X in F.items():
            if graph == "nobrain" and view != "photoreceptor":
                continue
            P = (torch.randn(PROJ, X.shape[1], generator=torch.Generator().manual_seed(hash(view) % 2**31)) /
                 np.sqrt(PROJ)).numpy().astype(np.float32)
            Z = X @ P.T
            sc = probe.standardize_pca(Z[:n_tr], Z[n_tr:], (256,))
            for K, (Str, Ste, k) in sc.items():
                accs = [float((probe.fit_probe(Str, y[:n_tr], Ste, seed)["pred"] == y[n_tr:]).mean()) for seed in (0, 1, 2)]
                res[f"{graph}:{view}"] = {"acc": accs, "mean": float(np.mean(accs)), "k": k}
                print(f"{graph:20s} {view:20s} acc {[round(a*100,1) for a in accs]} mean {np.mean(accs)*100:.2f}", flush=True)
        print(f"  ({graph} {time.time()-t0:.0f}s)", flush=True)
    (OUT / f"{TASK}_t{trial_seed}.json" if trial_seed else OUT / f"{TASK}.json").write_text(json.dumps({"task": TASK, "n": N, "chance": 1/len(set(y.tolist())), "res": res}, indent=1))
