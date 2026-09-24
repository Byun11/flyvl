"""P5: fly-domain visual tasks through the retina (no hand-made detectors), real vs matched shuffle vs no brain.

Stimuli (synthetic, parameterised, rendered onto the same eye model as every other experiment):
  looming_side : a dark disc expands at azimuth +-a  -> label = left / right            (2 classes)
  motion_dir   : a grating drifts front-to-back / back-to-front / up / down             (4 classes)
  loom_speed   : disc expands slowly / quickly (side randomised)                        (2 classes)
Basic visual characterisation (STATIC stimuli - no motion, so these measure spatial vision, which the
motion tasks cannot):
  orient_static: a stationary grating is horizontal or vertical                          (2 classes)
  acuity_f<F>  : orient_static at a FIXED spatial frequency F -> acuity curve vs F
  position     : a dark spot sits in one of 4 quadrants                                  (4 classes)
  size         : a dark disc is small or large, position randomised                      (2 classes)
Each trial has randomised nuisance parameters (position, size, speed, contrast, phase) so the label cannot be
read off a single global statistic.

Readout: time-mean evoked activity (image - blank) of a neuron set -> fixed random projection (1024) ->
standardize -> multinomial logistic regression (same probe code as P1). Conditions:
  real / matched_shuffle_s0 / nobrain (the photoreceptor input itself)
Views: descending (1,314), central_vnc, visual_projection.
usage: p5_flytask.py TASK [n_trials]
"""
import dataclasses
import hashlib
import os
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
SERIES_VIEWS = ("optic_lobe", "visual_projection", "central_vnc")   # also read WITHOUT the time average
# A per-step RANDOM PROJECTION destroys individual neurons' time courses, which is exactly what a
# delay-and-multiply motion computation needs. So the series readout keeps a fixed random SUBSET of
# neurons and their full trajectories instead: N_SUB x STEPS dims, no mixing across neurons.
# N_SUB was first set to 400 without checking the actual limit, which put the series readout at a
# 23x neuron disadvantage against the full-view time mean (9,201 for visual_projection). Raised: at
# N_SUB x STEPS = 50,000 dims the 1024-d projection matrix is ~205 MB, which is comfortable.
N_SUB = 2000
SKIP_SERIES = os.environ.get("SKIP_SERIES") == "1"   # the per-neuron time-series probes are the heavy part
NOISE_SEED = 777         # fixed so that photoreceptor noise is identical across runs and graphs
OUT = connectome.DATA_ROOT / "runs" / "p5"
OUT.mkdir(parents=True, exist_ok=True)


def view_seed(view: str) -> int:
    """Deterministic across processes: Python's str hash is randomised per interpreter (PYTHONHASHSEED),
    so hash(view) gave a different random projection on every run and absolute numbers drifted between
    runs (graph comparisons inside one run were still valid, since all graphs shared the projection)."""
    return int(hashlib.md5(view.encode()).hexdigest()[:8], 16) % 2**31


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
    elif task == "pixel_where16":
        # V3a: the V2 stimulus as a PIXEL video (what a VLM receives) fed through the eye's image path.
        y = rng.integers(0, 16, n)
        p["cell"] = y
        p["freq"] = rng.uniform(2.0, 5.0, n)
        p["speed"] = rng.choice([-1.0, 1.0], n) * rng.uniform(0.6, 1.4, n)
        p["phase"] = rng.uniform(0, 2 * np.pi, n)
        p["contrast"] = rng.uniform(0.04, 0.10, n)
        p["noise"] = np.full(n, 0.06)
    elif task == "motion_where16":
        # V2: same stimulus, but the drifting patch sits in one of a 4x4 grid of locations (16 classes)
        y = rng.integers(0, 16, n)
        az_c, el_c = np.array([-0.7, -0.25, 0.25, 0.7]), np.array([-0.6, -0.2, 0.2, 0.6])
        p["az"] = az_c[y % 4] + rng.uniform(-0.05, 0.05, n)
        p["el"] = el_c[y // 4] + rng.uniform(-0.05, 0.05, n)
        p["rad"] = rng.uniform(0.12, 0.16, n)
        p["freq"] = rng.uniform(2.0, 5.0, n)
        p["speed"] = rng.choice([-1.0, 1.0], n) * rng.uniform(0.6, 1.4, n)
        p["phase"] = rng.uniform(0, 2 * np.pi, n)
        p["contrast"] = rng.uniform(0.04, 0.10, n)
        p["noise"] = np.full(n, 0.06)
    elif task == "motion_where":
        # V1: static grating everywhere; only one quadrant's patch drifts. Low contrast + noise (P5 hard regime).
        y = rng.integers(0, 4, n)                             # quadrant: (left/right) x (down/up)
        p["az"] = np.where(y % 2 == 0, -1.0, 1.0) * rng.uniform(0.3, 0.6, n)
        p["el"] = np.where(y // 2 == 0, -1.0, 1.0) * rng.uniform(0.2, 0.5, n)
        p["rad"] = rng.uniform(0.2, 0.3, n)
        p["freq"] = rng.uniform(2.0, 5.0, n)
        p["speed"] = rng.choice([-1.0, 1.0], n) * rng.uniform(0.6, 1.4, n)
        p["phase"] = rng.uniform(0, 2 * np.pi, n)
        p["contrast"] = rng.uniform(0.04, 0.10, n)
        p["noise"] = np.full(n, 0.06)
    elif task == "orient_static" or task.startswith("acuity_f"):
        # No drift at all: the grating is frozen. Direction-selective machinery cannot help here.
        y = rng.integers(0, 2, n)
        p["orient"] = y                                       # 0 = varies along azimuth, 1 = along elevation
        p["freq"] = (rng.uniform(1.5, 6.0, n) if task == "orient_static"
                     else np.full(n, float(task.split("f")[1])))
        p["phase"] = rng.uniform(0, 2 * np.pi, n)
        p["contrast"] = rng.uniform(0.25, 0.45, n)
        p["noise"] = np.zeros(n)
    elif task == "position":
        y = rng.integers(0, 4, n)                             # quadrant: (left/right) x (down/up)
        p["az"] = np.where(y % 2 == 0, -1.0, 1.0) * rng.uniform(0.3, 0.6, n)
        p["el"] = np.where(y // 2 == 0, -1.0, 1.0) * rng.uniform(0.3, 0.6, n)
        p["rad"] = rng.uniform(0.12, 0.20, n)
        p["contrast"] = rng.uniform(0.6, 1.0, n)
    elif task == "size":
        y = rng.integers(0, 2, n)
        p["az"] = rng.uniform(-0.6, 0.6, n)
        p["el"] = rng.uniform(-0.5, 0.5, n)
        p["rad"] = np.where(y == 0, rng.uniform(0.08, 0.13, n), rng.uniform(0.22, 0.32, n))
        p["contrast"] = rng.uniform(0.6, 1.0, n)
    else:
        raise ValueError(TASK)
    return y, p


PIX = 128   # V3a video frame size (4 x 4 cells of 32 px)


def pixel_frames(p, t, device="cuda"):
    """(B, 1, PIX, PIX) video frame at time t: static grating everywhere, the texture drifts only inside
    one cell of a 4x4 grid, fresh pixel noise every frame (seeded per step like the eye-space tasks)."""
    B = len(p["cell"])
    g = lambda k: torch.as_tensor(p[k], dtype=torch.float32, device=device)[:, None, None]
    ax = torch.linspace(-1, 1, PIX, device=device)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    cell = torch.as_tensor(p["cell"], device=device)
    cx = ((xx + 1) / 2 * 4).clamp(max=3.999).floor()[None]
    cy = ((yy + 1) / 2 * 4).clamp(max=3.999).floor()[None]
    inside = ((cy * 4 + cx) == cell[:, None, None]).float()
    lum = GRAY + g("contrast") * torch.sin(2 * np.pi * g("freq") * (xx[None] - inside * g("speed") * t) + g("phase"))
    gen = torch.Generator(device=device).manual_seed(NOISE_SEED + int(round(t / 0.02)))
    lum = lum + torch.randn(lum.shape, generator=gen, device=device) * g("noise")
    return lum[:, None]


def render(retina, task, p, t, device="cuda"):
    """Luminance (n_driven, B) at time t for all trials."""
    phi, th = retina.phi[:, None], retina.theta[:, None]
    g = lambda k: torch.as_tensor(p[k], dtype=torch.float32, device=device)[None]
    if task.startswith("motion_dir"):
        pass
    if task == "orient_static" or task.startswith("acuity_f"):
        coord = torch.where(g("orient") > 0.5, th.expand_as(phi + g("orient")), phi.expand_as(th + g("orient")))
        return GRAY + g("contrast") * torch.sin(2 * np.pi * g("freq") * coord + g("phase"))
    if task == "pixel_where16":
        return retina.sample_images(pixel_frames(p, t, device), 0.0, "LR")
    if task in ("motion_where", "motion_where16"):
        inside = (torch.sqrt((phi - g("az")) ** 2 + ((th - g("el")) * 0.5) ** 2) < g("rad")).float()
        lum = GRAY + g("contrast") * torch.sin(2 * np.pi * g("freq") * (phi - inside * g("speed") * t) + g("phase"))
        gen = torch.Generator(device=lum.device).manual_seed(NOISE_SEED + int(round(t / 0.02)))
        return lum + torch.randn(lum.shape, generator=gen, device=lum.device, dtype=lum.dtype) * g("noise")
    if task in ("position", "size"):                        # static dark disc, no expansion
        d = torch.sqrt((phi - g("az")) ** 2 + ((th - g("el")) * 0.5) ** 2)
        return torch.where(d < g("rad"), (GRAY * (1 - g("contrast"))).expand_as(d),
                           torch.full_like(d, GRAY))
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
        # Seeded per (trial block, step): unseeded noise made the SAME condition vary by up to 3.6%p
        # between runs, which silently invalidated every across-run absolute comparison.
        gen = torch.Generator(device=lum.device).manual_seed(NOISE_SEED + int(round(t / 0.02)))
        lum = lum + torch.randn(lum.shape, generator=gen, device=lum.device, dtype=lum.dtype) * g("noise")
    return lum


@torch.no_grad()
def features(graph, y, p, c, cfg, views, batch=64):
    if graph == "nobrain":
        sim = None
    r16 = c.types(["R1-6"])
    driven = r16[c.column[r16, 0] >= 0]
    eye_cfg = frozen.load()[1]
    if TASK.startswith("pixel_"):   # spread the image over the whole field of view, no extra drift (the video moves)
        eye_cfg = dataclasses.replace(eye_cfg, image_half_width=1.0, drift_extent=0.0)
    retina = Retina(c, driven, eye_cfg, cfg.dt)
    series_views = [v for v in views if v in SERIES_VIEWS] if graph != "nobrain" else []
    # `{v}_submean` = time mean of the SAME N_SUB neurons as `{v}_series`, so that series-vs-mean is not
    # confounded by neuron count (the full view has 9k-95k neurons, the series subset has N_SUB).
    out = {k: [] for k in list(views) + [f"{v}_series" for v in series_views]
           + [f"{v}_submean" for v in series_views] + ["photoreceptor", "photoreceptor_series", "framediff"]
           + (["optic_lobe_energy"] if graph != "nobrain" and "optic_lobe" in views else [])
           + (["pixel_framediff"] if TASK.startswith("pixel_") and graph == "nobrain" else [])}
    sub = {v: np.random.default_rng(view_seed(v)).choice(len(views[v]),
                                                         min(N_SUB, len(views[v])), replace=False)
           for v in series_views}
    vidx = {v: torch.as_tensor(views[v][sub[v]], device="cuda") for v in series_views}
    ol_idx = torch.as_tensor(views["optic_lobe"], device="cuda") if "optic_lobe" in views else None
    sim = None if graph == "nobrain" else Sim(c, cfg, W=load_graph(c, graph, cfg), driven=driven)
    for s in range(0, len(y), batch):
        sl = slice(s, min(s + batch, len(y)))
        B = sl.stop - sl.start
        pb = {k: v[sl] for k, v in p.items()}
        retina.reset(B + 1)                                            # last column = blank
        if sim is not None:
            st = sim.zero_state(B + 1)
        acc = None
        pr, series, fd, prev, ol_en = 0, [], 0, None, 0
        pfd, pprev = 0, None
        vseries = {v: [] for v in series_views}
        for k in range(STEPS):
            lum = torch.cat([render(retina, TASK, pb, k * cfg.dt), torch.full((len(driven), 1), GRAY, device="cuda")], 1)
            eye = retina.transduce(lum)
            step_evoked = eye[:, :-1] - eye[:, -1:]
            pr = pr + step_evoked
            series.append(step_evoked)                                 # keep time, do not average
            # classic free motion detector: per-photoreceptor energy of the frame-to-frame change
            if prev is not None:
                fd = fd + (step_evoked - prev) ** 2
            prev = step_evoked
            if "pixel_framediff" in out:   # the free detector a VLM pipeline could run on the video itself
                fr = torch.nn.functional.avg_pool2d(pixel_frames(pb, k * cfg.dt), 2).flatten(1)
                if pprev is not None:
                    pfd = pfd + (fr - pprev) ** 2
                pprev = fr
            if sim is not None:
                st = sim.step(st, eye)
                x = torch.zeros(c.n, B + 1, device="cuda")
                x[torch.as_tensor(sim.gi, device="cuda")] = st.x
                x[torch.as_tensor(sim.li, device="cuda")] = st.s
                acc = x if acc is None else acc + x
                if "optic_lobe_energy" in out:
                    # V2 fix: a signed time mean cancels oscillating motion responses; read their energy
                    # instead, the same operation framediff applies to the eye.
                    xo = x[ol_idx]
                    ol_en = ol_en + (xo[:, :-1] - xo[:, -1:]) ** 2
                for v in series_views:                                 # evoked state of the SAME neurons
                    xv = x[vidx[v]]
                    vseries[v].append(xv[:, :-1] - xv[:, -1:])
        out["photoreceptor"].append((pr / STEPS).T.float().cpu())
        out["framediff"].append((fd / STEPS).T.float().cpu())
        if "pixel_framediff" in out:
            out["pixel_framediff"].append((pfd / STEPS).float().cpu())
        # same eye signal WITHOUT the time average: the honest no-brain baseline for a motion task,
        # since a time-averaged drifting grating cannot carry direction by construction.
        out["photoreceptor_series"].append(torch.stack(series).permute(2, 0, 1).reshape(B, -1).float().cpu())
        for v in series_views:
            st_v = torch.stack(vseries[v])                             # (STEPS, N_SUB, B)
            out[f"{v}_series"].append(st_v.permute(2, 0, 1).reshape(B, -1).float().cpu())
            out[f"{v}_submean"].append(st_v.mean(0).T.float().cpu())
        if sim is not None:
            if "optic_lobe_energy" in out:
                out["optic_lobe_energy"].append((ol_en / STEPS).T.float().cpu())
            ev = (acc[:, :-1] - acc[:, -1:]) / STEPS
            for k, idx in views.items():
                out[k].append(ev[torch.as_tensor(idx, device="cuda")].T.float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items() if v}


if __name__ == "__main__":
    c = connectome.load()
    cfg, _, _ = frozen.load()
    M = masks.build(c)
    views = {"descending": M["descending"], "central_vnc": M["central_vnc"],
             "visual_projection": M["visual_projection"],
             "optic_lobe": np.flatnonzero(c.graded)}     # read the optic lobe directly, bypassing the brain
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
            if graph == "nobrain" and not view.startswith(("photoreceptor", "framediff", "pixel_framediff")):
                continue
            # Keep the projection RATIO, not the width: a fixed 1024-d projection discards 98% of a
            # 50,000-d series but only 49% of a 2,000-d mean, which alone can make the series lose.
            width = min(int(round(X.shape[1] * PROJ / max(X.shape[1] // STEPS, 1))), X.shape[1])                 if view.endswith("_series") else PROJ
            width = min(max(width, PROJ), 32768)
            if SKIP_SERIES and view.endswith("_series"):
                continue
            # same seeded CPU matrix as always (results stay reproducible); the multiply runs on the GPU
            P = torch.randn(width, X.shape[1], generator=torch.Generator().manual_seed(view_seed(view))) / np.sqrt(width)
            Z = (torch.as_tensor(X, device="cuda") @ P.to("cuda").T).cpu().numpy()
            # PCA K is a second bottleneck: a series' variance is dominated by phase/noise, so the
            # label-relevant direction can fall outside the top 256 PCs. Report every K, never pick on test.
            sc = probe.standardize_pca(Z[:n_tr], Z[n_tr:], (256, 1024, 2048))
            for K, (Str, Ste, k) in sc.items():
                accs = [float((probe.fit_probe(Str, y[:n_tr], Ste, seed)["pred"] == y[n_tr:]).mean()) for seed in (0, 1, 2)]
                res[f"{graph}:{view}:K{K}"] = {"acc": accs, "mean": float(np.mean(accs)), "k": k}
                if K == 256:
                    res[f"{graph}:{view}"] = {"acc": accs, "mean": float(np.mean(accs)), "k": k}
                print(f"{graph:20s} {view:24s} K{K:<5d} k={k:<5d} mean {np.mean(accs)*100:.2f} "
                      f"{[round(a*100,1) for a in accs]}", flush=True)
        print(f"  ({graph} {time.time()-t0:.0f}s)", flush=True)
    # MERGE, do not overwrite: each invocation runs a subset of graphs, and overwriting silently dropped
    # the graphs measured by earlier runs of the same task (it cost the `real` row once already).
    path = OUT / (f"{TASK}_t{trial_seed}.json" if trial_seed else f"{TASK}.json")
    merged = json.loads(path.read_text())["res"] if path.exists() else {}
    merged.update(res)
    path.write_text(json.dumps({"task": TASK, "n": N, "chance": 1 / len(set(y.tolist())), "res": merged},
                               indent=1))
