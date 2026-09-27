"""E5': closed-loop virtual flight arena on the GPU (the fly's own turning changes what it sees next).

Task (optomotor stabilisation): a low-contrast, noisy panorama; an external yaw disturbance w_d(t) keeps
rotating the fly; the policy outputs a yaw command u(t); heading changes by (w_d + u) dt. Score = mean
retinal slip |w_d + u| on held-out disturbances (lower is better). The eye sees a 64x64 window of the
panorama (+-FOV/2 around the heading) every 20 ms.

Arms (all with a LINEAR policy u = w . f + b trained by the same CEM, same budget, same disturbances):
  flyvis  T4a-d/T5a-d responses averaged over all columns (like wide-field lobula plate cells) -> 8 + 8 energy
  hr      Hassenstein-Reichardt on 2 px-blurred, 3-frame-smoothed frames (support-matched, E2b), global
          horizontal / vertical opponent outputs and their energy -> 4 features
  blind   f = [] (only the bias; cannot see)
  oracle  u = -w_d (reference floor: 0 slip)

usage: e5_arena.py [contrast] [noise] [generations]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome  # noqa: E402
from flyvl.flyvis_encoder import T45, FlyvisEncoder  # noqa: E402
from flyvl.stimulus import _gaussian_blur  # noqa: E402

CONTRAST = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
NOISE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.06
GENS = int(sys.argv[3]) if len(sys.argv) > 3 else 25
DT, T, PIX, FOV = 0.02, 50, 64, 2.0          # 1 s episodes, 2 rad field of view
POP, ELITE = 48, 8
OUT = connectome.DATA_ROOT / "runs" / "e5"
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"


def world(n, seed):
    """Per-episode panorama texture (random harmonics in azimuth, weak elevation pattern) and disturbance."""
    g = np.random.default_rng(seed)
    k = np.arange(2, 13)
    amp = g.uniform(0.5, 1.0, (n, len(k))) / np.sqrt(len(k))
    ph = g.uniform(0, 2 * np.pi, (n, len(k)))
    # disturbance: a constant spin plus a slow sinusoid, rad/s
    w0 = g.choice([-1, 1], n) * g.uniform(0.5, 2.0, n)
    w1, wf, wp = g.uniform(0, 1.0, n), g.uniform(0.5, 2.0, n), g.uniform(0, 2 * np.pi, n)
    t = np.arange(T) * DT
    wd = w0[:, None] + w1[:, None] * np.sin(2 * np.pi * wf[:, None] * t + wp[:, None])
    f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    return {"k": f(k), "amp": f(amp), "ph": f(ph), "wd": f(wd), "seed": seed}


def frame(w, idx, heading, step):
    """(B, PIX, PIX) luminance of the window around each episode's heading."""
    ax = torch.linspace(-FOV / 2, FOV / 2, PIX, device=dev)
    az = heading[:, None] + ax[None]                                                  # (B, PIX)
    tex = (w["amp"][idx][:, :, None] * torch.sin(w["k"][None, :, None] * az[:, None] + w["ph"][idx][:, :, None])).sum(1)
    elev = torch.cos(3 * torch.linspace(-1, 1, PIX, device=dev))[None, :, None]          # mild vertical structure
    lum = 0.5 + CONTRAST * tex[:, None, :] * (0.7 + 0.3 * elev)
    gen = torch.Generator(device=dev).manual_seed(777 + 1000 * w["seed"] + step)
    return lum + NOISE * torch.randn(lum.shape, generator=gen, device=dev)


class FlyObs:
    def __init__(self):
        self.enc = FlyvisEncoder()
        self.idx = torch.as_tensor(np.stack([self.enc.idx[t] for t in T45]), device=dev)   # (8, 721)
        self.dim = 16

    def reset(self, B):
        with torch.device(dev), torch.no_grad():
            self.state = self.enc.net.steady_state(t_pre=0.5, dt=DT, batch_size=B, value=0.5)
        self.base, self.prev = None, None

    @torch.no_grad()
    def __call__(self, img):
        net = self.enc.net
        with torch.device(dev):
            movie = self.enc.eye(img[:, None].clamp(0, 1))                                # (B, 1, 1, 721)
            net.stimulus.zero(img.shape[0], 1)
            net.stimulus.add_input(movie)
            states = net(net.stimulus(), DT, self.state, as_states=True)
            self.state = states[-1]
            x = self.state.nodes.activity[:, self.idx]                                     # (B, 8, 721)
        if self.base is None:
            self.base = x
        ev = x - self.base
        en = torch.zeros_like(ev) if self.prev is None else (ev - self.prev) ** 2
        self.prev = ev
        return torch.cat([ev.mean(-1), en.mean(-1) * 10], 1)                              # (B, 16)


class HRObs:
    dim = 4

    def reset(self, B):
        self.sm, self.prev = None, None

    @torch.no_grad()
    def __call__(self, img):
        b = _gaussian_blur(img[:, None], 2)[:, 0]
        self.sm = b if self.sm is None else b / 3 + self.sm * (2 / 3)                       # 3-frame smoothing
        out = torch.zeros(img.shape[0], 4, device=dev)
        if self.prev is not None:
            a, c = self.prev, self.sm
            rx = (a[..., :-1] * c[..., 1:] - a[..., 1:] * c[..., :-1]).mean((1, 2))
            ry = (a[..., :-1, :] * c[..., 1:, :] - a[..., 1:, :] * c[..., :-1, :]).mean((1, 2))
            out = torch.stack([rx, ry, rx ** 2, ry ** 2], 1) * 100
        self.prev = self.sm
        return out


class BlindObs:
    dim = 0

    def reset(self, B):
        pass

    def __call__(self, img):
        return torch.zeros(img.shape[0], 0, device=dev)


CHUNK = 64      # flyvis carries (batch x 1.5M edges) tensors; larger batches run out of GPU memory


def rollout(obs, W, w, idx, oracle=False):
    """W (B, dim + 1) policies, one per episode in idx. Returns mean |slip| per episode (chunked)."""
    if len(idx) > CHUNK:
        return torch.cat([rollout(obs, None if W is None else W[s:s + CHUNK], w, idx[s:s + CHUNK], oracle)
                          for s in range(0, len(idx), CHUNK)])
    B = len(idx)
    obs.reset(B)
    heading = torch.zeros(B, device=dev)
    slip = torch.zeros(B, device=dev)
    u = torch.zeros(B, device=dev)
    for s in range(T):
        f = obs(frame(w, idx, heading, s))
        if oracle:
            u = -w["wd"][idx, s]
        else:
            u = (torch.cat([f, torch.ones(B, 1, device=dev)], 1) * W).sum(1).clamp(-4, 4)
        v = w["wd"][idx, s] + u
        heading = heading + v * DT
        slip += v.abs()
    return (slip / T)


def cem(obs, w_train, n_train, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    D = obs.dim + 1
    mu, sd = torch.zeros(D, device=dev), torch.ones(D, device=dev)
    ep = torch.arange(n_train, device=dev)
    for gen in range(GENS):
        cand = mu + sd * torch.randn(POP, D, generator=g, device=dev)
        # every candidate is scored on the same n_train episodes (batched: POP x n_train rollouts)
        Wb = cand.repeat_interleave(n_train, 0)
        idx = ep.repeat(POP)
        score = rollout(obs, Wb, w_train, idx).view(POP, n_train).mean(1)
        elite = cand[score.argsort()[:ELITE]]
        mu, sd = elite.mean(0), elite.std(0) + 0.02
        if gen % 5 == 0 or gen == GENS - 1:
            print(f"    gen {gen:2d} best train slip {score.min().item():.3f}", flush=True)
    return mu


if __name__ == "__main__":
    t0 = time.time()
    n_train, n_test = 16, 64
    w_tr, w_te = world(n_train, seed=1), world(n_test, seed=2)
    res = {"contrast": CONTRAST, "noise": NOISE, "generations": GENS}
    te = torch.arange(n_test, device=dev)
    res["no_control"] = float(w_te["wd"].abs().mean())
    res["oracle"] = float(rollout(BlindObs(), None, w_te, te, oracle=True).mean())
    for name, obs in (("blind", BlindObs()), ("hr", HRObs()), ("flyvis", FlyObs())):
        print(f"arm {name}", flush=True)
        mu = cem(obs, w_tr, n_train)
        s = rollout(obs, mu[None].expand(n_test, -1), w_te, te)
        res[name] = float(s.mean())
        res[f"{name}_sem"] = float(s.std() / np.sqrt(n_test))
        print(f"  {name:7s} test slip {res[name]:.3f} +- {res[name + '_sem']:.3f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"no control {res['no_control']:.3f}  oracle {res['oracle']:.3f}", flush=True)
    (OUT / f"e5_c{CONTRAST:g}_n{NOISE:g}.json").write_text(json.dumps(res, indent=1))
