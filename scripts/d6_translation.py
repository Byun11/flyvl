"""D6 (PROTOCOL_D6_translation.md): input interface (fly retina | AI patch) x wiring (Real | Rewired-L) on the P5
low-contrast drift task. Frozen whole-MaleCNS dynamics, VPN readout, same optimizer and budget in every cell.
usage: d6_translation.py data
       d6_translation.py run CELL SEED     A retina/real, B retina/rewired_l, C ai/real, D ai/rewired_l,
                                           Bm retina/rewired_m (secondary)
       d6_translation.py hr SEED           HR correlator reference (blur selected on val)
       d6_translation.py verdict
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import flygrapher as fg  # noqa: E402
from flyvl.flyvig import BN, Stem  # noqa: E402

OUT = fg.connectome.DATA_ROOT / "d6"
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
NTR, NVA, NTE, FRAMES, PIX, FRAME_DT, STEPS_PER_FRAME = 1800, 300, 900, 8, 32, 0.04, 2
EPOCHS, BS, LR, WD, DELTA = 50, 64, 1e-3, 0.05, 0.01
GAIN = 4.0                                     # v2 INPUT_SCALE
SIGN = np.array([-1.0, -1.0, -1.0, 1.0, 1.0])  # L1 L2 L3 invert the photoreceptor signal; R7 R8 are photoreceptors
CELLS = {"A": ("retina", "real"), "B": ("retina", "rewired_l"), "C": ("ai", "real"), "D": ("ai", "rewired_l"),
         "Bm": ("retina", "rewired_m")}


# ------------------------------------------------------------------ data
def make_videos(n, seed):
    """Full-field drifting sinusoidal gratings, 4 directions (0 L->R, 1 R->L, 2 top->bottom, 3 bottom->top)."""
    g = np.random.default_rng(seed)
    y = g.integers(0, 4, n)
    sf, v = g.uniform(2, 7, n), g.uniform(0.5, 1.6, n)
    ph, c = g.uniform(0, 2 * np.pi, n), g.uniform(0.04, 0.10, n)
    ax = (np.arange(PIX) + 0.5) / PIX
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    coord = np.where((y < 2)[:, None, None], xx[None], yy[None])
    sign = np.where(y % 2 == 0, 1.0, -1.0)[:, None, None]
    X = np.empty((n, FRAMES, PIX, PIX), np.float32)
    for k in range(FRAMES):
        t = k * FRAME_DT
        X[:, k] = 0.5 + c[:, None, None] * np.sin(2 * np.pi * sf[:, None, None] * (coord - sign * v[:, None, None] * t)
                                                  + ph[:, None, None]) + g.normal(0, 0.06, (n, PIX, PIX))
    return X.astype(np.float16), y


def data():
    f = OUT / "videos.npz"
    if not f.exists():
        parts = {k: make_videos(n, s) for k, n, s in (("train", NTR, 601), ("val", NVA, 602), ("test", NTE, 603))}
        np.savez(f, **{f"{k}_x": v[0] for k, v in parts.items()}, **{f"{k}_y": v[1] for k, v in parts.items()})
    d = np.load(f)
    return {k: (torch.as_tensor(d[f"{k}_x"]), torch.as_tensor(d[f"{k}_y"])) for k in ("train", "val", "test")}


# ------------------------------------------------------------------ brain (frozen: g = 1, alpha = 0.5)
class FrozenBrain:
    def __init__(self, brain, kind, seed):
        W, self.info = fg.build_graph(brain, kind, seed)
        self.W, self.WT = fg.to_csr(W, dev), fg.to_csr(W.T, dev)
        self.n = W.shape[0]
        self.inputs = torch.as_tensor(brain.inputs, device=dev)
        vpn = np.flatnonzero(np.isin(brain.superclass, ("visual_projection", "visual_projection_tbc")))
        self.vpn = torch.as_tensor(vpn, device=dev)

    def __call__(self, u):
        """u (FRAMES, n_in, B) -> time-mean f(V) of the VPN neurons (n_vpn, B)."""
        B = u.shape[2]
        V = torch.zeros(self.n, B, device=dev)
        acc = torch.zeros(len(self.vpn), B, device=dev)
        for f in range(FRAMES):
            for _ in range(STEPS_PER_FRAME):
                drive = fg._SpMM.apply(self.W, self.WT, fg.act(V)).index_add(0, self.inputs, u[f])
                V = V + 0.5 * (drive - V)
                acc = acc + fg.act(V)[self.vpn]
        return acc / (FRAMES * STEPS_PER_FRAME)


def retina_drive(frames, brain):
    """Fixed fly-native encoding: every input neuron reads the frame at its column's visual position (bilinear);
    drive = sign(class) * GAIN * (luminance - 0.5). frames (B, FRAMES, PIX, PIX) -> (FRAMES, n_in, B)."""
    B = frames.shape[0]
    pos = torch.as_tensor(brain.in_pos, dtype=torch.float32, device=dev)
    grid = torch.stack([pos[:, 0], -pos[:, 1]], 1).view(1, 1, -1, 2).expand(B * FRAMES, 1, -1, 2)
    lum = F.grid_sample(frames.reshape(B * FRAMES, 1, PIX, PIX), grid, mode="bilinear", align_corners=False)
    lum = lum.view(B, FRAMES, -1)
    sign = torch.as_tensor(SIGN[brain.in_class], dtype=torch.float32, device=dev)
    return (sign * GAIN * (lum - 0.5)).permute(1, 2, 0).contiguous()


class AIEncoder(nn.Module):
    """D5's interface: per frame, conv stem -> 256 patch tokens (D = 195) -> Norm -> rank-5 P_in -> input neurons."""

    def __init__(self, brain, D=195):
        super().__init__()
        self.stem, self.norm, self.p_in = Stem(1, D, PIX), BN(D), fg.PatchIn(1, D)
        self.register_buffer("in_idx", torch.as_tensor(brain.in_patch * fg.N_IN + brain.in_class))

    def forward(self, frames):
        B = frames.shape[0]
        tok = self.norm(self.stem(frames.reshape(B * FRAMES, 1, PIX, PIX)))
        U = self.p_in(tok)[0].reshape(B * FRAMES, fg.N_PATCH * fg.N_IN)[:, self.in_idx]
        return U.view(B, FRAMES, -1).permute(1, 2, 0).contiguous()


class Readout(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.bn, self.lin = nn.BatchNorm1d(d, affine=False, eps=1e-12), nn.Linear(d, 4)   # appendix A: real z-scoring

    def forward(self, feat):                                     # (B, d)
        return self.lin(self.bn(feat))


# ------------------------------------------------------------------ training (identical in every cell)
def fit(params, logits_of, n_train, xs_val, y_val, xs_test, y_test, modules):
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    spe = math.ceil(n_train / BS)
    steps, warm = EPOCHS * spe, max(1, EPOCHS * spe // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    g = torch.Generator().manual_seed(torch.initial_seed())
    best, best_state, curve = -1.0, None, []

    def evaluate(xs, y):
        for m in modules:
            m.eval()
        ok = []
        with torch.no_grad():
            for s in range(0, len(y), 150):
                ok.append(logits_of(xs, slice(s, s + 150), train=False).argmax(1).cpu() == y[s:s + 150])
        for m in modules:
            m.train()
        return torch.cat(ok)

    for ep in range(EPOCHS):
        order = torch.randperm(n_train, generator=g)
        for i in range(spe):
            b = order[i * BS:(i + 1) * BS]
            loss = F.cross_entropy(logits_of(None, b, train=True), logits_of.y_train[b].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        acc = float(evaluate(xs_val, y_val).float().mean())
        curve.append((ep + 1, round(float(loss), 4), acc))
        if acc > best:
            best, best_state = acc, [{k: v.clone() for k, v in m.state_dict().items()} for m in modules]
    for m, st in zip(modules, best_state):
        m.load_state_dict(st)
    ok = evaluate(xs_test, y_test)
    return best, curve, ok


def run(cell, seed):
    name = f"{cell}_s{seed}"
    if (OUT / f"{name}.json").exists():
        return
    t0 = time.time()
    interface, kind = CELLS[cell]
    d = data()
    brain = fg.load_malecns()
    net = FrozenBrain(brain, kind, seed)
    torch.manual_seed(seed)
    readout = Readout(len(net.vpn)).to(dev)
    if interface == "retina":
        feats = {}
        with torch.no_grad():
            for k, (x, _) in d.items():
                parts = [net(retina_drive(x[s:s + 128].to(dev).float(), brain)).T.cpu() for s in range(0, len(x), 128)]
                feats[k] = torch.cat(parts)

        def logits_of(xs, idx, train):
            src = feats["train"] if train else xs
            return readout(src[idx].to(dev))
        logits_of.y_train = d["train"][1]
        best, curve, ok = fit(list(readout.parameters()), logits_of, NTR, feats["val"], d["val"][1],
                              feats["test"], d["test"][1], [readout])
        extra = {"feature_abs_mean": float(feats["train"].abs().mean()), "feature_std": float(feats["train"].std())}
    else:
        enc = AIEncoder(brain).to(dev)

        def logits_of(xs, idx, train):
            x = (d["train"][0] if train else xs)[idx].to(dev).float()
            return readout(net(enc(x)).T)
        logits_of.y_train = d["train"][1]
        best, curve, ok = fit(list(enc.parameters()) + list(readout.parameters()), logits_of, NTR, d["val"][0],
                              d["val"][1], d["test"][0], d["test"][1], [enc, readout])
        extra = {"encoder_params": sum(p.numel() for p in enc.parameters())}
    rec = {"cell": cell, "interface": interface, "graph": kind, "seed": seed, "graph_info": net.info, "val_best": best,
           "test_acc": float(ok.float().mean()), "test_ok": ok.int().tolist(), "curve": curve, "minutes": (time.time() - t0) / 60,
           **extra}
    (OUT / f"{name}.json").write_text(json.dumps(rec))
    print(f"{name}: test {rec['test_acc']:.4f} (val {best:.4f}, {rec['minutes']:.1f} min)", flush=True)


# ------------------------------------------------------------------ HR reference
def hr_features(x, sigma):
    """Opponent delayed-product (Hassenstein-Reichardt) maps, horizontal and vertical, delays 1 and 2 frames, averaged
    over 4 x 4 regions -> 64 fixed features. x (B, FRAMES, PIX, PIX) on the GPU."""
    I = x - 0.5
    if sigma > 0:
        r = int(3 * sigma)
        k = torch.exp(-torch.arange(-r, r + 1, device=dev, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
        k = k / k.sum()
        B = I.shape[0]
        I = I.reshape(B * FRAMES, 1, PIX, PIX)
        I = F.conv2d(F.pad(I, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
        I = F.conv2d(F.pad(I, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1)).view(B, FRAMES, PIX, PIX)
    maps = []
    for dly in (1, 2):
        a, b = I[:, dly:], I[:, :-dly]                           # now, delayed
        h = (a[..., :-1] * b[..., 1:] - a[..., 1:] * b[..., :-1]).sum(1)
        v = (a[..., :-1, :] * b[..., 1:, :] - a[..., 1:, :] * b[..., :-1, :]).sum(1)
        maps += [F.adaptive_avg_pool2d(h[:, None], 4).flatten(1), F.adaptive_avg_pool2d(v[:, None], 4).flatten(1)]
    return torch.cat(maps, 1)


def hr(seed):
    name = f"HR_s{seed}"
    if (OUT / f"{name}.json").exists():
        return
    d = data()
    res = {}
    for sigma in (0, 1, 2):
        feats = {k: torch.cat([hr_features(x[s:s + 256].to(dev).float(), sigma).cpu() for s in range(0, len(x), 256)])
                 for k, (x, _) in d.items()}
        torch.manual_seed(seed)
        readout = Readout(feats["train"].shape[1]).to(dev)

        def logits_of(xs, idx, train):
            return readout((feats["train"] if train else xs)[idx].to(dev))
        logits_of.y_train = d["train"][1]
        best, curve, ok = fit(list(readout.parameters()), logits_of, NTR, feats["val"], d["val"][1], feats["test"],
                              d["test"][1], [readout])
        res[sigma] = (best, ok, curve)
    sigma = max(res, key=lambda s: res[s][0])
    best, ok, curve = res[sigma]
    rec = {"cell": "HR", "seed": seed, "sigma": sigma, "val_by_sigma": {s: r[0] for s, r in res.items()}, "val_best": best,
           "test_acc": float(ok.float().mean()), "test_ok": ok.int().tolist(), "curve": curve}
    (OUT / f"{name}.json").write_text(json.dumps(rec))
    print(f"{name}: test {rec['test_acc']:.4f} (sigma {sigma}, val {best:.4f})", flush=True)


# ------------------------------------------------------------------ verdict (PROTOCOL_D6 §7)
def verdict():
    seeds = (1, 2, 3)
    load = lambda c, s: json.loads((OUT / f"{c}_s{s}.json").read_text())
    ok = {c: [np.asarray(load(c, s)["test_ok"]) for s in seeds] for c in ("A", "B", "C", "D")}
    acc = {c: [float(v.mean()) for v in ok[c]] for c in ok}
    g = np.random.default_rng(0)
    boots = [g.integers(0, NTE, NTE) for _ in range(2000)]
    combos = {"A-B": {"A": 1, "B": -1}, "C-D": {"C": 1, "D": -1}, "I": {"A": 1, "B": -1, "C": -1, "D": 1}}
    rep = {"acc": acc, "mean": {c: float(np.mean(v)) for c, v in acc.items()}}
    for name, w in combos.items():
        per = [sum(wt * acc[c][i] for c, wt in w.items()) for i in range(3)]
        dist = np.array([np.mean([sum(wt * ok[c][i][b].mean() for c, wt in w.items()) for i in range(3)]) for b in boots])
        lo, hi = np.percentile(dist, [2.5, 97.5])
        rep[name] = {"per_seed": per, "mean": float(np.mean(per)), "ci95": [float(lo), float(hi)],
                     "effect": bool(all(x > 0 for x in per) and lo > DELTA)}
    means = list(rep["mean"].values())
    rep["floor_or_ceiling"] = bool(all(m <= 0.30 for m in means) or all(m >= 0.95 for m in means))
    eAB, eCD, eI = rep["A-B"]["effect"], rep["C-D"]["effect"], rep["I"]["effect"]
    if rep["floor_or_ceiling"]:
        rep["outcome"] = "uninformative (floor or ceiling for all cells)"
    elif eAB and eI:
        rep["outcome"] = "O1 retina only"
    elif not eAB and not eCD:
        rep["outcome"] = "O2 neither"
    elif eAB and eCD and not eI:
        rep["outcome"] = "O3 both"
    else:
        rep["outcome"] = "other"
    for c in ("HR", "Bm"):
        f = [OUT / f"{c}_s{s}.json" for s in seeds]
        if all(p.exists() for p in f):
            rep[c] = [json.loads(p.read_text())["test_acc"] for p in f]
    (OUT / "verdict.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "data":
        d = data()
        print({k: (tuple(v[0].shape), np.bincount(v[1].numpy()).tolist()) for k, v in d.items()})
    elif cmd == "run":
        run(sys.argv[2], int(sys.argv[3]))
    elif cmd == "hr":
        hr(int(sys.argv[2]))
    elif cmd == "verdict":
        verdict()
