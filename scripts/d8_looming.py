"""D8 (PROTOCOL_D8_looming.md): looming selectivity through the D7 harness (learned encoder -> frozen whole MaleCNS ->
readout of VPN or descending neurons), Real vs Rewired-L, with Bypass / GRU / looming detector / HR.
usage: d8_looming.py run MODEL SEED     MODEL in real_vpn rewired_l_vpn real_dn rewired_l_dn bypass gru detector hr
       d8_looming.py verdict
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
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flyvl import flygrapher as fg  # noqa: E402
import d6_translation as d6  # noqa: E402
import d7_screen as d7  # noqa: E402

OUT = fg.connectome.DATA_ROOT / "d8"
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
FR, PIX, DELTA = d7.FR, d7.PIX, 0.01
KINDS = ["looming", "looming_distractors", "recede", "translate", "linear", "zoom", "shake"]
POSITIVE = {"looming", "looming_distractors"}


# ------------------------------------------------------------------ stimuli
def gen(n, seed, kind):
    g = np.random.default_rng(seed)
    X = np.empty((n, FR, PIX, PIX), np.float32)
    yy, xx = np.meshgrid(np.arange(PIX) + 0.5, np.arange(PIX) + 0.5, indexing="ij")
    t = np.arange(FR)
    M, C = 8, PIX / 2
    for i in range(n):
        tex = g.normal(0, 1, (PIX + 2 * M, PIX + 2 * M))
        tt = torch.as_tensor(tex, dtype=torch.float32)[None, None]
        k = torch.exp(-torch.arange(-6, 7, dtype=torch.float32) ** 2 / 8.0)
        k = k / k.sum()
        tex = F.conv2d(F.conv2d(F.pad(tt, (6, 6, 6, 6), mode="reflect"), k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))[0, 0].numpy()
        tex = 0.05 * (tex - tex.mean()) / tex.std()
        c0 = g.uniform(8, 24, 2)
        r0, rho = g.uniform(1.5, 2.5), g.uniform(3, 5)
        zoom = np.ones(FR)
        off = np.zeros((FR, 2))
        centers = np.repeat(c0[None], FR, 0)
        if kind in ("looming", "looming_distractors"):
            r = r0 / (1 - (1 - 1 / rho) * t / (FR - 1))
        elif kind == "recede":
            r = (r0 / (1 - (1 - 1 / rho) * t / (FR - 1)))[::-1]
        elif kind == "linear":
            r = r0 + (rho - 1) * r0 * t / (FR - 1)
        elif kind == "translate":
            r = np.full(FR, g.uniform(2, 4))
            ang, sp = g.uniform(0, 2 * np.pi), g.uniform(0.3, 0.8) * PIX * d6.FRAME_DT
            v = np.array([np.cos(ang), np.sin(ang)]) * sp
            d = v * (FR - 1)
            start = np.array([g.uniform(4 - min(0, d[0]), 28 - max(0, d[0])), g.uniform(4 - min(0, d[1]), 28 - max(0, d[1]))])
            centers = start + t[:, None] * v
        elif kind == "zoom":
            r = np.full(FR, g.uniform(2, 4))
            zoom = 1 + (g.uniform(1.2, 1.5) - 1) * t / (FR - 1)
        elif kind == "shake":
            r = np.full(FR, g.uniform(2, 4))
            off = g.integers(-2, 3, (FR, 2)).astype(float)
        dis = []
        if kind == "looming_distractors":
            for _ in range(2):
                ang, sp = g.uniform(0, 2 * np.pi), g.uniform(0.25, 0.8) * PIX * d6.FRAME_DT
                v = np.array([np.cos(ang), np.sin(ang)]) * sp
                dis.append(g.uniform(4, 28, 2) + t[:, None] * v)
        for f in range(FR):
            sx = C + (xx - C) / zoom[f] + off[f, 0]                          # image -> scene coordinates
            sy = C + (yy - C) / zoom[f] + off[f, 1]
            frame = 0.5 + map_coordinates(tex, [sy - 0.5 + M, sx - 0.5 + M], order=1, mode="reflect")
            dist = np.hypot(sx - centers[f, 0], sy - centers[f, 1])
            frame = frame - 0.35 / (1 + np.exp(-(r[f] - dist) / 0.5))
            for p in dis:
                frame = frame + 0.35 * d7.blob(sx, sy, p[f, 0], p[f, 1])
            X[i, f] = frame + g.normal(0, 0.03, (PIX, PIX))
    return X.astype(np.float16)


def task_data():
    f = OUT / "looming.npz"
    if not f.exists():
        tr_n = {"looming": 450, "looming_distractors": 450, **{k: 180 for k in KINDS[2:]}}
        va_n = {"looming": 75, "looming_distractors": 75, **{k: 30 for k in KINDS[2:]}}
        parts = {}
        for split, counts, s0 in (("train", tr_n, 1), ("val", va_n, 2), ("test", {k: 450 for k in KINDS}, 3)):
            xs, ys, ks = [], [], []
            for j, kind in enumerate(KINDS):
                xs.append(gen(counts[kind], 900 + 10 * j + s0, kind))
                ys.append(np.full(counts[kind], int(kind in POSITIVE)))
                ks.append(np.full(counts[kind], j))
            parts[split] = (np.concatenate(xs), np.concatenate(ys), np.concatenate(ks))
        np.savez(f, **{f"{s}_{a}": v[i] for s, v in parts.items() for i, a in enumerate("xyk")})
    z = np.load(f)
    T = torch.as_tensor
    return {"train": (T(z["train_x"]), T(z["train_y"])), "val": (T(z["val_x"]), T(z["val_y"])),
            "test": (T(z["test_x"]), T(z["test_y"]), T(z["test_k"]), None)}


# ------------------------------------------------------------------ models
class DNBrain(d7.Brain):
    """Same frozen brain; readout population = descending neurons."""

    def __init__(self, brain, kind, seed):
        super().__init__(brain, kind, seed, "last")
        self.vpn = torch.as_tensor(np.flatnonzero(brain.superclass == "descending_neuron"), device=dev)


class FlyDN(nn.Module):
    def __init__(self, brain, kind, seed):
        super().__init__()
        self.enc = d6.AIEncoder(brain)
        self.net = DNBrain(brain, kind, seed)
        self.readout = d7.Readout(len(self.net.vpn), 2)

    def forward(self, x):
        return self.readout(self.net(self.enc(x)).T)


class Fixed(nn.Module):
    """Fixed features (no parameters) -> BatchNorm -> Linear."""

    def __init__(self, feat, d):
        super().__init__()
        self.feat, self.readout = feat, d7.Readout(d, 2)

    def forward(self, x):
        return self.readout(self.feat(x))


def size_features(x):
    k = torch.exp(-torch.arange(-3, 4, dtype=torch.float32, device=x.device) ** 2 / 2.0)
    k = k / k.sum()
    B = x.shape[0]
    xs = x.reshape(B * FR, 1, PIX, PIX)
    xs = F.conv2d(F.conv2d(F.pad(xs, (3, 3, 3, 3), mode="reflect"), k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))
    s = (xs < 0.5 - 0.15).float().sum((1, 2, 3)).view(B, FR)
    ls = torch.log((s + 1) / (s[:, :1] + 1))
    return torch.cat([ls[:, 1:], ls[:, 2:] - 2 * ls[:, 1:-1] + ls[:, :-2]], 1)


def run(model_name, seed):
    name = f"{model_name}_s{seed}"
    if (OUT / f"{name}.json").exists():
        return
    t0 = time.time()
    d = task_data()
    brain = fg.load_malecns()
    torch.manual_seed(seed)
    if model_name in ("real_vpn", "rewired_l_vpn"):
        model = d7.FlyNet(brain, model_name[:-4], seed, 2, "last")
    elif model_name in ("real_dn", "rewired_l_dn"):
        model = FlyDN(brain, model_name[:-3], seed)
    elif model_name == "bypass":
        model = d7.BypassNet(2, "last")
    elif model_name == "gru":
        fly_params = d7.n_params(d7.FlyNet(brain, "real", 1, 2, "last"))
        torch.manual_seed(seed)
        model = d7.GRUNet(2, d7.gru_hidden(2, fly_params))
    elif model_name == "detector":
        model = Fixed(size_features, 13)
    elif model_name == "hr":
        model = Fixed(lambda x: d6.hr_features(x, 0), 64)
    model = model.to(dev)
    best, curve, pred = d7.fit(model, d, 2)
    y, kind = d["test"][1], d["test"][2]
    ok = (pred == y).int()
    rec = {"model": model_name, "seed": seed, "val_best": best, "curve": curve, "params": d7.n_params(model),
           "test_pred": pred.tolist(), "minutes": (time.time() - t0) / 60,
           "rate_by_kind": {k: float(pred[kind == j].float().mean()) for j, k in enumerate(KINDS)}}
    pos = np.isin(np.asarray(kind), [KINDS.index(k) for k in POSITIVE])
    rec["balanced_acc"] = float(0.5 * (ok.numpy()[pos].mean() + ok.numpy()[~pos].mean()))
    (OUT / f"{name}.json").write_text(json.dumps(rec))
    print(f"{name}: balanced {rec['balanced_acc']:.3f} | 'looming' rate by kind "
          + " ".join(f"{k} {v:.2f}" for k, v in rec["rate_by_kind"].items()) + f" | params {rec['params']:,} ({rec['minutes']:.1f} min)", flush=True)


def verdict():
    seeds = (1, 2, 3)
    d = task_data()
    y, kind = d["test"][1].numpy(), d["test"][2].numpy()
    pos = np.isin(kind, [KINDS.index(k) for k in POSITIVE])
    load = lambda m: [np.asarray(json.loads((OUT / f"{m}_s{s}.json").read_text())["test_pred"]) for s in seeds]
    g = np.random.default_rng(0)
    boots = [g.integers(0, len(y), len(y)) for _ in range(2000)]

    def bal(pred, idx):
        ok, p = (pred[idx] == y[idx]), pos[idx]
        return 0.5 * (ok[p].mean() + ok[~p].mean())
    full = np.arange(len(y))
    rep = {}
    for pop in ("vpn", "dn"):
        P = {m: load(m) for m in (f"real_{pop}", f"rewired_l_{pop}", "bypass")}
        acc = {m: [bal(p, full) for p in v] for m, v in P.items()}

        def cmp(a, b):
            per = [acc[a][i] - acc[b][i] for i in range(3)]
            dist = np.array([np.mean([bal(P[a][i], bb) - bal(P[b][i], bb) for i in range(3)]) for bb in boots])
            lo, hi = np.percentile(dist, [2.5, 97.5])
            return {"per_seed": per, "mean": float(np.mean(per)), "ci95": [float(lo), float(hi)],
                    "effect": bool(all(x > 0 for x in per) and lo > DELTA)}
        r = {"balanced_acc": {m: [float(v) for v in a] for m, a in acc.items()}}
        r["real-rewired_l"] = cmp(f"real_{pop}", f"rewired_l_{pop}")
        r["real-bypass"] = cmp(f"real_{pop}", "bypass")
        r["rewired_l-bypass"] = cmp(f"rewired_l_{pop}", "bypass")
        mr, mw = np.mean(acc[f"real_{pop}"]), np.mean(acc[f"rewired_l_{pop}"])
        if (mr <= 0.55 and mw <= 0.55) or (mr >= 0.97 and mw >= 0.97):
            r["verdict"] = "UNINFORMATIVE (floor or ceiling)"
        elif r["real-rewired_l"]["effect"]:
            r["verdict"] = "TOPOLOGY EFFECT"
        elif r["real-bypass"]["effect"] and r["rewired_l-bypass"]["effect"]:
            r["verdict"] = "GENERIC RESERVOIR EFFECT"
        else:
            r["verdict"] = "NO EFFECT"
        r["false_alarm_linear_zoom"] = {m: [float(np.mean(p[np.isin(kind, [KINDS.index("linear"), KINDS.index("zoom")])] == 1))
                                            for p in P[m]] for m in (f"real_{pop}", f"rewired_l_{pop}")}
        rep[pop] = r
    for m in ("gru", "detector", "hr"):
        rep[m] = [bal(p, full) for p in load(m)]
    (OUT / "verdict.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "verdict":
        verdict()
    elif sys.argv[1] == "data":
        d = task_data()
        print({k: tuple(v[0].shape) for k, v in d.items()}, np.bincount(d["train"][1].numpy()))
