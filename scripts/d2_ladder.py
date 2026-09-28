"""D2 ladder: which InternViT depth can each input rebuild from its own 28x28 block?

D2 asked a block-local student (one fly per 28x28 block) to rebuild InternVL's FINAL visual tokens, which 24
attention layers have contextualised with the whole image. The pixel control failed as badly as the fly
(R2 0.002), so D2 could not say whether the fly carries anything. This ladder moves the target down the network:
  L0 = patch + position embedding (a linear function of the RGB patch), L1 ... L24 = hidden states of the 2x2
  patches under the block (4 x 1024), proj = the 896-d LLM token D2 used.
Inputs: fly (1170), pix (1568, the same jittered frames without the fly), grey block (784), RGB block (2352).
Ridge regression with position fixed effects (inputs and targets centred per block position), lambda picked on an
inner split of train (4500 / 500), refit on all 5000, R2 on the 500 validation images. All from second moments,
so no hidden state is stored.
usage: d2_ladder.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, teacher  # noqa: E402

D2 = connectome.DATA_ROOT / "d2"
OUT = ROOT / "results" / "d2"
OUT.mkdir(parents=True, exist_ok=True)
LAYERS = (0, 1, 2, 4, 6, 12, 18, 24)
TARGETS = [f"L{L}" for L in LAYERS] + ["proj"]
ARMS = ("fly", "pix", "grey", "rgb", "rand", "bank", "bank5")
BOOT = ("L2", "L4", "L6")
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
NFIT, NTRAIN, BS = 4500, 5000, 20
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
LUMA = torch.tensor([0.299, 0.587, 0.114], device=dev)


def to_blocks(x, c):
    """(b, 448, 448, c) -> (b, 256, 28*28*c), blocks row-major like D2's tokens."""
    b = x.shape[0]
    return x.view(b, 16, 28, 16, 28, c).permute(0, 1, 3, 2, 4, 5).reshape(b, 256, -1)


def _blur(x, sigma):
    r = int(3 * sigma + 0.5)
    k = torch.exp(-torch.arange(-r, r + 1, device=dev, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


_r3 = torch.arange(28, device=dev) * 3 // 28
REG3 = F.one_hot((_r3[:, None] * 3 + _r3[None, :]).flatten(), 9).float()
REG3 = REG3 / REG3.sum(0)                                                        # (784, 9) the fly's 3x3 regions
_r5 = torch.arange(28, device=dev) * 5 // 28
REG5 = F.one_hot((_r5[:, None] * 5 + _r5[None, :]).flatten(), 25).float()
REG5 = REG5 / REG5.sum(0)                                                        # (784, 25) finer pooling, 825 features


def v1_bank(pix, reg=None):
    """Classical V1-style energy bank on the same jittered frames, pooled on the fly's 3x3 regions:
    per time bin, ON / OFF centre-surround at 3 scales, oriented gradient energy (4 orientations x 2 scales) and
    mean luminance (15 maps); plus the between-bin change energy at 3 scales. 15 x 9 x 2 + 3 x 9 = 297 features."""
    b = pix.shape[0]
    x = pix.reshape(b * 256, 2, 28, 28)
    maps = []
    for t in range(2):
        f = x[:, t:t + 1]
        for sc, ss in ((0.5, 1.5), (1.0, 3.0), (2.0, 6.0)):
            d = _blur(f, sc) - _blur(f, ss)
            maps += [F.relu(d), F.relu(-d)]
        for s in (1.0, 2.0):
            g = _blur(f, s)
            gx = F.pad(g[..., :, 1:] - g[..., :, :-1], (0, 1))
            gy = F.pad(g[..., 1:, :] - g[..., :-1, :], (0, 0, 0, 1))
            maps += [gx ** 2, gy ** 2, (gx + gy) ** 2 / 2, (gx - gy) ** 2 / 2]
        maps.append(_blur(f, 2.0))
    for s in (0.5, 1.0, 2.0):
        maps.append((_blur(x[:, 1:2], s) - _blur(x[:, 0:1], s)) ** 2)
    m = torch.cat(maps, 1).flatten(2)                                            # (b*256, 33, 784)
    return (m @ (REG3 if reg is None else reg)).reshape(b, 256, -1)


def batch(model, s, i0, i1):
    """Inputs and targets for images i0:i1 of split s, each (b, 256, dim) float32 on the GPU."""
    img = torch.as_tensor(np.asarray(np.load(D2 / f"{s}_img.npy", mmap_mode="r")[i0:i1]), device=dev).float() / 255
    X = {"fly": np.load(D2 / f"{s}_fly_v1.npy", mmap_mode="r")[i0:i1],
         "pix": np.load(D2 / f"{s}_pix_v1.npy", mmap_mode="r")[i0:i1]}
    X = {k: torch.as_tensor(np.asarray(v, np.float32), device=dev) for k, v in X.items()}
    X["grey"] = to_blocks((img @ LUMA)[..., None], 1)
    X["rgb"] = to_blocks(img, 3)
    X["rand"] = F.relu(((X["pix"] - PIXMU) / PIXSD) @ RAND)
    X["bank"] = v1_bank(X["pix"])
    X["bank5"] = v1_bank(X["pix"], REG5)
    with torch.no_grad():
        px = ((img.permute(0, 3, 1, 2) - MEAN) / STD).to(torch.bfloat16)
        hs = model.model.vision_tower(pixel_values=px, output_hidden_states=True).hidden_states
    b = img.shape[0]
    Y = {f"L{L}": hs[L][:, 1:].float().view(b, 16, 2, 16, 2, 1024).permute(0, 1, 3, 2, 4, 5).reshape(b, 256, 4096)
         for L in LAYERS}
    Y["proj"] = VIT[s][i0:i1].to(dev).float()
    return X, Y


def passes(model):
    """Pass 1: per-position means on train. Pass 2: second moments for fit / inner / val."""
    mx = {a: 0 for a in ARMS}
    my = {t: 0 for t in TARGETS}
    for i in range(0, NTRAIN, BS):
        X, Y = batch(model, "train", i, min(i + BS, NTRAIN))
        for a in ARMS:
            mx[a] = mx[a] + X[a].double().sum(0)
        for t in TARGETS:
            my[t] = my[t] + Y[t].double().sum(0)
    mx = {a: (v / NTRAIN).float() for a, v in mx.items()}
    my = {t: (v / NTRAIN).float() for t, v in my.items()}
    print("pass 1 done", flush=True)
    acc = {}
    for part, s, lo, hi in (("fit", "train", 0, NFIT), ("inner", "train", NFIT, NTRAIN), ("val", "val", 0, 500)):
        A = {"n": 0, "xx": {}, "xy": {}, "y": {}, "yy": {}}
        for i in range(lo, hi, BS):
            X, Y = batch(model, s, i, min(i + BS, hi))
            Xc = {a: (X[a] - mx[a]).reshape(-1, X[a].shape[-1]) for a in ARMS}
            Yc = {t: (Y[t] - my[t]).reshape(-1, Y[t].shape[-1]) for t in TARGETS}
            A["n"] += len(Xc["fly"])
            for a in ARMS:
                A["xx"][a] = A["xx"].get(a, 0) + (Xc[a].T @ Xc[a]).double()
                for t in TARGETS:
                    A["xy"][(a, t)] = A["xy"].get((a, t), 0) + (Xc[a].T @ Yc[t]).double()
            for t in TARGETS:
                A["y"][t] = A["y"].get(t, 0) + Yc[t].double().sum(0)
                A["yy"][t] = A["yy"].get(t, 0) + Yc[t].double().pow(2).sum(0)
        acc[part] = A
        print(f"pass 2 {part} done", flush=True)
    return acc, mx, my


def r2(W, A, a, t):
    """Per-dim R2 (mean) and variance-weighted R2 of predictions X W on the part summarised by A."""
    sse = A["yy"][t] - 2 * (W * A["xy"][(a, t)]).sum(0) + ((A["xx"][a] @ W) * W).sum(0)
    sst = A["yy"][t] - A["y"][t] ** 2 / A["n"]
    return float((1 - sse / sst.clamp_min(1e-12)).mean()), float(1 - sse.sum() / sst.sum())


def solve(xx, xy, alpha):
    d = xx.shape[0]
    lam = alpha * xx.diagonal().mean()
    return torch.linalg.solve(xx + lam * torch.eye(d, device=dev, dtype=xx.dtype), xy)


def bootstrap(model, Ws, mx, my, n_boot=2000):
    """Per-image residuals on the 500 validation images for BOOT targets; image-level bootstrap of fly minus each
    other arm in variance-weighted R2."""
    sse = {k: [] for k in Ws}
    ys = {t: [] for t in BOOT}
    yy = {t: [] for t in BOOT}
    for i in range(0, 500, BS):
        X, Y = batch(model, "val", i, min(i + BS, 500))
        for t in BOOT:
            Yc = (Y[t] - my[t]).double()
            ys[t].append(Yc.sum(1).cpu())
            yy[t].append(Yc.pow(2).sum(1).cpu())
            for a in ARMS:
                pred = ((X[a] - mx[a]) @ Ws[(a, t)]).double()
                sse[(a, t)].append((Yc - pred).pow(2).sum((1, 2)).cpu())
    g = np.random.default_rng(0)
    out = {}
    for t in BOOT:
        S = torch.cat(ys[t]).numpy()
        Q = torch.cat(yy[t]).numpy()
        E = {a: torch.cat(sse[(a, t)]).numpy() for a in ARMS}
        n_img = len(S)
        def r2vw(idx, a):
            n = len(idx) * 256
            sst = (Q[idx].sum(0) - S[idx].sum(0) ** 2 / n).sum()
            return 1 - E[a][idx].sum() / sst
        full = np.arange(n_img)
        boots = [g.integers(0, n_img, n_img) for _ in range(n_boot)]
        for a in ARMS:
            if a == "fly":
                continue
            d = np.array([r2vw(b, "fly") - r2vw(b, a) for b in boots])
            out[f"fly-{a}/{t}"] = {"diff": float(r2vw(full, "fly") - r2vw(full, a)),
                                   "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
            print(f"fly - {a:5s} {t}: {out[f'fly-{a}/{t}']['diff']:+.4f}  95% CI [{out[f'fly-{a}/{t}']['ci95'][0]:+.4f}, "
                  f"{out[f'fly-{a}/{t}']['ci95'][1]:+.4f}]", flush=True)
    return out


def main():
    global VIT, PIXMU, PIXSD, RAND
    VIT = {s: torch.load(D2 / f"{s}_vit.pt") for s in ("train", "val")}
    p = torch.as_tensor(np.asarray(np.load(D2 / "train_pix_v1.npy", mmap_mode="r")[:500], np.float32), device=dev)
    PIXMU, PIXSD = p.mean((0, 1)), p.std((0, 1)) + 1e-6
    RAND = torch.randn(1568, 1170, generator=torch.Generator(device=dev).manual_seed(0), device=dev) / np.sqrt(1568)
    del p
    proc, model = teacher.load()
    acc, mx, my = passes(model)
    Ws = {}
    fit, inner, val = acc["fit"], acc["inner"], acc["val"]
    res = {}
    for a in ARMS:
        for t in TARGETS:
            best = max(ALPHAS, key=lambda al: r2(solve(fit["xx"][a], fit["xy"][(a, t)], al), inner, a, t)[1])
            W = solve(fit["xx"][a] + inner["xx"][a], fit["xy"][(a, t)] + inner["xy"][(a, t)], best)
            m, vw = r2(W, val, a, t)
            res[f"{a}/{t}"] = {"alpha": best, "r2_mean": m, "r2_vw": vw}
            if t in BOOT:
                Ws[(a, t)] = W.float()
        print(a, " ".join(f"{t}:{res[f'{a}/{t}']['r2_vw']:.3f}" for t in TARGETS), flush=True)
    res["bootstrap"] = bootstrap(model, Ws, mx, my)
    (OUT / "ladder.json").write_text(json.dumps(res, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for a, col in zip(ARMS, ("#d62728", "#1f77b4", "#7f7f7f", "#2ca02c", "#9467bd", "#ff7f0e")):
        ax.plot(range(len(TARGETS)), [res[f"{a}/{t}"]["r2_vw"] for t in TARGETS], "o-", color=col, label=a)
    ax.set_xticks(range(len(TARGETS)), [t.replace("L", "layer ") if t != "proj" else "LLM token" for t in TARGETS], rotation=30)
    ax.set_ylabel("held-out R2 (variance-weighted)")
    ax.set_title("How much of each InternViT depth a 28x28 block input rebuilds (ridge, position removed)")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend()
    fig.savefig(OUT / "ladder.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
