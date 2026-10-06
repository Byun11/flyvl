"""D2 filters: what each encoder pulls out of the pixels, drawn like ViT Fig. 7 / Fig. 10 (Dosovitskiy et al. 2021).

  vit   InternViT patch-embedding filters (1024 x RGB 14x14 px), first 28 principal components
  pos   InternViT position-embedding cosine similarity on its 32x32 patch grid
  tok   linear receptive fields of the 896 ViT token dims on the 28x28 grey block the token covers
        (ridge regression = the best a per-token linear adapter from pixels can do)
  fly   linear receptive fields of the 1170 fly features (cell type x 3x3 eye region x 2 time bins), same regression
No new simulation: features are regressed on the jittered, time-averaged grey block each fly actually saw (pix arm).
usage: d2_filters.py   (writes results/d2/*.png)
"""
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome  # noqa: E402

D2 = connectome.DATA_ROOT / "d2"
OUT = ROOT / "results" / "d2"
OUT.mkdir(parents=True, exist_ok=True)
NFIT, NTEST, NPC, LAM = 1000, 200, 28, 1e-2
dev = "cuda"


def vit_weights():
    from safetensors import safe_open
    f = safe_open(glob.glob(os.path.join(os.environ["HF_HOME"], "hub", "models--OpenGVLab--InternVL3-1B-hf",
                                         "snapshots", "*", "model.safetensors"))[0], "pt")
    return (f.get_tensor("vision_tower.embeddings.patch_embeddings.projection.weight").float(),
            f.get_tensor("vision_tower.embeddings.position_embeddings")[0, 1:].float())


def pcs(W):
    """W (n_filters, dim) -> top NPC components (NPC, dim) and cumulative explained variance."""
    W = W - W.mean(0)
    _, s, vt = torch.linalg.svd(W, full_matrices=False)
    ev = (s ** 2) / (s ** 2).sum()
    return vt[:NPC], ev.cumsum(0)


def fly_types():
    from flyvl.flyvis_encoder import FlyvisEncoder
    t = FlyvisEncoder().net.connectome.nodes.type[:]
    t = np.asarray([x.decode() if isinstance(x, bytes) else x for x in t])
    u = sorted(set(t))
    return u, {k: int((t == k).sum()) for k in u}


def receptive_fields():
    """Ridge-regress each target feature on the 784 block pixels. Returns {name: (W (784, K), test R2 per feature)}."""
    n = NFIT + NTEST
    P = np.load(D2 / "train_pix_v1.npy", mmap_mode="r")[:n]
    P = torch.as_tensor(np.asarray(P, np.float32).reshape(n, 256, 2, 784).mean(2)).to(dev)
    tgt = {"fly": torch.as_tensor(np.asarray(np.load(D2 / "train_fly_v1.npy", mmap_mode="r")[:n], np.float32)),
           "tok": torch.load(D2 / "train_vit.pt")[:n].float()}
    fit, test = slice(0, NFIT), slice(NFIT, n)
    X = P[fit].reshape(-1, 784)
    xm = X.mean(0)
    X = X - xm
    C = X.T @ X / len(X)
    C += LAM * C.diagonal().mean() * torch.eye(784, device=dev)
    Xt = (P[test].reshape(-1, 784) - xm)
    out = {}
    for k, Y in tgt.items():
        Y = Y.to(dev)
        Yf = Y[fit].reshape(-1, Y.shape[-1])
        ym, ys = Yf.mean(0), Yf.std(0) + 1e-6
        W = torch.linalg.solve(C, X.T @ ((Yf - ym) / ys) / len(X))                      # (784, K)
        Yt = (Y[test].reshape(-1, Y.shape[-1]) - ym) / ys
        r2 = 1 - ((Xt @ W - Yt) ** 2).mean(0) / Yt.var(0).clamp_min(1e-6)
        out[k] = (W.cpu(), r2.cpu())
        del Y, Yf, Yt
    return out


def show_grid(ax_grid, imgs, rgb=False):
    for ax, im in zip(ax_grid.flat, imgs):
        if rgb:
            im = (im - im.min()) / (im.max() - im.min() + 1e-9)
            ax.imshow(im.permute(1, 2, 0).numpy())
        else:
            v = float(im.abs().max()) + 1e-9
            ax.imshow(im.numpy(), cmap="gray", vmin=-v, vmax=v)
        ax.set_xticks([]), ax.set_yticks([])
    for ax in list(ax_grid.flat)[len(imgs):]:
        ax.axis("off")


def main():
    wv, pe = vit_weights()
    rf = receptive_fields()
    types, ncell = fly_types()
    T = len(types)
    assert rf["fly"][0].shape[1] == T * 9 * 2, (rf["fly"][0].shape, T)
    report = []

    # 1. filters side by side: ViT RGB patch embedding / ViT-token RF / fly-feature RF
    fig = plt.figure(figsize=(21, 4.6))
    sub = fig.subfigures(1, 3)
    panels = [("InternViT patch embedding\n(RGB 14x14 px, first 28 PCs)", wv.reshape(1024, -1), (3, 14, 14), True),
              ("ViT token linear RF\n(grey 28x28 px, first 28 PCs)", rf["tok"][0].T, (28, 28), False),
              ("fly feature linear RF\n(grey 28x28 px, first 28 PCs)", rf["fly"][0].T, (28, 28), False)]
    for sf, (title, W, shape, rgb) in zip(sub, panels):
        v, cum = pcs(W)
        axs = sf.subplots(4, 7)
        show_grid(axs, [x.reshape(shape) for x in v], rgb)
        sf.suptitle(f"{title}\nvariance in 5 / 28 PCs: {cum[4]:.0%} / {cum[NPC - 1]:.0%}", fontsize=11)
        report.append(f"{title.splitlines()[0]}: cumulative variance PC5 {cum[4]:.3f} PC10 {cum[9]:.3f} PC28 {cum[NPC - 1]:.3f}")
    fig.savefig(OUT / "filters_pca.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 2. fly receptive fields per cell type: centre eye region, late time bin (0.5-1 s)
    col = [k for k, t in enumerate(types) if ncell[t] == 721]
    W = rf["fly"][0]
    idx = [T * 9 + k * 9 + 4 for k in col]
    fig, axs = plt.subplots(int(np.ceil(len(col) / 10)), 10, figsize=(16, 1.9 * np.ceil(len(col) / 10)))
    show_grid(axs, [W[:, i].reshape(28, 28) for i in idx])
    for ax, k in zip(axs.flat, col):
        ax.set_title(f"{types[k]}  R2 {rf['fly'][1][T * 9 + k * 9 + 4]:.2f}", fontsize=8)
    fig.suptitle("fly linear receptive fields by cell type (centre eye region, 0.5-1 s); white = ON, black = OFF", fontsize=12)
    fig.savefig(OUT / "fly_rf_by_type.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 3. InternViT position-embedding similarity (every 2nd patch = the 16x16 token grid)
    pc = pe - pe.mean(0)                                                            # shared offset dominates raw cosine
    print(f"position embedding: mean vector holds {1 - pc.pow(2).sum() / pe.pow(2).sum():.0%} of the energy")
    pn = pc / pc.norm(dim=1, keepdim=True)
    S = (pn @ pn.T).reshape(32, 32, 32, 32)
    fig, axs = plt.subplots(16, 16, figsize=(12, 12))
    for i in range(16):
        for j in range(16):
            axs[i, j].imshow(S[2 * i, 2 * j].numpy(), cmap="viridis", vmin=-1, vmax=1)
            axs[i, j].set_xticks([]), axs[i, j].set_yticks([])
    fig.suptitle("InternViT position-embedding cosine similarity, mean removed (tile = one patch vs all 32x32 patches)", fontsize=12)
    fig.savefig(OUT / "vit_pos_similarity.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    for k in ("tok", "fly"):
        r2 = rf[k][1]
        report.append(f"{k} linear R2 on held-out images: mean {r2.mean():.3f} median {r2.median():.3f} "
                      f"frac>0.5 {(r2 > 0.5).float().mean():.3f}")
    r2 = rf["fly"][1][T * 9:].reshape(T, 9)[:, 4]
    top = sorted(((float(r2[k]), types[k]) for k in col), reverse=True)
    report.append("fly centre-region R2 top: " + ", ".join(f"{t} {v:.2f}" for v, t in top[:8]))
    report.append("fly centre-region R2 bottom: " + ", ".join(f"{t} {v:.2f}" for v, t in top[-8:]))
    (OUT / "filters_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
