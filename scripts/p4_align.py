"""P4 token distillation: frozen per-patch features -> trainable aligner -> InternVL3-1B visual tokens (4x4 pooled).
usage: p4_align.py REP [REP ...] [--seeds 0,1,2] [--tag NAME]
REP: GRAPH:VIEW (VIEW in central_vnc | visual_projection | optic_lobe | photoreceptor), pixels, cnn, mean
Evaluation on test: token cosine to pooled teacher, and zero-shot CIFAR accuracy of the frozen InternVL LLM given
the student tokens (4x4 -> nearest upsample to 16x16)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, teacher  # noqa: E402
from flyvl.swarm import G, PAD, PATCH, STRIDE  # noqa: E402

P4 = connectome.DATA_ROOT / "p4"
RUNS = connectome.DATA_ROOT / "runs" / "p4"
RUNS.mkdir(parents=True, exist_ok=True)
EPOCHS, BATCH, LR, WD, D_MODEL = 150, 64, 3e-4, 0.05, 256


def rgb_patches(split):
    from flyvl.data import cifar10
    idx = np.load(P4 / f"{split}_indices.npy")
    X, _ = cifar10(split != "test")
    x = torch.from_numpy(X[idx]).permute(0, 3, 1, 2).float().div(255)
    return x


def to_grid4(t):
    """(N, g*g, D) -> (N, 16, D) by nearest upsampling on the token grid (g in 1, 2, 4)."""
    n = t.shape[1]
    g = int(round(n ** 0.5))
    if g == G:
        return t
    x = t.reshape(len(t), g, g, -1).repeat_interleave(G // g, 1).repeat_interleave(G // g, 2)
    return x.reshape(len(t), G * G, -1)


def load_rep(rep, split):
    if rep.startswith("pixels-g"):                             # non-overlapping patch pixels for grid 2 / 1
        grid = int(rep[len("pixels-g"):])
        x = rgb_patches(split)
        size = 32 // grid
        p = x.unfold(2, size, size).unfold(3, size, size)
        return to_grid4(p.permute(0, 2, 3, 1, 4, 5).reshape(len(x), grid * grid, -1) - 0.5)
    if "+" in rep:                                             # concatenation of per-token features
        return torch.cat([load_rep(r, split) for r in rep.split("+")], -1)
    if rep == "pixels":
        x = F.pad(rgb_patches(split), (PAD,) * 4, mode="reflect")
        p = x.unfold(2, PATCH, STRIDE).unfold(3, PATCH, STRIDE)                # (N, 3, 4, 4, 16, 16)
        return p.permute(0, 2, 3, 1, 4, 5).reshape(len(x), G * G, -1) - 0.5
    if rep in ("cnn", "mean"):
        return rgb_patches(split)
    graph, view = rep.split(":")
    return to_grid4(torch.load(P4 / f"{split}_{graph}.pt")[view].float())


class Aligner(nn.Module):
    def __init__(self, d_in, rep):
        super().__init__()
        self.rep = rep
        if rep == "cnn":                                       # ordinary small CNN on the full image -> 4x4 tokens
            self.enc = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                                     nn.Conv2d(64, 128, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                                     nn.Conv2d(128, D_MODEL, 3, padding=1), nn.GELU(), nn.MaxPool2d(2))
        else:
            self.enc = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, D_MODEL))
        self.pos = nn.Parameter(torch.zeros(1, G * G, D_MODEL))
        layer = nn.TransformerEncoderLayer(D_MODEL, 4, 2 * D_MODEL, dropout=0.1, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, 2)
        self.out = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, 896))

    def forward(self, x):
        if self.rep == "cnn":
            h = self.enc(x).flatten(2).transpose(1, 2)
        else:
            h = self.enc(x)
        return self.out(self.body(h + self.pos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reps", nargs="+")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--ntrain", type=int, default=0, help="use only the first ntrain/10 train images per class")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    T = {s: teacher.pool_tokens(torch.load(P4 / f"{s}_teacher256.pt"), G) for s in ("train", "val", "test")}
    keep = None
    if a.ntrain:
        ytr = np.load(P4 / "train_labels.npy")
        keep = torch.as_tensor(np.sort(np.concatenate([np.flatnonzero(ytr == k)[:a.ntrain // 10] for k in range(10)])))
        T["train"] = T["train"][keep]
    mu, sd = T["train"].mean((0, 1)), T["train"].std((0, 1)) + 1e-6
    Z = {s: (t - mu) / sd for s, t in T.items()}
    y_test = np.load(P4 / "test_labels.npy")
    proc, vlm = teacher.load()
    zs = teacher.ZeroShot(proc, vlm)
    for p in vlm.parameters():
        p.requires_grad_(False)

    def zero_shot(tokens_raw):
        lp = zs.logprobs(teacher.unpool_tokens(tokens_raw, G), batch=10)
        return float((lp.argmax(1).numpy() == y_test).mean()), lp.argmax(1).numpy()

    ceiling_path = RUNS / "ceiling.json"
    if not ceiling_path.exists():
        acc, _ = zero_shot(T["test"])
        full = float((zs.logprobs(torch.load(P4 / "test_teacher256.pt").float(), batch=10).argmax(1).numpy() == y_test).mean())
        ceiling_path.write_text(json.dumps({"teacher_pooled4x4": acc, "teacher_full256": full}, indent=1))
    print("CEILING", ceiling_path.read_text().replace("\n", ""), flush=True)

    for rep in a.reps:
        X = {s: load_rep(rep, s) for s in ("train", "val", "test")}
        if keep is not None:
            X["train"] = X["train"][keep]
        for seed in seeds:
            out = RUNS / f"{a.tag}__{rep.replace(':', '-').replace('+', '__plus__')}__s{seed}.json"
            if out.exists():
                print("SKIP", out.name)
                continue
            torch.manual_seed(seed)
            t0 = time.time()
            if keep is not None:
                res_extra = {"ntrain": int(len(keep))}
            else:
                res_extra = {}
            if rep == "mean":
                pred = mu.expand(len(y_test), G * G, -1).clone()
                res = {"rep": rep, "seed": seed, "params": 0}
            else:
                d_in = X["train"].shape[-1]
                model = Aligner(d_in, rep).cuda()
                if rep != "cnn":
                    xm, xs = X["train"].mean((0, 1)), X["train"].std((0, 1)) + 1e-6
                else:
                    xm, xs = 0.0, 1.0
                norm = {s: ((X[s] - xm) / xs).cuda() for s in X}
                zc = {s: Z[s].cuda() for s in Z}
                opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

                def loss_fn(p, t):
                    return F.mse_loss(p, t) + (1 - F.cosine_similarity(p, t, dim=-1)).mean()

                best, best_state, log = 1e9, None, []
                g = torch.Generator().manual_seed(seed)
                for ep in range(EPOCHS):
                    model.train()
                    perm = torch.randperm(len(norm["train"]), generator=g)
                    for s in range(0, len(perm), BATCH):
                        b = perm[s:s + BATCH].cuda()
                        loss = loss_fn(model(norm["train"][b]), zc["train"][b])
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    sched.step()
                    model.eval()
                    with torch.no_grad():
                        vl = float(loss_fn(model(norm["val"]), zc["val"]))
                    log.append(vl)
                    if vl < best:
                        best, best_ep = vl, ep
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    pz = model(norm["test"]).cpu()
                pred = pz * sd + mu
                res = {"rep": rep, "seed": seed, "params": sum(p.numel() for p in model.parameters()),
                       "best_epoch": best_ep, "val_loss": best, "val_curve": log[::10]}
            with torch.no_grad():
                cos = float(F.cosine_similarity(pred, T["test"], dim=-1).mean())
                zcos = float(F.cosine_similarity((pred - mu) / sd, Z["test"], dim=-1).mean())
            acc, preds = zero_shot(pred)
            res.update(res_extra)
            res.update({"test_token_cos": cos, "test_centered_cos": zcos, "test_zeroshot_acc": acc,
                        "sec": time.time() - t0})
            np.save(out.with_suffix(".preds.npy"), preds)
            out.write_text(json.dumps(res, indent=1))
            print("RESULT", json.dumps({k: v for k, v in res.items() if k != "val_curve"}), flush=True)


if __name__ == "__main__":
    main()
