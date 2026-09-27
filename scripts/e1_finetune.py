"""E1: train the fly end to end. flyvis's cell-type parameters (connectome-constrained optic lobe, wiring
fixed) are fine-tuned together with the readout MLP on the M2 figure-ground task, through differentiable
pooling by measured column geometry. Everything else is M2 POOL=1: same videos, same 4 quadrant glimpses,
same 16-cell x 8 T4/T5-type features, same MLP, same test set.

Reference points at the same budget (M2 pooled, frozen): n=300 flyvis 35.8, HR 74.4, CNN 27.1;
n=1000 flyvis 79.8, HR 98.2.

usage: e1_finetune.py N_TRAIN [epochs] [lr_fly]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ["V3_TASK"] = "pixel_fg16"
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 300
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
LR_FLY = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
sys.argv = [sys.argv[0], "pixel_fg16"]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v3b_vlm as V  # noqa: E402
from flyvl import connectome  # noqa: E402
from flyvl.flyvis_encoder import T45, FlyvisEncoder, hexal_xy  # noqa: E402

P = V.P
DT, STEPS, N = 0.02, 25, 3000
OUT = connectome.DATA_ROOT / "runs" / "e1"
OUT.mkdir(parents=True, exist_ok=True)


class FlyReadout(torch.nn.Module):
    def __init__(self, enc: FlyvisEncoder):
        super().__init__()
        self.net, self.eye = enc.net, enc.eye
        xy = hexal_xy(enc)
        sub = (xy[:, 1] >= 0.5).astype(int) * 2 + (xy[:, 0] >= 0.5).astype(int)
        m = torch.as_tensor(np.stack([sub == k for k in range(4)]), dtype=torch.float32)
        self.register_buffer("masks", (m / m.sum(1, keepdim=True)).T.contiguous())      # (721, 4)
        self.register_buffer("idx", torch.as_tensor(np.stack([enc.idx[t] for t in T45])))  # (8, 721)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                                       torch.nn.Linear(256, 16))
        self.register_buffer("mu", torch.zeros(256))
        self.register_buffer("sd", torch.ones(256))

    def features(self, video):
        """video (B, T, 128, 128) -> (B, 256), differentiable w.r.t. the flyvis parameters."""
        B, H = video.shape[0], video.shape[-1] // 2
        feat = video.new_zeros(B, 16, 8, 2)
        for q in range(4):
            qi, qj = divmod(q, 2)
            movie = self.eye(video[..., qi * H:(qi + 1) * H, qj * H:(qj + 1) * H])
            with torch.no_grad():
                state = self.net.steady_state(t_pre=0.5, dt=DT, batch_size=B, value=0.5)
            # same input path as Network.simulate, minus its no-grad/eval guard
            self.net.stimulus.zero(B, movie.shape[1])
            self.net.stimulus.add_input(movie)
            r = self.net(self.net.stimulus(), DT, state)                                 # (B, T, n_cells)
            x = r[..., self.idx]                                                        # (B, T, 8, 721)
            ev = x - x[:, :1]
            sm = ev[:, 5:].mean(1) @ self.masks                                         # (B, 8, 4)
            en = (ev[:, 1:] - ev[:, :-1]).pow(2).mean(1) @ self.masks
            for k in range(4):
                cy, cx = divmod(k, 2)
                cell = (2 * qi + cy) * 4 + (2 * qj + cx)
                feat[:, cell, :, 0], feat[:, cell, :, 1] = sm[..., k], en[..., k]
        return feat.flatten(1)

    def forward(self, video):
        return self.mlp((self.features(video) - self.mu) / self.sd)


def make_video(p, idx):
    pb = {k: v[idx] for k, v in p.items()}
    return torch.stack([V.letter_frames(pb, k * DT)[:, 0] for k in range(STEPS)], 1).clamp(0, 1)


if __name__ == "__main__":
    t0 = time.time()
    y, p = P.trial_params(np.random.default_rng(0), "pixel_fg16", N)       # same trials as M2
    p["letters"] = np.random.default_rng(1).integers(0, 26, (N, 16))
    tr, te = np.arange(N_TRAIN), np.arange(2100, N)
    enc = FlyvisEncoder()
    with torch.device("cuda"):
        model = FlyReadout(enc).cuda()
        with torch.no_grad():                                               # standardise with frozen-fly stats
            f = torch.cat([model.features(make_video(p, tr[s:s + 32])) for s in range(0, len(tr), 32)])
            model.mu.copy_(f.mean(0)), model.sd.copy_(f.std(0) + 1e-6)
        fly_params = [q for n_, q in model.named_parameters() if not n_.startswith("mlp")]
        opt = torch.optim.Adam([{"params": fly_params, "lr": LR_FLY}, {"params": model.mlp.parameters(), "lr": 1e-3}],
                               weight_decay=1e-4)
        yt = torch.as_tensor(y, device="cuda")
        g = torch.Generator().manual_seed(0)
        log = []

        def evaluate():
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(make_video(p, te[s:s + 32])).argmax(1) for s in range(0, len(te), 32)])
            model.train()
            return float((pred.cpu().numpy() == y[te]).mean()), pred.cpu().numpy()

        acc0, _ = evaluate()
        print(f"n={N_TRAIN} fly params {sum(q.numel() for q in fly_params)} | before training (random MLP) {acc0*100:.1f}", flush=True)
        B = 16
        for ep in range(EPOCHS):
            perm = torch.as_tensor(tr, device="cpu")[torch.randperm(len(tr), generator=g, device="cpu")]
            losses = []
            for s in range(0, len(tr), B):
                b = perm[s:s + B].numpy()
                loss = F.cross_entropy(model(make_video(p, b)), yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if ep % 2 == 1 or ep == EPOCHS - 1:
                acc, _ = evaluate()
                log.append({"epoch": ep, "loss": float(np.mean(losses)), "test": acc})
                print(f"epoch {ep:3d} loss {np.mean(losses):.3f} test {acc*100:.1f} ({(time.time()-t0)/60:.0f} min)", flush=True)
        acc, pred = evaluate()
    res = {"n_train": N_TRAIN, "epochs": EPOCHS, "lr_fly": LR_FLY, "test": acc, "log": log,
           "ref_frozen": {"300": {"flyvis": 0.358, "hr": 0.744, "cnn": 0.271}, "1000": {"flyvis": 0.798, "hr": 0.982}},
           "minutes": (time.time() - t0) / 60}
    (OUT / f"e1_n{N_TRAIN}_lr{LR_FLY:g}.json").write_text(json.dumps(res, indent=1))
    np.save(OUT / f"e1_n{N_TRAIN}_pred.npy", pred)
    print(f"FINAL n={N_TRAIN} fine-tuned fly {acc*100:.2f}  (frozen fly / HR / CNN at n=300: 35.8 / 74.4 / 27.1)", flush=True)
