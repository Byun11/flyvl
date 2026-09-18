"""Learned no-brain baselines on the same hard motion task: how much of the fly circuit's advantage
survives once the comparison model is allowed to LEARN from the eye movie?

Input for every baseline: the evoked photoreceptor signal (25 steps x n_driven), i.e. exactly what the
fly's brain receives — no brain, no connectome. Trials, labels and split are identical to p5_flytask's
`motion_dir_hard` so the numbers sit in the same table as the brain views.

baselines
  linear_series : logistic regression on the flattened movie (this is the probe used for the brain views)
  mlp           : 2-layer MLP on the flattened movie
  tempconv      : per-photoreceptor temporal conv (this is the computation a motion detector needs:
                  a learned temporal filter, then pooling over the eye) -> linear
usage: p5_cnn_baseline.py [n_trials] [trial_seed]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from flyvl import connectome, frozen, masks  # noqa: E402

import p5_flytask as P5  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
TRIAL_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
EPOCHS, BATCH = 60, 64
P5.TASK = "motion_dir_hard"
OUT = connectome.DATA_ROOT / "runs" / "p5"


class TempConv(torch.nn.Module):
    """Learned temporal filters shared across photoreceptors, then a learned spatial pooling."""
    def __init__(self, n_pr, steps, n_filt=16, n_cls=4):
        super().__init__()
        self.n_pr, self.steps = n_pr, steps
        self.conv = torch.nn.Conv1d(1, n_filt, 7, padding=3)
        self.pool = torch.nn.Linear(n_pr, 32)
        self.head = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Flatten(),
                                        torch.nn.Linear(n_filt * 32, n_cls))
        self.n_filt = n_filt

    def forward(self, x):                                  # x (B, steps * n_pr)
        B = x.shape[0]
        x = x.reshape(B, self.steps, self.n_pr).permute(0, 2, 1).reshape(B * self.n_pr, 1, self.steps)
        h = self.conv(x).mean(-1)                          # (B*n_pr, n_filt) time-filtered energy
        h = h.reshape(B, self.n_pr, self.n_filt).permute(0, 2, 1)
        h = self.pool(h)                                   # (B, n_filt, 32) learned eye pooling
        return self.head(h)


def mlp(d_in, n_cls=4, h=256):
    return torch.nn.Sequential(torch.nn.Linear(d_in, h), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                               torch.nn.Linear(h, n_cls))


def train_eval(model, Xtr, ytr, Xte, yte, seed, lr=1e-3):
    torch.manual_seed(seed)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best = 0.0
    for ep in range(EPOCHS):
        model.train()
        order = torch.randperm(len(ytr))
        for s in range(0, len(ytr), BATCH):
            b = order[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(Xte).argmax(1) == yte).float().mean())
        best = max(best, acc)                              # optimistic: no val split, reported as an upper bound
    return best


if __name__ == "__main__":
    c = connectome.load()
    cfg, _, _ = frozen.load()
    M = masks.build(c)
    y, p = P5.trial_params(np.random.default_rng(TRIAL_SEED), "motion_dir_hard", N)
    t0 = time.time()
    F = P5.features("nobrain", y, p, c, cfg, {})
    X = F["photoreceptor_series"]
    steps = P5.STEPS
    n_pr = X.shape[1] // steps
    print(f"eye movie {X.shape} = {steps} steps x {n_pr} photoreceptors  ({time.time()-t0:.0f}s)", flush=True)
    n_tr = int(N * 0.7)
    mu, sd = X[:n_tr].mean(0), X[:n_tr].std(0) + 1e-6
    Xn = torch.as_tensor((X - mu) / sd, dtype=torch.float32).cuda()
    yt = torch.as_tensor(y).cuda()
    Xtr, ytr, Xte, yte = Xn[:n_tr], yt[:n_tr], Xn[n_tr:], yt[n_tr:]
    res = {}
    for name, build in [("linear", lambda: torch.nn.Linear(X.shape[1], 4)),
                        ("mlp", lambda: mlp(X.shape[1])),
                        ("tempconv", lambda: TempConv(n_pr, steps))]:
        accs = [train_eval(build(), Xtr, ytr, Xte, yte, s) for s in (0, 1, 2)]
        res[name] = {"acc": accs, "mean": float(np.mean(accs))}
        print(f"{name:10s} best-epoch acc {[round(a*100,1) for a in accs]} mean {np.mean(accs)*100:.2f}",
              flush=True)
    (OUT / f"cnn_baseline_t{TRIAL_SEED}.json").write_text(
        json.dumps({"n": N, "chance": 0.25, "note": "no brain; input = eye movie; best epoch on test = upper bound",
                    "res": res}, indent=1))
