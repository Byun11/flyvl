"""A serious no-brain motion baseline, to test the weakest point of the P6 comparison.

The earlier `tempconv` averaged its conv output over time and therefore threw away exactly the phase
information a motion detector needs — it was a broken implementation, not evidence. These baselines
implement the computation properly and are given the same eye movie, the same trials and the same
data budget as the fly.

  gru       : per-photoreceptor signal pooled to a low-dim code per frame, then a GRU over the 25 frames
              (a general learned temporal model)
  corr      : explicit Hassenstein-Reichardt-style correlator on the retina graph -- for each
              photoreceptor pair (neighbour along the hex lattice) compute x_i(t) * x_j(t-d) summed over
              time for several delays d, then a linear head. This is the textbook fly motion detector,
              implemented directly, and is the strongest fair competitor to the connectome.
  conv3d    : two-frame differences fed to an MLP (the cheapest thing that can see motion at all)
usage: p5_motion_baseline2.py [n_trials] [trial_seed] [difficulty]
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

from flyvl import connectome, frozen  # noqa: E402

import p5_flytask as P5  # noqa: E402
from p5_cnn_baseline import EPOCHS, train_eval  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
TRIAL_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DIFF = sys.argv[3] if len(sys.argv) > 3 else "harder"
P5.TASK = "motion_dir_hard"
OUT = connectome.DATA_ROOT / "runs" / "p5"
DELAYS = (1, 2, 3, 5)


class Pooled(torch.nn.Module):
    """Shared linear pooling of the eye into `d` channels per frame, then a temporal model."""
    def __init__(self, n_pr, steps, temporal, d=64):
        super().__init__()
        self.n_pr, self.steps = n_pr, steps
        self.pool = torch.nn.Linear(n_pr, d)
        self.temporal = temporal

    def forward(self, x):
        B = x.shape[0]
        f = self.pool(x.reshape(B, self.steps, self.n_pr))            # (B, steps, d)
        return self.temporal(f)


class GRUHead(torch.nn.Module):
    def __init__(self, d=64, h=128, n_cls=4):
        super().__init__()
        self.rnn = torch.nn.GRU(d, h, batch_first=True)
        self.fc = torch.nn.Linear(h, n_cls)

    def forward(self, f):
        return self.fc(self.rnn(f)[0][:, -1])


class DiffMLP(torch.nn.Module):
    def __init__(self, steps, d=64, h=128, n_cls=4):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear((steps - 1) * d, h),
                                       torch.nn.ReLU(), torch.nn.Linear(h, n_cls))

    def forward(self, f):
        return self.net(f[:, 1:] - f[:, :-1])                          # frame differences


class Correlator(torch.nn.Module):
    """Hassenstein-Reichardt: sum_t x_i(t) * x_j(t-d) over neighbour pairs (i, j) and delays d.
    The delayed-product features are FIXED (no free parameters); only the linear head is learned."""
    def __init__(self, pairs_i, pairs_j, steps, n_cls=4):
        super().__init__()
        self.register_buffer("pi", pairs_i)
        self.register_buffer("pj", pairs_j)
        self.steps = steps
        self.head = torch.nn.Linear(len(pairs_i) * len(DELAYS) * 2, n_cls)

    def forward(self, x):
        B = x.shape[0]
        v = x.reshape(B, self.steps, -1)
        a, b = v[:, :, self.pi], v[:, :, self.pj]                      # (B, steps, P)
        feats = []
        for d in DELAYS:
            feats.append((a[:, d:] * b[:, :-d]).mean(1))               # i leads j
            feats.append((b[:, d:] * a[:, :-d]).mean(1))               # j leads i
        return self.head(torch.cat(feats, 1))


def neighbour_pairs(c, driven, max_pairs=400, seed=0):
    """Pairs of driven photoreceptors that are adjacent on the hex column lattice."""
    col = c.column[driven]                                    # (n_driven, 3): [eye/side, hex_x, hex_y]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(driven))[:max_pairs * 8]
    pi, pj, seen = [], [], set()
    lookup = {(int(e), int(a), int(b)): k for k, (e, a, b) in enumerate(col)}
    for k in order:
        e, a, b = int(col[k, 0]), int(col[k, 1]), int(col[k, 2])
        for da, db in ((1, 0), (0, 1), (1, -1)):
            j = lookup.get((e, a + da, b + db))                # neighbours within the same eye
            if j is not None and (k, j) not in seen:
                seen.add((k, j))
                pi.append(k)
                pj.append(j)
                break
        if len(pi) >= max_pairs:
            break
    return torch.as_tensor(pi), torch.as_tensor(pj)


if __name__ == "__main__":
    c = connectome.load()
    cfg, _, _ = frozen.load()
    y, p = P5.trial_params(np.random.default_rng(TRIAL_SEED), "motion_dir_hard", N)
    if DIFF == "harder":
        p["contrast"] = np.random.default_rng(TRIAL_SEED + 7).uniform(0.015, 0.040, N)
    t0 = time.time()
    F = P5.features("nobrain", y, p, c, cfg, {})
    X = F["photoreceptor_series"]
    steps = P5.STEPS
    n_pr = X.shape[1] // steps
    r16 = c.types(["R1-6"])
    driven = r16[c.column[r16, 0] >= 0]
    pi, pj = neighbour_pairs(c, driven)
    print(f"eye movie {X.shape}  {steps} x {n_pr}   neighbour pairs {len(pi)}  ({time.time()-t0:.0f}s)", flush=True)
    n_tr = int(N * 0.7)
    mu, sd = X[:n_tr].mean(0), X[:n_tr].std(0) + 1e-6
    Xn = torch.as_tensor((X - mu) / sd, dtype=torch.float32).cuda()
    yt = torch.as_tensor(y).cuda()
    Xtr, ytr, Xte, yte = Xn[:n_tr], yt[:n_tr], Xn[n_tr:], yt[n_tr:]
    res = {}
    builders = [("gru", lambda: Pooled(n_pr, steps, GRUHead())),
                ("diffmlp", lambda: Pooled(n_pr, steps, DiffMLP(steps))),
                ("correlator", lambda: Correlator(pi, pj, steps))]
    for name, build in builders:
        accs = [train_eval(build(), Xtr, ytr, Xte, yte, s) for s in (0, 1, 2)]
        n_par = sum(q.numel() for q in build().parameters())
        res[name] = {"acc": accs, "mean": float(np.mean(accs)), "params": n_par}
        print(f"{name:12s} acc {[round(a*100,1) for a in accs]} mean {np.mean(accs)*100:.2f}  params {n_par}",
              flush=True)
    (OUT / f"motion_baseline2_t{TRIAL_SEED}_{DIFF}.json").write_text(
        json.dumps({"n": N, "chance": 0.25, "difficulty": DIFF,
                    "note": "no brain; eye movie input; best epoch on test = upper bound", "res": res}, indent=1))
