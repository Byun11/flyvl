"""P8: push the connectome as a VLM/vision encoder with real training capacity.

Everything before this trained 47,524 type-level parameters on <=3,000 images and concluded the
connectome does not contribute. That was the right experiment for that question and the wrong one for
"can this be made to work". Here the wiring and synapse signs stay frozen but a per-synapse gain
(24,472,770), a learned pixel->photoreceptor encoder and the readout are trained on full CIFAR-10 -
the same recipe the public chess projects use.

The matched-shuffle control trains identically. If it matches real, the capacity is doing the work
and the connectome is a substrate, not a prior; if real pulls ahead, the measured wiring helps.

usage: p8_train_cifar.py GRAPH [epochs] [n_train]     GRAPH: real | matched_shuffle_s0 | cnn | pixels
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, data, frozen  # noqa: E402
from flyvl.extract import load_graph, to_luma  # noqa: E402
from flyvl.v3 import FlyVL3  # noqa: E402

GRAPH = sys.argv[1] if len(sys.argv) > 1 else "real"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
N_TRAIN = int(sys.argv[3]) if len(sys.argv) > 3 else 45000
BATCH, STEPS = 64, 16
OUT = connectome.DATA_ROOT / "runs" / "p8"
OUT.mkdir(parents=True, exist_ok=True)


def small_cnn(n_classes=10):
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(), torch.nn.Linear(128, n_classes)).cuda()


@torch.no_grad()
def evaluate(model, X, y, is_fly):
    model.eval()
    correct = 0
    for s in range(0, len(y), 256):
        xb = X[s:s + 256].cuda()
        out = model(xb if is_fly else xb.view(-1, 1, 32, 32))
        correct += (out.argmax(1).cpu() == y[s:s + 256]).sum().item()
    model.train()
    return correct / len(y)


if __name__ == "__main__":
    Xtr_raw, ytr_raw = data.cifar10(True)
    Xte_raw, yte_raw = data.cifar10(False)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ytr_raw))
    tr, va = perm[:N_TRAIN], perm[N_TRAIN:N_TRAIN + 5000]
    X = {"train": to_luma(Xtr_raw[tr]).reshape(len(tr), -1),
         "val": to_luma(Xtr_raw[va]).reshape(len(va), -1),
         "test": to_luma(Xte_raw).reshape(len(yte_raw), -1)}
    Y = {"train": torch.as_tensor(ytr_raw[tr]), "val": torch.as_tensor(ytr_raw[va]),
         "test": torch.as_tensor(yte_raw)}
    mu, sd = X["train"].mean(), X["train"].std()
    X = {k: (v - mu) / sd for k, v in X.items()}

    torch.manual_seed(0)
    is_fly = GRAPH not in ("cnn", "pixels")
    if is_fly:
        c = connectome.load()
        cfg, _, _ = frozen.load()
        model = FlyVL3(c, load_graph(c, GRAPH, cfg), steps=STEPS, view="all")
        groups = model.param_groups()
    elif GRAPH == "cnn":
        model = small_cnn()
        # list(...), not the generator: `n_par` below iterates the group and would exhaust a
        # generator before Adam ever saw it, leaving the optimiser with zero parameters. That is
        # exactly what happened on the first run - pixels and cnn "trained" without updating at all.
        groups = [{"params": list(model.parameters()), "lr": 1e-3}]
    else:
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1024, 10)).cuda()
        groups = [{"params": list(model.parameters()), "lr": 1e-3}]
    n_par = sum(p.numel() for g in groups for p in g["params"])
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    print(f"graph={GRAPH} epochs={EPOCHS} n_train={N_TRAIN} trainable={n_par:,}", flush=True)

    ckpt = OUT / f"{GRAPH}_ckpt.pt"
    log, start, best_val = [], 0, 0.0
    if ckpt.exists():
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        log, start, best_val = st["log"], st["epoch"] + 1, st["best_val"]
        print(f"resumed at epoch {start}", flush=True)

    for epoch in range(start, EPOCHS):
        t0 = time.time()
        order = torch.as_tensor(np.random.default_rng(1000 + epoch).permutation(N_TRAIN))
        losses = []
        for s in range(0, N_TRAIN, BATCH):
            b = order[s:s + BATCH]
            xb = X["train"][b].cuda()
            opt.zero_grad(set_to_none=True)
            out = model(xb if is_fly else xb.view(-1, 1, 32, 32))
            loss = torch.nn.functional.cross_entropy(out, Y["train"][b].cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            losses.append(loss.item())
            if not np.isfinite(losses[-1]):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
        sched.step()
        val = evaluate(model, X["val"], Y["val"], is_fly)
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), "val": val,
                    "sec": time.time() - t0})
        print(f"epoch {epoch:2d} loss {np.mean(losses):.4f} val {val*100:5.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if val >= best_val:
            best_val = val
            torch.save(model.state_dict(), OUT / f"{GRAPH}_best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "log": log, "epoch": epoch, "best_val": best_val}, ckpt)

    model.load_state_dict(torch.load(OUT / f"{GRAPH}_best.pt", weights_only=False))
    test = evaluate(model, X["test"], Y["test"], is_fly)
    res = {"graph": GRAPH, "trainable": n_par, "epochs": EPOCHS, "n_train": N_TRAIN,
           "steps": STEPS, "best_val": best_val, "test": test, "log": log}
    (OUT / f"{GRAPH}.json").write_text(json.dumps(res, indent=1))
    print(f"TEST {GRAPH} {test*100:.2f}  (val {best_val*100:.2f}, {n_par:,} params)", flush=True)
