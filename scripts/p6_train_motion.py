"""P6: learn the unknown physiology on FROZEN connectome wiring, on the low-contrast motion task.

The wiring is never modified: W is the frozen effective graph (or, for the control, a pre-built rewired
graph used only as a null hypothesis). Trainable = per-cell-type g_pre, g_post, tau, bias (47,524) and
the linear readout.

conditions
  init    : dynamics frozen at the common init (g = 1), readout only  -> what P5 measured
  trained : dynamics + readout
usage: p6_train_motion.py GRAPH SEED CONDITION [epochs] [n_train]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, frozen  # noqa: E402
from flyvl.extract import load_graph  # noqa: E402
from flyvl.v2motion import MotionFlyVL2, trial_params  # noqa: E402

GRAPH, SEED, COND = sys.argv[1], int(sys.argv[2]), sys.argv[3]
assert COND in ("trained", "init")
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
N_TRAIN = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
N_VAL, N_TEST, BATCH = 500, 1000, 32
LR_DYN, LR_READOUT, CLIP = 3e-3, 1e-3, 1.0
OUT = connectome.DATA_ROOT / "runs" / "p6" / f"{GRAPH}_s{SEED}_{COND}"
OUT.mkdir(parents=True, exist_ok=True)

# Trials are drawn once from fixed seeds, so every graph/condition/seed sees the SAME stimuli.
P, Y = {}, {}
for name, (n, s) in {"train": (N_TRAIN, 100), "val": (N_VAL, 101), "test": (N_TEST, 102)}.items():
    P[name], Y[name] = trial_params(np.random.default_rng(s), n)

c = connectome.load()
cfg, _, _ = frozen.load()
torch.manual_seed(SEED)                       # readout init identical across graphs at the same seed
model = MotionFlyVL2(c, load_graph(c, GRAPH, cfg))
groups = [{"params": model.readout.parameters(), "lr": LR_READOUT}]
if COND == "trained":
    groups.append({"params": model.dynamics_parameters(), "lr": LR_DYN})
else:
    for p in model.dynamics_parameters():
        p.requires_grad_(False)
opt = torch.optim.Adam(groups)
ckpt_path = OUT / "ckpt.pt"
log, start = [], 0
if ckpt_path.exists():
    ck = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    log, start = ck["log"], ck["epoch"] + 1


@torch.no_grad()
def evaluate(split):
    model.eval()
    torch.cuda.manual_seed(12345)              # same sensor-noise draw for every model at eval
    preds = []
    for s in range(0, len(Y[split]), BATCH):
        f = model.features(P[split][s:s + BATCH], grad=False)
        preds.append(model.readout(f).argmax(1).cpu())
    model.train()
    return float((torch.cat(preds) == Y[split]).float().mean())


print(f"graph={GRAPH} seed={SEED} cond={COND} epochs={EPOCHS} n_train={N_TRAIN} "
      f"trainable={sum(p.numel() for g in groups for p in g['params'])}", flush=True)
for epoch in range(start, EPOCHS):
    t0 = time.time()
    order = torch.as_tensor(np.random.default_rng(SEED * 1000 + epoch).permutation(N_TRAIN))
    losses = []
    for s in range(0, N_TRAIN, BATCH):
        b = order[s:s + BATCH]
        opt.zero_grad(set_to_none=True)
        f = model.features(P["train"][b])
        loss = torch.nn.functional.cross_entropy(model.readout(f), Y["train"][b].cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], CLIP)
        opt.step()
        losses.append(loss.item())
        if not np.isfinite(losses[-1]):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
    val_acc = evaluate("val")
    dyn = {k: float(v) for k, v in model.dynamics_summary().items()} if hasattr(model, "dynamics_summary") else {}
    log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": val_acc,
                "sec": time.time() - t0, **dyn})
    print(f"epoch {epoch:2d} loss {np.mean(losses):.4f} val {val_acc*100:5.2f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "log": log, "epoch": epoch}, ckpt_path)
    if val_acc >= max(r["val_acc"] for r in log):
        torch.save(model.state_dict(), OUT / "best.pt")

model.load_state_dict(torch.load(OUT / "best.pt", weights_only=False))
best = max(log, key=lambda r: (r["val_acc"], -r["epoch"]))
result = {"graph": GRAPH, "seed": SEED, "condition": COND, "best_epoch": best["epoch"],
          "val_acc": best["val_acc"], "test_acc": evaluate("test"), "chance": 0.25,
          "n_train": N_TRAIN, "epochs": EPOCHS, "log": log}
(OUT / "result.json").write_text(json.dumps(result, indent=1))
print("TEST", json.dumps({k: v for k, v in result.items() if k != "log"}), flush=True)
