"""P2 training per PROTOCOL_v2.md.
usage: p2_train.py GRAPH SEED CONDITION      CONDITION: trained | init (dynamics frozen at init, readout only)
Resumable: checkpoint after every epoch. Final test uses the best-validation epoch."""
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
from flyvl.v2 import FlyVL2  # noqa: E402

GRAPH, SEED, COND = sys.argv[1], int(sys.argv[2]), sys.argv[3]
assert COND in ("trained", "init")
EPOCHS, BATCH = 20, 64
LR_DYN, LR_READOUT, CLIP = 3e-3, 1e-3, 1.0
OUT = connectome.DATA_ROOT / "runs" / "p2_mini" / f"{GRAPH}_s{SEED}_{COND}"
OUT.mkdir(parents=True, exist_ok=True)

# data: P1-mini images; fixed stratified 450/50 split of the 5k train (split seed 0 for every run)
images_tr, labels_tr = data.cifar10(True)
images_te, labels_te = data.cifar10(False)
itr = data.first_per_class(labels_tr, 500)
ite = data.first_per_class(labels_te, 100)
split = np.random.default_rng(0)
val_mask = np.zeros(len(itr), bool)
for k in range(10):
    val_mask[split.permutation(np.flatnonzero(labels_tr[itr] == k))[:50]] = True
tr_idx, va_idx = itr[~val_mask], itr[val_mask]
X = {"train": to_luma(images_tr[tr_idx]), "val": to_luma(images_tr[va_idx]), "test": to_luma(images_te[ite])}
Y = {"train": torch.as_tensor(labels_tr[tr_idx]), "val": torch.as_tensor(labels_tr[va_idx]),
     "test": torch.as_tensor(labels_te[ite])}

c = connectome.load()
cfg, _, _ = frozen.load()
torch.manual_seed(SEED)                       # readout init: identical for real / shuffle at the same seed
model = FlyVL2(c, load_graph(c, GRAPH, cfg))
groups = [{"params": model.readout.parameters(), "lr": LR_READOUT}]
if COND == "trained":
    groups.append({"params": model.dynamics_parameters(), "lr": LR_DYN})
else:
    for p in model.dynamics_parameters():
        p.requires_grad_(False)
opt = torch.optim.Adam(groups)

ckpt_path, log_path = OUT / "last.pt", OUT / "log.json"
log, start = [], 0
if ckpt_path.exists():
    ck = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    log, start = ck["log"], ck["epoch"] + 1


@torch.no_grad()
def evaluate(split_name, cache=None):
    model.eval()
    preds = []
    for s in range(0, len(X[split_name]), BATCH):
        f = cache[s:s + BATCH] if cache is not None else model.features(X[split_name][s:s + BATCH].cuda(), grad=False)
        preds.append(model.readout(f).argmax(1).cpu())
    p = torch.cat(preds)
    return float((p == Y[split_name]).float().mean()), p.numpy()


cache = {}
if COND == "init":                           # dynamics never change: compute features once
    with torch.no_grad():
        for k in X:
            cache[k] = torch.cat([model.features(X[k][s:s + BATCH].cuda(), grad=False)
                                  for s in range(0, len(X[k]), BATCH)])

for epoch in range(start, EPOCHS):
    t0 = time.time()
    order = torch.as_tensor(np.random.default_rng(SEED * 1000 + epoch).permutation(len(X["train"])))
    model.train()
    losses = []
    for s in range(0, len(order), BATCH):
        b = order[s:s + BATCH]
        f = cache["train"][b.cuda()] if COND == "init" else model.features(X["train"][b].cuda())
        loss = torch.nn.functional.cross_entropy(model.readout(f), Y["train"][b].cuda())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CLIP)
        opt.step()
        losses.append(loss.item())
        if not np.isfinite(losses[-1]):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
    val_acc, _ = evaluate("val", cache.get("val"))
    improved = not log or val_acc > max(r["val_acc"] for r in log)       # ties keep the earlier epoch
    with torch.no_grad():
        dyn = {"g_pre": torch.nn.functional.softplus(model.g_pre_raw).mean().item(),
               "g_post": torch.nn.functional.softplus(model.g_post_raw).mean().item(),
               "tau": (0.02 + torch.nn.functional.softplus(model.tau_raw)).mean().item(),
               "bias_abs": model.bias.abs().mean().item()}
    log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": val_acc, "sec": time.time() - t0, **dyn})
    print(json.dumps(log[-1]), flush=True)
    if improved:
        torch.save(model.state_dict(), OUT / "best.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "log": log, "epoch": epoch}, ckpt_path)
    log_path.write_text(json.dumps(log, indent=1))

best = max(log, key=lambda r: (r["val_acc"], -r["epoch"]))
model.load_state_dict(torch.load(OUT / "best.pt", weights_only=False))
if COND == "init":
    cache_test = cache["test"]
else:
    cache_test = None
test_acc, preds = evaluate("test", cache_test)
np.save(OUT / "test_preds.npy", preds)
result = {"graph": GRAPH, "seed": SEED, "condition": COND, "best_epoch": best["epoch"], "val_acc": best["val_acc"],
          "test_acc": test_acc}
(OUT / "result.json").write_text(json.dumps(result, indent=1))
print("RESULT", json.dumps(result), flush=True)
