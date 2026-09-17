"""P2-C front+back translators per PROTOCOL_v2C_translators.md.
usage: p2c_train.py BODY SEED      BODY: real | global_shuffle_s0 | nobrain
Same data split, epochs, batch order, optimizer and best-val selection as scripts/p2b_train.py."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, data, frozen  # noqa: E402
from flyvl.extract import load_graph  # noqa: E402
from flyvl.v2c import BrainTranslators, NoBrainTranslators  # noqa: E402

GRAPH, SEED = sys.argv[1], int(sys.argv[2])
MODE = "translators"
EPOCHS, BATCH, LR, CLIP = 20, 64, 1e-3, 1.0
OUT = connectome.DATA_ROOT / "runs" / "p2c_mini" / f"{GRAPH}_s{SEED}"
OUT.mkdir(parents=True, exist_ok=True)


def rgb(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0


images_tr, labels_tr = data.cifar10(True)
images_te, labels_te = data.cifar10(False)
itr = data.first_per_class(labels_tr, 500)
ite = data.first_per_class(labels_te, 100)
split = np.random.default_rng(0)
val_mask = np.zeros(len(itr), bool)
for k in range(10):
    val_mask[split.permutation(np.flatnonzero(labels_tr[itr] == k))[:50]] = True
tr_idx, va_idx = itr[~val_mask], itr[val_mask]
X = {"train": rgb(images_tr[tr_idx]), "val": rgb(images_tr[va_idx]), "test": rgb(images_te[ite])}
Y = {"train": torch.as_tensor(labels_tr[tr_idx]), "val": torch.as_tensor(labels_tr[va_idx]),
     "test": torch.as_tensor(labels_te[ite])}

c = connectome.load()
cfg, _, _ = frozen.load()
torch.manual_seed(SEED)                      # readout + input projection init identical across graphs at a seed
model = NoBrainTranslators(c) if GRAPH == "nobrain" else BrainTranslators(c, load_graph(c, GRAPH, cfg))
counts = model.trainable_counts()
assert counts["total"] == counts["total_requires_grad"], counts
init_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                    for p in list(model.input_proj.parameters()) + list(model.readout.parameters())
                                    )).hexdigest()
meta = {"graph": GRAPH, "seed": SEED, "mode": MODE, "n_entry": int(model.input_proj.out_features),
        "trainable": counts, "init_params_sha256": init_hash}
print("META", json.dumps(meta), flush=True)
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

ckpt_path = OUT / "last.pt"
log, start = [], 0
if ckpt_path.exists():
    ck = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    log, start = ck["log"], ck["epoch"] + 1


@torch.no_grad()
def evaluate(name):
    model.eval()
    preds = [model.readout(model.features(X[name][s:s + BATCH].cuda(), grad=False)).argmax(1).cpu()
             for s in range(0, len(X[name]), BATCH)]
    p = torch.cat(preds)
    return float((p == Y[name]).float().mean()), p.numpy()


for epoch in range(start, EPOCHS):
    t0 = time.time()
    order = torch.as_tensor(np.random.default_rng(SEED * 1000 + epoch).permutation(len(X["train"])))
    model.train()
    losses, correct = [], 0
    for s in range(0, len(order), BATCH):
        b = order[s:s + BATCH]
        logits = model(X["train"][b].cuda())
        y = Y["train"][b].cuda()
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CLIP)
        opt.step()
        losses.append(loss.item())
        correct += (logits.argmax(1) == y).sum().item()
        if not np.isfinite(losses[-1]):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
    val_acc, _ = evaluate("val")
    improved = not log or val_acc > max(r["val_acc"] for r in log)
    log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "train_acc": correct / len(order),
                "val_acc": val_acc, "sec": time.time() - t0})
    print(json.dumps(log[-1]), flush=True)
    if improved:
        torch.save(model.state_dict(), OUT / "best.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "log": log, "epoch": epoch}, ckpt_path)
    (OUT / "log.json").write_text(json.dumps(log, indent=1))

best = max(log, key=lambda r: (r["val_acc"], -r["epoch"]))
model.load_state_dict(torch.load(OUT / "best.pt", weights_only=False))
test_acc, preds = evaluate("test")
np.save(OUT / "test_preds.npy", preds)
result = {**meta, "best_epoch": best["epoch"], "val_acc": best["val_acc"], "test_acc": test_acc}
(OUT / "result.json").write_text(json.dumps(result, indent=1))
print("RESULT", json.dumps(result), flush=True)
