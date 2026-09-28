"""D4 (PROTOCOL_D4_flyvig.md): train FlyViG conditions on the smoke task (CIFAR-100) and the registered motion videos.

Videos (the M1 / M2 stimuli of this project, re-implemented here without letters): 4 x 4 cells, drawn at 128 px with
fresh pixel noise every 20 ms frame, averaged to 64 px, 25 frames.
  fg16   the whole background drifts, one cell moves differently -> which cell
  mix16  one cell moves (random direction), another only flickers  -> which cell moves
usage: d4_train.py cifar COND SEED
       d4_train.py video TASK LO,HI COND SEED NTRAIN
       d4_train.py calib TASK            (Stage 0: vig, seed 0, 2,100 videos, contrast ladder until test <= 90%)
COND in vig | dense | dense_small | real | rewired | random
"""
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome  # noqa: E402
from flyvl.flyvig import FlyViG, TypeFFN, random_graph, rewire  # noqa: E402

D4 = connectome.DATA_ROOT / "d4"
OUT = Path(os.environ.get("D4_OUT", connectome.DATA_ROOT / "runs" / "d4"))
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
T, DT, PIX, OUTPIX = 25, 0.02, 128, 64
NTR, NVAL, NTE = 2100, 900, 900
BS = 32
STEPS_VIDEO = int(os.environ.get("D4_STEPS", 2500))                  # registered budget (PROTOCOL_D4 appendix A)
EVAL_EVERY = STEPS_VIDEO // 20
LADDER = ("0.04,0.10", "0.02,0.05", "0.01,0.03")
TASK_SEED = {"fg16": 100, "mix16": 200}


def graph_bank():
    d = np.load(D4 / "type_graph.npz")
    names, M = list(d["names"]), d["M"]
    return names, M, round(TypeFFN(M).active_params() / (2 * len(names) * 3))


def make_model(cond, seed, cin, H, n_cls):
    names, M, hs = graph_bank()
    masks = {"real": M, "rewired": rewire(M, seed), "random": random_graph(M, seed)}
    mode = "graph" if cond in masks else cond
    return FlyViG(names, mode, masks.get(cond), cin=cin, H=H, n_cls=n_cls, hid_small=hs).to(dev)


# ---------------------------------------------------------------- videos
def trial_params(task, n, seed, contrast):
    g = np.random.default_rng(seed)
    y = g.integers(0, 16, n)
    p = {"cell": y}
    if task == "fg16":
        p["bgdir"], p["odd"] = g.integers(0, 4, n), g.integers(1, 4, n)          # target dir = (bgdir + odd) % 4
    else:
        p["cell2"], p["dir"], p["flick"] = (y + g.integers(1, 16, n)) % 16, g.integers(0, 4, n), g.uniform(4, 8, n)
    p["freq"], p["speed"], p["phase"] = g.uniform(2, 5, n), g.uniform(0.6, 1.4, n), g.uniform(0, 2 * np.pi, n)
    lo, hi = contrast
    p["contrast"] = g.uniform(lo, hi, n)
    return y, p


def frame(p, t, noise_gen):
    B = len(p["cell"])
    f = lambda k: torch.as_tensor(p[k], dtype=torch.float32, device=dev)[:, None, None]
    ax = torch.linspace(-1, 1, PIX, device=dev)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    cx = ((xx + 1) / 2 * 4).clamp(max=3.999).floor()[None]
    cy = ((yy + 1) / 2 * 4).clamp(max=3.999).floor()[None]
    inside = ((cy * 4 + cx) == torch.as_tensor(p["cell"], device=dev)[:, None, None]).float()

    def drift(dirs):
        d = torch.as_tensor(dirs, device=dev)[:, None, None]
        vertical = (d >= 2).float()
        sign = torch.where((d == 1) | (d == 3), -1.0, 1.0)
        coord = xx[None] * (1 - vertical) + yy[None] * vertical
        return torch.sin(2 * np.pi * f("freq") * (coord - sign * f("speed") * t) + f("phase"))

    if "bgdir" in p:
        tex = inside * drift((np.asarray(p["bgdir"]) + np.asarray(p["odd"])) % 4) + (1 - inside) * drift(p["bgdir"])
    else:
        still = torch.sin(2 * np.pi * f("freq") * xx[None] + f("phase"))
        inside2 = ((cy * 4 + cx) == torch.as_tensor(p["cell2"], device=dev)[:, None, None]).float()
        flick = torch.cos(2 * np.pi * f("flick") * t)
        tex = inside * drift(p["dir"]) + inside2 * still * flick + (1 - inside - inside2) * still
    lum = 0.5 + f("contrast") * tex + 0.06 * torch.randn(B, PIX, PIX, generator=noise_gen, device=dev)
    return F.avg_pool2d(lum[:, None], PIX // OUTPIX)[:, 0]


def video_set(task, contrast, split):
    lo, hi = (float(v) for v in contrast.split(","))
    f = D4 / f"{task}_c{contrast}_{split}.pt"
    if f.exists():
        return torch.load(f)
    seed = TASK_SEED[task] + {"train": 1, "val": 2, "test": 3}[split]
    n = {"train": NTR, "val": NVAL, "test": NTE}[split]
    y, p = trial_params(task, n, seed, (lo, hi))
    X = torch.zeros(n, T, OUTPIX, OUTPIX, dtype=torch.float16)
    for s in range(0, n, 700):
        pb = {k: v[s:s + 700] for k, v in p.items()}
        gen = torch.Generator(device=dev).manual_seed(seed * 1000 + s)
        for k in range(T):
            X[s:s + 700, k] = frame(pb, k * DT, gen).half().cpu()
    d = {"x": X, "y": torch.as_tensor(y)}
    torch.save(d, f)
    return d


# ---------------------------------------------------------------- training
def evaluate(model, X, y, prep, bs=64):
    model.eval()
    ok = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for s in range(0, len(X), bs):
            ok.append((model(prep(X[s:s + bs])).argmax(1).cpu() == y[s:s + bs]))
    model.train()
    return torch.cat(ok)


def fit(model, Xtr, ytr, Xval, yval, prep, steps, eval_every, lr=1e-3, augment=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    warm = steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    g = torch.Generator().manual_seed(torch.initial_seed())                   # set by torch.manual_seed(seed)
    order, pos = torch.randperm(len(Xtr), generator=g), 0
    best, best_state, curve = -1, None, []
    for step in range(steps):
        if pos + BS > len(order):
            order, pos = torch.randperm(len(Xtr), generator=g), 0
        b = order[pos:pos + BS]
        pos += BS
        x = prep(Xtr[b])
        if augment:
            x = augment(x)
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(model(x), ytr[b].to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if Xval is not None and ((step + 1) % eval_every == 0 or step == steps - 1):
            acc = float(evaluate(model, Xval, yval, prep).float().mean())
            curve.append((step + 1, round(float(loss), 4), acc))
            if acc > best:
                best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    step {step + 1} loss {float(loss):.3f} val {acc:.3f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, curve


def run_video(task, contrast, cond, seed, ntrain):
    name = f"{task}_c{contrast}_n{ntrain}_{cond}_s{seed}"
    if (OUT / f"{name}.json").exists():
        return json.loads((OUT / f"{name}.json").read_text())
    t0 = time.time()
    tr, va, te = (video_set(task, contrast, s) for s in ("train", "val", "test"))
    prep = lambda x: ((x.to(dev).float() - 0.5) / 0.1)[:, :, None]
    torch.manual_seed(seed)
    model = make_model(cond, seed, 1, OUTPIX, 16)
    best, curve = fit(model, tr["x"][:ntrain], tr["y"][:ntrain], va["x"], va["y"], prep, STEPS_VIDEO, EVAL_EVERY)
    ok = evaluate(model, te["x"], te["y"], prep)
    rec = {"task": task, "contrast": contrast, "cond": cond, "seed": seed, "ntrain": ntrain, "steps": STEPS_VIDEO,
           "val_best": best, "test_acc": float(ok.float().mean()), "test_ok": ok.int().tolist(), "curve": curve,
           "params": sum(p.numel() for p in model.parameters()), "minutes": (time.time() - t0) / 60}
    (OUT / f"{name}.json").write_text(json.dumps(rec))
    print(f"{name}: test {rec['test_acc']:.3f} (val {best:.3f}, {rec['minutes']:.1f} min)", flush=True)
    return rec


# ---------------------------------------------------------------- CIFAR-100 smoke test
def cifar(split):
    f = D4 / f"cifar100_{split}.npz"
    if not f.exists():
        import pyarrow.parquet as pq
        from PIL import Image
        t = pq.read_table(connectome.DATA_ROOT / "datasets" / "cifar100" / f"{split}.parquet").to_pydict()
        X = np.stack([np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in t["img"]])
        np.savez(f, x=X, y=np.asarray(t["fine_label"]))
    d = np.load(f)
    return torch.as_tensor(d["x"]), torch.as_tensor(d["y"])


def run_cifar(cond, seed, epochs=int(os.environ.get("D4_CIFAR_EPOCHS", 15))):
    name = f"cifar100_{cond}_s{seed}"
    t0 = time.time()
    (Xtr, ytr), (Xte, yte) = cifar("train"), cifar("test")
    mu = torch.tensor([0.507, 0.487, 0.441], device=dev).view(1, 1, 3, 1, 1)
    sd = torch.tensor([0.267, 0.256, 0.276], device=dev).view(1, 1, 3, 1, 1)
    prep = lambda x: ((x.to(dev).permute(0, 3, 1, 2).float() / 255)[:, None] - mu) / sd

    def augment(x):
        B = x.shape[0]
        x = F.pad(x[:, 0], (4, 4, 4, 4), mode="reflect")
        i, j = torch.randint(0, 9, (2,)).tolist()
        x = x[..., i:i + 32, j:j + 32]
        flip = torch.rand(B, device=dev) < 0.5
        return torch.where(flip[:, None, None, None], x.flip(-1), x)[:, None]

    torch.manual_seed(seed)
    model = make_model(cond, seed, 3, 32, 100)
    global BS
    BS = 128
    fit(model, Xtr, ytr, None, None, prep, epochs * math.ceil(len(Xtr) / BS), 10 ** 9, augment=augment)
    acc = float(evaluate(model, Xte, yte, prep).float().mean())
    rec = {"task": "cifar100", "cond": cond, "seed": seed, "epochs": epochs, "test_acc": acc,
           "params": sum(p.numel() for p in model.parameters()), "minutes": (time.time() - t0) / 60}
    (OUT / f"{name}.json").write_text(json.dumps(rec))
    print(f"{name}: test {acc:.3f} ({rec['minutes']:.1f} min)", flush=True)


def calib(task):
    res = {}
    for c in LADDER:
        r = run_video(task, c, "vig", 0, NTR)
        res[c] = r["test_acc"]
        if r["test_acc"] <= 0.90:
            break
    choice = next((c for c in LADDER if res.get(c, 1) <= 0.90), LADDER[-1])
    (OUT / f"calib_{task}.json").write_text(json.dumps({"ladder": res, "choice": choice}))
    print(f"CALIB {task}: {res} -> {choice}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cifar":
        run_cifar(sys.argv[2], int(sys.argv[3]))
    elif cmd == "video":
        run_video(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
    elif cmd == "calib":
        calib(sys.argv[2])
