"""D7 (PROTOCOL_D7_screen.md): falsification screen for connectome-specific computations, one harness for every task
(D6's AI row: learned per-frame encoder -> frozen whole MaleCNS -> VPN readout).
usage: d7_screen.py run TASK MODEL SEED     TASK in 0 | N | A;  MODEL in real | rewired_l | bypass | gru | engineered
       d7_screen.py verdict TASK
Task 0 reuses D6 (cells C / D are Real / Rewired-L of this harness, HR its engineered baseline).
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flyvl import flygrapher as fg  # noqa: E402
from flyvl.flyvig import BN, Stem  # noqa: E402
import d6_translation as d6  # noqa: E402

OUT = fg.connectome.DATA_ROOT / "d7"
OUT.mkdir(parents=True, exist_ok=True)
D6 = fg.connectome.DATA_ROOT / "d6"
dev = "cuda"
FR, PIX, SPF = d6.FRAMES, d6.PIX, d6.STEPS_PER_FRAME
EPOCHS, BS, LR, WD, DELTA = d6.EPOCHS, d6.BS, d6.LR, d6.WD, 0.01
TASKS = {  # n classes, readout window, in-distribution test conditions, out-of-distribution conditions
    "0": (4, "mean", ["drift"], []),
    "N": (4, "mean", ["shapes"], []),
    "A": (16, "last", ["clean", "noise", "distractors", "crossing", "occlusion", "jitter"], ["ood_speed"]),
}
NTR, NVA, NTE = 1800, 300, 450


# ------------------------------------------------------------------ task data
def blob(xx, yy, x, y, s=1.2):
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))


def gen_A(n, seed, cond):
    """Dark target moving at constant velocity; label = its 4 x 4 bin in the final frame (image coordinates)."""
    g = np.random.default_rng(seed)
    X = np.empty((n, FR, PIX, PIX), np.float32)
    y = np.empty(n, np.int64)
    fin = np.empty((n, 2), np.float32)
    yy, xx = np.meshgrid(np.arange(PIX) + 0.5, np.arange(PIX) + 0.5, indexing="ij")
    lo_s, hi_s = (1.0, 1.6) if cond == "ood_speed" else (0.25, 0.8)
    noise = 0.10 if cond == "noise" else 0.03
    M = 4
    for i in range(n):
        tex = g.normal(0, 1, (PIX + 2 * M, PIX + 2 * M))
        tex = torch.as_tensor(tex, dtype=torch.float32)[None, None]
        k = torch.exp(-torch.arange(-6, 7, dtype=torch.float32) ** 2 / 8.0)
        k = k / k.sum()
        tex = F.conv2d(F.conv2d(F.pad(tex, (6, 6, 6, 6), mode="reflect"), k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))[0, 0].numpy()
        tex = 0.05 * (tex - tex.mean()) / tex.std()

        def path(lo, hi):
            ang, sp = g.uniform(0, 2 * np.pi), g.uniform(lo, hi) * PIX * d6.FRAME_DT
            v = np.array([np.cos(ang), np.sin(ang)]) * sp
            d = v * (FR - 1)
            x0 = g.uniform(3 - min(0, d[0]), 28 - max(0, d[0]))
            y0 = g.uniform(3 - min(0, d[1]), 28 - max(0, d[1]))
            return np.array([x0, y0]) + np.arange(FR)[:, None] * v
        tgt = path(lo_s, hi_s)
        others = []
        if cond == "distractors":
            others = [path(0.25, 0.8) for _ in range(3)]
        elif cond == "crossing":
            tc = g.integers(3, 5)
            ang, sp = g.uniform(0, 2 * np.pi), g.uniform(0.25, 0.8) * PIX * d6.FRAME_DT
            v = np.array([np.cos(ang), np.sin(ang)]) * sp
            others = [tgt[tc] + (np.arange(FR) - tc)[:, None] * v]
        off = g.integers(-2, 3, (FR, 2)) if cond == "jitter" else np.zeros((FR, 2), int)
        for t in range(FR):
            ox, oy = off[t]
            frame = 0.5 + tex[M + oy:M + oy + PIX, M + ox:M + ox + PIX]
            if not (cond == "occlusion" and t >= 5):
                frame = frame - 0.35 * blob(xx, yy, tgt[t, 0] - ox, tgt[t, 1] - oy)
            for o in others:
                frame = frame + 0.35 * blob(xx, yy, o[t, 0] - ox, o[t, 1] - oy)
            X[i, t] = frame + g.normal(0, noise, (PIX, PIX))
        fx, fy = np.clip(tgt[-1] - off[-1], 0, PIX - 1e-3)
        fin[i] = (fx, fy)
        y[i] = int(fy // 8) * 4 + int(fx // 8)
    return X.astype(np.float16), y, fin


def gen_N(n, seed, cond="shapes"):
    """Static shape (circle, square, triangle, cross), random size / position / polarity, repeated for 8 frames."""
    g = np.random.default_rng(seed)
    X = np.empty((n, FR, PIX, PIX), np.float32)
    y = g.integers(0, 4, n)
    yy, xx = np.meshgrid(np.arange(PIX) + 0.5, np.arange(PIX) + 0.5, indexing="ij")
    for i in range(n):
        s = g.uniform(8, 14)
        cx, cy = g.uniform(s / 2 + 1, PIX - s / 2 - 1, 2)
        dx, dy = xx - cx, yy - cy
        if y[i] == 0:
            m = dx ** 2 + dy ** 2 <= (s / 2) ** 2
        elif y[i] == 1:
            m = (abs(dx) <= s / 2) & (abs(dy) <= s / 2)
        elif y[i] == 2:
            m = (dy <= s / 2) & (dy >= -s / 2) & (abs(dx) <= (dy + s / 2) / 2)
        else:
            m = ((abs(dx) <= s / 8) & (abs(dy) <= s / 2)) | ((abs(dy) <= s / 8) & (abs(dx) <= s / 2))
        img = 0.5 + g.choice([-0.3, 0.3]) * m
        X[i] = img[None] + g.normal(0, 0.05, (FR, PIX, PIX))
    return X.astype(np.float16), y, None


def task_data(task):
    """{'train': (x, y), 'val': (x, y), 'test': (x, y, cond_index, final_pos)}; conditions in TASKS order."""
    if task == "0":
        d = d6.data()
        return {"train": d["train"], "val": d["val"], "test": (*d["test"], torch.zeros(len(d["test"][1]), dtype=torch.long), None)}
    f = OUT / f"task{task}.npz"
    gen = {"A": gen_A, "N": gen_N}[task]
    conds = TASKS[task][2] + TASKS[task][3]
    if not f.exists():
        base = {"A": 700, "N": 800}[task]
        tr = [gen(NTR // len(TASKS[task][2]), base + 10 * j + 1, c) for j, c in enumerate(TASKS[task][2])]
        va = [gen(NVA // len(TASKS[task][2]), base + 10 * j + 2, c) for j, c in enumerate(TASKS[task][2])]
        te = [gen(NTE, base + 10 * j + 3, c) for j, c in enumerate(conds)]
        cat = lambda parts, k: np.concatenate([p[k] for p in parts])
        np.savez(f, trx=cat(tr, 0), try_=cat(tr, 1), vax=cat(va, 0), vay=cat(va, 1), tex=cat(te, 0), tey=cat(te, 1),
                 tec=np.repeat(np.arange(len(conds)), NTE),
                 tepos=cat(te, 2) if te[0][2] is not None else np.zeros((len(conds) * NTE, 2), np.float32))
    z = np.load(f)
    T = lambda a: torch.as_tensor(a)
    return {"train": (T(z["trx"]), T(z["try_"])), "val": (T(z["vax"]), T(z["vay"])),
            "test": (T(z["tex"]), T(z["tey"]), T(z["tec"]), z["tepos"])}


# ------------------------------------------------------------------ models
class Brain(d6.FrozenBrain):
    """D6's frozen brain with a registered readout window: 'mean' (all 16 steps) or 'last' (the final frame's steps)."""

    def __init__(self, brain, kind, seed, window):
        super().__init__(brain, kind, seed)
        self.window = window

    def __call__(self, u):
        B = u.shape[2]
        V = torch.zeros(self.n, B, device=dev)
        acc = torch.zeros(len(self.vpn), B, device=dev)
        for f in range(FR):
            for _ in range(SPF):
                drive = fg._SpMM.apply(self.W, self.WT, fg.act(V)).index_add(0, self.inputs, u[f])
                V = V + 0.5 * (drive - V)
                if self.window == "mean" or f == FR - 1:
                    acc = acc + fg.act(V)[self.vpn]
        return acc / (FR * SPF if self.window == "mean" else SPF)


class Readout(nn.Module):
    def __init__(self, d, n_cls):
        super().__init__()
        self.bn, self.lin = nn.BatchNorm1d(d, affine=False, eps=1e-12), nn.Linear(d, n_cls)

    def forward(self, feat):
        return self.lin(self.bn(feat))


class FlyNet(nn.Module):
    def __init__(self, brain, kind, seed, n_cls, window):
        super().__init__()
        self.enc = d6.AIEncoder(brain)
        self.net = Brain(brain, kind, seed, window)
        self.readout = Readout(len(self.net.vpn), n_cls)

    def forward(self, x):
        return self.readout(self.net(self.enc(x)).T)


class BypassNet(nn.Module):
    """Same encoder (stem, Norm, rank-5 P_in), no brain: the patch drives in the same window -> readout."""

    def __init__(self, n_cls, window, D=195):
        super().__init__()
        self.stem, self.norm, self.p_in = Stem(1, D, PIX), BN(D), fg.PatchIn(1, D)
        self.window = window
        self.readout = Readout(fg.N_PATCH * fg.N_IN, n_cls)

    def forward(self, x):
        B = x.shape[0]
        U = self.p_in(self.norm(self.stem(x.reshape(B * FR, 1, PIX, PIX))))[0].reshape(B, FR, -1)
        return self.readout(U.mean(1) if self.window == "mean" else U[:, -1])


class GRUNet(nn.Module):
    def __init__(self, n_cls, hidden, D=195):
        super().__init__()
        self.stem, self.norm = Stem(1, D, PIX), BN(D)
        self.squeeze = nn.Linear(D, 4)
        self.gru = nn.GRU(fg.N_PATCH * 4, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_cls)

    def forward(self, x):
        B = x.shape[0]
        z = self.squeeze(self.norm(self.stem(x.reshape(B * FR, 1, PIX, PIX)))).reshape(B, FR, -1)
        return self.head(self.gru(z)[0][:, -1])


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def gru_hidden(n_cls, fly_params):
    """Largest hidden size whose GRU net does not exceed the fly harness's trainable parameters."""
    h = 1
    while n_params(GRUNet(n_cls, h + 1)) <= fly_params:
        h += 1
    return h


# ------------------------------------------------------------------ training (the D6 recipe; returns predictions)
def fit(model, d, n_cls):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    spe = math.ceil(len(d["train"][1]) / BS)
    steps, warm = EPOCHS * spe, max(1, EPOCHS * spe // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    g = torch.Generator().manual_seed(torch.initial_seed())
    Xtr, ytr = d["train"]

    def predict(x):
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(x), 150):
                out.append(model(x[s:s + 150].to(dev).float()).argmax(1).cpu())
        model.train()
        return torch.cat(out)

    best, best_state, curve = -1.0, None, []
    for ep in range(EPOCHS):
        order = torch.randperm(len(ytr), generator=g)
        for i in range(spe):
            b = order[i * BS:(i + 1) * BS]
            loss = F.cross_entropy(model(Xtr[b].to(dev).float()), ytr[b].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        acc = float((predict(d["val"][0]) == d["val"][1]).float().mean())
        curve.append((ep + 1, round(float(loss), 4), acc))
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best, curve, predict(d["test"][0])


def tracker(d):
    """Engineered baseline for A: per frame, blur and take the darkest point if it is dark enough; fit constant
    velocity to the visible frames and predict the final-frame position."""
    x = d["test"][0].float()
    k = torch.exp(-torch.arange(-3, 4, dtype=torch.float32) ** 2 / 2.0)
    k = k / k.sum()
    B = x.shape[0]
    xs = x.reshape(B * FR, 1, PIX, PIX)
    xs = F.conv2d(F.conv2d(F.pad(xs, (3, 3, 3, 3), mode="reflect"), k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1)).reshape(B, FR, -1)
    mn, arg = xs.min(2)
    vis = (mn < 0.5 - 0.15).numpy()
    py, px = (arg // PIX).numpy() + 0.5, (arg % PIX).numpy() + 0.5
    pred = np.empty(B, np.int64)
    t = np.arange(FR)
    for i in range(B):
        v = vis[i]
        if v.sum() >= 2:
            fx, fy = np.polyval(np.polyfit(t[v], px[i, v], 1), FR - 1), np.polyval(np.polyfit(t[v], py[i, v], 1), FR - 1)
        elif v.sum() == 1:
            fx, fy = px[i, v][0], py[i, v][0]
        else:
            fx = fy = PIX / 2
        fx, fy = np.clip(fx, 0, PIX - 1e-3), np.clip(fy, 0, PIX - 1e-3)
        pred[i] = int(fy // 8) * 4 + int(fx // 8)
    return torch.as_tensor(pred)


def run(task, model_name, seed):
    name = f"task{task}_{model_name}_s{seed}"
    if (OUT / f"{name}.json").exists():
        return
    t0 = time.time()
    n_cls, window, _, _ = TASKS[task]
    d = task_data(task)
    torch.manual_seed(seed)
    brain = fg.load_malecns()
    fly_params = n_params(FlyNet(brain, "real", 1, n_cls, window)) if model_name in ("gru",) else None
    torch.manual_seed(seed)
    if model_name == "engineered":
        if task != "A":
            raise ValueError("engineered baseline: task 0 uses D6 HR; task N has none")
        best, curve, pred = None, None, tracker(d)
        params = 0
    else:
        if model_name in ("real", "rewired_l"):
            model = FlyNet(brain, model_name, seed, n_cls, window).to(dev)
        elif model_name == "bypass":
            model = BypassNet(n_cls, window).to(dev)
        elif model_name == "gru":
            model = GRUNet(n_cls, gru_hidden(n_cls, fly_params)).to(dev)
        params = n_params(model)
        best, curve, pred = fit(model, d, n_cls)
    ok = (pred == d["test"][1]).int()
    rec = {"task": task, "model": model_name, "seed": seed, "val_best": best, "curve": curve, "params": params,
           "test_ok": ok.tolist(), "test_pred": pred.tolist(), "minutes": (time.time() - t0) / 60}
    conds = TASKS[task][2] + TASKS[task][3]
    rec["by_condition"] = {c: float(ok[d["test"][2] == j].float().mean()) for j, c in enumerate(conds)}
    if task == "A":
        pos = d["test"][3]
        centers = np.stack([(pred.numpy() % 4) * 8 + 4, (pred.numpy() // 4) * 8 + 4], 1)
        err = np.linalg.norm(centers - pos, axis=1)
        rec["pos_err_px"] = {c: float(err[(d["test"][2] == j).numpy()].mean()) for j, c in enumerate(conds)}
    (OUT / f"{name}.json").write_text(json.dumps(rec))
    print(f"{name}: " + " ".join(f"{c} {v:.3f}" for c, v in rec["by_condition"].items())
          + f" | params {params:,} ({rec['minutes']:.1f} min)", flush=True)


# ------------------------------------------------------------------ verdict (PROTOCOL_D7 §3)
def verdict(task):
    seeds = (1, 2, 3)
    n_cls, _, ind, ood = TASKS[task]
    conds = ind + ood

    def load(model):
        if task == "0" and model in ("real", "rewired_l"):
            cell = {"real": "C", "rewired_l": "D"}[model]
            return [np.asarray(json.loads((D6 / f"{cell}_s{s}.json").read_text())["test_ok"]) for s in seeds]
        return [np.asarray(json.loads((OUT / f"task{task}_{model}_s{s}.json").read_text())["test_ok"]) for s in seeds]
    ok = {m: load(m) for m in ("real", "rewired_l", "bypass")}
    n_te = len(ok["real"][0])
    cond_idx = np.repeat(np.arange(len(conds)), NTE) if task != "0" else np.zeros(n_te, int)
    primary = cond_idx < len(ind)
    g = np.random.default_rng(0)
    idx_p = np.flatnonzero(primary)
    boots = [g.choice(idx_p, len(idx_p)) for _ in range(2000)]
    rep = {"task": task, "chance": 1 / n_cls}

    def compare(a, b, mask_boot=None):
        per = [float(ok[a][i][primary].mean() - ok[b][i][primary].mean()) for i in range(3)]
        dist = np.array([np.mean([ok[a][i][bb].mean() - ok[b][i][bb].mean() for i in range(3)]) for bb in boots])
        lo, hi = np.percentile(dist, [2.5, 97.5])
        return {"per_seed": per, "mean": float(np.mean(per)), "ci95": [float(lo), float(hi)],
                "effect": bool(all(x > 0 for x in per) and lo > DELTA)}
    rep["acc_primary"] = {m: [float(v[primary].mean()) for v in ok[m]] for m in ok}
    rep["real-rewired_l"] = compare("real", "rewired_l")
    rep["real-bypass"] = compare("real", "bypass")
    rep["rewired_l-bypass"] = compare("rewired_l", "bypass")
    ch = 1 / n_cls
    mr, mw = np.mean(rep["acc_primary"]["real"]), np.mean(rep["acc_primary"]["rewired_l"])
    rep["uninformative"] = bool((mr <= ch + 0.05 and mw <= ch + 0.05) or (mr >= 0.97 and mw >= 0.97))
    if rep["uninformative"]:
        rep["verdict"] = "UNINFORMATIVE (floor or ceiling)"
    elif rep["real-rewired_l"]["effect"]:
        rep["verdict"] = "TOPOLOGY EFFECT"
    elif rep["real-bypass"]["effect"] and rep["rewired_l-bypass"]["effect"]:
        rep["verdict"] = "GENERIC RESERVOIR EFFECT"
    else:
        rep["verdict"] = "NO EFFECT"
    by_cond = {}
    for j, c in enumerate(conds):
        m = cond_idx == j
        per = [float(ok["real"][i][m].mean() - ok["rewired_l"][i][m].mean()) for i in range(3)]
        by_cond[c] = {"real": float(np.mean([ok["real"][i][m].mean() for i in range(3)])),
                      "rewired_l": float(np.mean([ok["rewired_l"][i][m].mean() for i in range(3)])),
                      "diff_per_seed": per}
    rep["by_condition"] = by_cond
    for m in ("gru", "engineered"):
        f = [OUT / f"task{task}_{m}_s{s}.json" for s in seeds]
        if all(p.exists() for p in f):
            rep[m] = [float(np.asarray(json.loads(p.read_text())["test_ok"])[primary].mean()) for p in f]
    if task == "0":
        rep["engineered"] = [json.loads((D6 / f"HR_s{s}.json").read_text())["test_acc"] for s in seeds]
    (OUT / f"verdict_task{task}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("task", "acc_primary", "real-rewired_l", "real-bypass", "rewired_l-bypass", "verdict")}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1] == "verdict":
        verdict(sys.argv[2])
    elif sys.argv[1] == "data":
        d = task_data(sys.argv[2])
        print({k: tuple(v[0].shape) for k, v in d.items()})
