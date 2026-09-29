"""D5 architecture smoke test (PROTOCOL_D5_flygrapher.md §8): the gate before the main run.
Seed 0, CIFAR-100 (the D4 train / val split); nothing here is used for any verdict.
usage: d5_smoke.py prep                 label-free: control graphs (seed 0) and their checks, patch assignment, Grid k
       d5_smoke.py train NAME            NAME in real | rewired_l | grid | bypass | none | vig | real_shuffled
                                         400 iterations at K = 1 with checks 1-4 (+ the brain-state shuffle diagnostic)
       d5_smoke.py timing                check 5: K = 1, 2, 4 and one brain with 4 channels, 50 iterations each
       d5_smoke.py summary               pass / fail per check -> <FLYVL_DATA>/d5/smoke/summary.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flyvl import flygrapher as fg  # noqa: E402
from d4_train import cifar, cifar_prep, crop_flip  # noqa: E402

OUT = fg.D5 / "smoke"
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
ITERS, BS, LR, WARM, SEED = 400, 128, 1e-3, 20, 0
torch.backends.cuda.matmul.allow_tf32 = True
CATEGORY = [("OL", ("ol_",)), ("VPN", ("visual_projection",)), ("VCN", ("visual_centrifugal",)), ("central", ("cb_",)),
            ("VNC", ("vnc_",)), ("neck/other", ("",))]


def category(sc):
    return next(name for name, pre in CATEGORY if any(sc.startswith(p) for p in pre))


def data():
    X, y = cifar("train")
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(12345))
    tr, va = perm[:45000], perm[45000:]
    return X[tr], y[tr], X[va][:2000], y[va][:2000]


def make(name, brain=None, K=1, tied=False):
    cond = {"real": "fly", "rewired_l": "fly", "real_shuffled": "fly"}.get(name, name)
    graph = grid_A = None
    if cond == "fly":
        W, _ = fg.build_graph(brain, "rewired_l" if name == "rewired_l" else "real", SEED)
        graph = fg.BrainGraph(W, brain)
    if cond == "grid":
        prep = json.loads((OUT / "prep.json").read_text())
        grid_A = fg.grid_adjacency(prep["grid_k"])
    torch.manual_seed(SEED)
    model = fg.D5Net(cond, K=K, graph=graph, grid_A=grid_A, tied=tied).to(dev)
    return model, graph


def group_categories(brain):
    cats = np.empty(brain.n_groups, object)
    sc_of = {}
    for gid, sc in zip(brain.group, brain.superclass):
        sc_of.setdefault(gid, {}).setdefault(sc, 0)
        sc_of[gid][sc] += 1
    for gid, d in sc_of.items():
        cats[gid] = category(max(d, key=d.get))
    return cats


def optimizer(model):
    phys, other = model.branch_params()
    opt = torch.optim.AdamW([{"params": other, "weight_decay": 0.05}, {"params": phys, "weight_decay": 0.0}], lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM))
    return opt, sched


@torch.no_grad()
def val_loss(model, Xv, yv, prep, bs=250):
    model.eval()
    tot = 0.0
    for s in range(0, len(Xv), bs):
        with torch.autocast("cuda", torch.bfloat16):
            tot += float(F.cross_entropy(model(prep(Xv[s:s + bs])).float(), yv[s:s + bs].to(dev), reduction="sum"))
    model.train()
    return tot / len(Xv)


@torch.no_grad()
def brain_health(model, x):
    """Check 3 on one batch: per block, fraction of the readout's energy that varies across images, and saturation."""
    out = []
    for gr in model.graphers:
        gr.keep_state = True
    model.eval()
    with torch.autocast("cuda", torch.bfloat16):
        model(x)
    model.train()
    for gr in model.graphers:
        a, fin = gr.last["abar"], gr.last["final"]                    # (n_ro, B)
        var = a.var(1, unbiased=False).sum()
        energy = (a ** 2).mean(1).sum()
        by_sc = {}
        for cat in ("OL", "VPN", "central"):
            m = torch.as_tensor(np.array([category(s) == cat for s in gr.graph.ro_superclass]), device=a.device)
            by_sc[cat] = float(a[m].abs().mean())
        out.append({"varying_energy_fraction": float(var / energy.clamp_min(1e-30)),
                    "saturated_fraction": float((fin.abs() > 0.49).float().mean()),
                    "mean_abs_readout": float(a.abs().mean()), "mean_abs_by_region": by_sc})
        gr.keep_state, gr.last = False, {}
    return out


@torch.no_grad()
def output_energy(model, x):
    """Per block, fraction of the branch output's energy (after P_out) that varies across images."""
    model.capture = []
    model.eval()
    model(x)
    model.train()
    out = [float(y.var(0, unbiased=False).sum() / (y ** 2).mean(0).sum().clamp_min(1e-30)) for _, y in model.capture]
    model.capture = None
    return out


@torch.no_grad()
def relay_r2(model, x):
    """Check 4b: R^2 of each block's branch output fitted as a linear function of the same patch's input token."""
    model.capture = []
    model.eval()
    model(x)
    model.train()
    r2 = []
    for xh, y in model.capture:
        Xm = torch.cat([xh.reshape(-1, xh.shape[-1]), torch.ones(xh.shape[0] * xh.shape[1], 1, device=dev)], 1).double()
        Ym = y.reshape(-1, y.shape[-1]).double()
        beta = torch.linalg.lstsq(Xm, Ym).solution
        res = ((Ym - Xm @ beta) ** 2).sum()
        tot = ((Ym - Ym.mean(0)) ** 2).sum()
        r2.append(float(1 - res / tot))
    model.capture = None
    return r2


def grad_report(model, cats):
    rep = []
    for gr in model.graphers:
        if not hasattr(gr, "g_pre"):
            ps = gr.physiology()
            rep.append({"norms": [float(p.grad.norm()) if p.grad is not None else 0.0 for p in ps]})
            continue
        norms = {k: float(getattr(gr, k).grad.norm()) if getattr(gr, k).grad is not None else 0.0 for k in ("g_pre", "g_post", "tau")}
        nz = (gr.g_pre.grad.abs().sum(0) > 0).cpu().numpy() if gr.g_pre.grad is not None else np.zeros(len(cats), bool)
        frac = {c: float(nz[cats == c].mean()) for c, _ in CATEGORY if (cats == c).any()}
        rep.append({"norms": norms, "finite": all(math.isfinite(v) for v in norms.values()), "nonzero_group_fraction": frac})
    return rep


def train(name):
    brain = fg.load_malecns() if name in ("real", "rewired_l", "real_shuffled") else None
    Xtr, ytr, Xv, yv = data()
    prep = cifar_prep()
    model, graph = make(name, brain)
    cats = group_categories(brain) if brain is not None else None
    if name == "real_shuffled":
        perm = torch.randperm(len(graph.readout), generator=torch.Generator().manual_seed(1)).to(dev)
        for gr in model.graphers:
            gr.shuffle = perm
    calib_x = prep(Xtr[:BS])
    scales = fg.calibrate_output_scale(model, calib_x)
    rec = {"name": name, "params": sum(p.numel() for p in model.parameters()), "output_scales_at_init": scales}
    probe_x = prep(Xv[:256])
    if brain is not None:
        rec["health_init"] = brain_health(model, probe_x)
    opt, sched = optimizer(model)
    g = torch.Generator().manual_seed(SEED)
    order = torch.cat([torch.randperm(len(Xtr), generator=g), torch.randperm(len(Xtr), generator=g)])
    losses, grads, t0 = [], {}, time.time()
    for it in range(ITERS):
        b = order[it * BS:(it + 1) * BS]
        x = crop_flip(prep(Xtr[b]))
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(model(x).float(), ytr[b].to(dev), label_smoothing=0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if brain is not None and it + 1 in (1, 100, ITERS):
            grads[it + 1] = grad_report(model, cats)
        elif name == "grid" and it + 1 in (1, ITERS):
            grads[it + 1] = grad_report(model, None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss))
        if not math.isfinite(losses[-1]):
            break
        if (it + 1) % 50 == 0:
            print(f"{name} it {it + 1} loss {np.mean(losses[-50:]):.3f} ({(time.time() - t0) / (it + 1):.2f} s/it)", flush=True)
    rec.update(losses=losses, sec_per_iter=(time.time() - t0) / len(losses), grads=grads,
               val_loss=val_loss(model, Xv, yv, prep))
    if name in ("real", "rewired_l", "real_shuffled", "grid", "bypass"):
        rec["output_energy_trained"] = output_energy(model, probe_x)
    if brain is not None:
        rec["health_trained"] = brain_health(model, probe_x)
        rec["relay_r2"] = relay_r2(model, prep(Xv[:64]))
        for gr in model.graphers:
            gr.brain_off = True
        rec["val_loss_brain_off"] = val_loss(model, Xv, yv, prep)
        for gr in model.graphers:
            gr.brain_off = False
        if name != "real_shuffled":
            perm = torch.randperm(len(graph.readout), generator=torch.Generator().manual_seed(1)).to(dev)
            for gr in model.graphers:
                gr.shuffle = perm
            rec["val_loss_shuffled_eval"] = val_loss(model, Xv, yv, prep)
            rec["output_energy_shuffled_eval"] = output_energy(model, probe_x)
            for gr in model.graphers:
                gr.shuffle = None
    elif name in ("grid", "bypass"):
        for gr in model.graphers:
            gr.brain_off = True
        rec["val_loss_brain_off"] = val_loss(model, Xv, yv, prep) if name == "grid" else None
    (OUT / f"{name}.json").write_text(json.dumps(rec))
    print(f"{name}: final loss {np.mean(losses[-50:]):.3f} (first 10: {np.mean(losses[:10]):.3f}), val loss {rec['val_loss']:.3f}", flush=True)


def prep_cmd():
    brain = fg.load_malecns()
    rec = {"patches": brain.info, "n_groups": brain.n_groups}
    real = fg.BrainGraph(brain.W, brain)
    rec["grid_k"] = fg.grid_k(real)
    rec["readout_neurons"] = int(len(real.readout))
    rec["footprint_nonzero_fraction"] = float((real.Fp[real.readout].sum(1) > 0).float().mean())
    del real
    torch.cuda.empty_cache()
    graphs = {}
    for kind in ("rewired_l", "rewired_m", "random"):
        t0 = time.time()
        R, info = fg.build_graph(brain, kind, SEED)
        if kind == "rewired_l" and info["changed_fraction"] < 0.8:          # registered: widen the tiles once
            R, info = fg.build_graph(brain, kind, SEED, tile_side=4)
        info["out_degree_exact"] = bool((np.diff(R.tocsc().indptr) == np.diff(brain.W.tocsc().indptr)).all())
        sign_ok = np.sign(np.asarray(R.tocsc().sum(0)).ravel()) * np.sign(np.asarray(brain.W.tocsc().sum(0)).ravel())
        info["presynaptic_sign_kept"] = bool((sign_ok >= 0).all())
        info["minutes"] = (time.time() - t0) / 60
        graphs[kind] = info
        print(kind, json.dumps(info), flush=True)
    rec["graphs"] = graphs
    (OUT / "prep.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "graphs"}, indent=1))


def timing():
    brain = fg.load_malecns()
    Xtr, ytr, _, _ = data()
    prep = cifar_prep()
    W, _ = fg.build_graph(brain, "real", SEED)
    graph = fg.BrainGraph(W, brain)
    res = {}
    for label, K, tied in (("K1", 1, False), ("K2", 2, False), ("K4", 4, False), ("c4", 4, True)):
        torch.manual_seed(SEED)
        model = fg.D5Net("fly", K=K, graph=graph, tied=tied).to(dev)
        opt, _ = optimizer(model)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for it in range(53):
            b = torch.arange(it * BS, (it + 1) * BS)
            torch.cuda.synchronize()
            t = time.time()
            with torch.autocast("cuda", torch.bfloat16):
                loss = F.cross_entropy(model(crop_flip(prep(Xtr[b]))).float(), ytr[b].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            if it >= 3:
                times.append(time.time() - t)
        res[label] = {"sec_per_iter": float(np.mean(times)), "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                      "params": sum(p.numel() for p in model.parameters())}
        print(label, json.dumps(res[label]), flush=True)
        del model, opt
        torch.cuda.empty_cache()
    (OUT / "timing.json").write_text(json.dumps(res, indent=1))


def summary():
    r = {n: json.loads((OUT / f"{n}.json").read_text()) for n in ("real", "rewired_l", "grid", "bypass", "none", "vig", "real_shuffled")
         if (OUT / f"{n}.json").exists()}
    tm = json.loads((OUT / "timing.json").read_text()) if (OUT / "timing.json").exists() else None
    first = lambda n: float(np.mean(r[n]["losses"][:10]))
    last = lambda n: float(np.mean(r[n]["losses"][-50:]))
    s = {}
    s["1_loss_decreases"] = {n: {"first10": first(n), "last50": last(n), "pass": last(n) <= 0.9 * first(n)}
                             for n in ("real", "rewired_l", "grid", "bypass", "none", "vig") if n in r}
    s["2_gradients"] = {}
    for n in ("real", "rewired_l"):
        if n in r:
            fin = r[n]["grads"][str(ITERS)] if str(ITERS) in r[n]["grads"] else r[n]["grads"][ITERS]
            ok = all(b["finite"] and all(v > 0 for v in b["norms"].values()) for b in fin)
            ok &= all(b["nonzero_group_fraction"].get("OL", 0) >= 0.5 and b["nonzero_group_fraction"].get("VPN", 0) >= 0.5 for b in fin)
            s["2_gradients"][n] = {"pass": ok, "last": fin}
    s["3_not_constant"] = {}
    for n in ("real", "rewired_l"):
        if n in r:
            h = r[n]["health_init"] + r[n]["health_trained"]
            s["3_not_constant"][n] = {"pass": all(b["varying_energy_fraction"] >= 0.1 and b["saturated_fraction"] < 0.2 for b in h),
                                      "init": r[n]["health_init"], "trained": r[n]["health_trained"]}
    s["4_no_bypass"] = {}
    for n in ("real", "rewired_l"):
        if n in r:
            d = r[n]["val_loss_brain_off"] - r[n]["val_loss"]
            s["4_no_bypass"][n] = {"brain_off_delta": d, "relay_r2": r[n]["relay_r2"],
                                   "pass": d >= 0.01 and max(r[n]["relay_r2"]) < 0.9}
    if "bypass" in r:
        s["4_no_bypass"]["bypass_last50_vs_real"] = {"bypass": last("bypass"), "real": last("real") if "real" in r else None}
    if tm:
        base = tm["K1"]["sec_per_iter"]
        s["5_scaling"] = {k: {"ratio_to_K1": v["sec_per_iter"] / base, "peak_gb": v["peak_gb"]} for k, v in tm.items()}
        s["5_scaling"]["pass"] = all(0.75 * K <= tm[k]["sec_per_iter"] / base <= 1.25 * K for k, K in (("K2", 2), ("K4", 4), ("c4", 4))) \
            and all(v["peak_gb"] < 80 for v in tm.values())
    if "real" in r and "rewired_l" in r:
        dr, dw = first("real") - last("real"), first("rewired_l") - last("rewired_l")
        s["6_real_only_failure"] = {"real_drop": dr, "rewired_l_drop": dw, "flag": dr < 0.5 * dw}
    diag = {}
    for n in ("real", "rewired_l"):
        if n in r and "val_loss_shuffled_eval" in r[n]:
            diag[n] = {"val_loss_change_when_shuffled": r[n]["val_loss_shuffled_eval"] - r[n]["val_loss"],
                       "output_energy": r[n]["output_energy_trained"], "output_energy_shuffled": r[n]["output_energy_shuffled_eval"]}
    if "real_shuffled" in r and "real" in r:
        diag["trained_with_shuffled_identity"] = {"last50_shuffled": last("real_shuffled"), "last50_real": last("real")}
    s["diagnostic_shuffle"] = diag
    gate = [s["1_loss_decreases"].get(n, {}).get("pass", False) for n in ("real", "rewired_l", "grid")]
    gate += [v["pass"] for v in s["2_gradients"].values()] + [v["pass"] for v in s["3_not_constant"].values()]
    gate += [v["pass"] for k, v in s["4_no_bypass"].items() if k in ("real", "rewired_l")]
    gate += [s.get("5_scaling", {}).get("pass", False)]
    s["gate_1_to_5_pass"] = bool(all(gate))
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1)[:6000])


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"prep": prep_cmd, "timing": timing, "summary": summary}.get(cmd, lambda: train(sys.argv[2]))()
