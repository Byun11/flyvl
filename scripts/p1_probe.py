"""P1 probe per PROTOCOL.md. usage: p1_probe.py mini"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, data, masks, probe  # noqa: E402
from flyvl.extract import to_luma  # noqa: E402

subset = sys.argv[1]
KS = (256, 1024, 4096)
SEEDS = (0, 1, 2)
VIEWS = ("all", "no_photoreceptor", "central_vnc", "visual_projection", "descending")
GRAPHS = {"mini": ("real", "global_shuffle_s0")}[subset]
PRIMARY = {"view": "central_vnc", "K": 1024, "a": "real", "b": "global_shuffle_s0"}
FEAT = connectome.DATA_ROOT / "features"
OUT = connectome.DATA_ROOT / "runs" / f"p1_{subset}"
OUT.mkdir(parents=True, exist_ok=True)

c = connectome.load()
M = masks.build(c)
r16 = c.types(["R1-6"])
driven = r16[c.column[r16, 0] >= 0]


def load(graph, split):
    d = FEAT / graph / f"cifar10_{subset}_{split}"
    assert json.loads((d / "progress.json").read_text())["done"] == len(np.load(d / "labels.npy"))
    return np.load(d / "features.fp16.npy", mmap_mode="r"), np.load(d / "labels.npy"), np.load(d / "indices.npy")


reps = {}
Fr_tr, ytr, itr = load("real", "train")
Fr_te, yte, ite = load("real", "test")
imgs_tr, _ = data.cifar10(True)
imgs_te, _ = data.cifar10(False)
pix_tr = to_luma(imgs_tr[itr]).reshape(len(itr), -1).numpy()
pix_te = to_luma(imgs_te[ite]).reshape(len(ite), -1).numpy()
reps["pixels"] = lambda: (pix_tr, pix_te)
rng = np.random.default_rng(0)
P = rng.standard_normal((1024, 16384)).astype(np.float32) / np.sqrt(1024)
reps["randproj"] = lambda: (np.maximum((pix_tr - 0.5) @ P, 0), np.maximum((pix_te - 0.5) @ P, 0))
reps["stim_only"] = lambda: (np.asarray(Fr_tr[:, driven], np.float32), np.asarray(Fr_te[:, driven], np.float32))
for g in GRAPHS:
    Ftr, gy, gi = load(g, "train")
    Fte, gyt, git = load(g, "test")
    assert (gi == itr).all() and (git == ite).all()
    for v in VIEWS:
        reps[f"{g}:{v}"] = (lambda Ftr=Ftr, Fte=Fte, v=v: (np.asarray(Ftr[:, M[v]], np.float32),
                                                            np.asarray(Fte[:, M[v]], np.float32)))

results, correct = {}, {}
for name, get in reps.items():
    t0 = time.time()
    Xtr, Xte = get()
    scores = probe.standardize_pca(Xtr, Xte, KS)
    for K, (Str, Ste, k) in scores.items():
        runs = [probe.fit_probe(Str, ytr, Ste, seed) for seed in SEEDS]
        accs = [float((r["pred"] == yte).mean()) for r in runs]
        correct[(name, K)] = np.mean([r["pred"] == yte for r in runs], 0)
        results[f"{name}|K={K}"] = {"rep": name, "K": K, "k_used": k, "dim": Xtr.shape[1], "acc_mean": float(np.mean(accs)),
                                    "acc_seeds": accs, "lambda": [r["lambda"] for r in runs],
                                    "cv_best": [max(r["cv"].values()) for r in runs]}
        print(f"{name:32s} K={K:5d} (k={k}) acc={np.mean(accs):.4f} {accs} lam={[r['lambda'] for r in runs]} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    del Xtr, Xte, scores

a, b = f"{PRIMARY['a']}:{PRIMARY['view']}", f"{PRIMARY['b']}:{PRIMARY['view']}"
primary = probe.paired_bootstrap(correct[(a, PRIMARY["K"])], correct[(b, PRIMARY["K"])])
secondary = {}
for v in VIEWS:
    for K in KS:
        secondary[f"{v}|K={K}"] = probe.paired_bootstrap(correct[(f"real:{v}", K)], correct[(f"global_shuffle_s0:{v}", K)])
for ref in ("stim_only", "pixels", "randproj"):
    secondary[f"real:central_vnc - {ref}|K=1024"] = probe.paired_bootstrap(correct[(a, 1024)], correct[(ref, 1024)])
decision = "STOP" if primary["diff"] <= 0 else "GO to P1-full"
report = {"subset": subset, "primary": PRIMARY | primary, "decision": decision, "secondary": secondary,
          "results": results, "chance": 0.1}
(OUT / "results.json").write_text(json.dumps(report, indent=1))
print(json.dumps({"primary": report["primary"], "decision": decision}, indent=1))
