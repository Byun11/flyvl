"""P7b: does driving the eye properly change the CIFAR encoder result?

P7 found that the eye's loss versus raw pixels is caused by DRIFT EXTENT, not by aliasing,
resolution or adaptation: with the frozen drift of 0.2 (10% of the image width) the eye scores
25.77 against 29.17 for pixels, and at drift 1.2 it reaches 28.00. The 437 fixed viewing directions
only cover more of the image if the image moves across them.

Every CIFAR and InternVL result so far used the frozen drift_extent = 0.2, i.e. an under-driven eye.
This runs the FULL pipeline (eye -> connectome -> pre-registered readout masks) at both drift values,
so the encoder conclusion can be re-checked on equal footing with pixels.

CONTROLS ARE REQUIRED HERE. "The fly beats pixels" is not evidence for the connectome: P1-mini already
showed visual_projection real 31.9 > pixels 30.5, but matched shuffle reached 32.7, global shuffle 38.7
and a plain random projection of the pixels 35.1 - i.e. any high-dimensional expansion helps a linear
probe. The only question this script can answer is whether driving the eye properly helps REAL wiring
MORE than it helps a shuffled one.

usage: p7_drift_brain.py [per_class] [drift ...] [--graphs=a,b]
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, data, frozen, masks, probe  # noqa: E402
from flyvl.extract import load_graph, to_luma  # noqa: E402
from flyvl.sim import Sim  # noqa: E402
from flyvl.stimulus import GRAY, Retina  # noqa: E402

GRAPHS = ["real", "matched_shuffle_s0"]
_g = [a for a in sys.argv if a.startswith("--graphs=")]
if _g:                                     # strip flags BEFORE parsing positional args
    GRAPHS = _g[0].split("=")[1].split(",")
    sys.argv.remove(_g[0])
PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
DRIFTS = [float(d) for d in sys.argv[2:]] or [0.2, 1.2]
STEPS, KS = 25, (1024,)
OUT = connectome.DATA_ROOT / "runs" / "p7"
OUT.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def brain_features(c, cfg, sim, retina, imgs, dirs, views, batch=32):
    """Evoked activity averaged over the drift directions, per readout view."""
    out = {k: [] for k in list(views) + ["photoreceptor"]}
    for s in range(0, len(imgs), batch):
        x = imgs[s:s + batch].cuda()
        B = x.shape[0]
        acc_v, acc_pr = None, 0
        for d in dirs:
            retina.reset(B + 1)
            st = sim.zero_state(B + 1)
            acc, pr = None, 0
            for k in range(STEPS):
                lum = retina.sample_images(x, k * cfg.dt, d)
                lum = torch.cat([lum, torch.full((lum.shape[0], 1), GRAY, device="cuda")], 1)
                eye = retina.transduce(lum)
                pr = pr + (eye[:, :-1] - eye[:, -1:])
                st = sim.step(st, eye)
                full = torch.zeros(c.n, B + 1, device="cuda")
                full[torch.as_tensor(sim.gi, device="cuda")] = st.x
                full[torch.as_tensor(sim.li, device="cuda")] = st.s
                acc = full if acc is None else acc + full
            ev = (acc[:, :-1] - acc[:, -1:]) / STEPS
            acc_v = ev if acc_v is None else acc_v + ev
            acc_pr = acc_pr + pr / STEPS
        for name, idx in views.items():
            out[name].append((acc_v[torch.as_tensor(idx, device="cuda")] / len(dirs)).T.float().cpu())
        out["photoreceptor"].append((acc_pr / len(dirs)).T.float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


if __name__ == "__main__":
    c = connectome.load()
    cfg, eye_cfg, _ = frozen.load()
    M = masks.build(c)
    views = {"central_vnc": M["central_vnc"], "visual_projection": M["visual_projection"]}
    r16 = c.types(["R1-6"])
    driven = r16[c.column[r16, 0] >= 0]
    Xtr, ytr = data.cifar10(True)
    Xte, yte = data.cifar10(False)
    itr = data.first_per_class(ytr, PER_CLASS)
    ite = data.first_per_class(yte, max(PER_CLASS // 3, 20))
    imgs_tr, imgs_te = to_luma(Xtr[itr]), to_luma(Xte[ite])
    ltr, lte = ytr[itr], yte[ite]
    res = {}

    a = imgs_tr.reshape(len(itr), -1).numpy()
    b = imgs_te.reshape(len(ite), -1).numpy()
    (Str, Ste, k), = probe.standardize_pca(a, b, KS).values()
    accs = [float((probe.fit_probe(Str, ltr, Ste, s)["pred"] == lte).mean()) for s in (0, 1, 2)]
    res["pixels"] = {"acc": accs, "mean": float(np.mean(accs))}
    print(f"pixels                        mean {np.mean(accs)*100:.2f}", flush=True)

    # random-projection control: pixels expanded to the same width as a brain view, no brain at all
    rp = torch.randn(9201, a.shape[1], generator=torch.Generator().manual_seed(0)) / np.sqrt(9201)
    ra, rb = (a @ rp.T.numpy()).clip(0, None), (b @ rp.T.numpy()).clip(0, None)
    (Str, Ste, k), = probe.standardize_pca(ra, rb, KS).values()
    accs = [float((probe.fit_probe(Str, ltr, Ste, s)["pred"] == lte).mean()) for s in (0, 1, 2)]
    res["randproj"] = {"acc": accs, "mean": float(np.mean(accs))}
    print(f"randproj (no brain)           mean {np.mean(accs)*100:.2f}", flush=True)

    for graph in GRAPHS:
        sim = Sim(c, cfg, W=load_graph(c, graph, cfg), driven=driven)
        for drift in DRIFTS:
            t0 = time.time()
            ec = dataclasses.replace(eye_cfg, drift_extent=drift)
            retina = Retina(c, driven, ec, cfg.dt)
            F_tr = brain_features(c, cfg, sim, retina, imgs_tr, ec.drift_directions, views)
            F_te = brain_features(c, cfg, sim, retina, imgs_te, ec.drift_directions, views)
            for view in F_tr:
                (Str, Ste, k), = probe.standardize_pca(F_tr[view], F_te[view], KS).values()
                accs = [float((probe.fit_probe(Str, ltr, Ste, s)["pred"] == lte).mean()) for s in (0, 1, 2)]
                res[f"{graph}:drift{drift:g}:{view}"] = {"acc": accs, "mean": float(np.mean(accs)), "k": k}
                print(f"{graph:20s} drift {drift:<5g} {view:20s} mean {np.mean(accs)*100:.2f} "
                      f"{[round(x*100,1) for x in accs]}", flush=True)
            print(f"  ({graph} drift {drift:g}: {time.time()-t0:.0f}s)", flush=True)
            path = OUT / f"drift_brain_{PER_CLASS}.json"          # merge: runs cover different graphs
            merged = json.loads(path.read_text())["res"] if path.exists() else {}
            merged.update(res)
            path.write_text(json.dumps({"per_class": PER_CLASS, "chance": 0.1, "res": merged}, indent=1))
