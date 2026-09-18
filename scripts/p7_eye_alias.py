"""P7: is the CIFAR encoder failure an EYE problem (aliasing) rather than a circuit problem?

The retina point-samples: 3,241 R1-6 photoreceptors carry only 800 distinct viewing directions
(292 left / 508 right), of which 437 see the image at all - i.e. a 32x32 = 1,024 pixel image is
sampled at 0.43 samples per pixel with no low-pass filter. That is the textbook condition for
aliasing, and it happens BEFORE the connectome sees anything.

This script measures the eye alone, no brain: CIFAR luminance -> photoreceptor evoked response
(4-way drift, same as the frozen v1 image path) -> the same probe used everywhere else, with and
without a pre-sampling Gaussian blur. Reference points from P1: pixels 27.87, eye input 24.40.

If blur moves the eye readout toward pixels, aliasing was destroying the image before the circuit
had a chance. usage: p7_eye_alias.py [n_per_class] [blur_px ...]
       p7_eye_alias.py adapt [n_per_class]     -> sweep tau_adapt x drift_extent instead

Hypotheses tested here, in order:
  1. aliasing        -> REJECTED: blur recovers at most 1.0pp of the 3.4pp loss.
  2. resolution      -> REJECTED: pixels at 21x21 (441 px, the eye's sampling) score 30.60 vs
                        29.17 at 32x32, i.e. CIFAR-10 linear decoding needs no more than ~13x13.
  3. adaptation      -> the photoreceptor is a temporal high-pass (tau_adapt = 0.5 s) and the stimulus
                        lasts exactly 0.5 s, while the image is drifted by only 10% of its width. A
                        static photo may simply be washed out by design.
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
from flyvl import connectome, data, frozen, probe  # noqa: E402
from flyvl.extract import to_luma  # noqa: E402
from flyvl.stimulus import GRAY, Retina  # noqa: E402

MODE = "adapt" if len(sys.argv) > 1 and sys.argv[1] == "adapt" else "blur"
_args = sys.argv[2:] if MODE == "adapt" else sys.argv[1:]
PER_CLASS = int(_args[0]) if _args else 300
BLURS = [float(b) for b in _args[1:]] or [0.0, 0.5, 0.7, 1.0, 1.5]
# (tau_adapt seconds, drift_extent in image units; the image spans 2, so 0.2 = 10% of its width)
ADAPT_GRID = [(0.5, 0.2), (0.5, 0.6), (0.5, 1.2), (2.0, 0.2), (2.0, 0.6), (1e6, 0.2), (1e6, 0.6)]
STEPS, KS = 25, (1024,)
OUT = connectome.DATA_ROOT / "runs" / "p7"
OUT.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def eye_features(retina, imgs, dirs, batch=128):
    """Evoked photoreceptor response, averaged over the 4 drift directions: (B, n_driven)."""
    outs = []
    for s in range(0, len(imgs), batch):
        x = imgs[s:s + batch].cuda()
        B = x.shape[0]
        acc = 0
        for d in dirs:
            retina.reset(B + 1)
            pr = 0
            for k in range(STEPS):
                lum = retina.sample_images(x, k * 0.02, d)
                lum = torch.cat([lum, torch.full((lum.shape[0], 1), GRAY, device="cuda")], 1)
                e = retina.transduce(lum)
                pr = pr + (e[:, :-1] - e[:, -1:])
            acc = acc + pr / STEPS
        outs.append((acc / len(dirs)).T.float().cpu())
    return torch.cat(outs).numpy()


if __name__ == "__main__":
    c = connectome.load()
    cfg, eye_cfg, _ = frozen.load()
    r16 = c.types(["R1-6"])
    driven = r16[c.column[r16, 0] >= 0]
    Xtr, ytr = data.cifar10(True)
    Xte, yte = data.cifar10(False)
    itr = data.first_per_class(ytr, PER_CLASS)
    ite = data.first_per_class(yte, PER_CLASS // 3)
    imgs_tr, imgs_te = to_luma(Xtr[itr]), to_luma(Xte[ite])
    ltr, lte = ytr[itr], yte[ite]
    res = {}

    pix_tr = imgs_tr.reshape(len(itr), -1).numpy()
    pix_te = imgs_te.reshape(len(ite), -1).numpy()
    for name, (A, B) in {"pixels": (pix_tr, pix_te)}.items():
        sc = probe.standardize_pca(A, B, KS)
        for K, (Str, Ste, k) in sc.items():
            accs = [float((probe.fit_probe(Str, ltr, Ste, s)["pred"] == lte).mean()) for s in (0, 1, 2)]
            res[name] = {"acc": accs, "mean": float(np.mean(accs)), "k": k}
            print(f"{name:16s} mean {np.mean(accs)*100:.2f} {[round(a*100,1) for a in accs]}", flush=True)

    if MODE == "adapt":
        for tau, extent in ADAPT_GRID:
            t0 = time.time()
            ec = dataclasses.replace(eye_cfg, tau_adapt=tau, drift_extent=extent)
            retina = Retina(c, driven, ec, cfg.dt)
            A = eye_features(retina, imgs_tr, ec.drift_directions)
            B = eye_features(retina, imgs_te, ec.drift_directions)
            sc = probe.standardize_pca(A, B, KS)
            for K, (Str, Ste, k) in sc.items():
                accs = [float((probe.fit_probe(Str, ltr, Ste, s)["pred"] == lte).mean()) for s in (0, 1, 2)]
                res[f"tau{tau:g}_drift{extent:g}"] = {"acc": accs, "mean": float(np.mean(accs)), "k": k}
                print(f"tau {tau:<8g} drift {extent:<5g} mean {np.mean(accs)*100:.2f} "
                      f"{[round(a*100,1) for a in accs]} ({time.time()-t0:.0f}s)", flush=True)
            (OUT / f"eye_adapt_{PER_CLASS}.json").write_text(
                json.dumps({"per_class": PER_CLASS, "chance": 0.1, "res": res}, indent=1))
        sys.exit()

    for blur in BLURS:
        t0 = time.time()
        retina = Retina(c, driven, dataclasses.replace(eye_cfg, image_blur_px=blur), cfg.dt)
        A = eye_features(retina, imgs_tr, eye_cfg.drift_directions)
        B = eye_features(retina, imgs_te, eye_cfg.drift_directions)
        sc = probe.standardize_pca(A, B, KS)
        for K, (Str, Ste, k) in sc.items():
            accs = [float((probe.fit_probe(Str, ltr, Ste, s)["pred"] == lte).mean()) for s in (0, 1, 2)]
            res[f"eye_blur{blur:g}"] = {"acc": accs, "mean": float(np.mean(accs)), "k": k}
            print(f"eye blur {blur:<5g} mean {np.mean(accs)*100:.2f} {[round(a*100,1) for a in accs]} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        (OUT / f"eye_alias_{PER_CLASS}.json").write_text(
            json.dumps({"per_class": PER_CLASS, "n_train": len(itr), "n_test": len(ite),
                        "chance": 0.1, "res": res}, indent=1))
