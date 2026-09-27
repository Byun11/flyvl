"""V3b: the full pipeline for the first time - video -> fly picks where the motion is -> only that cell
goes to the VLM, which reads it.

Video (128 x 128, 4 x 4 cells of 32 px, 25 frames): every cell holds a static letter (contrast 0.3,
readable), the V3c low-contrast grating covers the page and drifts inside ONE cell only, and fresh pixel
noise is added every frame. Static letters cancel in frame differences and adapt away in the eye, so
finding the cell is exactly V3c's problem. Question to InternVL3-1B: which letter is in the given
image? The right answer is the letter of the moving cell.

Selectors (each hands the VLM ONE 32 x 32 cell of the last frame, 1/16 of the page):
  fly       - V3c: 4 quadrant glimpses, optic-lobe response energy, linear probe
  framediff - pixel frame-difference energy, linear probe
  random    - a random cell
  oracle    - the true moving cell (reading upper bound)
Reference: the whole last frame (one still image cannot show which part moves).

usage: v3b_vlm.py [n] [warm] [change]   warm = static adaptation steps (V3d, hurt); change = V3e readout
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import os
TASK = os.environ.get("V3_TASK", "pixel_where16")   # pixel_dir16 = S1 (which cell moves LEFT)
if sys.argv[1:2] != [TASK]:
    sys.argv = [sys.argv[0], TASK] + sys.argv[1:]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5_flytask as P  # noqa: E402
from v3c_glimpse import glimpse_energy  # noqa: E402
from flyvl import connectome, frozen, probe, teacher  # noqa: E402
from flyvl.sim import Sim  # noqa: E402
from flyvl.stimulus import GRAY, Retina  # noqa: E402

LETTERS = [chr(ord("A") + i) for i in range(26)]
C = P.PIX // 4                               # cell size in pixels
L_CONTRAST = 0.3
OUT = connectome.DATA_ROOT / "runs" / "video"


def glyphs():
    """(26, C, C) float masks of bold capitals (rendered once)."""
    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 24)
    out = []
    for ch in LETTERS:
        im = Image.new("L", (C, C), 0)
        d = ImageDraw.Draw(im)
        l, t, r, b = d.textbbox((0, 0), ch, font=font)
        d.text(((C - (r - l)) / 2 - l, (C - (b - t)) / 2 - t), ch, fill=255, font=font)
        out.append(np.asarray(im, np.float32) / 255)
    return torch.as_tensor(np.stack(out), device="cuda")


GLYPH = None


PHOTON_FLUX = float(os.environ.get("PHOTON_FLUX", "0"))   # E2: mean photons per pixel per frame at luminance 1 (0 = off)


def letter_frames(p, t, device="cuda"):
    """V3c video plus a static letter in every cell. With PHOTON_FLUX > 0 each frame is a Poisson photon
    count (low-light camera): lum -> Poisson(lum * flux) / flux, seeded per frame like the other noise."""
    global GLYPH
    GLYPH = glyphs() if GLYPH is None else GLYPH
    base = P.pixel_frames(p, t, device)                                     # (B, 1, PIX, PIX)
    let = torch.as_tensor(p["letters"], device=device)                      # (B, 16)
    B = len(let)
    ink = GLYPH[let].view(B, 4, 4, C, C).permute(0, 1, 3, 2, 4).reshape(B, 1, P.PIX, P.PIX)
    lum = base - L_CONTRAST * ink
    if PHOTON_FLUX > 0:
        gen = torch.Generator(device=device).manual_seed(P.NOISE_SEED + 5000 + int(round(t / 0.02)))
        lum = torch.poisson(lum.clamp(min=0) * PHOTON_FLUX, generator=gen) / PHOTON_FLUX
    return lum


@torch.no_grad()
def rf_centers(c, sim, retina, driven, hops=8):
    """Receptive-field centre (x, y in image coords) of every graded (optic-lobe) unit, estimated by
    propagating photoreceptor view directions through |W|: each unit takes the input-weighted mean
    position of its presynaptic partners that already have one. Only ~8% of optic-lobe units have a
    column in the MaleCNS table (L1, R7, R8), so T4/T5 etc. need this estimate."""
    n = c.n
    A = abs(c.W).tocsr()
    A = torch.sparse_csr_tensor(torch.as_tensor(A.indptr, dtype=torch.int64), torch.as_tensor(A.indices, dtype=torch.int64),
                                torch.as_tensor(A.data, dtype=torch.float32), size=A.shape).to("cuda")
    pos = torch.zeros(n, 2, device="cuda")
    known = torch.zeros(n, 1, device="cuda")
    d = torch.as_tensor(driven, device="cuda")
    pos[d, 0], pos[d, 1] = retina.phi, -retina.theta          # image x = phi (half width 1), y = -theta
    known[d] = 1
    for _ in range(hops):
        num, den = A @ (pos * known), A @ known
        new = (den[:, 0] > 0) & (known[:, 0] == 0)
        pos[new] = num[new] / den[new]
        known[new] = 1
    g = torch.as_tensor(sim.gi, device="cuda")
    return pos[g].cpu().numpy(), known[g, 0].bool().cpu().numpy()


def pool_cells(X, rf, has):
    """X (N, 4*Ng) glimpse-major optic-lobe energies -> (N, 16) mean energy per page cell."""
    Ng = len(rf)
    cx = np.clip(((rf[:, 0] + 1) / 2 * 2).astype(int), 0, 1)
    cy = np.clip(((rf[:, 1] + 1) / 2 * 2).astype(int), 0, 1)
    out = np.zeros((len(X), 16), np.float32)
    for q in range(4):
        qi, qj = divmod(q, 2)
        cell = (2 * qi + cy) * 4 + (2 * qj + cx)
        E = X[:, q * Ng:(q + 1) * Ng]
        for k in range(16):
            m = has & (cell == k)
            if m.any():
                out[:, k] += E[:, m].mean(1)
    return out


def cell_crop(frame, cell):
    """(B, 1, PIX, PIX), (B,) -> (B, 1, C, C)"""
    r, c = cell // 4, cell % 4
    return torch.stack([frame[b, :, r[b] * C:(r[b] + 1) * C, c[b] * C:(c[b] + 1) * C] for b in range(len(cell))])


@torch.no_grad()
def ask(proc, model, imgs, batch=32):
    """imgs (B, 1, h, w) luminance -> predicted letter per image ('' if none)."""
    msgs = [{"role": "user", "content": [{"type": "image"},
            {"type": "text", "text": "Which single capital letter is shown in this image? Answer with the letter only."}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    out = []
    for s in range(0, len(imgs), batch):
        x = F.interpolate(imgs[s:s + batch].clamp(0, 1), size=448, mode="bicubic").clamp(0, 1)
        pil = [Image.fromarray((im[0].cpu().numpy() * 255).astype(np.uint8)).convert("RGB") for im in x]
        inp = proc(images=pil, text=[prompt] * len(pil), return_tensors="pt", padding=True).to("cuda", torch.bfloat16)
        gen = model.generate(**inp, max_new_tokens=3, do_sample=False)
        txt = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        out += [next((ch for ch in t.strip().upper() if ch in LETTERS), "") for t in txt]
    return out


if __name__ == "__main__":
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    WARM = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    CHANGE = len(sys.argv) > 4 and sys.argv[4] == "change"
    t0 = time.time()
    rng = np.random.default_rng(0)
    y, p = P.trial_params(rng, TASK, N)                          # same motion trials as V3a/V3c
    p["letters"] = np.random.default_rng(1).integers(0, 26, (N, 16))
    n_tr = int(N * 0.7)
    te = np.arange(n_tr, N)
    res = {}

    # --- selector 1: the fly (V3c) ---
    c = connectome.load()
    cfg, eye_cfg, _ = frozen.load()
    eye_cfg = dataclasses.replace(eye_cfg, image_half_width=1.0, drift_extent=0.0)
    r16 = c.types(["R1-6"])
    driven = r16[c.column[r16, 0] >= 0]
    sim, retina = Sim(c, cfg, driven=driven), Retina(c, driven, eye_cfg, cfg.dt)
    ol = torch.as_tensor(np.arange(sim.Ng), device="cuda")
    X = np.concatenate([glimpse_energy(c, cfg, sim, retina, p, q, ol, frames=letter_frames, warm=WARM, change=CHANGE) for q in range(4)], 1)
    print(f"  fly features {(time.time()-t0)/60:.1f} min", flush=True)

    # --- selector 2: pixel frame difference, plain and spatially blurred (V3f: stronger free baselines) ---
    from flyvl.stimulus import _gaussian_blur
    BLURS = (0, 1, 2, 4)
    fds = {}
    for blur in BLURS:
        fd, prev = 0, None
        for k in range(P.STEPS):
            parts = []
            for s in range(0, N, 500):
                fr = letter_frames({kk: v[s:s + 500] for kk, v in p.items()}, k * cfg.dt)
                fr = _gaussian_blur(fr, blur) if blur else fr
                parts.append(F.avg_pool2d(fr, 2).flatten(1))
            fr = torch.cat(parts)
            if prev is not None:
                fd = fd + (fr - prev) ** 2
            prev = fr
        fds[blur] = (fd / P.STEPS).cpu().numpy()

    def select(feat, name):
        Pm = torch.randn(1024, feat.shape[1], generator=torch.Generator().manual_seed(P.view_seed(name))) / np.sqrt(1024)
        Z = (torch.as_tensor(feat, device="cuda") @ Pm.to("cuda").T).cpu().numpy()
        (Str, Ste, _), = probe.standardize_pca(Z[:n_tr], Z[n_tr:], (1024,)).values()
        return probe.fit_probe(Str, y[:n_tr], Ste, 0)["pred"]

    rf, has = rf_centers(c, sim, retina, driven)
    pooled = pool_cells(X, rf, has)
    print(f"  rf estimated for {has.mean()*100:.0f}% of optic-lobe units", flush=True)
    pz = (pooled - pooled[:n_tr].mean(0)) / pooled[:n_tr].std(0)
    (Str, Ste, _), = probe.standardize_pca(pooled[:n_tr], pooled[n_tr:], (16,)).values()
    # --- selector 3 (S1): direction-selective classical baseline = Hassenstein-Reichardt array on blurred
    # pixels (equivalent to Adelson-Bergen motion energy; van Santen & Sperling 1985). Signed: + right, - left.
    hr, prev, D = 0, None, 2
    for k in range(P.STEPS):
        fr = torch.cat([_gaussian_blur(letter_frames({kk: v[s:s + 500] for kk, v in p.items()}, k * cfg.dt), 2)
                        for s in range(0, N, 500)])[:, 0]
        if prev is not None:
            hr = hr + prev[..., :-D] * fr[..., D:] - prev[..., D:] * fr[..., :-D]
        prev = fr
    hr = F.avg_pool2d((hr / P.STEPS)[:, None], 2).flatten(1).cpu().numpy()

    pick = {"fly_pooled_argmax": pz[n_tr:].argmax(1),                      # training-free: the loudest cell
            "fly_pooled": probe.fit_probe(Str, y[:n_tr], Ste, 0)["pred"],
            "fly": select(X, "v3b_fly"), "framediff": select(fds[0], "v3b_framediff"),
            **{f"framediff_blur{b}": select(fds[b], "v3b_framediff") for b in BLURS if b},
            "hr_pixels_blur2": select(hr, "v3b_hr"),
            "random": np.random.default_rng(2).integers(0, 16, len(te)), "oracle": y[te]}
    for k, v in pick.items():
        res[f"locate_{k}"] = float((v == y[te]).mean())
        print(f"locate  {k:10s} {res[f'locate_{k}']*100:.2f}", flush=True)

    # --- the VLM reads what each selector handed it ---
    proc, model = teacher.load()
    pt = {k: v[te] for k, v in p.items()}
    last = letter_frames(pt, (P.STEPS - 1) * cfg.dt)
    answer = [LETTERS[i] for i in p["letters"][te, y[te]]]
    for k, v in pick.items():
        pred = ask(proc, model, cell_crop(last, torch.as_tensor(v, device="cuda")))
        res[f"vqa_{k}"] = float(np.mean([a == b for a, b in zip(pred, answer)]))
        print(f"VQA     {k:10s} {res[f'vqa_{k}']*100:.2f}", flush=True)
    pred = ask(proc, model, last)
    res["vqa_whole_frame"] = float(np.mean([a == b for a, b in zip(pred, answer)]))
    print(f"VQA     whole frame {res['vqa_whole_frame']*100:.2f}", flush=True)
    import os
    res.update({"mw_contrast": os.environ.get("MW_CONTRAST", "0.04,0.10"), "mw_noise": os.environ.get("MW_NOISE", "0.06"),
                "warm": WARM, "change": CHANGE, "n": N, "n_test": len(te), "letter_contrast": L_CONTRAST, "minutes": (time.time() - t0) / 60})
    (OUT / f"{TASK}_warm{WARM}{'_change' if CHANGE else ''}_c{os.environ.get('MW_CONTRAST', '0.04,0.10')}_n{os.environ.get('MW_NOISE', '0.06')}.json").write_text(json.dumps(res, indent=1))
    print(f"done in {res['minutes']:.1f} min", flush=True)
