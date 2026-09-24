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

usage: v3b_vlm.py [n] [warm]   warm = static adaptation steps before motion (V3d)
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

if sys.argv[1:2] != ["pixel_where16"]:
    sys.argv = [sys.argv[0], "pixel_where16"] + sys.argv[1:]
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


def letter_frames(p, t, device="cuda"):
    """V3c video plus a static letter in every cell."""
    global GLYPH
    GLYPH = glyphs() if GLYPH is None else GLYPH
    base = P.pixel_frames(p, t, device)                                     # (B, 1, PIX, PIX)
    let = torch.as_tensor(p["letters"], device=device)                      # (B, 16)
    B = len(let)
    ink = GLYPH[let].view(B, 4, 4, C, C).permute(0, 1, 3, 2, 4).reshape(B, 1, P.PIX, P.PIX)
    return base - L_CONTRAST * ink


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
    t0 = time.time()
    rng = np.random.default_rng(0)
    y, p = P.trial_params(rng, "pixel_where16", N)                          # same motion trials as V3a/V3c
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
    X = np.concatenate([glimpse_energy(c, cfg, sim, retina, p, q, ol, frames=letter_frames, warm=WARM) for q in range(4)], 1)
    print(f"  fly features {(time.time()-t0)/60:.1f} min", flush=True)

    # --- selector 2: pixel frame difference ---
    fd, prev = 0, None
    for k in range(P.STEPS):
        fr = torch.cat([F.avg_pool2d(letter_frames({kk: v[s:s + 500] for kk, v in p.items()}, k * cfg.dt), 2).flatten(1)
                        for s in range(0, N, 500)])
        if prev is not None:
            fd = fd + (fr - prev) ** 2
        prev = fr
    fd = (fd / P.STEPS).cpu().numpy()

    def select(feat, name):
        Pm = torch.randn(1024, feat.shape[1], generator=torch.Generator().manual_seed(P.view_seed(name))) / np.sqrt(1024)
        Z = (torch.as_tensor(feat, device="cuda") @ Pm.to("cuda").T).cpu().numpy()
        (Str, Ste, _), = probe.standardize_pca(Z[:n_tr], Z[n_tr:], (1024,)).values()
        return probe.fit_probe(Str, y[:n_tr], Ste, 0)["pred"]

    pick = {"fly": select(X, "v3b_fly"), "framediff": select(fd, "v3b_framediff"),
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
    res.update({"warm": WARM, "n": N, "n_test": len(te), "letter_contrast": L_CONTRAST, "minutes": (time.time() - t0) / 60})
    (OUT / f"v3b_vlm_warm{WARM}.json").write_text(json.dumps(res, indent=1))
    print(f"done in {res['minutes']:.1f} min", flush=True)
