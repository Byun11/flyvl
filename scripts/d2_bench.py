"""D2: can a swarm of flyvis flies stand in for InternVL3-1B's vision encoder on real VLM benchmarks?

Controlled setup (every arm gets the same thing): InternVL3-1B frozen end to end (ViT, MLP connector, LLM); one
448x448 tile -> the 256 visual-token slots (16x16 grid, one token = a 28x28 px block); the same prompt and yes/no
scoring; the same training images, the same per-token linear adapter, the same seed. Only the 256 tokens differ:
  vit        original tokens (ceiling)            vit_gray   original ViT on the grey image (cost of losing colour)
  fly        256 flyvis flies, one per token block: 0.5 s grey steady state, then the block under random-walk
             fixational jitter (0.5 px/frame, +-3 px) for VIEW s; response minus grey steady state, per cell type x
             3x3 eye region x 2 time bins -> linear adapter
  pix        the SAME jittered grey frames, 2 time bins x 28x28 px -> the same linear adapter (the key control)
  mean       mean training token (floor)
Heads (D2_HEAD, same for every arm): linear = per-token linear map, each token sees only its own fly/block;
mixer = all 256 tokens mixed by an MLP-Mixer (token-mixing across the grid + channel MLP), linear head as skip.
Adapter training: Flickr30k (no COCO, so no overlap with POPE's COCO val2014 images), 5000 train / 500 val.
Benchmarks: POPE (9000 yes/no, 500 images), MME (yes/no).
usage: d2_bench.py prep | teacher | fly | merge | align ARM | eval ARM
multi-GPU: D2_SHARD=k/m d2_bench.py fly on machine k, copy the .part.s{k}of{m}.npy files together, then
D2_SHARD=0/m d2_bench.py merge. D2_FLYB=256 fits an 80 GB A100.
"""
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome  # noqa: E402

BENCH = connectome.DATA_ROOT / "datasets" / "bench"
D2 = connectome.DATA_ROOT / "d2"
D2.mkdir(parents=True, exist_ok=True)
SIZE, TILE, NTOK, NTRAIN, NVAL = 448, 28, 256, 5000, 500
VIEW = float(os.environ.get("D2_VIEW", "1.0"))
DT, FLYB = 0.02, int(os.environ.get("D2_FLYB", "32"))          # 64 flies ~11 GB of GPU memory
STEPS = int(round(VIEW / DT))
SHARD = os.environ.get("D2_SHARD", "")                                  # "k/m": this machine does the k-th of m image ranges
SUF = f".s{SHARD.replace('/', 'of')}" if SHARD else ""
TAG = f"v{VIEW:g}"
HEAD = os.environ.get("D2_HEAD", "linear")                              # linear | mixer (align/eval)
HSUF = "" if HEAD == "linear" else f"_{HEAD}"
SETS = ("train", "val", "pope", "mme")
PROMPT = "\nAnswer the question using a single word or phrase."
dev = "cuda"


def _img(b):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC))


def prep():
    import pyarrow.parquet as pq
    z = zipfile.ZipFile(BENCH / "flickr30k" / "flickr30k-images.zip")
    names = sorted(n for n in z.namelist() if n.endswith(".jpg") and not n.startswith("__MACOSX"))[:NTRAIN + NVAL]
    for name, sl in (("train", slice(0, NTRAIN)), ("val", slice(NTRAIN, NTRAIN + NVAL))):
        np.save(D2 / f"{name}_img.npy", np.stack([_img(z.read(n)) for n in names[sl]]))
    # POPE: three splits share images; keep one image per image_source
    qs, imgs, key = [], [], {}
    for f in sorted((BENCH / "POPE" / "Full").glob("*.parquet")):
        for r in pq.read_table(f).to_pylist():
            if r["image_source"] not in key:
                key[r["image_source"]] = len(imgs)
                imgs.append(_img(r["image"]["bytes"]))
            qs.append({"q": r["question"], "a": r["answer"].strip().lower(), "img": key[r["image_source"]],
                       "cat": r["category"]})
    np.save(D2 / "pope_img.npy", np.stack(imgs))
    (D2 / "pope_q.json").write_text(json.dumps(qs))
    qs, imgs, key = [], [], {}
    for f in sorted((BENCH / "MME" / "data").glob("*.parquet")):
        for r in pq.read_table(f).to_pylist():
            k = (r["category"], r["question_id"])
            if k not in key:
                key[k] = len(imgs)
                imgs.append(_img(r["image"]["bytes"]))
            qs.append({"q": r["question"], "a": r["answer"].strip().lower(), "img": key[k], "cat": r["category"]})
    np.save(D2 / "mme_img.npy", np.stack(imgs))
    (D2 / "mme_q.json").write_text(json.dumps(qs))
    for s in SETS:
        print(s, np.load(D2 / f"{s}_img.npy", mmap_mode="r").shape, flush=True)


def load_vlm():
    from flyvl import teacher
    return teacher.load()


def teacher_cmd():
    from flyvl import teacher
    proc, model = load_vlm()
    for s in SETS:
        X = np.load(D2 / f"{s}_img.npy", mmap_mode="r")
        torch.save(teacher.teacher_tokens(model, np.asarray(X), batch=16), D2 / f"{s}_vit.pt")
        if s in ("pope", "mme"):
            g = (np.asarray(X).astype(np.float32) @ np.array([0.299, 0.587, 0.114])).round().clip(0, 255).astype(np.uint8)
            torch.save(teacher.teacher_tokens(model, np.repeat(g[..., None], 3, -1), batch=16), D2 / f"{s}_vit_gray.pt")
        print(s, "teacher done", flush=True)


def fly_cmd():
    from flyvl.flyvis_encoder import FlyvisEncoder, hexal_xy
    enc = FlyvisEncoder()
    net = enc.net
    types = np.asarray([t.decode() if isinstance(t, bytes) else t for t in net.connectome.nodes.type[:]])
    utypes = sorted(set(types))
    xy = hexal_xy(enc, TILE)
    reg = np.minimum((xy[:, 1] * 3).astype(int), 2) * 3 + np.minimum((xy[:, 0] * 3).astype(int), 2)
    # (cells, types*9) pooling matrix: 721-column types by eye region, other types averaged
    Pm = np.zeros((len(types), len(utypes) * 9), np.float32)
    for k, t in enumerate(utypes):
        cells = np.flatnonzero(types == t)
        if len(cells) == 721:
            for r in range(9):
                m = cells[reg == r]
                Pm[m, k * 9 + r] = 1 / max(len(m), 1)
        else:
            Pm[cells, k * 9:(k + 1) * 9] = 1 / len(cells)
    Pm = torch.as_tensor(Pm, device=dev)
    DF, DP = Pm.shape[1] * 2, 2 * TILE * TILE
    with torch.no_grad(), torch.device(dev):
        state0 = net.steady_state(t_pre=0.5, dt=DT, batch_size=FLYB, value=0.5)
    base = state0.nodes.activity.clone()
    off = torch.arange(TILE, device=dev, dtype=torch.float32) - (TILE - 1) / 2
    gy, gx = torch.meshgrid(off, off, indexing="ij")
    cen = torch.arange(16, device=dev, dtype=torch.float32) * TILE + TILE / 2 - 0.5
    cy, cx = torch.meshgrid(cen, cen, indexing="ij")
    centers = torch.stack([cx.flatten(), cy.flatten()], 1)                                # (256, 2) row-major
    for s in SETS:
        pf = D2 / f"{s}_fly_{TAG}.npy"
        if pf.exists():
            continue
        X = np.load(D2 / f"{s}_img.npy", mmap_mode="r")
        lo, hi = shard_range(len(X))
        part = D2 / f"{s}_fly_{TAG}.part{SUF}.npy"
        mode = "r+" if part.exists() else "w+"                                            # resume a killed run
        fo = np.lib.format.open_memmap(part, mode, np.float16, (len(X), NTOK, DF))
        po = np.lib.format.open_memmap(D2 / f"{s}_pix_{TAG}.part{SUF}.npy", mode, np.float16, (len(X), NTOK, DP))
        start = lo
        while mode == "r+" and start < hi and np.any(fo[start, -1, :8]):
            start += 1
        start = max(start - 1, lo)                                                        # redo the last, maybe partial, image
        t0 = time.time()
        for i in range(start, hi):
            img = torch.as_tensor(np.asarray(X[i]), device=dev).float().div(255) @ torch.tensor([0.299, 0.587, 0.114], device=dev)
            gen = torch.Generator().manual_seed(SETS.index(s) * 10**6 + i)                   # per-image jitter, resumable
            walk = torch.randn(NTOK, STEPS, 2, generator=gen).mul(0.5).cumsum(1).clamp(-3, 3).to(dev)
            for b in range(0, NTOK, FLYB):
                c = centers[b:b + FLYB]
                with torch.no_grad(), torch.device(dev):
                    state = net.steady_state(t_pre=0.5, dt=DT, batch_size=FLYB, value=0.5)
                    acc = torch.zeros(2, FLYB, len(types), device=dev)
                    pacc = torch.zeros(2, FLYB, TILE, TILE, device=dev)
                    for k in range(STEPS):
                        pos = c[:, None, None] + walk[b:b + FLYB, k][:, None, None] + torch.stack([gx, gy], -1)
                        fr = F.grid_sample(img[None, None].expand(FLYB, 1, SIZE, SIZE), pos / (SIZE - 1) * 2 - 1,
                                           mode="bilinear", padding_mode="border", align_corners=True)[:, 0]
                        net.stimulus.zero(FLYB, 1)
                        net.stimulus.add_input(enc.eye(fr[:, None]))
                        state = net(net.stimulus(), DT, state, as_states=True)[-1]
                        h = k * 2 // STEPS
                        acc[h] += state.nodes.activity - base
                        pacc[h] += fr
                    acc /= STEPS // 2
                    pacc /= STEPS // 2
                fo[i, b:b + FLYB] = (acc @ Pm).permute(1, 0, 2).reshape(FLYB, -1).half().cpu().numpy()
                po[i, b:b + FLYB] = pacc.permute(1, 0, 2, 3).reshape(FLYB, -1).half().cpu().numpy()
            if i % 50 == 0:
                print(s, i, len(X), f"{(time.time() - t0) / (i - start + 1):.1f} s/img", flush=True)
        fo.flush(), po.flush()
        del fo, po
        if SHARD:
            print(s, f"shard {SHARD} done", f"{time.time() - t0:.0f}s", flush=True)
            continue
        for a in ("fly", "pix"):
            os.replace(D2 / f"{s}_{a}_{TAG}.part.npy", D2 / f"{s}_{a}_{TAG}.npy")
        print(s, "fly done", f"{time.time() - t0:.0f}s", flush=True)


def shard_range(n):
    if not SHARD:
        return 0, n
    k, m = map(int, SHARD.split("/"))
    return n * k // m, n * (k + 1) // m


def merge_cmd():
    """Combine shard files (copied into D2 from every machine) into the final arrays."""
    m = int(SHARD.split("/")[1])
    for s in SETS:
        n = len(np.load(D2 / f"{s}_img.npy", mmap_mode="r"))
        for a in ("fly", "pix"):
            parts = [np.load(D2 / f"{s}_{a}_{TAG}.part.s{k}of{m}.npy", mmap_mode="r") for k in range(m)]
            out = np.lib.format.open_memmap(D2 / f"{s}_{a}_{TAG}.npy", "w+", np.float16, parts[0].shape)
            for k, p in enumerate(parts):
                lo, hi = n * k // m, n * (k + 1) // m
                assert np.any(p[hi - 1, -1, :8]), f"{s} {a} shard {k} incomplete"
                out[lo:hi] = p[lo:hi]
            out.flush()
            print(s, a, "merged", out.shape, flush=True)


class Linear(nn.Module):
    """Per-token linear map shared over the 256 positions, plus a learned per-position bias."""

    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.lin = nn.Linear(d, 896)
        self.pos = nn.Parameter(torch.zeros(NTOK, 896))

    def forward(self, x):
        return self.lin(self.norm(x)) + self.pos


class Mixer(nn.Module):
    """All 256 tokens in one shared space without shrinking it: per-token projection to 896 + learned position
    embedding, then MLP-Mixer blocks (token-mixing MLP across the 16x16 grid, channel-mixing MLP per token).
    The Linear head is kept as a skip path and the mixer output starts at zero, so training starts from Linear."""

    def __init__(self, d, w=896, depth=2, tok_h=512, ch_h=1792, drop=0.1):
        super().__init__()
        self.skip = Linear(d)
        self.inp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, w))
        self.pos = nn.Parameter(torch.randn(NTOK, w) * 0.02)
        self.blocks = nn.ModuleList(nn.ModuleList([
            nn.LayerNorm(w), nn.Sequential(nn.Linear(NTOK, tok_h), nn.GELU(), nn.Dropout(drop), nn.Linear(tok_h, NTOK)),
            nn.LayerNorm(w), nn.Sequential(nn.Linear(w, ch_h), nn.GELU(), nn.Dropout(drop), nn.Linear(ch_h, w))])
            for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 896))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, x):
        h = self.inp(x) + self.pos
        for n1, tm, n2, cm in self.blocks:
            h = h + tm(n1(h).transpose(1, 2)).transpose(1, 2)
            h = h + cm(n2(h))
        return self.skip(x) + self.out(h)


HEADS = {"linear": Linear, "mixer": Mixer}


def align_cmd(arm, seed=0):
    T = {s: torch.load(D2 / f"{s}_vit.pt").float() for s in ("train", "val")}
    mu, sd = T["train"].mean((0, 1)), T["train"].std((0, 1)) + 1e-6
    out = {}
    if arm == "mean":
        for s in ("pope", "mme"):
            n = len(np.load(D2 / f"{s}_img.npy", mmap_mode="r"))
            out[s] = T["train"].mean(0, keepdim=True).expand(n, -1, -1).half()
        torch.save(out, D2 / f"pred_{arm}.pt")
        return
    X = {s: torch.as_tensor(np.load(D2 / f"{s}_{arm}_{TAG}.npy")) for s in SETS}
    xm = X["train"].float().mean((0, 1))
    xs = X["train"].float().std((0, 1)) + 1e-6
    torch.manual_seed(seed)
    model = HEADS[HEAD](X["train"].shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    EP, BS = 40, 32
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    prep_x = lambda x: ((x.float() - xm) / xs).to(dev)
    zt = lambda t: ((t - mu) / sd).to(dev)
    loss_fn = lambda p, t: F.mse_loss(p, t) + (1 - F.cosine_similarity(p, t, dim=-1)).mean()
    best, best_state = 1e9, None
    for ep in range(EP):
        model.train()
        for b in torch.randperm(NTRAIN).split(BS):
            loss = loss_fn(model(prep_x(X["train"][b])), zt(T["train"][b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([float(loss_fn(model(prep_x(X["val"][b])), zt(T["val"][b]))) for b in torch.arange(NVAL).split(100)])
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(arm, ep, f"val {vl:.4f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    res = {"arm": arm, "view": VIEW, "head": HEAD, "params": sum(p.numel() for p in model.parameters()),
           "best_epoch": best_ep, "val_loss": best}
    torch.save(best_state, D2 / f"model_{arm}_{TAG}{HSUF}.pt")
    with torch.no_grad():
        for s in ("val", "pope", "mme"):
            p = torch.cat([model(prep_x(X[s][b])).cpu() for b in torch.arange(len(X[s])).split(100)]) * sd + mu
            if s == "val":
                res["val_token_cos"] = float(F.cosine_similarity(p, T["val"], dim=-1).mean())
            else:
                out[s] = p.half()
    torch.save(out, D2 / f"pred_{arm}_{TAG}{HSUF}.pt")
    print("ALIGN", json.dumps(res), flush=True)


@torch.no_grad()
def eval_cmd(arm):
    proc, model = load_vlm()
    tok = proc.tokenizer
    tok.padding_side = "left"
    img_id = tok.convert_tokens_to_ids(proc.image_token)
    yes = [tok.convert_tokens_to_ids(t) for t in ("Yes", "yes")]
    no = [tok.convert_tokens_to_ids(t) for t in ("No", "no")]
    if arm in ("vit", "vit_gray"):
        preds = {s: torch.load(D2 / f"{s}_{arm}.pt") for s in ("pope", "mme")}
        name = arm
    else:
        preds = torch.load(D2 / (f"pred_{arm}.pt" if arm == "mean" else f"pred_{arm}_{TAG}{HSUF}.pt"))
        name = arm if arm == "mean" else f"{arm}_{TAG}{HSUF}"
    emb = model.get_input_embeddings()
    res = {"arm": name}
    for s in ("pope", "mme"):
        qs = json.loads((D2 / f"{s}_q.json").read_text())
        texts = []
        for q in qs:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q["q"] + PROMPT}]}]
            texts.append(proc.apply_chat_template(msgs, add_generation_prompt=True).replace(proc.image_token, proc.image_token * NTOK))
        said = []
        for b in range(0, len(qs), 16):
            enc = tok(texts[b:b + 16], add_special_tokens=False, return_tensors="pt", padding=True).to(dev)
            e = emb(enc.input_ids)
            vis = torch.stack([preds[s][q["img"]] for q in qs[b:b + 16]]).to(dev, e.dtype)             # (B, 256, 896)
            e[enc.input_ids == img_id] = vis.reshape(-1, vis.shape[-1])
            lg = model(inputs_embeds=e, attention_mask=enc.attention_mask, logits_to_keep=1).logits[:, -1].float()
            said += (lg[:, yes].max(1).values > lg[:, no].max(1).values).tolist()
        said = np.array(said)
        gold = np.array([q["a"] == "yes" for q in qs])
        ok = said == gold
        r = {"acc": float(ok.mean()), "yes_rate": float(said.mean()), "n": len(qs)}
        for c in sorted({q["cat"] for q in qs}):
            m = np.array([q["cat"] == c for q in qs])
            r[f"acc_{c}"] = float(ok[m].mean())
        if s == "pope":
            tp = (said & gold).sum()
            r["f1"] = float(2 * tp / (said.sum() + gold.sum()))
        else:                                           # MME acc+: both questions of an image right
            per = {}
            for q, o in zip(qs, ok):
                per.setdefault(q["img"], []).append(o)
            r["acc_plus"] = float(np.mean([all(v) for v in per.values()]))
        res[s] = r
    (D2 / f"eval_{name}.json").write_text(json.dumps(res, indent=1))
    print("EVAL", json.dumps(res), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"prep": prep, "teacher": teacher_cmd, "fly": fly_cmd, "merge": merge_cmd}.get(cmd, lambda: None)()
    if cmd == "align":
        align_cmd(sys.argv[2])
    if cmd == "eval":
        eval_cmd(sys.argv[2])
