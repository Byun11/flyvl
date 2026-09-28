"""D3 (PROTOCOL_D3_swarm.md): an agent with fly eyes pursues a low-contrast moving letter; the VLM reads its
stabilised fovea. Stage 1 = one agent; arms differ only in the eye's motion features.

Canvas 256 px: static low-contrast texture + 8 static distractor letters + 1 moving target letter (all contrast c),
fresh pixel noise every frame. The agent sees a 64 px window and moves it by (dx, dy) in [-4, 4] px per 20 ms frame.
Features (13): 3x3-region motion energy, signed global horizontal / vertical motion, previous action. Linear policy
trained by CEM on mean tracking error. Reading: the window's central 32 px averaged over the last 25 frames ->
InternVL3-1B, argmax over the 26 capital-letter tokens.

usage: d3_pursuit.py stage0
       d3_pursuit.py run CONTRAST CEM_SEED FAMILY [SETTING,...]   FAMILY in flyvis | hr | framediff | base
       d3_pursuit.py read                                         VLM-read every saved fovea set
Every setting stores its train error, per-episode test error and test foveae; selection on train happens in the
verdict, so a family can be split across GPUs.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome  # noqa: E402
from flyvl.stimulus import _gaussian_blur  # noqa: E402

DT, T, CAN, WIN, FOV, READ = 0.02, 50, 256, 64, 32, 25
VMAX = 4.0
NOISE = float(os.environ.get("D3_NOISE", "0.06"))
POP, ELITE, GENS = 48, 8, int(os.environ.get("D3_GENS", "25"))
N_TR, N_TE, N_CAL = int(os.environ.get("D3_NTR", "64")), 128, 128
CHUNK = int(os.environ.get("D3_CHUNK", "256"))
LO, HI = WIN / 2, CAN - WIN / 2                        # window centre range; the target moves in the same range
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LETTERS = [chr(ord("A") + i) for i in range(26)]
MEMBERS = ("000", "001", "002", "003", "004")
BLURS, TAUS = (1, 2, 4), (1, 3, 6, 12, 24)
OUT = Path(os.environ.get("D3_OUT", connectome.DATA_ROOT / "runs" / "d3"))
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"


def glyphs():
    font = ImageFont.truetype(FONT, 24)
    out = []
    for ch in LETTERS:
        im = Image.new("L", (FOV, FOV), 0)
        d = ImageDraw.Draw(im)
        l, t, r, b = d.textbbox((0, 0), ch, font=font)
        d.text(((FOV - (r - l)) / 2 - l, (FOV - (b - t)) / 2 - t), ch, fill=255, font=font)
        out.append(np.asarray(im, np.float32) / 255)
    return torch.as_tensor(np.stack(out), device=dev)


def world(n, seed, c):
    """Episodes: static canvas (texture + distractors), target letter and trajectory, agent start, sensor noise."""
    g = np.random.default_rng(seed)
    G = glyphs()
    ax = torch.arange(CAN, device=dev, dtype=torch.float32)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    K = 12
    freq, ang = g.uniform(2, 12, (n, K)), g.uniform(0, np.pi, (n, K))
    ph, amp = g.uniform(0, 2 * np.pi, (n, K)), g.uniform(0.5, 1.0, (n, K))
    tex = torch.zeros(n, CAN, CAN, device=dev)
    for k in range(K):
        f_, a_, p_, m_ = (torch.as_tensor(v[:, k], dtype=torch.float32, device=dev)[:, None, None] for v in (freq, ang, ph, amp))
        tex += m_ * torch.sin(2 * np.pi * f_ * (xx * torch.cos(a_) + yy * torch.sin(a_)) / CAN + p_)
    tex /= torch.as_tensor(np.sqrt((amp ** 2).sum(1) / 2), dtype=torch.float32, device=dev)[:, None, None]
    canvas = 0.5 + c * tex
    target = g.integers(0, 26, n)
    for i in range(n):
        others = g.choice([k for k in range(26) if k != target[i]], 8, replace=False)
        for k, (x, y) in zip(others, g.integers(0, CAN - FOV, (8, 2))):
            canvas[i, y:y + FOV, x:x + FOV] -= c * G[k]
    # target: Ornstein-Uhlenbeck velocity (stationary sd 2 px/frame per axis), bounces inside [LO, HI]
    p = g.uniform(LO + 16, HI - 16, (n, 2))
    v = g.normal(0, 2, (n, 2))
    traj = np.zeros((n, T, 2))
    for t in range(T):
        traj[:, t] = p
        v = 0.95 * v + np.sqrt(1 - 0.95 ** 2) * 2 * g.normal(0, 1, (n, 2))
        p = p + v
        lo, hi = p < LO, p > HI
        p[lo], v[lo] = 2 * LO - p[lo], -v[lo]
        p[hi], v[hi] = 2 * HI - p[hi], -v[hi]
    start = np.clip(traj[:, 0] + g.uniform(-12, 12, (n, 2)), LO, HI)
    gen = torch.Generator(device=dev).manual_seed(10_000 + seed)
    noise = torch.randn(n, T, WIN, WIN, generator=gen, device=dev)
    f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    return {"canvas": canvas, "glyph": G[torch.as_tensor(target, device=dev)], "target": target, "traj": f(traj),
            "start": f(start), "noise": noise, "c": c, "n": n}


OFF = torch.arange(WIN, device=dev, dtype=torch.float32) - (WIN - 1) / 2


def render(w, idx, pos, s):
    """(B, WIN, WIN) luminance of each agent's window at frame s."""
    X = pos[:, 0, None] + OFF[None]
    Y = pos[:, 1, None] + OFF[None]
    grid = torch.stack(torch.broadcast_tensors(X[:, None, :], Y[:, :, None]), -1)            # (B, WIN, WIN, (x, y))
    static = F.grid_sample(w["canvas"][idx][:, None], grid / (CAN - 1) * 2 - 1, mode="bilinear",
                           padding_mode="border", align_corners=True)[:, 0]
    gl = (grid - w["traj"][idx, s][:, None, None] + (FOV - 1) / 2) / (FOV - 1) * 2 - 1
    ink = F.grid_sample(w["glyph"][idx][:, None], gl, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]
    return static - w["c"] * ink + NOISE * w["noise"][idx, s]


_ry = torch.arange(WIN, device=dev) * 3 // WIN
MPIX = F.one_hot((_ry[:, None] * 3 + _ry[None, :]).flatten(), 9).float()
MPIX = MPIX / MPIX.sum(0)                                                                     # (WIN*WIN, 9) region means


class FlyEye:
    """flyvis member: T4a-d / T5a-d. Region energy = frame-to-frame change squared, summed over the 8 types;
    global = (T4a+T5a-T4b-T5b), (T4c+T5c-T4d-T5d) relative to the first frame."""
    dim = 11

    def __init__(self, member):
        from flyvl.flyvis_encoder import T45, FlyvisEncoder, hexal_xy
        self.enc = FlyvisEncoder(model=f"flow/0000/{member}")
        self.idx = torch.as_tensor(np.stack([self.enc.idx[t] for t in T45]), device=dev)       # (8, 721)
        xy = hexal_xy(self.enc, WIN)
        reg = np.minimum((xy[:, 1] * 3).astype(int), 2) * 3 + np.minimum((xy[:, 0] * 3).astype(int), 2)
        M = F.one_hot(torch.as_tensor(reg, device=dev), 9).float()
        self.M = M / M.sum(0)

    def reset(self, B):
        with torch.device(dev), torch.no_grad():
            self.state = self.enc.net.steady_state(t_pre=0.5, dt=DT, batch_size=B, value=0.5)
        self.base = self.prev = None

    @torch.no_grad()
    def __call__(self, img):
        net = self.enc.net
        with torch.device(dev):
            net.stimulus.zero(img.shape[0], 1)
            net.stimulus.add_input(self.enc.eye(img[:, None].clamp(0, 1)))
            self.state = net(net.stimulus(), DT, self.state, as_states=True)[-1]
            x = self.state.nodes.activity[:, self.idx]                                         # (B, 8, 721)
        if self.base is None:
            self.base = self.prev = x
        E = ((x - self.prev) ** 2).sum(1) @ self.M
        ev = x - self.base
        H = (ev[:, 0] + ev[:, 4] - ev[:, 1] - ev[:, 5]).mean(-1, keepdim=True)
        V = (ev[:, 2] + ev[:, 6] - ev[:, 3] - ev[:, 7]).mean(-1, keepdim=True)
        self.prev = x
        return torch.cat([E, H, V], 1)


class PixEye:
    """Blur b px, smooth over tau frames, then HR correlation (hr) or frame difference (framediff)."""
    dim = 11

    def __init__(self, kind, blur, tau):
        self.kind, self.blur, self.tau = kind, blur, tau

    def reset(self, B):
        self.sm = self.prev = None

    @torch.no_grad()
    def __call__(self, img):
        b = _gaussian_blur(img[:, None], self.blur)[:, 0]
        self.sm = b if self.sm is None else b / self.tau + self.sm * (1 - 1 / self.tau)
        B = img.shape[0]
        if self.prev is None:
            out = torch.zeros(B, 11, device=dev)
        elif self.kind == "hr":
            a, c = self.prev, self.sm
            rx = a[..., :-1] * c[..., 1:] - a[..., 1:] * c[..., :-1]
            ry = a[..., :-1, :] * c[..., 1:, :] - a[..., 1:, :] * c[..., :-1, :]
            en = F.pad(rx ** 2, (0, 1)) + F.pad(ry ** 2, (0, 0, 0, 1))
            out = torch.cat([en.flatten(1) @ MPIX, rx.mean((1, 2))[:, None], ry.mean((1, 2))[:, None]], 1)
        else:
            d = self.sm - self.prev
            out = torch.cat([(d ** 2).flatten(1) @ MPIX, torch.zeros(B, 2, device=dev)], 1)
        self.prev = self.sm
        return out


def rollout(eye, Wp, w, idx, mode="policy", read=False):
    """Returns (mean tracking error per episode, averaged fovea per episode or None). mode: policy | static |
    random | oracle | calib (random actions, returns the feature matrix instead)."""
    if len(idx) > CHUNK:
        parts = [rollout(eye, None if Wp is None else Wp[s:s + CHUNK], w, idx[s:s + CHUNK], mode, read)
                 for s in range(0, len(idx), CHUNK)]
        if mode == "calib":
            return torch.cat(parts)
        return torch.cat([p[0] for p in parts]), (torch.cat([p[1] for p in parts]) if read else None)
    B = len(idx)
    if eye is not None:
        eye.reset(B)
    pos = w["start"][idx].clone()
    a = torch.zeros(B, 2, device=dev)
    err = torch.zeros(B, device=dev)
    fov = torch.zeros(B, FOV, FOV, device=dev) if read else None
    rg = torch.Generator(device=dev).manual_seed(int(idx[0]) + 7)
    feats = []
    ones = torch.ones(B, 1, device=dev)
    for s in range(T):
        if mode == "oracle":
            pos = w["traj"][idx, s].clamp(LO, HI)
        img = render(w, idx, pos, s)
        err += (pos - w["traj"][idx, s]).norm(dim=1)
        if read and s >= T - READ:
            fov += img[:, (WIN - FOV) // 2:(WIN + FOV) // 2, (WIN - FOV) // 2:(WIN + FOV) // 2]
        if mode in ("policy", "calib"):
            f = torch.cat([eye(img), a / VMAX], 1)
        if mode == "policy":
            f = (f - eye.mu) / eye.sd
            a = (Wp.view(B, 2, -1) * torch.cat([f, ones], 1)[:, None]).sum(-1).clamp(-VMAX, VMAX)
        elif mode in ("random", "calib"):
            if mode == "calib":
                feats.append(f)
            a = (torch.rand(B, 2, generator=rg, device=dev) * 2 - 1) * VMAX
        else:
            a = torch.zeros(B, 2, device=dev)
        pos = (pos + a).clamp(LO, HI)
    if mode == "calib":
        return torch.cat(feats)
    return err / T, (fov / READ if read else None)


def calibrate(eye, w):
    F_ = rollout(eye, None, w, torch.arange(w["n"], device=dev), mode="calib")
    eye.mu, eye.sd = F_.mean(0), F_.std(0) + 1e-8


def cem(eye, w, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    D = 2 * (eye.dim + 2 + 1)
    mu, sd = torch.zeros(D, device=dev), torch.ones(D, device=dev)
    ep = torch.arange(w["n"], device=dev)
    for gen in range(GENS):
        cand = mu + sd * torch.randn(POP, D, generator=g, device=dev)
        score = rollout(eye, cand.repeat_interleave(w["n"], 0), w, ep.repeat(POP))[0].view(POP, w["n"]).mean(1)
        elite = cand[score.argsort()[:ELITE]]
        mu, sd = elite.mean(0), elite.std(0) + 0.02
        if gen % 5 == 0 or gen == GENS - 1:
            print(f"    gen {gen:2d} best train err {score.min().item():.2f} px", flush=True)
    return mu


class Reader:
    def __init__(self):
        from flyvl import teacher
        self.proc, self.model = teacher.load()
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text",
                 "text": "Which capital letter is in the image? Answer with a single letter."}]}]
        self.prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        tok = self.proc.tokenizer
        self.ids = [tok.convert_tokens_to_ids(ch) for ch in LETTERS]
        assert len(set(self.ids)) == 26 and tok.unk_token_id not in self.ids, self.ids

    @torch.no_grad()
    def __call__(self, fov):
        """fov (N, FOV, FOV) -> predicted letter index per fovea."""
        fov = fov.to(dev).float()
        z = (fov - fov.mean((1, 2), keepdim=True)) / (fov.std((1, 2), keepdim=True) + 1e-6)
        img = (0.5 + 0.2 * z).clamp(0, 1)
        out = []
        for s in range(0, len(img), 32):
            x = F.interpolate(img[s:s + 32, None], size=448, mode="bicubic").clamp(0, 1)
            pil = [Image.fromarray((im[0].cpu().numpy() * 255).astype(np.uint8)).convert("RGB") for im in x]
            inp = self.proc(images=pil, text=[self.prompt] * len(pil), return_tensors="pt", padding=True).to(dev, torch.bfloat16)
            out.append(self.model(**inp).logits[:, -1][:, self.ids].argmax(1).cpu())
        return torch.cat(out).numpy()


def stage0():
    reader = Reader()
    res = {"noise": NOISE}
    for c in (0.3, 0.1, 0.05, 0.03, 0.02):
        w = world(N_CAL, 3, c)
        idx = torch.arange(N_CAL, device=dev)
        for mode in ("oracle", "static"):
            err, fov = rollout(None, None, w, idx, mode=mode, read=True)
            acc = float((reader(fov) == w["target"]).mean())
            res[f"c{c:g}_{mode}"] = {"read_acc": acc, "track_err": float(err.mean())}
            print(f"c {c:g} {mode:6s} read {acc:.3f} track err {err.mean():.1f} px", flush=True)
    (OUT / f"stage0_n{NOISE:g}.json").write_text(json.dumps(res, indent=1))


def settings(family):
    if family == "flyvis":
        return [(m, lambda m=m: FlyEye(m)) for m in MEMBERS]
    if family in ("hr", "framediff"):
        return [(f"b{b}_t{t}", lambda b=b, t=t: PixEye(family, b, t)) for b in BLURS for t in TAUS]
    raise ValueError(family)


def run(c, seed, family, only=None):
    t0 = time.time()
    w_tr, w_te = world(N_TR, 1, c), world(N_TE, 2, c)
    te = torch.arange(N_TE, device=dev)
    if family == "base":
        for mode in ("static", "random", "oracle"):
            err, fov = rollout(None, None, w_te, te, mode=mode, read=True)
            save(c, seed, "base", mode, {"test_err_episodes": err.cpu().tolist()}, fov, w_te)
        return
    for name, make in settings(family):
        if only and name not in only:
            continue
        f = OUT / f"c{c:g}_s{seed}_{family}_{name}.json"
        if f.exists():
            continue
        eye = make()
        calibrate(eye, w_tr)
        mu = cem(eye, w_tr, seed)
        tr = rollout(eye, mu[None].expand(N_TR, -1), w_tr, torch.arange(N_TR, device=dev))[0]
        err, fov = rollout(eye, mu[None].expand(N_TE, -1), w_te, te, read=True)
        save(c, seed, family, name, {"train_err": float(tr.mean()), "test_err_episodes": err.cpu().tolist(),
                                     "policy": mu.cpu().tolist()}, fov, w_te)
        print(f"  {family} {name}: train {tr.mean():.2f} px  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        del eye
        torch.cuda.empty_cache()


def save(c, seed, family, name, rec, fov, w):
    stem = f"c{c:g}_s{seed}_{family}_{name}"                  # a string: Path.with_suffix would cut at "0.1"
    rec.update({"contrast": c, "cem_seed": seed, "family": family, "setting": name, "gens": GENS, "n_train": N_TR,
                "noise": NOISE, "target": w["target"].tolist()})
    np.save(OUT / f"{stem}_fov.npy", fov.cpu().numpy().astype(np.float32))
    (OUT / f"{stem}.json").write_text(json.dumps(rec))


def read_all():
    reader = None
    for f in sorted(OUT.glob("c*_s*_*.json")):
        rec = json.loads(f.read_text())
        if "read" in rec:
            continue
        reader = reader or Reader()
        pred = reader(torch.as_tensor(np.load(str(f)[:-5] + "_fov.npy")))
        rec["read"] = (pred == np.asarray(rec["target"])).astype(int).tolist()
        f.write_text(json.dumps(rec))
        print(f.name, f"read {np.mean(rec['read']):.3f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "stage0":
        stage0()
    elif cmd == "run":
        run(float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5].split(",") if len(sys.argv) > 5 else None)
    elif cmd == "read":
        read_all()
