"""flyvis (Lappalainen et al. 2024) as the fly encoder: a connectome-constrained optic lobe (FIB25/19 column
template x 721 columns, 45,669 cells) whose 734 cell-type parameters were trained on optic flow, so its
T4/T5 are direction selective - which our invented v1 physiology never achieved (P0).

features(): video -> per-column T4a-d / T5a-d responses, signed time mean and change energy, optionally
over 4 quadrant glimpses (each quadrant fills the whole eye), concatenated.
"""
from __future__ import annotations

import numpy as np
import torch

from . import flyvis_compat  # noqa: F401  (Windows fix, sets FLYVIS_ROOT_DIR)
from flyvis import NetworkView
from flyvis.datasets.rendering import BoxEye

T45 = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")


class FlyvisEncoder:
    def __init__(self, model: str = "flow/0000/000", device: str = "cuda"):
        with torch.device(device):
            self.net = NetworkView(model).init_network().to(device)
            self.eye = BoxEye()
        types = np.asarray([t.decode() if isinstance(t, bytes) else t for t in self.net.connectome.nodes.type[:]])
        self.idx = {t: np.flatnonzero(types == t) for t in T45}                     # all columns of each type
        self.device = device

    @torch.no_grad()
    def run(self, frames: torch.Tensor, dt: float) -> dict:
        """frames (B, T, H, W) in [0, 1] -> {"mean": (B, 8*721), "energy": (B, 8*721)}"""
        B, T = frames.shape[:2]
        with torch.device(self.device):
            movie = self.eye(frames.float().to(self.device))                     # (B, T, 1, 721)
            r = self.net.simulate(movie, dt=dt)                                      # (B, T, n_cells)
        sel = torch.cat([r[..., torch.as_tensor(self.idx[t], device=r.device)] for t in T45], -1)
        base = sel[:, :1]                                                            # response to the first frame
        ev = sel - base
        return {"mean": ev.mean(1).float().cpu(), "energy": (ev[:, 1:] - ev[:, :-1]).pow(2).mean(1).float().cpu()}


def features(enc: FlyvisEncoder, frames_fn, p: dict, N: int, steps: int, dt: float, glimpses: bool = True,
             batch: int = 32) -> dict:
    """frames_fn(p_batch, t) -> (B, 1, H, W). Returns numpy features over all N trials."""
    out = {"mean": [], "energy": []}
    for s in range(0, N, batch):
        pb = {k: v[s:s + batch] for k, v in p.items()}
        video = torch.stack([frames_fn(pb, k * dt)[:, 0] for k in range(steps)], 1).clamp(0, 1)   # (B, T, H, W)
        H = video.shape[-1] // 2
        views = [video[..., i * H:(i + 1) * H, j * H:(j + 1) * H] for i in (0, 1) for j in (0, 1)] if glimpses else [video]
        res = [enc.run(v, dt) for v in views]
        for k in out:
            out[k].append(torch.cat([r[k] for r in res], 1))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def hexal_xy(enc: FlyvisEncoder, size: int = 64) -> np.ndarray:
    """(721, 2) x, y in [0, 1] that each eye column samples, measured by feeding coordinate ramps."""
    ax = torch.linspace(0, 1, size)
    with torch.device(enc.device):
        x = enc.eye(ax[None, :].repeat(size, 1)[None, None].to(enc.device))[0, 0, 0]
        y = enc.eye(ax[:, None].repeat(1, size)[None, None].to(enc.device))[0, 0, 0]
    return torch.stack([x, y], 1).cpu().numpy()


@torch.no_grad()
def pooled_features(enc: FlyvisEncoder, frames_fn, p: dict, N: int, steps: int, dt: float, batch: int = 32) -> np.ndarray:
    """4 quadrant glimpses; T4a-d/T5a-d responses averaged over the columns that look at each page cell.
    Returns (N, 16 cells x 8 types x 2) = signed late mean and change energy. The per-type columns are in
    hexal order (checked: T4a cell i sits on hexal i)."""
    xy = hexal_xy(enc)
    sub = (xy[:, 1] >= 0.5).astype(int) * 2 + (xy[:, 0] >= 0.5).astype(int)      # which quarter of the glimpse
    masks = torch.as_tensor(np.stack([sub == k for k in range(4)]), dtype=torch.float32)
    masks = (masks / masks.sum(1, keepdim=True)).to(enc.device)                     # (4, 721) averaging weights
    out = []
    for s in range(0, N, batch):
        pb = {k: v[s:s + batch] for k, v in p.items()}
        video = torch.stack([frames_fn(pb, k * dt)[:, 0] for k in range(steps)], 1).clamp(0, 1)
        B, H = video.shape[0], video.shape[-1] // 2
        feat = torch.zeros(B, 16, len(T45), 2, device=enc.device)
        for q in range(4):
            qi, qj = divmod(q, 2)
            with torch.device(enc.device):
                r = enc.net.simulate(enc.eye(video[..., qi * H:(qi + 1) * H, qj * H:(qj + 1) * H].float().to(enc.device)), dt=dt)
            for ti, t in enumerate(T45):
                x = r[..., torch.as_tensor(enc.idx[t], device=r.device)]                  # (B, T, 721)
                ev = x - x[:, :1]
                sm, en = ev[:, 5:].mean(1), (ev[:, 1:] - ev[:, :-1]).pow(2).mean(1)      # (B, 721)
                for k in range(4):
                    cy, cx = divmod(k, 2)
                    cell = (2 * qi + cy) * 4 + (2 * qj + cx)
                    feat[:, cell, ti, 0] = sm @ masks[k]
                    feat[:, cell, ti, 1] = en @ masks[k]
        out.append(feat.flatten(1).cpu())
    return torch.cat(out).numpy()
