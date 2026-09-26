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
