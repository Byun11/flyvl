"""Fly swarm feature extraction (frozen), for P4 token distillation.

Image (32x32 RGB) -> luminance, reflect-pad 4 -> 16 overlapping 16x16 patches centered on the 4x4 token grid
(centers 4, 12, 20, 28 px) -> each patch is shown to one fly through the v1/v2 eye model (R1-6 retinotopy,
4-way drift, 25 steps) -> MaleCNS rate dynamics of FlyVL-v2 frozen at the common init (g = 1)
-> time/direction-mean evoked f(V) per neuron -> fixed Gaussian projections (seed 0, 1024-d) of several views.
All flies share the same connectome (weight sharing), so a batch of patches is a batch of flies.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import connectome, masks
from .v2 import DT, STIM_STEPS, FlyVL2, act

VIEWS = ("central_vnc", "visual_projection", "optic_lobe")
PROJ = 1024
PATCH, STRIDE, PAD, G = 16, 8, 4, 4


GRIDS = {4: (16, 8, 4), 2: (16, 16, 0), 1: (32, 32, 0)}                     # grid -> (patch, stride, pad)


def patches(images_uint8: np.ndarray, grid: int = G) -> torch.Tensor:
    """(N, 32, 32, 3) -> (N * grid^2, 1, P, P) luminance patches, row-major.
    grid 4: 16px patches, stride 8, reflect pad 4 (default); grid 2: 16px non-overlapping; grid 1: whole image."""
    patch, stride, pad = GRIDS[grid]
    x = torch.from_numpy(images_uint8).float().div(255) @ torch.tensor([0.299, 0.587, 0.114])
    x = x[:, None]
    if pad:
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    p = x.unfold(2, patch, stride).unfold(3, patch, stride)
    assert p.shape[2:4] == (grid, grid)
    return p.permute(0, 2, 3, 1, 4, 5).reshape(-1, 1, patch, patch).contiguous()


def view_indices(c: connectome.Connectome) -> dict:
    M = masks.build(c)
    return {"central_vnc": M["central_vnc"], "visual_projection": M["visual_projection"],
            "optic_lobe": np.flatnonzero(c.graded)}


class SwarmExtractor:
    def __init__(self, c, W, device="cuda"):
        self.model = FlyVL2(c, W, device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.views = {k: torch.as_tensor(v, device=device) for k, v in view_indices(c).items()}
        g = torch.Generator().manual_seed(0)
        self.proj = {k: (torch.randn(PROJ, len(v), generator=g) / np.sqrt(PROJ)).to(device)
                     for k, v in self.views.items()}
        self.device = device

    @torch.no_grad()
    def run(self, patch_batch: torch.Tensor) -> dict:
        """(B, 1, 16, 16) -> {view: (B, 1024)} plus 'photoreceptor': (B, 1024) of the eye input itself."""
        m = self.model
        B, D = patch_batch.shape[0], len(m.dirs)
        g_pre, g_post, alpha, bias = m._neuron_params()
        eye = m.eye_sequence(patch_batch.to(self.device))                      # (T, n_driven, D*B+1)
        V = torch.zeros(len(m.tid), D * B + 1, device=self.device)
        acc = torch.zeros_like(V)
        for k in range(STIM_STEPS):
            V = m._step(V, eye[k], g_pre, g_post, alpha, bias)
            acc += act(V)
        mean = acc / STIM_STEPS
        ev = (mean[:, :-1] - mean[:, -1:]).reshape(len(m.tid), D, B).mean(1)  # (N_neurons, B)
        out = {k: (self.proj[k] @ ev[idx]).T.float().cpu() for k, idx in self.views.items()}
        pr = (eye[:, :, :-1] - eye[:, :, -1:]).mean(0).reshape(len(m.driven), D, B).mean(1)
        if "photoreceptor" not in self.proj:
            g = torch.Generator().manual_seed(1)
            self.proj["photoreceptor"] = (torch.randn(PROJ, len(m.driven), generator=g) / np.sqrt(PROJ)).to(self.device)
        out["photoreceptor"] = (self.proj["photoreceptor"] @ pr).T.float().cpu()
        out["evoked_abs_mean"] = {k: float(ev[idx].abs().mean()) for k, idx in self.views.items()}
        return out
