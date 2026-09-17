"""P2-C: BPU-style front + back translators, with a no-brain control (see PROTOCOL_v2C_translators.md).

front: RGB 3072 (centered) -> trainable Linear (no bias) -> 8,884 lamina L1-L5 entry values   (as P2-B)
body:  real | global_shuffle_s0  -> frozen FlyVL-v2 dynamics at the common init, time-mean f(V) of central_vnc
       nobrain                    -> f(front output) directly (no connectome)
back:  fixed Gaussian random projection -> 1024 (seed 0) -> BatchNorm(no affine)
       -> trainable Linear 1024->1024 -> ReLU -> Linear 1024->10
Trainable parameter count is identical in all three conditions (front + back MLP).
"""
from __future__ import annotations

import numpy as np
import torch

from . import connectome
from .v2 import PROJ_DIM, act
from .v2b import N_PIXELS, EyeBypass, entry_neurons


def back_translator(device) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.BatchNorm1d(PROJ_DIM, affine=False), torch.nn.Linear(PROJ_DIM, PROJ_DIM),
                               torch.nn.ReLU(), torch.nn.Linear(PROJ_DIM, 10)).to(device)


class BrainTranslators(EyeBypass):
    def __init__(self, c, W, device="cuda"):
        super().__init__(c, W, "visual_entry", device)
        self.readout = back_translator(device)


class NoBrainTranslators(torch.nn.Module):
    def __init__(self, c: connectome.Connectome, device="cuda"):
        super().__init__()
        self.device = device
        n_entry = len(entry_neurons(c, "visual_entry"))
        # same RNG consumption order as BrainTranslators: v2 readout Linear, input_proj, back translator
        torch.nn.Linear(PROJ_DIM, 10)
        self.input_proj = torch.nn.Linear(N_PIXELS, n_entry, bias=False).to(device)
        self.readout = back_translator(device)
        g = torch.Generator().manual_seed(0)
        proj = torch.randn(PROJ_DIM, n_entry, generator=g) / np.sqrt(PROJ_DIM)
        self.register_buffer("proj", proj.to(device))

    def features(self, rgb, grad=True):
        with torch.set_grad_enabled(grad):
            h = act(self.input_proj(rgb.reshape(rgb.shape[0], -1) - 0.5))      # (B, n_entry)
            return h @ self.proj.T

    def forward(self, rgb):
        return self.readout(self.features(rgb))

    def trainable_counts(self):
        n_in = sum(p.numel() for p in self.input_proj.parameters())
        n_cls = sum(p.numel() for p in self.readout.parameters() if p.requires_grad)
        return {"input_projection": n_in, "classifier": n_cls, "total": n_in + n_cls,
                "total_requires_grad": sum(p.numel() for p in self.parameters() if p.requires_grad)}
