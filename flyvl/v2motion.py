"""FlyVL-v2 on the fly-domain motion task: the connectome wiring is FROZEN (never rewired, never
re-weighted); the only trainable things are the unknown physiological parameters (per cell type:
g_pre, g_post, tau, bias) and the linear readout.

Difference from flyvl.v2: the stimulus is a drifting grating whose direction IS the label, so there is
no 4-way drift augmentation (D = 1) and the eye input is rendered from per-trial parameters instead of
from an image. Rendering matches scripts/p5_flytask.render for `motion_dir`, so the frozen-parameter
numbers of P5 are the init point of this model.

Trial parameter layout (B, 5): [dir(0..3), freq, speed, phase, contrast].
"""
from __future__ import annotations

import numpy as np
import torch

from . import connectome
from .stimulus import GRAY
from .v2 import DT, INPUT_SCALE, PROJ_DIM, STIM_STEPS, FlyVL2

PARAM_COLS = ("dir", "freq", "speed", "phase", "contrast")


class MotionFlyVL2(FlyVL2):
    def __init__(self, c: connectome.Connectome, W, noise: float = 0.06, n_classes: int = 4,
                 device: str = "cuda"):
        super().__init__(c, W, device=device)
        self.dirs = (0,)                      # the grating direction is the label, not an augmentation
        self.noise = noise
        self.readout = torch.nn.Sequential(torch.nn.BatchNorm1d(PROJ_DIM, affine=False),
                                           torch.nn.Linear(PROJ_DIM, n_classes)).to(device)
        phi, th = self.retina.phi[:, None], self.retina.theta[:, None]
        fb = (phi.abs() - 0.05) / 0.95
        # coordinate along which the grating moves, one column per direction (ftb, btf, up, down)
        self.register_buffer("along", torch.cat([2 * fb - 1, 1 - 2 * fb, th, -th], 1))

    def eye_sequence(self, params: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
        """params (B, 5) -> (STIM_STEPS, n_driven, B + 1); the last column is the blank branch."""
        p = params.to(self.device)
        idx = p[:, 0].long()
        freq, speed, phase, contrast = (p[:, i][None] for i in (1, 2, 3, 4))
        coord = self.along[:, idx]                                     # (n_driven, B)
        blank = torch.full((len(self.driven), 1), GRAY, device=self.device)
        self.retina.reset(p.shape[0] + 1)
        seq = []
        with torch.no_grad():
            for k in range(STIM_STEPS):
                t = k * DT
                lum = GRAY + contrast * torch.sin(2 * np.pi * freq * (coord - speed * t) + phase)
                if self.noise > 0:
                    lum = lum + torch.randn(lum.shape, generator=gen, device=self.device) * self.noise
                seq.append(INPUT_SCALE * self.retina.transduce(torch.cat([lum, blank], 1)))
        return torch.stack(seq)


def trial_params(rng: np.random.Generator, n: int, contrast: tuple[float, float] = (0.04, 0.10)):
    """Same distributions as p5_flytask's motion_dir_hard. Returns (params (n, 5) float32, labels (n,))."""
    y = rng.integers(0, 4, n)
    p = np.stack([y, rng.uniform(2.0, 7.0, n), rng.uniform(0.5, 1.6, n),
                  rng.uniform(0, 2 * np.pi, n), rng.uniform(*contrast, n)], 1).astype(np.float32)
    return torch.from_numpy(p), torch.from_numpy(y.astype(np.int64))
