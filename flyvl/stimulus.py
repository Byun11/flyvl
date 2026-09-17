"""Eye model: world luminance -> R1-6 photoreceptor signal (achromatic only).

Retina geometry: each driven R1-6 cell sits on a MaleCNS hex column (eye, h1, h2).
  horizontal = (h1 - h2) * sqrt(3)/2   (mirror-symmetric between eyes; |r| ~ 0.85 with EM x)
  vertical   = (h1 + h2) / 2           (|r| = 1.0 with EM y; larger = dorsal)
World coordinates: azimuth phi in [-1, 1] (left eye < 0 < right eye, 0 = straight ahead),
elevation theta in [-1, 1] (dorsal > 0). Whether larger horizontal index is frontal or
posterior is not in the data. P0-1 could not decide it (no T4/T5 direction tuning), so v1 fixes
front_sign = +1; the 4-way symmetric drift ensemble makes image features invariant to that choice
up to a left/right mirror of the visual field.

Transduction: x = k_pr * (lum - A),  A <- A + dt/tau_adapt * (lum - A)   (A starts at gray).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .connectome import Connectome

GRAY = 0.5


@dataclass
class EyeConfig:
    front_sign: int = 1
    k_pr: float = 1.0
    tau_adapt: float = 0.5
    # static images: 4-way symmetric drift ensemble (same image, same warm state, 4 trajectories)
    image_half_width: float = 0.5      # image spans phi in [-w, w], theta in [-1, 1]
    drift_extent: float = 0.2          # total drift in normalized image units (image spans 2) = 10% of image
    drift_time: float = 0.4            # seconds to complete the drift, then hold
    drift_directions: tuple = ("LR", "RL", "UD", "DU")

    def spec(self) -> dict:
        return asdict(self)


class Retina:
    def __init__(self, c: Connectome, driven: np.ndarray, cfg: EyeConfig, dt: float, device="cuda"):
        col = c.column[driven]
        assert (col[:, 0] >= 0).all(), "every driven photoreceptor needs a column"
        eye = col[:, 0]
        h1, h2 = col[:, 1].astype(np.float32), col[:, 2].astype(np.float32)
        hor = (h1 - h2) * np.sqrt(3) / 2
        ver = (h1 + h2) / 2
        f = np.empty_like(hor)
        for e in (0, 1):
            m = eye == e
            f[m] = (hor[m] - hor[m].min()) / (hor[m].max() - hor[m].min())
        if cfg.front_sign < 0:
            f = 1 - f                                  # f: 0 = front, 1 = back
        phi = (0.05 + 0.95 * f) * np.where(eye == 0, -1, 1)
        theta = 2 * (ver - ver.min()) / (ver.max() - ver.min()) - 1
        self.eye = torch.as_tensor(eye, device=device)
        self.phi = torch.as_tensor(phi, dtype=torch.float32, device=device)
        self.theta = torch.as_tensor(theta, dtype=torch.float32, device=device)
        self.cfg, self.dt, self.device = cfg, dt, device
        self.adapt = None

    def reset(self, batch: int):
        self.adapt = torch.full((len(self.phi), batch), GRAY, device=self.device)

    def transduce(self, lum: torch.Tensor) -> torch.Tensor:
        """lum (n, B) -> photoreceptor deviation (n, B); advances adaptation by one step."""
        x = self.cfg.k_pr * (lum - self.adapt)
        self.adapt = self.adapt + (self.dt / self.cfg.tau_adapt) * (lum - self.adapt)
        return x

    # ---- image frames ----
    def sample_images(self, images: torch.Tensor, t: float, direction: str) -> torch.Tensor:
        """images (B, 1, H, W) luminance in [0, 1] -> lum (n, B) at time t (s) after onset.
        direction: the image moves LR (toward +phi), RL, UD (top to bottom = toward -theta) or DU."""
        cfg = self.cfg
        d = -cfg.drift_extent / 2 + cfg.drift_extent * min(t / cfg.drift_time, 1.0)
        sx, sy = {"LR": (d, 0.0), "RL": (-d, 0.0), "UD": (0.0, d), "DU": (0.0, -d)}[direction]
        gx = self.phi / cfg.image_half_width - sx          # grid_sample: x in [-1, 1] left -> right
        gy = -self.theta - sy                               # grid_sample: y in [-1, 1] top -> bottom
        grid = torch.stack([gx, gy], -1)[None, None].expand(images.shape[0], 1, -1, 2)
        out = F.grid_sample(images - GRAY, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return (out[:, 0, 0] + GRAY).T.contiguous()


# ---- synthetic stimuli (world luminance as a function of phi, theta, t) ----
def eye_relative_phi(r: Retina) -> torch.Tensor:
    """Front-to-back coordinate, the same in both eyes (0 front ... 1 back)."""
    return (r.phi.abs() - 0.05) / 0.95


def synthetic(name: str, r: Retina, t: float) -> torch.Tensor:
    """Luminance (n,) for a named synthetic stimulus at time t. Directions are eye-relative:
    'ftb' front-to-back (progressive), 'btf' back-to-front, 'up', 'down'."""
    fb, th = eye_relative_phi(r), r.theta
    k, speed = 4.0, 1.0                      # cycles per unit, units per second
    if name == "blank":
        return torch.full_like(fb, GRAY)
    if name == "flash":
        return torch.full_like(fb, 1.0 if t < 0.2 else GRAY)
    # each direction as a coordinate that increases along the motion
    along = {"ftb": 2 * fb - 1, "btf": 1 - 2 * fb, "up": th, "down": -th}
    if name.startswith("grating_"):
        coord = along[name.split("_")[1]]
        return GRAY + 0.4 * torch.sin(2 * np.pi * k * (coord - speed * t))
    if name.startswith("on_edge_") or name.startswith("off_edge_"):
        polarity, d = name.split("_edge_")
        passed = along[d] < -1.1 + 2.2 * speed * t
        level = 0.9 if polarity == "on" else 0.1
        return torch.where(passed, torch.tensor(level, device=fb.device), torch.tensor(GRAY, device=fb.device))
    if name.startswith("loom_"):
        side = -1 if name.endswith("L") else 1
        center = side * 0.4
        radius = min(0.02 + 0.6 * t, 0.5)
        d = torch.sqrt((r.phi - center) ** 2 + (r.theta * 0.5) ** 2)
        return torch.where(d < radius, torch.tensor(0.05, device=fb.device), torch.tensor(GRAY, device=fb.device))
    raise ValueError(name)
