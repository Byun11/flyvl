"""P2-B eye-bypass baseline (see PROTOCOL_v2B_eye_bypass.md).

RGB pixels (3072, centered: x - 0.5) -> trainable Linear (no bias) -> input current on entry neurons
-> MaleCNS dynamics of FlyVL-v2 frozen at the common init (g_pre = g_post = 1, tau = 40 ms, bias = 0)
-> time-mean f(V) of central_vnc over STIM_STEPS -> same fixed random projection as v2-A -> BatchNorm (no affine)
-> Linear -> 10.
No virtual eye: no column sampling, no photoreceptor transduction, no drift. The image is a constant input.
With bias = 0 and zero input, V = 0 is the exact resting state, so no warm-up and no blank branch are needed
(evoked activity = activity).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import sparse
from torch.utils.checkpoint import checkpoint

from . import connectome
from .v2 import STIM_STEPS, FlyVL2, _FixedSpMM, act

VISUAL_ENTRY_TYPES = ("L1", "L2", "L3", "L4", "L5")      # lamina monopolar cells (direct R1-6 targets)
N_PIXELS = 32 * 32 * 3


def entry_neurons(c: connectome.Connectome, mode: str) -> np.ndarray:
    """Neuron IDs (indices) that receive the learned input. Defined by identity only, so equal for every graph."""
    if mode == "visual_entry":
        idx = np.flatnonzero(np.isin(c.cell_type.astype(str), VISUAL_ENTRY_TYPES))
    elif mode == "all_sensory":
        idx = np.flatnonzero(np.char.find(c.superclass.astype(str), "sensory") >= 0)
    else:
        raise ValueError(mode)
    return idx


class EyeBypass(FlyVL2):
    def __init__(self, c: connectome.Connectome, W: sparse.csr_matrix, mode: str, device: str = "cuda"):
        super().__init__(c, W, device)          # dynamics params at INIT, same projection/readout layout as v2-A
        for p in self.dynamics_parameters():
            p.requires_grad_(False)
        self.mode = mode
        self.register_buffer("entry", torch.as_tensor(entry_neurons(c, mode), device=device))
        self.input_proj = torch.nn.Linear(N_PIXELS, len(self.entry), bias=False).to(device)

    def trainable_counts(self) -> dict:
        n_in = sum(p.numel() for p in self.input_proj.parameters())
        n_cls = sum(p.numel() for p in self.readout.parameters() if p.requires_grad)
        return {"input_projection": n_in, "classifier": n_cls, "total": n_in + n_cls,
                "total_requires_grad": sum(p.numel() for p in self.parameters() if p.requires_grad)}

    def features(self, rgb: torch.Tensor, grad: bool = True) -> torch.Tensor:
        """rgb (B, 3, 32, 32) in [0, 1] -> (B, PROJ_DIM)."""
        B = rgb.shape[0]
        g_pre, g_post, alpha, bias = self._neuron_params()
        with torch.set_grad_enabled(grad):
            inp = self.input_proj(rgb.reshape(B, -1) - 0.5).T          # (n_entry, B)
            V = torch.zeros(len(self.tid), B, device=self.device)
            acc = torch.zeros(len(self.central), B, device=self.device)
            for _ in range(STIM_STEPS):
                if grad:
                    V = checkpoint(self._entry_step, V, inp, g_pre, g_post, alpha, bias, use_reentrant=False)
                else:
                    V = self._entry_step(V, inp, g_pre, g_post, alpha, bias)
                acc = acc + act(V[self.central])
            return (self.proj @ (acc / STIM_STEPS)).T

    def _entry_step(self, V, inp, g_pre, g_post, alpha, bias):
        syn = g_post * _FixedSpMM.apply(self.W, self.WT, g_pre * act(V))
        drive = (syn + bias).index_add(0, self.entry, inp)
        return V + alpha * (drive - V)

    @torch.no_grad()
    def health(self, rgb: torch.Tensor) -> dict:
        g = self._neuron_params()
        inp = self.input_proj(rgb.reshape(rgb.shape[0], -1) - 0.5).T
        V = torch.zeros(len(self.tid), rgb.shape[0], device=self.device)
        for _ in range(STIM_STEPS):
            V = self._entry_step(V, inp, *g)
        r = act(V)
        return {"finite": bool(torch.isfinite(V).all()), "input_abs_mean": inp.abs().mean().item(),
                "entry_act_abs_mean": r[self.entry].abs().mean().item(),
                "central_act_abs_mean": r[self.central].abs().mean().item(),
                "central_frac>1e-3": (r[self.central].abs().mean(1) > 1e-3).float().mean().item(),
                "frac_saturated": (r.abs() > 0.49).float().mean().item()}
