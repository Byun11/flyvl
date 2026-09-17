"""FlyVL-v2: connectome-constrained, task-optimized whole-MaleCNS rate model (see PROTOCOL_v2.md).

Fixed: topology, |w| from synapse counts (input-normalized), E/I sign, effective-graph rules (v1 files),
eye mapping, 4-way drift stimulus.
Trained (shared per cell type, identical count and meaning for every graph):
  g_pre[type] > 0, g_post[type] > 0 (softplus), tau[type] > dt (dt + softplus), bias[type].
Dynamics for every neuron (FlyVis-style graded units):
  V <- V + (dt/tau) * ( -V + g_post * W (g_pre * f(V)) + bias + input ),  f(V) = 0.5 tanh(2V)
  (f: slope 1 at rest, bounded +-0.5, f(0) = 0 so V = 0 is the exact resting state when bias = 0.
   A sigmoid f was tried first: label-free checks showed ~10x attenuation per layer on the real graph
   (central evoked ~1e-5, 0% of central units > 1e-3) and oscillating rest for larger gains.)
Readout: time- and direction-mean evoked f(V) (image - blank) of central_vnc neurons ->
  fixed Gaussian random projection (seed 0, same for every graph) -> 1024 -> BatchNorm (no affine) -> Linear -> 10.
  (The standardization removes feature-scale differences between graphs: at the common init, real central evoked
   activity is ~100x smaller than shuffle's, which would otherwise handicap the readout's optimization only.)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from torch.utils.checkpoint import checkpoint

from . import connectome, frozen, masks
from .sim import _to_torch_csr
from .stimulus import GRAY, Retina

DT = 0.02
WARM_STEPS = 50          # no-grad settle to the current parameters' resting state
STIM_STEPS = 25
PROJ_DIM = 1024
INPUT_SCALE = 4.0        # eye signal (about +-0.5) -> photoreceptor input current
# Common init for every graph: g_pre = g_post = 1. Spectral check: real W has spectral radius 1.0 (an isolated
# 2-neuron ENS loop) and a band of optic-lobe feedback modes at ~0.91; global shuffle has 0.29. Any uniform gain
# > ~1.1 makes rest unstable on the real graph only (an earlier label-free pick g_post = 6 was in that regime),
# so the largest gain stable for both graphs is used and gain growth is left to training.
INIT = {"g_pre": 1.0, "g_post": 1.0, "tau": 0.04, "bias": 0.0}


def act(V: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.tanh(2 * V)


def _inv_softplus(y: float) -> float:
    return float(np.log(np.expm1(y)))


def cell_type_ids(c: connectome.Connectome) -> np.ndarray:
    ct = c.cell_type.astype(str)
    key = np.where(ct == "", np.char.add("untyped:", c.superclass.astype(str)), ct)
    return np.unique(key, return_inverse=True)[1]


class _FixedSpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, WT, x):
        ctx.WT = WT
        return W @ x

    @staticmethod
    def backward(ctx, g):
        return None, None, ctx.WT @ g


class FlyVL2(torch.nn.Module):
    def __init__(self, c: connectome.Connectome, W: sparse.csr_matrix, device: str = "cuda"):
        super().__init__()
        cfg, eye_cfg, _ = frozen.load()
        self.device = device
        self.W = _to_torch_csr(W, device)
        self.WT = _to_torch_csr(W.T.tocsr(), device)
        tid = cell_type_ids(c)
        self.n_types = int(tid.max()) + 1
        self.register_buffer("tid", torch.as_tensor(tid, device=device))
        r16 = c.types(["R1-6"])
        driven = r16[c.column[r16, 0] >= 0]
        self.register_buffer("driven", torch.as_tensor(driven, device=device))
        self.retina = Retina(c, driven, eye_cfg, DT, device=device)
        self.dirs = eye_cfg.drift_directions
        central = masks.build(c)["central_vnc"]
        self.register_buffer("central", torch.as_tensor(central, device=device))
        g = torch.Generator().manual_seed(0)
        proj = torch.randn(PROJ_DIM, len(central), generator=g) / np.sqrt(PROJ_DIM)
        self.register_buffer("proj", proj.to(device))
        nT = self.n_types
        self.g_pre_raw = torch.nn.Parameter(torch.full((nT,), _inv_softplus(INIT["g_pre"]), device=device))
        self.g_post_raw = torch.nn.Parameter(torch.full((nT,), _inv_softplus(INIT["g_post"]), device=device))
        self.tau_raw = torch.nn.Parameter(torch.full((nT,), _inv_softplus(INIT["tau"] - DT), device=device))
        self.bias = torch.nn.Parameter(torch.full((nT,), INIT["bias"], device=device))
        self.readout = torch.nn.Sequential(torch.nn.BatchNorm1d(PROJ_DIM, affine=False),
                                           torch.nn.Linear(PROJ_DIM, 10)).to(device)

    def dynamics_parameters(self):
        return [self.g_pre_raw, self.g_post_raw, self.tau_raw, self.bias]

    def _neuron_params(self):
        t = self.tid
        g_pre = F.softplus(self.g_pre_raw)[t][:, None]
        g_post = F.softplus(self.g_post_raw)[t][:, None]
        alpha = (DT / (DT + F.softplus(self.tau_raw)))[t][:, None]
        return g_pre, g_post, alpha, self.bias[t][:, None]

    def _step(self, V, inp, g_pre, g_post, alpha, bias):
        syn = g_post * _FixedSpMM.apply(self.W, self.WT, g_pre * act(V))
        drive = syn + bias
        drive = drive.index_add(0, self.driven, inp)
        return V + alpha * (drive - V)

    def eye_sequence(self, images: torch.Tensor) -> torch.Tensor:
        """images (B, 1, H, W) -> (STIM_STEPS, n_driven, 4B + 1) input currents; last column is blank."""
        B = images.shape[0]
        self.retina.reset(len(self.dirs) * B + 1)
        seq = []
        with torch.no_grad():
            for k in range(STIM_STEPS):
                t = k * DT
                lum = torch.cat([self.retina.sample_images(images, t, d) for d in self.dirs]
                                + [torch.full((len(self.driven), 1), GRAY, device=self.device)], 1)
                seq.append(INPUT_SCALE * self.retina.transduce(lum))
        return torch.stack(seq)

    def features(self, images: torch.Tensor, grad: bool = True) -> torch.Tensor:
        """Evoked central_vnc activity projected to PROJ_DIM: (B, PROJ_DIM)."""
        B, D = images.shape[0], len(self.dirs)
        g_pre, g_post, alpha, bias = self._neuron_params()
        zero_in = torch.zeros(len(self.driven), 1, device=self.device)
        with torch.no_grad():
            V = torch.zeros(len(self.tid), 1, device=self.device)
            for _ in range(WARM_STEPS):
                V = self._step(V, zero_in, g_pre, g_post, alpha, bias)
        V = V.expand(-1, D * B + 1).contiguous()
        eye = self.eye_sequence(images)
        acc = torch.zeros(len(self.central), D * B + 1, device=self.device)
        with torch.set_grad_enabled(grad):
            for k in range(STIM_STEPS):
                if grad:
                    V = checkpoint(self._step, V, eye[k], g_pre, g_post, alpha, bias, use_reentrant=False)
                else:
                    V = self._step(V, eye[k], g_pre, g_post, alpha, bias)
                acc = acc + act(V[self.central])
            mean = acc / STIM_STEPS
            evoked = (mean[:, :-1] - mean[:, -1:]).reshape(len(self.central), D, B).mean(1)   # (Nc, B)
            return (self.proj @ evoked).T

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.readout(self.features(images))

    @torch.no_grad()
    def health(self, images: torch.Tensor) -> dict:
        """Label-free state summary for stability checks."""
        B, D = images.shape[0], len(self.dirs)
        g_pre, g_post, alpha, bias = self._neuron_params()
        V = torch.zeros(len(self.tid), 1, device=self.device)
        zero_in = torch.zeros(len(self.driven), 1, device=self.device)
        for _ in range(WARM_STEPS):
            V_prev, V = V, self._step(V, zero_in, g_pre, g_post, alpha, bias)
        warm_delta = (V - V_prev).abs().max().item()
        V = V.expand(-1, D * B + 1).contiguous()
        eye = self.eye_sequence(images)
        for k in range(STIM_STEPS):
            V = self._step(V, eye[k], g_pre, g_post, alpha, bias)
        r = act(V)
        ev = (r[:, :-1] - r[:, -1:]).abs()
        return {"finite": bool(torch.isfinite(V).all()), "warm_last_step_max_change": warm_delta,
                "act_abs_mean": r.abs().mean().item(), "frac_saturated(|f|>0.49)": (r.abs() > 0.49).float().mean().item(),
                "central_evoked_abs_mean": ev[self.central].mean().item(),
                "central_frac_evoked>1e-3": (ev[self.central].mean(1) > 1e-3).float().mean().item(),
                "photoreceptor_evoked_abs_mean": ev[self.driven].mean().item()}
