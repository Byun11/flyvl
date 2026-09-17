"""FlyVL-simple: MaleCNS topology + signed input-normalized weights (flybrain) +
graded optic lobe + LIF central brain / VNC. Batched, deterministic, torch sparse CSR.

Graded units (ol_intrinsic, ol_sensory) carry a deviation x from a resting rate `rest`:
    x <- x + a * (clamp(g_gg W_gg x + g_sg W_sg s, -rest, r_max - rest) - x),  a = dt / tau_graded
  clamp(-rest) is the rectification at zero rate; the upper bound is off (inf) unless
  `saturation` is set. Photoreceptors driven by the eye are clamped to the eye signal.
LIF units follow flybrain (ornata/fly):
    v <- exp(-dt/tau) v + g_ss W_ss s + g_gs W_gs x + tonic + noise;  v >= 1 -> spike, v = 0
Noise is "frozen": a fixed Bernoulli pattern per step index, identical for every batch element
and every branch, so image - blank differences are exact and runs are bit-reproducible.
Every deviation from this definition is a named flag in SimConfig.

backend (implementation only, not part of the model): where the sparse matrix products run.
  cuda_fast          cuSPARSE on the GPU; fast but not bit-reproducible (run-to-run diffs <= ~2e-7).
  cpu_deterministic  torch CPU sparse CSR; bit-reproducible. State and everything else stay on `device`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import warnings

import numpy as np
import torch
from scipy import sparse

from .connectome import Connectome, effective_W

warnings.filterwarnings("ignore", message="Sparse")

BACKENDS = ("cuda_fast", "cpu_deterministic")


@dataclass
class SimConfig:
    dt: float = 0.020
    # LIF (flybrain defaults, calibrated at dt = 0.020)
    tau_lif: float = 0.100
    tonic: float = 0.14
    g_ss: float = 3.0
    noise: str = "frozen"            # "frozen" | "off"
    noise_hz: float = 1.2
    noise_amp: float = 0.22
    noise_seed: int = 64
    # graded optic lobe
    tau_graded: float = 0.040
    rest: float = 0.5
    g_gg: float = 1.0
    g_sg: float = 0.5
    g_gs: float = 3.0
    # explicit corrections (all OFF by default)
    saturation: float | None = None                  # r_max for graded units
    tau_graded_by_type: dict = field(default_factory=dict)
    cut_input_to_nonvisual_sensory: bool = False     # flybrain's sensory_input=False, minus the eye
    kc_kc_scale: float = 1.0                         # scale KC -> KC (axo-axonal) weights; runaway loop in LIF
    dtype: str = "float32"

    def spec(self) -> dict:
        return asdict(self)


def _to_torch_csr(M: sparse.csr_matrix, device) -> torch.Tensor:
    M = M.tocsr().astype(np.float32)
    M.sort_indices()
    return torch.sparse_csr_tensor(torch.from_numpy(M.indptr.astype(np.int32)),
                                   torch.from_numpy(M.indices.astype(np.int32)),
                                   torch.from_numpy(M.data), size=M.shape, device=device)


class State:
    def __init__(self, x, v, s, step):
        self.x, self.v, self.s, self.step = x, v, s, step   # x (Ng, B), v/s (Nl, B)

    def clone(self) -> "State":
        return State(self.x.clone(), self.v.clone(), self.s.clone(), self.step)

    def expand(self, batch: int) -> "State":
        """A batch-1 snapshot copied into `batch` identical elements."""
        return State(*(t.expand(-1, batch).contiguous() for t in (self.x, self.v, self.s)), self.step)


class Sim:
    def __init__(self, c: Connectome, cfg: SimConfig, device: str = "cuda", W: sparse.csr_matrix | None = None,
                 driven: np.ndarray | None = None, backend: str = "cuda_fast"):
        """W: an EFFECTIVE weight matrix (flags already applied, e.g. a control graph rewired from
        effective_W); default effective_W(c, cfg). Matrix-level flags are never re-applied here.
        driven: photoreceptor indices clamped to the eye signal (default R1-6)."""
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.c, self.cfg, self.device, self.backend = c, cfg, device, backend
        mat_device = "cpu" if backend == "cpu_deterministic" else device
        W = effective_W(c, cfg) if W is None else W.tocsr()
        graded = c.graded
        self.gi = np.flatnonzero(graded)          # global index of each graded unit
        self.li = np.flatnonzero(~graded)         # global index of each LIF unit
        self.local = np.empty(c.n, np.int64)
        self.local[self.gi] = np.arange(len(self.gi))
        self.local[self.li] = np.arange(len(self.li))
        Wg, Wl = W[self.gi], W[self.li]
        self.W_gg = _to_torch_csr(Wg[:, self.gi], mat_device)
        self.W_sg = _to_torch_csr(Wg[:, self.li], mat_device)    # spiking pre -> graded post
        self.W_gs = _to_torch_csr(Wl[:, self.gi], mat_device)    # graded pre -> spiking post
        self.W_ss = _to_torch_csr(Wl[:, self.li], mat_device)
        self.Ng, self.Nl = len(self.gi), len(self.li)

        driven = c.types(["R1-6"]) if driven is None else driven
        assert graded[driven].all()
        self.driven = torch.as_tensor(self.local[driven], device=device)

        self.set_config(cfg)

    def set_config(self, cfg: SimConfig) -> None:
        """Change scalar parameters without rebuilding the matrices (the sensory cut is fixed at init)."""
        assert cfg.cut_input_to_nonvisual_sensory == self.cfg.cut_input_to_nonvisual_sensory
        assert cfg.kc_kc_scale == self.cfg.kc_kc_scale, "matrix-level flag: build a new Sim"
        self.cfg = cfg
        a = np.full(self.Ng, cfg.dt / cfg.tau_graded, np.float32)
        for t, tau in cfg.tau_graded_by_type.items():
            a[self.c.cell_type[self.gi] == t] = cfg.dt / tau
        self.alpha = torch.as_tensor(np.minimum(a, 1.0), device=self.device)[:, None]
        self.decay = float(np.exp(-cfg.dt / cfg.tau_lif))
        self.tonic = cfg.tonic * (1 - np.exp(-cfg.dt / cfg.tau_lif)) / (1 - np.exp(-0.020 / cfg.tau_lif))
        self.hi = float("inf") if cfg.saturation is None else cfg.saturation - cfg.rest

    # ---- state ----
    def zero_state(self, batch: int = 1) -> State:
        z = lambda n: torch.zeros(n, batch, device=self.device)
        return State(z(self.Ng), z(self.Nl), z(self.Nl), 0)

    def _noise(self, step: int) -> torch.Tensor | None:
        if self.cfg.noise == "off":
            return None
        g = torch.Generator(device=self.device).manual_seed(self.cfg.noise_seed * 1_000_003 + step)
        p = self.cfg.noise_hz * self.cfg.dt
        return (torch.rand(self.Nl, 1, generator=g, device=self.device) < p).float() * self.cfg.noise_amp

    def _mm(self, M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "cpu_deterministic":
            return (M @ x.cpu()).to(self.device, non_blocking=False)
        return M @ x

    # ---- dynamics ----
    @torch.no_grad()
    def step(self, st: State, eye: torch.Tensor | None = None, inject=()) -> State:
        """eye: (n_driven, B) photoreceptor deviation from rest, or None (= rest).
        inject: (global neuron indices, voltage) pairs added to LIF units this step."""
        cfg = self.cfg
        x, v, s = st.x, st.v, st.s
        in_g = cfg.g_gg * self._mm(self.W_gg, x) + cfg.g_sg * self._mm(self.W_sg, s)
        in_l = cfg.g_ss * self._mm(self.W_ss, s) + cfg.g_gs * self._mm(self.W_gs, x)

        x = x + self.alpha * (in_g.clamp(-cfg.rest, self.hi) - x)
        x[self.driven] = 0.0 if eye is None else eye

        v = v * self.decay + in_l + self.tonic
        noise = self._noise(st.step)
        if noise is not None:
            v = v + noise
        for idx, amount in inject:
            v[torch.as_tensor(self.local[idx], device=self.device)] += amount
        s = (v >= 1.0).float()
        v = v * (1.0 - s)
        return State(x, v, s, st.step + 1)

    def spec(self) -> dict:
        return {"model": "FlyVL-simple", "backend": self.backend, "weights_sha256": self.c.weights_sha256, "graded_units": self.Ng,
                "lif_units": self.Nl, "driven_photoreceptors": int(len(self.driven)), **self.cfg.spec()}
