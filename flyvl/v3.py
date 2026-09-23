"""FlyVL-v3: train the connectome hard, the way the public projects that report numbers do.

Every earlier experiment here deliberately tied its own hands - only 47,524 type-level physiology
parameters were trainable - because the question was "does the wiring contribute?". That gave a clean
negative answer and never once tried to build something that works. This module is the other
experiment: give the network as much trainable capacity as the wiring allows and see how far it goes.

FROZEN: which neurons connect to which, and each synapse's sign. That is the measured connectome and
        it is never modified.
TRAINED: a per-synapse gain (24,472,770 values, kept positive so signs cannot flip), a linear
        image -> photoreceptor encoder, per-cell-type time constants and biases, and the readout.

The matched-shuffle control trains identically, so any gain that is not specific to the real wiring
still shows up in the control.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

from . import connectome, frozen, masks
from .v2 import DT, act, cell_type_ids


class _SparseMM(torch.autograd.Function):
    """y = W(values) @ x over a FIXED sparsity pattern, with a gradient for `values`.

    Rebuilding a COO tensor and coalescing on every step is far too slow at 24.5M edges, and
    materialising the per-edge outer product (nnz x batch) costs gigabytes. Both are avoided:
    forward uses a prebuilt CSR, and d(loss)/d(values) is exactly (g @ x^T) sampled at the pattern,
    which `torch.sparse.sampled_addmm` computes without materialising the dense product.
    """

    @staticmethod
    def forward(ctx, values, x, crow, col, crow_t, col_t, perm_t, shape):
        W = torch.sparse_csr_tensor(crow, col, values, size=shape)
        ctx.save_for_backward(values, x, crow, col, crow_t, col_t, perm_t)
        ctx.shape = shape
        return W @ x

    @staticmethod
    def backward(ctx, g):
        values, x, crow, col, crow_t, col_t, perm_t = ctx.saved_tensors
        g = g.contiguous()
        n = ctx.shape[0]
        WT = torch.sparse_csr_tensor(crow_t, col_t, values[perm_t], size=(ctx.shape[1], n))
        grad_x = WT @ g
        # grad_values[k] = sum_b g[row_k, b] * x[col_k, b]  -- sampled at the sparsity pattern
        pattern = torch.sparse_csr_tensor(crow, col, torch.zeros_like(values), size=ctx.shape)
        grad_values = torch.sparse.sampled_addmm(pattern, g, x.t().contiguous(), beta=0.0).values()
        return grad_values, grad_x, None, None, None, None, None, None


class TrainableSpMM(torch.nn.Module):
    """The measured graph with a learnable positive gain per synapse; pattern and signs are buffers."""

    def __init__(self, W: sparse.csr_matrix, device: str = "cuda"):
        super().__init__()
        W = W.tocsr()
        W.sort_indices()
        self.shape = W.shape
        self.register_buffer("crow", torch.from_numpy(W.indptr.astype(np.int32)).to(device))
        self.register_buffer("col", torch.from_numpy(W.indices.astype(np.int32)).to(device))
        self.register_buffer("sign", torch.from_numpy(np.sign(W.data).astype(np.float32)).to(device))
        mag = np.abs(W.data).astype(np.float32).clip(1e-6)
        # softplus(raw) == mag at init, so the model starts exactly at the measured graph
        self.raw = torch.nn.Parameter(torch.from_numpy(np.log(np.expm1(mag).clip(1e-6))).to(device))
        # transpose pattern + the permutation that reorders values into it (for the backward pass)
        order = np.arange(W.nnz)
        WT = sparse.csr_matrix((order, W.indices, W.indptr), shape=W.shape).T.tocsr()
        self.register_buffer("crow_t", torch.from_numpy(WT.indptr.astype(np.int32)).to(device))
        self.register_buffer("col_t", torch.from_numpy(WT.indices.astype(np.int32)).to(device))
        self.register_buffer("perm_t", torch.from_numpy(WT.data.astype(np.int64)).to(device))

    def forward(self, x):
        values = self.sign * F.softplus(self.raw)
        return _SparseMM.apply(values, x, self.crow, self.col,
                               self.crow_t, self.col_t, self.perm_t, self.shape)


class FlyVL3(torch.nn.Module):
    def __init__(self, c: connectome.Connectome, W: sparse.csr_matrix, n_classes: int = 10,
                 img: int = 32, steps: int = 16, view: str = "all", device: str = "cuda"):
        """`view` is one of the pre-registered readout masks. Default "all": tracing activity at init
        shows the central brain receiving a signal ~1,000x weaker than the optic lobe
        (1.2e-5 vs 1.3e-2), so a central-only readout hands the optimiser a near-zero feature vector,
        LayerNorm amplifies it into noise, and training diverges (loss 17 at the first attempt).
        Reading every neuron keeps the optic lobe - where the signal actually is - in scope.
        Activity saturates by about step 15, hence steps=16."""
        super().__init__()
        self.device, self.steps = device, steps
        self.spmm = TrainableSpMM(W, device)
        tid = cell_type_ids(c)
        self.register_buffer("tid", torch.as_tensor(tid, device=device))
        nT = int(tid.max()) + 1
        r16 = c.types(["R1-6"])
        driven = r16[c.column[r16, 0] >= 0]
        self.register_buffer("driven", torch.as_tensor(driven, device=device))
        readout_idx = masks.build(c)[view]
        self.register_buffer("central", torch.as_tensor(readout_idx, device=device))
        # learned encoder: pixels -> photoreceptor drive (P7 showed the fixed retina under-drove the eye)
        self.encoder = torch.nn.Linear(img * img, len(driven)).to(device)
        self.tau_raw = torch.nn.Parameter(
            torch.full((nT,), float(np.log(np.expm1(0.04 - DT))), device=device))
        self.bias = torch.nn.Parameter(torch.zeros(nT, device=device))
        self.readout = torch.nn.Sequential(torch.nn.LayerNorm(len(readout_idx)),
                                           torch.nn.Linear(len(readout_idx), n_classes)).to(device)
        self.n = W.shape[0]

    def param_groups(self, lr_syn=3e-4, lr_other=1e-3):
        other = [self.tau_raw, self.bias, *self.encoder.parameters(), *self.readout.parameters()]
        return [{"params": [self.spmm.raw], "lr": lr_syn}, {"params": other, "lr": lr_other}]

    def forward(self, images: torch.Tensor):
        """images (B, img*img) -> logits (B, n_classes)."""
        B = images.shape[0]
        alpha = (DT / (DT + F.softplus(self.tau_raw)))[self.tid][:, None]
        bias = self.bias[self.tid][:, None]
        inp = self.encoder(images).t().contiguous()        # (n_driven, B), held over the sweep
        V = torch.zeros(self.n, B, device=self.device)
        acc = 0
        for _ in range(self.steps):
            drive = self.spmm(act(V)) + bias
            drive = drive.index_add(0, self.driven, inp)
            V = V + alpha * (drive - V)
            acc = acc + act(V[self.central])
        return self.readout((acc / self.steps).t())
