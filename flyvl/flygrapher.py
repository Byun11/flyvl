"""D5 (PROTOCOL_D5_flygrapher.md): the Grapher of a Vision GNN replaced by K whole-MaleCNS graph processors.

One fly = the whole MaleCNS connectome (166,700 neurons, 25,582,938 connections) run as a recurrent graph processor
with the ViG Grapher's contract, N patches x D -> N x D:
  P_in   rank-5 patch-local linear map: one drive per input class (L1, L2, L3, R7, R8) per patch, shared by all patches
  brain  V <- V + a * (-V + g_post * W (g_pre * f(V)) + E u), T steps from V = 0. W is fixed (topology, sign, share of
         the neuron's input synapses); g_pre, g_post and the time constants are learned per cell-type group.
  P_out  fixed pooling Q (each neuron's visual-input footprint over the patches, computed from the graph), then one
         linear map of rank <= 16 shared by all patches.
K flies per block share only W; their outputs are averaged (no fusion parameters).
Controls with the same interface: rewired graphs (Rewired-L / Rewired-M / Random), Grid (a fixed local graph over the
patches), Bypass (no processor), and one brain with K signal channels whose physiology is shared (replication vs width).
The brain always runs in fp32 (autocast off); all K flies of a block go through one sparse product of width K * batch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch.utils.checkpoint import checkpoint

from . import connectome
from .flyvig import BN, DenseFFN, Grapher, Stem

SRC = connectome.CONNECTOME
D5 = connectome.DATA_ROOT / "d5"
INPUT_TYPES = ("L1", "L2", "L3", "R7", "R8")
N_SIDE = 16
N_PATCH = N_SIDE * N_SIDE
N_IN = len(INPUT_TYPES)
T_STEPS, DT, TAU0, RANK = 16, 0.02, 0.04, 16
GRID_UNITS = 190                      # 190^2 = 36,100 mixing weights ~ the fly's 3 x 11,881 physiology parameters
GLOBAL_EFF_PATCHES = 64               # footprint wider than this -> "global" tile (Rewired-L)
INV_SP1 = math.log(math.expm1(1.0))   # softplus^-1(1)
INV_SP_TAU = math.log(math.expm1(TAU0 - DT))


def act(v):
    return 0.5 * torch.tanh(2 * v)


# ------------------------------------------------------------------------------------------------ connectome
@dataclass
class MaleCNS:
    W: sparse.csr_matrix              # rows = post, cols = pre; neurons in D5 order (superclass, eye, column, type)
    group: np.ndarray                 # cell-type group per neuron (named type, or "untyped:<superclass>")
    superclass: np.ndarray
    scside: np.ndarray                # superclass x side id (the matched-shuffle block key)
    inputs: np.ndarray                # input neurons (L1 L2 L3 R7 R8 with a column)
    in_class: np.ndarray              # 0..4
    in_patch: np.ndarray              # 0..255
    n_groups: int
    in_pos: np.ndarray = None         # (n_inputs, 2) visual position (phi, theta) in [-1, 1], used by D6's retina
    info: dict = field(default_factory=dict)

    @property
    def n(self):
        return self.W.shape[0]


def assign_patches(col):
    """Input neuron columns (eye, h1, h2) -> patch 0..255. Left eye -> patch columns 0-7, right eye -> 8-15, same
    coordinates as stimulus.Retina. A patch left without columns takes the nearest column of a patch that has >= 3."""
    eye, h1, h2 = col[:, 0], col[:, 1].astype(float), col[:, 2].astype(float)
    hor, ver = (h1 - h2) * math.sqrt(3) / 2, (h1 + h2) / 2
    f = np.empty_like(hor)
    for e in (0, 1):
        m = eye == e
        f[m] = (hor[m] - hor[m].min()) / (hor[m].max() - hor[m].min())
    phi = (0.05 + 0.95 * f) * np.where(eye == 0, -1.0, 1.0)
    theta = 2 * (ver - ver.min()) / (ver.max() - ver.min()) - 1
    cols, first, inv = np.unique(col, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    cphi, cth = phi[first], theta[first]
    cx = np.clip(np.floor((cphi + 1) / 2 * N_SIDE), 0, N_SIDE - 1).astype(int)
    cy = np.clip(np.floor((1 - cth) / 2 * N_SIDE), 0, N_SIDE - 1).astype(int)
    patch = cy * N_SIDE + cx
    moved, empty0 = 0, int((np.bincount(patch, minlength=N_PATCH) == 0).sum())
    for p in np.flatnonzero(np.bincount(patch, minlength=N_PATCH) == 0):
        counts = np.bincount(patch, minlength=N_PATCH)
        donors = np.flatnonzero(counts[patch] >= 3)
        if len(donors) == 0:
            break
        px, py = (p % N_SIDE + 0.5) / N_SIDE * 2 - 1, 1 - (p // N_SIDE + 0.5) / N_SIDE * 2
        j = donors[np.argmin((cphi[donors] - px) ** 2 + (cth[donors] - py) ** 2)]
        patch[j] = p
        moved += 1
    counts = np.bincount(patch, minlength=N_PATCH)
    info = {"columns": int(len(cols)), "empty_patches_before": empty0, "columns_moved": moved, "pos": np.stack([phi, theta], 1),
            "empty_patches_after": int((counts == 0).sum()), "columns_per_patch_min": int(counts.min()),
            "columns_per_patch_median": float(np.median(counts)), "columns_per_patch_max": int(counts.max())}
    return patch[inv], info


def load_malecns() -> MaleCNS:
    meta = np.load(SRC / "brain.npz")
    W = sparse.load_npz(SRC / "weights.npz").tocsr().astype(np.float32)
    assert W.shape[0] == connectome.EXPECTED_NEURONS and W.nnz == connectome.EXPECTED_EDGES
    ct, sc, side = meta["cell_type"].astype(str), meta["superclass"].astype(str), meta["side"].astype(str)
    col = np.load(D5 / "input_columns.npy")                       # written by scripts/d5_coverage.py
    key = np.where(ct == "", np.char.add("untyped:", sc), ct)
    _, group = np.unique(key, return_inverse=True)
    _, sid = np.unique(sc, return_inverse=True)
    _, scside = np.unique(np.char.add(np.char.add(sc, "|"), side), return_inverse=True)
    order = np.lexsort((group, col[:, 2], col[:, 1], col[:, 0], sid))   # locality order for the sparse products
    W = W[order][:, order].tocsr()
    W.sort_indices()
    ct, sc, col, group, scside = ct[order], sc[order], col[order], group[order], scside[order]
    inputs = np.flatnonzero(np.isin(ct, INPUT_TYPES) & (col[:, 0] >= 0))
    in_class = np.array([INPUT_TYPES.index(t) for t in ct[inputs]])
    in_patch, info = assign_patches(col[inputs])
    in_pos = info.pop("pos")
    info.update(n_inputs=int(len(inputs)), inputs_per_class={t: int((in_class == i).sum()) for i, t in enumerate(INPUT_TYPES)})
    return MaleCNS(W, group.astype(np.int64), sc, scside.astype(np.int64), inputs, in_class, in_patch,
                   int(group.max()) + 1, info=info, in_pos=in_pos)


def to_csr(M, device):
    M = M.tocsr().astype(np.float32)
    M.sort_indices()
    return torch.sparse_csr_tensor(torch.as_tensor(M.indptr.astype(np.int32)), torch.as_tensor(M.indices.astype(np.int32)),
                                   torch.as_tensor(M.data), size=M.shape, device=device)


@torch.no_grad()
def footprint(W, brain: MaleCNS, T=T_STEPS, device="cuda"):
    """F[i, p]: probability that a walk backwards along neuron i's inputs (each step picks a presynaptic partner with
    probability = its share of i's input synapses) reaches, within T steps, an input neuron that sits in patch p.
    Input neurons absorb (S2's visual input connectivity, split by source patch). Returns (n, 256) on `device`."""
    n = W.shape[0]
    P = abs(W).tocsr()
    is_in = np.zeros(n, bool)
    is_in[brain.inputs] = True
    S = sparse.csr_matrix((np.ones(len(brain.inputs), np.float32), (brain.inputs, brain.in_patch)), shape=(n, N_PATCH))
    b = torch.as_tensor((P @ S).toarray(), device=device)
    b[torch.as_tensor(is_in, device=device)] = 0
    keep = sparse.diags((~is_in).astype(np.float32))
    Pna = to_csr(keep @ P @ keep, device)
    v, acc = b, b.clone()
    for _ in range(T - 1):
        v = Pna @ v
        acc += v
    return acc


@torch.no_grad()
def pooling(Fp, brain: MaleCNS, readout):
    """Q (256, n_readout): Q[p, i] = F[i, p] / sum_j F[j, p] over all readout neurons j (fixed, not learned): patch p
    reads the activity of the neurons its visual input reaches, weighted by how much of it reaches each one.
    (Protocol appendix E: the registered per-(patch, type) mean gave every type full weight at every patch, including
    types the patch barely reaches, and spread the output almost uniformly over the patches.)"""
    Fr = Fp[readout]
    Q = Fr / Fr.sum(0, keepdim=True).clamp_min(1e-30)
    return Q.T.contiguous()


class BrainGraph:
    """Everything a fly needs that is fixed by the graph (kept out of the state dict: shared by all flies)."""

    def __init__(self, W, brain: MaleCNS, readout="all", T=T_STEPS, device="cuda"):
        self.n, self.n_groups, self.T = W.shape[0], brain.n_groups, T
        self.W, self.WT = to_csr(W, device), to_csr(W.T, device)
        self.inputs = torch.as_tensor(brain.inputs, device=device)
        self.in_idx = torch.as_tensor(brain.in_patch * N_IN + brain.in_class, device=device)
        self.group = torch.as_tensor(brain.group, device=device)
        is_in = np.zeros(self.n, bool)
        is_in[brain.inputs] = True
        keep = ~is_in if readout == "all" else (~is_in & (brain.superclass == "ol_intrinsic"))   # "ol" = ablation O2
        ro = np.flatnonzero(keep)
        self.Fp = footprint(W, brain, T, device)
        self.readout = torch.as_tensor(ro, device=device)
        self.ro_group = self.group[self.readout]
        self.Q = pooling(self.Fp, brain, ro)
        self.ro_superclass = brain.superclass[ro]


# ------------------------------------------------------------------------------------------------ control graphs
def rewire_blocks(W, gkey, seed, max_iter=300):
    """Keep every connection's presynaptic neuron, weight and sign; permute postsynaptic ends among connections in the
    same block (block = gkey[pre], gkey[post]). In- and out-degree stay exact. Duplicates and self-loops are then
    removed by swapping posts inside the block; leftovers after max_iter swap with any connection (reported).
    Rows are rescaled to the original sum of |w| (as controls.rewire). No KC rule: D5 uses the raw graph."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    coo = W.tocoo()
    pre, post0, w = coo.col.astype(np.int64), coo.row.astype(np.int64), coo.data.copy()
    G = int(gkey.max()) + 1
    block = gkey[pre] * G + gkey[post0]
    o1 = np.argsort(block, kind="stable")
    o2 = np.lexsort((rng.random(len(pre)), block))
    post = post0.copy()
    post[o1] = post0[o2]
    sb = block[o1]
    starts = np.r_[0, np.flatnonzero(np.diff(sb)) + 1]
    bstart = np.empty(len(pre), np.int64)
    bcount = np.empty(len(pre), np.int64)
    lens = np.diff(np.r_[starts, len(sb)])
    bstart[o1] = np.repeat(starts, lens)
    bcount[o1] = np.repeat(lens, lens)

    def bad(post):
        key = pre * n + post
        srt = np.argsort(key, kind="stable")
        dup = np.zeros(len(key), bool)
        dup[srt[1:]] = key[srt[1:]] == key[srt[:-1]]
        return np.flatnonzero(dup | (pre == post))

    history, fallback = [], 0
    for it in range(max_iter + 100):
        b = bad(post)
        history.append(len(b))
        if len(b) == 0:
            break
        if it < max_iter:
            partner = o1[bstart[b] + (rng.random(len(b)) * bcount[b]).astype(np.int64)]
        else:
            fallback += 1
            partner = rng.integers(0, len(pre), len(b))
        ok = ~np.isin(partner, b) & (partner != b)
        _, first = np.unique(partner[ok], return_index=True)
        a, p = b[ok][first], partner[ok][first]
        post[a], post[p] = post[p].copy(), post[a].copy()
    else:
        raise RuntimeError(f"rewiring did not converge: {history[-5:]}")
    R = sparse.csr_matrix((w, (post, pre)), shape=W.shape, dtype=np.float32)
    target = np.asarray(abs(W).sum(1)).ravel()
    have = np.asarray(abs(R).sum(1)).ravel()
    R = (sparse.diags(np.divide(target, have, out=np.zeros_like(target), where=have > 0).astype(np.float32)) @ R).tocsr()
    R.sort_indices()
    indeg_ok = bool((np.bincount(post, minlength=n) == np.bincount(post0, minlength=n)).all())
    info = {"seed": seed, "blocks": int(len(starts)), "changed_fraction": float((post != post0).mean()),
            "fix_iterations": len(history), "bad_initial": history[0], "fallback_iterations": fallback,
            "edges_violating_block": int((gkey[pre] * G + gkey[post] != block).sum()), "nnz": int(R.nnz),
            "in_degree_exact": indeg_ok, "self_loops": int((pre == post).sum())}
    return R, info


def tiles_from_footprint(Fp, tile_side=8):
    """Tile of each neuron for Rewired-L: tile of its footprint centroid (tile_side x tile_side tiles over the 16 x 16
    patches), 'global' if the footprint is wider than GLOBAL_EFF_PATCHES effective patches, 'none' if it has none."""
    F_ = Fp.float()
    tot = F_.sum(1)
    pn = F_ / tot.clamp_min(1e-30)[:, None]
    ent = -(pn * torch.log(pn.clamp_min(1e-30))).sum(1)
    eff = torch.exp(ent)
    r = torch.arange(N_PATCH, device=F_.device) // N_SIDE
    c = torch.arange(N_PATCH, device=F_.device) % N_SIDE
    cy, cx = (pn * r).sum(1), (pn * c).sum(1)
    per = N_SIDE // tile_side
    tile = (torch.clamp((cy / per).floor(), max=tile_side - 1) * tile_side + torch.clamp((cx / per).floor(), max=tile_side - 1)).long()
    n_t = tile_side * tile_side
    tile = torch.where(eff > GLOBAL_EFF_PATCHES, torch.full_like(tile, n_t), tile)
    tile = torch.where(tot <= 0, torch.full_like(tile, n_t + 1), tile)
    return tile.cpu().numpy(), n_t + 2


def build_graph(brain: MaleCNS, kind, seed, device="cuda", tile_side=8):
    """real | rewired_l | rewired_m | random. Control graphs are cached in <FLYVL_DATA>/d5/graphs."""
    if kind == "real":
        return brain.W, {"kind": "real"}
    f = D5 / "graphs" / f"{kind}{'' if kind != 'rewired_l' or tile_side == 8 else f'_t{tile_side}'}_s{seed}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        R = sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=brain.W.shape)
        return R, d["info"].item()
    if kind == "random":
        gkey = np.zeros(brain.n, np.int64)
    elif kind == "rewired_m":
        gkey = brain.scside
    elif kind == "rewired_l":
        tile, n_t = tiles_from_footprint(footprint(brain.W, brain, T_STEPS, device), tile_side)
        gkey = brain.scside * n_t + tile
    else:
        raise ValueError(kind)
    R, info = rewire_blocks(brain.W, gkey, seed)
    info["kind"], info["tile_side"] = kind, tile_side
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, data=R.data, indices=R.indices, indptr=R.indptr, info=np.array(info, dtype=object))
    return R, info


def grid_k(graph: BrainGraph):
    """Registered degree of the Grid control: E = Q F (output patch <- input patch influence of the real fly); k = median
    over output patches of the number of input patches giving >= 1% of that patch's influence."""
    E = graph.Q @ graph.Fp[graph.readout]
    return int(torch.median((E >= 0.01 * E.sum(1, keepdim=True)).sum(1)).item())


def grid_adjacency(k):
    """Row-normalised fixed local graph over the 16 x 16 patches: each patch reads its k nearest patches (itself included)."""
    r = torch.arange(N_PATCH) // N_SIDE
    c = torch.arange(N_PATCH) % N_SIDE
    d = (r[:, None] - r[None]) ** 2 + (c[:, None] - c[None]) ** 2
    d = d.float() + torch.arange(N_PATCH)[None].float() * 1e-6             # deterministic tie break
    idx = d.topk(k, largest=False).indices
    A = torch.zeros(N_PATCH, N_PATCH).scatter_(1, idx, 1.0 / k)
    return A


# ------------------------------------------------------------------------------------------------ graphers
class _SpMM(torch.autograd.Function):
    """y = W x with a fixed W; the backward pass is one product with the fixed transpose."""

    @staticmethod
    def forward(ctx, W, WT, x):
        ctx.WT = WT
        return W @ x

    @staticmethod
    def backward(ctx, g):
        return None, None, ctx.WT @ g.contiguous()


class PatchIn(nn.Module):
    """P_in: per head, one linear map D -> 5 applied to every patch (rank 5, no hidden layer, no nonlinearity)."""

    def __init__(self, K, D):
        super().__init__()
        self.a = nn.Parameter(torch.randn(K, N_IN, D) / math.sqrt(D))
        self.b = nn.Parameter(torch.zeros(K, N_IN))

    def forward(self, xh):                                                  # (B, 256, D) -> (K, B, 256, 5)
        return torch.einsum("bpd,kcd->kbpc", xh, self.a) + self.b[:, None, None, :]


class FlyGrapher(nn.Module):
    """K whole-MaleCNS heads (or, with tied=True, one brain carrying K signal channels with shared physiology)."""

    def __init__(self, graph: BrainGraph, D, K, tied=False, checkpointing=True):
        super().__init__()
        self.graph, self.K, self.tied, self.ckpt = graph, K, tied, checkpointing
        P, G = (1 if tied else K), graph.n_groups
        self.p_in = PatchIn(K, D)
        self.g_pre = nn.Parameter(torch.full((P, G), INV_SP1))
        self.g_post = nn.Parameter(torch.full((P, G), INV_SP1))
        self.tau = nn.Parameter(torch.full((P, G), INV_SP_TAU))
        self.v_out = nn.Parameter(torch.randn(K, G, RANK) / math.sqrt(RANK))
        self.u_out = nn.Parameter(torch.randn(K, RANK, D) / math.sqrt(RANK))
        self.c_out = nn.Parameter(torch.zeros(K, D))
        self.brain_off, self.shuffle, self.keep_state = False, None, False
        self.last = {}

    def physiology(self):
        return [self.g_pre, self.g_post, self.tau]

    def _brain(self, u, g_pre, g_post, tau):
        g, K = self.graph, self.K
        KB = u.shape[1]
        B = KB // K
        gp = F.softplus(g_pre)[:, g.group].T.expand(-1, K)[:, :, None] if self.tied else F.softplus(g_pre)[:, g.group].T[:, :, None]
        gq = F.softplus(g_post)[:, g.group].T.expand(-1, K)[:, :, None] if self.tied else F.softplus(g_post)[:, g.group].T[:, :, None]
        al = (DT / (DT + F.softplus(tau)))[:, g.group].T
        al = (al.expand(-1, K) if self.tied else al)[:, :, None]
        V = torch.zeros(g.n, KB, device=u.device)
        acc = torch.zeros_like(V)
        for _ in range(g.T):
            s = (act(V).view(g.n, K, B) * gp).view(g.n, KB)
            drive = (_SpMM.apply(g.W, g.WT, s).view(g.n, K, B) * gq).view(g.n, KB).index_add(0, g.inputs, u)
            V = (V.view(g.n, K, B) + al * (drive - V).view(g.n, K, B)).view(g.n, KB)
            acc = acc + act(V)
        return acc / g.T, V

    def forward(self, xh):
        g, K = self.graph, self.K
        B = xh.shape[0]
        with torch.autocast("cuda", enabled=False):
            U = self.p_in(xh.float())                                        # (K, B, 256, 5)
            u = U.reshape(K, B, N_PATCH * N_IN)[:, :, g.in_idx].permute(2, 0, 1).reshape(-1, K * B)
            if self.ckpt and torch.is_grad_enabled():
                abar, V = checkpoint(self._brain, u, self.g_pre, self.g_post, self.tau, use_reentrant=False)
            else:
                abar, V = self._brain(u, self.g_pre, self.g_post, self.tau)
            abar = abar[g.readout]
            if self.keep_state:
                self.last = {"abar": abar.detach(), "final": act(V[g.readout]).detach()}
            if self.shuffle is not None:
                abar = abar[self.shuffle]
            if self.brain_off:
                abar = torch.zeros_like(abar)
            Y = 0
            for k in range(K):
                vg = self.v_out[k][g.ro_group]                                # (n_ro, RANK)
                H = (abar[:, k * B:(k + 1) * B, None] * vg[:, None, :]).reshape(len(g.readout), B * RANK)
                z = (g.Q @ H).view(N_PATCH, B, RANK).permute(1, 0, 2)
                Y = Y + z @ self.u_out[k] + self.c_out[k]
            return Y / K


class GridGrapher(nn.Module):
    """Same interface and dynamics as a fly, but the processor is a fixed local graph over the 256 patches with
    GRID_UNITS units per patch (units 0-4 receive P_in, the rest are read out) and learned channel mixing."""

    def __init__(self, A, D, K, T=T_STEPS):
        super().__init__()
        h, self.K, self.T = GRID_UNITS, K, T
        self.register_buffer("A", A)
        self.p_in = PatchIn(K, D)
        self.w_ch = nn.Parameter(torch.randn(K, h, h) / math.sqrt(h))
        self.tau = nn.Parameter(torch.full((K, h), INV_SP_TAU))
        self.v_out = nn.Parameter(torch.randn(K, h - N_IN, RANK) / math.sqrt(RANK))
        self.u_out = nn.Parameter(torch.randn(K, RANK, D) / math.sqrt(RANK))
        self.c_out = nn.Parameter(torch.zeros(K, D))
        self.brain_off = False

    def physiology(self):
        return [self.w_ch, self.tau]

    def _run(self, U, w_ch, tau):
        K, B = U.shape[:2]
        pad = F.pad(U, (0, GRID_UNITS - N_IN))
        al = (DT / (DT + F.softplus(tau)))[:, None, None, :]
        V = torch.zeros(K, B, N_PATCH, GRID_UNITS, device=U.device)
        acc = torch.zeros_like(V)
        for _ in range(self.T):
            m = torch.einsum("qp,kbph->kbqh", self.A, act(V))
            drive = torch.einsum("kbqh,khj->kbqj", m, w_ch) + pad
            V = V + al * (drive - V)
            acc = acc + act(V)
        return acc[..., N_IN:] / self.T

    def forward(self, xh):
        with torch.autocast("cuda", enabled=False):
            U = self.p_in(xh.float())                                        # (K, B, 256, 5)
            if torch.is_grad_enabled():
                abar = checkpoint(self._run, U, self.w_ch, self.tau, use_reentrant=False)
            else:
                abar = self._run(U, self.w_ch, self.tau)
            if self.brain_off:
                abar = torch.zeros_like(abar)
            z = torch.einsum("kbph,khr->kbpr", abar, self.v_out)
            return (torch.einsum("kbpr,krd->kbpd", z, self.u_out) + self.c_out[:, None, None, :]).mean(0)


class BypassGrapher(nn.Module):
    """P_in -> P_out with no processor: the 5 drives of each patch go straight to the rank-16 output map."""

    def __init__(self, D, K):
        super().__init__()
        self.p_in = PatchIn(K, D)
        self.v_out = nn.Parameter(torch.randn(K, N_IN, RANK) / math.sqrt(RANK))
        self.u_out = nn.Parameter(torch.randn(K, RANK, D) / math.sqrt(RANK))
        self.c_out = nn.Parameter(torch.zeros(K, D))

    def physiology(self):
        return []

    def forward(self, xh):
        with torch.autocast("cuda", enabled=False):
            U = self.p_in(xh.float())
            z = torch.einsum("kbpc,kcr->kbpr", U, self.v_out)
            return (torch.einsum("kbpr,krd->kbpd", z, self.u_out) + self.c_out[:, None, None, :]).mean(0)


# ------------------------------------------------------------------------------------------------ network
class D5Net(nn.Module):
    """D4's ViG backbone (stem, 6 blocks, D = 195, ViG FFN, head) with the Grapher of every block chosen by `cond`:
    vig (original dynamic KNN Grapher) | none | bypass | grid | fly (graph given) — the last three with K heads."""

    def __init__(self, cond, K=1, cin=3, H=32, n_cls=100, graph=None, grid_A=None, tied=False, depth=6, D=195):
        super().__init__()
        self.cond, self.depth = cond, depth
        self.stem = Stem(cin, D, H)
        if cond == "vig":
            self.graphers = nn.ModuleList(Grapher(D, 1, 9) for _ in range(depth))
        elif cond == "none":
            self.graphers = None
        else:
            make = {"fly": lambda: FlyGrapher(graph, D, K, tied), "grid": lambda: GridGrapher(grid_A, D, K),
                    "bypass": lambda: BypassGrapher(D, K)}[cond]
            self.graphers = nn.ModuleList(make() for _ in range(depth))
            self.norm_in = nn.ModuleList(BN(D) for _ in range(depth))
            self.norm_out = nn.ModuleList(BN(D) for _ in range(depth))
        self.ffns = nn.ModuleList(DenseFFN(D, 4 * D) for _ in range(depth))
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, n_cls))
        self.capture = None

    def forward(self, x):
        """x (B, 1, cin, H, W) -> logits. (Video, stage 2, is added separately.)"""
        h = self.stem(x[:, 0])
        for l in range(self.depth):
            if self.cond == "vig":
                h = self.graphers[l](h)
            elif self.graphers is not None:
                xh = self.norm_in[l](h)
                y = self.graphers[l](xh)
                if self.capture is not None:
                    self.capture.append((xh.detach().float(), y.detach().float()))
                h = h + self.norm_out[l](y)
            h = self.ffns[l](h)
        return self.head(h.mean(1))

    def branch_params(self):
        """(processor-internal parameters without weight decay, all others)."""
        phys = [p for g in (self.graphers or []) if hasattr(g, "physiology") for p in g.physiology()]
        ids = {id(p) for p in phys}
        return phys, [p for p in self.parameters() if id(p) not in ids]


@torch.no_grad()
def calibrate_output_scale(model: D5Net, x):
    """Label-free, identical for fly / grid / bypass: rescale each block's rank-16 output map so the branch output has
    unit standard deviation on one batch at initialisation (branches start with the same weight whatever the graph)."""
    if model.cond in ("vig", "none"):
        return []
    scales = []
    for l in range(model.depth):
        model.capture = []
        model(x)
        y = model.capture[l][1]
        s = float(y.std())
        model.graphers[l].u_out.mul_(1.0 / max(s, 1e-12))
        scales.append(s)
    model.capture = None
    return scales
