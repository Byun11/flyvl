"""FlyViG (PROTOCOL_D4_flyvig.md): a Vision GNN whose spatial graph stays dynamic (KNN in feature space, as in ViG)
while the fly connectome decides how feature channels exchange information.

Channels are 65 flyvis cell types x 3. Spatial message passing is per type (grouped), so types meet only in the
channel mixer, whose first layer is masked by the cell-type graph (type u reads itself and the types that synapse
onto it) and whose second layer is block-diagonal. Patch features enter only the photoreceptor types (R1-R8).
Each block keeps a per-type leaky state over video frames, s_t = (1 - a) s_{t-1} + a out_t.
Modes: vig (original dense ViG, input to every channel) | dense | dense_small | graph (mask = real / rewired / random).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PER, HID = 3, 12                     # channels per cell type, hidden units per type in the channel mixer
R_TYPES = tuple(f"R{i}" for i in range(1, 9))


def build_type_graph(path):
    """Save the flyvis cell-type graph: M[u, t] = 1 if type t synapses onto type u (off-diagonal)."""
    from . import flyvis_compat  # noqa: F401
    from flyvis import NetworkView
    c = NetworkView("flow/0000/000").init_network().connectome
    t = np.asarray([x.decode() if isinstance(x, bytes) else x for x in c.nodes.type[:]])
    names = sorted(set(t))
    ix = {k: i for i, k in enumerate(names)}
    M = np.zeros((len(names), len(names)), bool)
    M[np.vectorize(ix.get)(t[c.edges.target_index[:]]), np.vectorize(ix.get)(t[c.edges.source_index[:]])] = True
    np.fill_diagonal(M, False)
    np.savez(path, names=np.asarray(names), M=M)


def rewire(M, seed, swaps_per_edge=20):
    """Directed double-edge swaps: keeps every type's in-degree and out-degree, no self-loops, no duplicates."""
    g = np.random.default_rng(seed)
    A = M.copy()
    E = np.argwhere(A)
    target, done, tries = swaps_per_edge * len(E), 0, 0
    while done < target and tries < 50 * target:
        tries += 1
        i, j = g.integers(0, len(E), 2)
        (u1, t1), (u2, t2) = E[i], E[j]
        if u1 == u2 or t1 == t2 or u1 == t2 or u2 == t1 or A[u1, t2] or A[u2, t1]:
            continue
        A[u1, t1] = A[u2, t2] = False
        A[u1, t2] = A[u2, t1] = True
        E[i], E[j] = (u1, t2), (u2, t1)
        done += 1
    return A


def random_graph(M, seed):
    """Same number of off-diagonal edges, placed uniformly at random."""
    g = np.random.default_rng(seed)
    n = M.shape[0]
    off = np.flatnonzero(~np.eye(n, dtype=bool))
    A = np.zeros(n * n, bool)
    A[g.choice(off, int(M.sum()), replace=False)] = True
    return A.reshape(n, n)


class GLinear(nn.Module):
    """Grouped linear on the channel axis, (..., G*i) -> (..., G*o), run as one dense matmul with a block-diagonal
    mask (65 tiny groups are slow as an einsum). G = 1 is an ordinary dense layer. `w` / `b` hold only the
    active weights; the dense matrix is rebuilt each call."""

    def __init__(self, G, i, o):
        super().__init__()
        self.G, self.i, self.o = G, i, o
        self.w = nn.Parameter(torch.randn(G, i, o) / math.sqrt(i))
        self.b = nn.Parameter(torch.zeros(G, o))

    def forward(self, x):
        if self.G == 1:
            return F.linear(x, self.w[0].t(), self.b[0])
        W = torch.block_diag(*self.w.transpose(1, 2))                                          # (G*o, G*i)
        return F.linear(x, W, self.b.reshape(-1))


class BN(nn.GroupNorm):
    """Norm on (B, N, C) with one group per cell type (3 channels x all nodes of one sample): no statistics are
    shared across types or across the batch. Batch norm failed here: channels that are still ~constant (types the
    input has not reached yet) get ~0 running variance, and at eval time tiny deviations blow up (smoke test)."""

    def __init__(self, C):
        # eps = 1: a near-constant group is passed through (x - mean) instead of being blown up by 1/sqrt(eps);
        # with eps 1e-5 those groups multiplied gradients by ~316 per norm and the norm overflowed (smoke test)
        super().__init__(C // PER, C, eps=1.0)

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class Grapher(nn.Module):
    """ViG Grapher: fc -> dynamic KNN (k, on all channels) -> max-relative graph conv -> fc. Grouped when G > 1."""

    def __init__(self, C, G, k):
        super().__init__()
        g = C // G
        self.k, self.G = k, G
        self.fc1, self.bn1 = GLinear(G, g, g), BN(C)
        self.gc, self.bn2 = GLinear(G, 2 * g, g), BN(C)
        self.fc2, self.bn3 = GLinear(G, g, g), BN(C)

    def forward(self, x):
        B, N, C = x.shape
        y = self.bn1(self.fc1(x))
        with torch.no_grad():
            f = F.normalize(y.float(), dim=-1)
            idx = (f @ f.transpose(1, 2)).topk(self.k, dim=-1).indices                      # (B, N, k), includes self
        nb = y.reshape(B * N, C)[(idx + torch.arange(B, device=x.device)[:, None, None] * N).reshape(-1)]
        rel = (nb.reshape(B, N, self.k, C) - y[:, :, None]).amax(2)
        z = torch.cat([y.reshape(B, N, self.G, -1), rel.reshape(B, N, self.G, -1)], -1).reshape(B, N, 2 * C)
        z = F.gelu(self.bn2(self.gc(z)))
        return x + self.bn3(self.fc2(z))


class DenseFFN(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        self.l1, self.l2, self.bn = nn.Linear(C, H), nn.Linear(H, C), BN(C)

    def forward(self, x):
        return x + self.bn(self.l2(F.gelu(self.l1(x))))


class TypeFFN(nn.Module):
    """Type u's HID hidden units read the channels of u and of every type t with mask[u, t]; they write only u."""

    def __init__(self, mask):
        super().__init__()
        G = mask.shape[0]
        m = torch.as_tensor(np.kron(mask | np.eye(G, dtype=bool), np.ones((HID, PER), bool)), dtype=torch.float32)
        self.register_buffer("m", m)                                                            # (G*HID, G*PER)
        fan = m.sum(1, keepdim=True)
        self.w1 = nn.Parameter(torch.randn(m.shape) / fan.sqrt())
        self.b1 = nn.Parameter(torch.zeros(m.shape[0]))
        self.l2, self.bn = GLinear(G, HID, PER), BN(G * PER)

    def forward(self, x):
        return x + self.bn(self.l2(F.gelu(F.linear(x, self.w1 * self.m, self.b1))))

    def active_params(self):
        return int(self.m.sum()) + self.b1.numel() + self.l2.w.numel() + self.l2.b.numel()


class Stem(nn.Module):
    """(B, cin, H, W) -> (B, 256, cout) on a 16 x 16 node grid."""

    def __init__(self, cin, cout, H):
        super().__init__()
        n_down = int(round(math.log2(H // 16)))
        layers, c = [], cin
        for i, co in enumerate((32, 64)):
            layers += [nn.Conv2d(c, co, 3, 2 if i < n_down else 1, 1), nn.BatchNorm2d(co), nn.GELU()]
            c = co
        layers += [nn.Conv2d(c, cout, 3, 1, 1), nn.BatchNorm2d(cout)]
        self.net = nn.Sequential(*layers)
        self.pos = nn.Parameter(torch.randn(1, 256, cout) * 0.02)

    def forward(self, x):
        return self.net(x).flatten(2).transpose(1, 2) + self.pos


class FlyViG(nn.Module):
    def __init__(self, names, mode, mask=None, cin=1, H=64, n_cls=16, depth=6, k=9, hid_small=None, inp=None, space=None):
        """inp: "r" = patch features enter the photoreceptor channels only, "all" = every channel.
        space: "grouped" = spatial message passing per cell type, "dense" = the original ViG Grapher.
        Defaults: vig -> all / dense, every other mode -> r / grouped (the registered D4 design, variant V1)."""
        super().__init__()
        names = list(names)
        G = len(names)
        C = G * PER
        self.mode, self.C = mode, C
        inp = inp or ("all" if mode == "vig" else "r")
        space = space or ("dense" if mode == "vig" else "grouped")
        self.r_only = inp == "r"
        r = [names.index(t) * PER + j for t in R_TYPES for j in range(PER)]
        self.register_buffer("ridx", torch.as_tensor(r))
        keep = torch.ones(C, dtype=torch.bool)
        keep[r] = False
        self.register_buffer("readout", torch.nonzero(keep if self.r_only else torch.ones(C, dtype=torch.bool))[:, 0])
        self.stem = Stem(cin, len(r) if self.r_only else C, H)
        self.graphers = nn.ModuleList(Grapher(C, G if space == "grouped" else 1, k) for _ in range(depth))
        if mode == "graph":
            self.ffns = nn.ModuleList(TypeFFN(mask) for _ in range(depth))
        else:
            h = {"vig": 4 * C, "dense": 4 * C, "dense_small": hid_small}[mode]
            self.ffns = nn.ModuleList(DenseFFN(C, h) for _ in range(depth))
        self.alpha = nn.Parameter(torch.zeros(depth, G))                                         # sigmoid(0) = 0.5
        self.head = nn.Sequential(nn.LayerNorm(len(self.readout)), nn.Linear(len(self.readout), n_cls))

    def forward(self, x):
        """x (B, T, cin, H, W) -> logits (B, n_cls)."""
        B, T = x.shape[:2]
        z = self.stem(x.flatten(0, 1))                                                           # (B*T, 256, c)
        if self.r_only:
            h = z.new_zeros(z.shape[0], z.shape[1], self.C)
            h[..., self.ridx] = z
        else:
            h = z
        for l, (gr, ff) in enumerate(zip(self.graphers, self.ffns)):
            out = ff(gr(h)).reshape(B, T, *h.shape[1:])
            a = torch.sigmoid(self.alpha[l]).repeat_interleave(PER)
            s, states = out[:, 0], [out[:, 0]]
            for t in range(1, T):
                s = (1 - a) * s + a * out[:, t]
                states.append(s)
            h = torch.stack(states, 1).flatten(0, 1)
        last = h.reshape(B, T, *h.shape[1:])[:, -1]
        return self.head(last.mean(1)[:, self.readout])
