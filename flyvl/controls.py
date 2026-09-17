"""Control graphs, all rewired FROM the effective FlyVL graph (effective_W), never from raw MaleCNS.

Rewiring keeps, relative to W_eff:
  * the same neurons and the same number of edges (no duplicates are created),
  * every neuron's in-degree and out-degree (configuration model: post endpoints are permuted
    across edges, and each edge keeps its presynaptic neuron, weight and therefore sign),
  * each neuron's total |input| (rows are rescaled to the W_eff row sums of |w|),
  * the KC constraint: no KC -> KC edge is created (W_eff has none),
  * no self-loops (W_eff has 101; the rewired graphs have 0 — reported, negligible).
global_shuffle: posts permuted across all edges.
matched_shuffle: posts permuted only among edges in the same (pre group, post group) block,
  group = superclass x side, so region / superclass / side wiring statistics are preserved.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .connectome import Connectome, kc_mask


def _blocks(c: Connectome, pre: np.ndarray, post: np.ndarray, matched: bool) -> np.ndarray:
    if not matched:
        return np.zeros(len(pre), np.int64)
    _, group = np.unique(np.char.add(np.char.add(c.superclass.astype(str), "|"), c.side.astype(str)),
                         return_inverse=True)
    return group[pre].astype(np.int64) * (group.max() + 1) + group[post]


def rewire(W_eff: sparse.csr_matrix, c: Connectome, seed: int, matched: bool, max_iter: int = 500) -> tuple:
    rng = np.random.default_rng(seed)
    n = W_eff.shape[0]
    coo = W_eff.tocoo()
    pre, post, w = coo.col.astype(np.int64), coo.row.astype(np.int64), coo.data.copy()
    block = _blocks(c, pre, post, matched)

    # permute post endpoints within each block
    order = np.argsort(block, kind="stable")
    bounds = np.flatnonzero(np.diff(block[order])) + 1
    new_post = post.copy()
    for seg in np.split(order, bounds):
        new_post[seg] = post[rng.permutation(seg)]
    post = new_post

    kc = kc_mask(c)
    segs = {b: s for b, s in zip(block[order][np.r_[0, bounds]], np.split(order, bounds))}

    def bad_mask(post):
        key = pre * n + post
        srt = np.argsort(key, kind="stable")
        dup = np.zeros(len(key), bool)
        dup[srt[1:]] = key[srt[1:]] == key[srt[:-1]]            # every repeat after the first
        return dup | (kc[pre] & kc[post]) | (pre == post)

    history, global_phase = [], 0
    all_edges = np.arange(len(post))
    for it in range(max_iter + 200):
        bad = np.flatnonzero(bad_mask(post))
        history.append(len(bad))
        if len(bad) == 0:
            break
        # swap each bad edge's post with a random edge's post: from the same block first; if a dense block cannot
        # be made duplicate-free, fall back to any edge for the leftovers (degree-exact, block rule relaxed)
        partner = np.empty_like(bad)
        if it < max_iter:
            for b in np.unique(block[bad]):
                m = block[bad] == b
                partner[m] = rng.choice(segs[b], m.sum())
        else:
            global_phase += 1
            partner[:] = rng.choice(all_edges, len(bad))
        keep = ~np.isin(partner, bad)
        _, first = np.unique(partner[keep], return_index=True)
        a, p = bad[keep][first], partner[keep][first]
        post[a], post[p] = post[p].copy(), post[a].copy()
    else:
        raise RuntimeError(f"rewiring did not converge: {history[-5:]}")
    block_violations = int((block != _blocks(c, pre, post, matched)).sum()) if matched else 0

    R = sparse.csr_matrix((w, (post, pre)), shape=W_eff.shape, dtype=np.float32)
    target = np.asarray(abs(W_eff).sum(1)).ravel()
    have = np.asarray(abs(R).sum(1)).ravel()
    scale = np.divide(target, have, out=np.zeros_like(target), where=have > 0)
    R = (sparse.diags(scale.astype(np.float32)) @ R).tocsr()
    R.sort_indices()
    info = {"seed": seed, "matched": matched, "fix_iterations": len(history), "bad_initial": history[0],
            "nnz": int(R.nnz), "nnz_eff": int(W_eff.nnz),
            "global_fallback_iterations": global_phase, "edges_violating_block": block_violations,
            "bad_before_fallback": history[min(max_iter, len(history)) - 1]}
    return R, info


def signflip(W_eff: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    """Same edges and magnitudes; each presynaptic neuron's sign replaced by a random permutation of
    the neurons' signs (keeps the E/I proportion across neurons)."""
    rng = np.random.default_rng(seed)
    coo = W_eff.tocoo()
    n = W_eff.shape[0]
    sign = np.zeros(n, np.float32)
    has_out = np.bincount(coo.col, minlength=n) > 0
    first = np.full(n, 0.0, np.float32)
    first[coo.col[::-1]] = np.sign(coo.data[::-1])
    sign[has_out] = rng.permutation(first[has_out])
    return sparse.csr_matrix((np.abs(coo.data) * sign[coo.col], (coo.row, coo.col)), shape=W_eff.shape,
                             dtype=np.float32)
