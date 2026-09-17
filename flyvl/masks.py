"""Readout masks, fixed by neuron identity on the REAL graph and reused for every control graph."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .connectome import Connectome

CENTRAL_VNC = ("cb_intrinsic", "vnc_intrinsic", "visual_projection", "visual_centrifugal",
               "descending_neuron", "ascending_neuron", "cb_motor", "vnc_motor")


def hops_from(W: sparse.csr_matrix, sources: np.ndarray, max_hops: int = 8) -> np.ndarray:
    """Shortest feed-forward hop count from `sources` (-1 = unreachable within max_hops)."""
    A = (W != 0).astype(np.float32).tocsr()    # rows = post, cols = pre
    hops = np.full(W.shape[0], -1, np.int32)
    hops[sources] = 0
    frontier = np.zeros(W.shape[0], bool)
    frontier[sources] = True
    for h in range(1, max_hops + 1):
        reached = (A @ frontier.astype(np.float32)) > 0
        new = reached & (hops < 0)
        if not new.any():
            break
        hops[new] = h
        frontier = new
    return hops


def build(c: Connectome, hop_min: int = 3) -> dict[str, np.ndarray]:
    sc = c.superclass
    pr = np.zeros(c.n, bool)
    pr[c.photoreceptors] = True
    r16 = c.types(["R1-6"])
    hops = hops_from(c.W, r16)
    masks = {
        "all": np.ones(c.n, bool),
        "no_photoreceptor": ~pr,
        "no_ol_intrinsic": ~np.isin(sc, ("ol_intrinsic", "ol_sensory")),
        "central_vnc": np.isin(sc, CENTRAL_VNC),
        "visual_projection": sc == "visual_projection",
        "descending": sc == "descending_neuron",
        f"hop_ge{hop_min}": hops >= hop_min,
    }
    masks = {k: np.flatnonzero(v) for k, v in masks.items()}
    masks["_hops"] = hops
    return masks
