"""Visual Input Contribution (VIC), following the structural metric of the MaleCNS visual-pathway paper
(bioRxiv 2025.12.22.696097): the summed product of input-normalized synaptic connections along all paths
from the visual input seeds to a downstream neuron.

Seeds (as in that paper): L1, L2, L3, R7, R8 (+ dorsal-rim R7d/R8d and HB eyelet if present as cell types).
We propagate magnitudes (|w|) so that excitatory/inhibitory paths do not cancel, over `hops` steps.

Deviation from the paper: their threshold comes from left/right homologue consistency of the scores.
We instead take the top-N central neurons by VIC, with N chosen to match the reported scale (~11k neurons),
and we compute both sides. This is a reproduction of the *ranking*, not of their exact threshold.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .connectome import Connectome

SEED_TYPES = ("L1", "L2", "L3", "R7", "R8", "R7d", "R8d", "HBeyelet")


def seeds(c: Connectome) -> np.ndarray:
    return np.flatnonzero(np.isin(c.cell_type.astype(str), SEED_TYPES))


def vic(W: sparse.csr_matrix, seed_idx: np.ndarray, hops: int = 5) -> np.ndarray:
    """|W| rows = post, cols = pre. Returns the accumulated contribution score per neuron."""
    A = abs(W).tocsr().astype(np.float64)
    x = np.zeros(W.shape[0])
    x[seed_idx] = 1.0
    total = np.zeros_like(x)
    for _ in range(hops):
        x = A @ x
        total += x
    return total
