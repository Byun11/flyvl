"""MaleCNS connectome as used by FlyVL-simple.

Source: fly.ai / flybrain prebuilt files (MaleCNS v1.0, CC BY 4.0):
  weights.npz  CSR, rows = postsynaptic, synapse count x transmitter sign,
               divided by each neuron's total incoming synapse count.
  brain.npz    ids, cell_type, side, superclass, positions, photoreceptor indices.
Plus the MaleCNS optic column table (hex coordinates h1, h2 per eye) for retinotopy.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

DATA_ROOT = Path(os.environ.get("FLYVL_DATA", r"D:\flyvl_data"))
CONNECTOME = DATA_ROOT / "connectome"

EXPECTED_NEURONS = 166_700
EXPECTED_EDGES = 25_582_938

# Graded (non-spiking) in FlyVL-simple: optic-lobe intrinsic cells and optic-lobe sensory cells.
GRADED_SUPERCLASSES = ("ol_intrinsic", "ol_sensory")


@dataclass
class Connectome:
    W: sparse.csr_matrix          # (n, n) rows = post, cols = pre
    ids: np.ndarray               # MaleCNS body ids
    cell_type: np.ndarray
    side: np.ndarray
    superclass: np.ndarray
    photoreceptors: np.ndarray    # indices of R1-6 / R7 / R8
    column: np.ndarray            # (n, 3) int: eye (0=L, 1=R, -1 none), h1, h2
    weights_sha256: str

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def graded(self) -> np.ndarray:
        return np.isin(self.superclass, GRADED_SUPERCLASSES)

    def types(self, names, side: str | None = None) -> np.ndarray:
        mask = np.isin(self.cell_type, names)
        if side:
            mask &= self.side == side
        return np.flatnonzero(mask)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _columns(W: sparse.csr_matrix, ids: np.ndarray, photoreceptors: np.ndarray, root: Path) -> np.ndarray:
    """Hex column of every column-assigned neuron (L1, R7, R8 in the table); R1-6 inherit the
    column of their strongest column-assigned postsynaptic partner, as flybrain.build does."""
    from flybrain.build import optic_columns

    table = optic_columns(root / "optic-columns.xlsx")
    column = np.full((len(ids), 3), -1, np.int32)
    index = {int(b): i for i, b in enumerate(ids)}
    for body, (eye, h1, h2) in table.items():
        if body in index:
            column[index[body]] = (0 if eye == "L" else 1, h1, h2)
    known = np.flatnonzero(column[:, 0] >= 0)
    to_known = abs(W[known][:, photoreceptors]).tocsc()   # rows: known post, cols: photoreceptors
    for k, pr in enumerate(photoreceptors):
        if column[pr, 0] >= 0:
            continue
        a, b = to_known.indptr[k:k + 2]
        if b > a:
            column[pr] = column[known[to_known.indices[a + np.argmax(to_known.data[a:b])]]]
    return column


def load(root: Path | str = CONNECTOME) -> Connectome:
    root = Path(root)
    cache = root / "flyvl_columns.npy"
    meta = np.load(root / "brain.npz")
    W = sparse.load_npz(root / "weights.npz").tocsr().astype(np.float32)
    photoreceptors = meta["visual"].astype(np.int64)
    if cache.exists():
        column = np.load(cache)
    else:
        column = _columns(W, meta["ids"], photoreceptors, root)
        np.save(cache, column)
    return Connectome(W=W, ids=meta["ids"], cell_type=meta["cell_type"], side=meta["side"],
                      superclass=meta["superclass"], photoreceptors=photoreceptors, column=column,
                      weights_sha256=_sha256(root / "weights.npz"))


def effective_W(c: Connectome, cfg) -> sparse.csr_matrix:
    """The graph FlyVL actually simulates: MaleCNS weights with matrix-level flags applied
    (cut_input_to_nonvisual_sensory, kc_kc_scale). Control graphs are rewired FROM this matrix."""
    W = c.W.tocoo(copy=True)
    keep = np.ones(W.nnz, bool)
    if cfg.cut_input_to_nonvisual_sensory:
        sensory = (np.char.find(c.superclass.astype(str), "sensory") >= 0) & (c.superclass != "ol_sensory")
        keep &= ~sensory[W.row]
    if cfg.kc_kc_scale != 1.0:
        both = kc_mask(c)[W.row] & kc_mask(c)[W.col]
        W.data[both] *= cfg.kc_kc_scale
        keep &= W.data != 0
    out = sparse.csr_matrix((W.data[keep], (W.row[keep], W.col[keep])), shape=W.shape, dtype=np.float32)
    out.sort_indices()
    return out


def kc_mask(c: Connectome) -> np.ndarray:
    return np.char.startswith(c.cell_type.astype(str), "KC")
