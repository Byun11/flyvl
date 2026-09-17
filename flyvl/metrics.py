"""Label-free representation diagnostics."""
from __future__ import annotations

import numpy as np


def spectrum(X: np.ndarray) -> np.ndarray:
    Xc = X.astype(np.float64) - X.mean(0, dtype=np.float64)
    s = np.linalg.svd(Xc, compute_uv=False)
    return s ** 2


def participation_ratio(X: np.ndarray) -> float:
    lam = spectrum(X)
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.sum() > 0 else 0.0


def entropy_rank(X: np.ndarray) -> float:
    lam = spectrum(X)
    if lam.sum() <= 0:
        return 0.0
    p = lam / lam.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def pairwise_cosine_distance(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    U = np.divide(X, norm, out=np.zeros_like(X), where=norm > 0)
    D = 1 - U @ U.T
    return D[np.triu_indices(len(X), 1)]


def summary(X: np.ndarray) -> dict:
    d = pairwise_cosine_distance(X)
    var = X.astype(np.float64).var(0)
    return {
        "dim": int(X.shape[1]),
        "frac_images_nonzero": float((np.abs(X).sum(1) > 0).mean()),
        "frac_units_varying": float((var > 0).mean()),
        "participation_ratio": round(participation_ratio(X), 3),
        "entropy_rank": round(entropy_rank(X), 3),
        "cos_dist_median": round(float(np.median(d)), 5),
        "cos_dist_p05": round(float(np.percentile(d, 5)), 5),
        "cos_dist_min": round(float(d.min()), 5),
    }
