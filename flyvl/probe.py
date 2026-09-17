"""Linear probe per PROTOCOL.md: standardize -> train-only PCA to K -> multinomial logistic regression
(L2 strength chosen by 5-fold stratified CV on train) -> test accuracy; paired bootstrap CIs.

PCA is fit once on the full training set (unsupervised, no labels) and reused inside the CV folds;
the test set never enters any fit.
"""
from __future__ import annotations

import numpy as np
import torch

LAMBDAS = (1e3, 1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3)     # fitted strongest first, warm-started


@torch.no_grad()
def standardize_pca(Xtr: np.ndarray, Xte: np.ndarray, Ks, device="cuda") -> dict:
    """Returns {K: (train scores, test scores)} for each K (capped at n_train - 1)."""
    tr = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    te = torch.as_tensor(Xte, dtype=torch.float32, device=device)
    mu, sd = tr.mean(0), tr.std(0)
    sd = torch.where(sd > 0, sd, torch.ones_like(sd))
    tr = (tr - mu) / sd
    te = (te - mu) / sd
    gram = (tr @ tr.T).double()
    evals, U = torch.linalg.eigh(gram)
    evals, U = evals.flip(0).clamp(min=0), U.flip(1)
    out = {}
    for K in Ks:
        k = min(K, tr.shape[0] - 1, int((evals > 1e-9 * evals[0]).sum()))
        s = evals[:k].sqrt()
        V = tr.T @ (U[:, :k] / s).float()                               # (D, k) principal axes
        str_ = (U[:, :k] * s).float()
        ste = te @ V
        scale = str_.std(0).mean()
        out[K] = ((str_ / scale).cpu().numpy(), (ste / scale).cpu().numpy(), k)
    del tr, te, gram, U
    torch.cuda.empty_cache()
    return out


def _fit(X: torch.Tensor, y: torch.Tensor, lam: float, W0: torch.Tensor | None = None, iters: int = 150) -> torch.Tensor:
    n, d = X.shape
    W = (torch.zeros(d + 1, 10, device=X.device, dtype=torch.float64) if W0 is None else W0.clone()).requires_grad_(True)
    Xb = torch.cat([X, torch.ones(n, 1, device=X.device, dtype=X.dtype)], 1)
    opt = torch.optim.LBFGS([W], lr=1, max_iter=iters, tolerance_grad=1e-7, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xb @ W, y) + 0.5 * lam * (W[:-1] ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach()


def _predict(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return (torch.cat([X, torch.ones(len(X), 1, device=X.device, dtype=X.dtype)], 1) @ W).argmax(1)


def fit_probe(Str: np.ndarray, ytr: np.ndarray, Ste: np.ndarray, seed: int, device="cuda") -> dict:
    X = torch.as_tensor(Str, dtype=torch.float64, device=device)
    y = torch.as_tensor(ytr, device=device)
    rng = np.random.default_rng(seed)
    folds = np.empty(len(ytr), np.int64)
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(ytr == c))
        folds[idx] = np.arange(len(idx)) % 5
    acc = {lam: [] for lam in LAMBDAS}
    for f in range(5):
        tr, va = torch.as_tensor(folds != f, device=device), torch.as_tensor(folds == f, device=device)
        W = None
        for lam in LAMBDAS:
            W = _fit(X[tr], y[tr], lam, W)
            acc[lam].append((_predict(W, X[va]) == y[va]).double().mean().item())
    cv = {lam: float(np.mean(a)) for lam, a in acc.items()}
    best = max(cv, key=cv.get)
    W = None
    for lam in LAMBDAS[:LAMBDAS.index(best) + 1]:                # same warm-start path as in CV
        W = _fit(X, y, lam, W)
    pred = _predict(W, torch.as_tensor(Ste, dtype=torch.float64, device=device)).cpu().numpy()
    return {"lambda": best, "cv": cv, "pred": pred}


def paired_bootstrap(correct_a: np.ndarray, correct_b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """correct_*: per-test-image correctness (averaged over probe seeds). CI of mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    idx = rng.integers(0, n, (n_boot, n))
    diff = correct_a[idx].mean(1) - correct_b[idx].mean(1)
    return {"diff": float(correct_a.mean() - correct_b.mean()),
            "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
