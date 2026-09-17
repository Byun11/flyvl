"""CIFAR-10 subsets with fixed, label-balanced selection (first k per class in file order)."""
from __future__ import annotations

import numpy as np

from .connectome import DATA_ROOT

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def cifar10(train: bool) -> tuple[np.ndarray, np.ndarray]:
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root=str(DATA_ROOT / "datasets"), train=train, download=True)
    return ds.data, np.asarray(ds.targets)


def first_per_class(labels: np.ndarray, k: int) -> np.ndarray:
    idx = np.concatenate([np.flatnonzero(labels == c)[:k] for c in range(10)])
    return np.sort(idx)
