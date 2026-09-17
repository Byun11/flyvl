"""CIFAR-10 subsets with fixed, label-balanced selection (first k per class in file order).

Source: HuggingFace mirror uoft-cs/cifar10 (plain_text parquet, 50k train / 10k test), decoded once
to D:/flyvl_data/datasets/cifar10_{split}.npz. (The toronto.edu tarball downloaded at ~90 kB/s.)
"""
from __future__ import annotations

import io

import numpy as np

from .connectome import DATA_ROOT

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def cifar10(train: bool) -> tuple[np.ndarray, np.ndarray]:
    split = "train" if train else "test"
    cache = DATA_ROOT / "datasets" / f"cifar10_{split}.npz"
    if not cache.exists():
        import pyarrow.parquet as pq
        from PIL import Image

        table = pq.read_table(DATA_ROOT / "datasets" / "hf_cifar10" / f"{split}.parquet").to_pydict()
        images = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in table["img"]])
        labels = np.asarray(table["label"], np.int64)
        assert images.shape == ((50000 if train else 10000), 32, 32, 3)
        np.savez(cache, images=images, labels=labels)
    d = np.load(cache)
    return d["images"], d["labels"]


def first_per_class(labels: np.ndarray, k: int) -> np.ndarray:
    idx = np.concatenate([np.flatnonzero(labels == c)[:k] for c in range(10)])
    return np.sort(idx)
