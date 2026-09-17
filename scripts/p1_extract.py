"""Extract P1 features. usage: p1_extract.py GRAPH SUBSET   (SUBSET: mini -> train 500/class, test 100/class)"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, data  # noqa: E402
from flyvl.extract import Extractor, extract_to_memmap  # noqa: E402

graph, subset = sys.argv[1], sys.argv[2]
per_class = {"mini": (500, 100), "full": (5000, 1000)}[subset]
c = connectome.load()
ex = Extractor(c, graph, backend="cpu_deterministic")
for split, k in zip(("train", "test"), per_class):
    images, labels = data.cifar10(train=split == "train")
    idx = data.first_per_class(labels, k)
    out = connectome.DATA_ROOT / "features" / graph / f"cifar10_{subset}_{split}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "indices.npy", idx)
    np.save(out / "labels.npy", labels[idx])
    extract_to_memmap(ex, images[idx], out, batch=64)
    t = np.load(out / "tail_frac.npy")
    print(graph, split, len(idx), "tail>40Hz max", float(t.max()), flush=True)
