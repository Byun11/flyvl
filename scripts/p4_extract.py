"""P4 extraction: teacher tokens (InternVL3-1B) and frozen fly-swarm features.
usage: p4_extract.py teacher | GRAPH [GRAPH ...]     (GRAPH: real, global_shuffle_s0, matched_shuffle_s0)
Splits (fixed, small): train = first 200/class of CIFAR train, val = next 50/class of CIFAR train,
test = first 100/class of CIFAR test."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, data, frozen  # noqa: E402

OUT = connectome.DATA_ROOT / "p4"
OUT.mkdir(parents=True, exist_ok=True)


def splits():
    Xtr, ytr = data.cifar10(True)
    Xte, yte = data.cifar10(False)
    first250 = data.first_per_class(ytr, 250)
    tr = np.sort(np.concatenate([np.flatnonzero(ytr == k)[:200] for k in range(10)]))
    va = np.sort(np.concatenate([np.flatnonzero(ytr == k)[200:250] for k in range(10)]))
    assert set(tr) | set(va) == set(first250)
    te = data.first_per_class(yte, 100)
    return {"train": (Xtr[tr], ytr[tr], tr), "val": (Xtr[va], ytr[va], va), "test": (Xte[te], yte[te], te)}


if __name__ == "__main__":
    S = splits()
    for name, (X, y, idx) in S.items():
        np.save(OUT / f"{name}_labels.npy", y)
        np.save(OUT / f"{name}_indices.npy", idx)
    if sys.argv[1] == "teacher":
        from flyvl import teacher
        proc, model = teacher.load()
        for name, (X, y, idx) in S.items():
            t0 = time.time()
            T = teacher.teacher_tokens(model, X)
            torch.save(T, OUT / f"{name}_teacher256.pt")
            print(name, tuple(T.shape), f"{time.time() - t0:.0f}s", flush=True)
        sys.exit()

    from flyvl.extract import load_graph
    from flyvl.swarm import G, SwarmExtractor, patches
    c = connectome.load()
    cfg, _, _ = frozen.load()
    for graph in sys.argv[1:]:
        ex = SwarmExtractor(c, load_graph(c, graph, cfg))
        for name, (X, y, idx) in S.items():
            path = OUT / f"{name}_{graph}.pt"
            if path.exists():
                continue
            t0 = time.time()
            P = patches(X)                                                  # (N*16, 1, 16, 16)
            feats, stats = {}, []
            for s in range(0, len(P), 64):
                o = ex.run(P[s:s + 64])
                stats.append(o.pop("evoked_abs_mean"))
                for k, v in o.items():
                    feats.setdefault(k, []).append(v)
            feats = {k: torch.cat(v).reshape(len(X), G * G, -1).half() for k, v in feats.items()}
            torch.save(feats, path)
            ev = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
            print(graph, name, {k: tuple(v.shape) for k, v in feats.items()}, "evoked|mean|", ev,
                  f"{time.time() - t0:.0f}s", flush=True)
            (OUT / f"{name}_{graph}_stats.json").write_text(json.dumps(ev))
