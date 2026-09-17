"""P4 variant: fly-swarm features from the frozen FlyVL-simple-v1 model (graded optic lobe + LIF central/VNC,
gains chosen in P0-1) instead of the v2 g=1 rate model. cuda_fast backend (exploratory; not bit-reproducible).
Output files: p4/{split}_{GRAPH}-v1.pt with the same views and projections as p4_extract.py.
usage: p4_extract_v1.py GRAPH [GRAPH ...]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome  # noqa: E402
from flyvl.extract import Extractor  # noqa: E402
from flyvl.swarm import G, PROJ, patches, view_indices  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from p4_extract import OUT, splits  # noqa: E402

S = splits()
c = connectome.load()
views = view_indices(c)
g = torch.Generator().manual_seed(0)
proj = {k: (torch.randn(PROJ, len(v), generator=g) / np.sqrt(PROJ)).cuda() for k, v in views.items()}
views = {k: torch.as_tensor(v, device="cuda") for k, v in views.items()}

for graph in sys.argv[1:]:
    ex = Extractor(c, graph, backend="cuda_fast")
    for name, (X, y, idx) in S.items():
        path = OUT / f"{name}_{graph}-v1.pt"
        if path.exists():
            continue
        t0 = time.time()
        P = patches(X)
        feats, mags = {k: [] for k in views}, {k: [] for k in views}
        for s in range(0, len(P), 64):
            f, _ = ex.features(P[s:s + 64].cuda())                       # (B, N) numpy
            f = torch.from_numpy(f).cuda()
            for k, idx_t in views.items():
                sub = f[:, idx_t]
                feats[k].append((sub @ proj[k].T).cpu())
                mags[k].append(float(sub.abs().mean()))
        out = {k: torch.cat(v).reshape(len(X), G * G, -1).half() for k, v in feats.items()}
        torch.save(out, path)
        ev = {k: float(np.mean(v)) for k, v in mags.items()}
        print(graph, "v1", name, "evoked|mean|", ev, f"{time.time() - t0:.0f}s", flush=True)
        (OUT / f"{name}_{graph}-v1_stats.json").write_text(json.dumps(ev))
