"""P4 variant: swarm features read out from VIC-selected neurons only (paper-grounded readout set).
views:
  vic_central : top 11k of central_vnc by VIC (mostly visual projection neurons)
  vic_cb      : top 11k of central_vnc EXCLUDING visual_projection (closest to the paper's VCBN set)
Output: p4/{split}_{GRAPH}-vic.pt
usage: p4_extract_vic.py GRAPH [GRAPH ...]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, frozen, masks, vic  # noqa: E402
from flyvl.extract import load_graph  # noqa: E402
from flyvl.swarm import G, PROJ, SwarmExtractor, patches  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from p4_extract import OUT, splits  # noqa: E402

TOP = 11000
c = connectome.load()
cfg, _, _ = frozen.load()
score_path = connectome.DATA_ROOT / "runs" / "vic_score.npy"
score = np.load(score_path) if score_path.exists() else vic.vic(connectome.effective_W(c, cfg), vic.seeds(c))
central = masks.build(c)["central_vnc"]
cb = central[c.superclass[central] != "visual_projection"]
views = {"vic_central": central[np.argsort(-score[central])][:TOP],
         "vic_cb": cb[np.argsort(-score[cb])][:TOP]}
print({k: (len(v), float(score[v].min())) for k, v in views.items()}, flush=True)

S = splits()
for graph in sys.argv[1:]:
    ex = SwarmExtractor(c, load_graph(c, graph, cfg))
    g = torch.Generator().manual_seed(0)
    ex.views = {k: torch.as_tensor(v, device="cuda") for k, v in views.items()}
    ex.proj = {k: (torch.randn(PROJ, len(v), generator=g) / np.sqrt(PROJ)).cuda() for k, v in views.items()}
    for name, (X, y, idx) in S.items():
        path = OUT / f"{name}_{graph}-vic.pt"
        if path.exists():
            continue
        t0 = time.time()
        P = patches(X)
        feats = {}
        for s in range(0, len(P), 64):
            o = ex.run(P[s:s + 64])
            o.pop("evoked_abs_mean")
            o.pop("photoreceptor", None)
            for k, v in o.items():
                feats.setdefault(k, []).append(v)
        feats = {k: torch.cat(v).reshape(len(X), G * G, -1).half() for k, v in feats.items()}
        torch.save(feats, path)
        print(graph, "vic", name, {k: tuple(v.shape) for k, v in feats.items()}, f"{time.time() - t0:.0f}s", flush=True)
