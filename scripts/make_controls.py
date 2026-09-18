"""Build control graphs from the frozen effective graph and verify their constraints.
usage: make_controls.py global|matched|ol|central SEED [SEED ...]
  ol / central: matched rewiring restricted to optic-lobe-internal / non-optic-lobe edges."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, controls, frozen  # noqa: E402

kind, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
c = connectome.load()
cfg, _, _ = frozen.load()
W = connectome.effective_W(c, cfg)
out = connectome.DATA_ROOT / "graphs"
out.mkdir(parents=True, exist_ok=True)
kc = connectome.kc_mask(c)


def degrees(M):
    M = M.tocsr()
    return np.diff(M.indptr), np.bincount(M.indices, minlength=M.shape[1])


din, dout = degrees(W)
for seed in seeds:
    scope = kind if kind in ("ol", "central") else None
    R, info = controls.rewire(W, c, seed=seed, matched=(kind != "global"), scope=scope)
    rin, rout = degrees(R)
    coo = R.tocoo()
    checks = {
        "nnz_equal": R.nnz == W.nnz,
        "in_degree_equal": bool((rin == din).all()),
        "out_degree_equal": bool((rout == dout).all()),
        "kc_kc_edges": int((kc[coo.row] & kc[coo.col]).sum()),
        "self_loops": int((coo.row == coo.col).sum()),
        "row_abs_sum_maxdiff": float(np.abs(np.asarray(abs(R).sum(1)) - np.asarray(abs(W).sum(1))).max()),
        "inhibitory_frac": [float((W.data < 0).mean()), float((R.data < 0).mean())],
        "frac_edges_unchanged": float((R != 0).multiply(W != 0).nnz / W.nnz),
    }
    info.update(checks)
    assert checks["nnz_equal"] and checks["in_degree_equal"] and checks["out_degree_equal"]
    assert checks["kc_kc_edges"] == 0 and checks["row_abs_sum_maxdiff"] < 1e-4
    name = f"shuffle_{kind}_s{seed}" if scope else f"{kind}_shuffle_s{seed}"
    sparse.save_npz(out / f"{name}.npz", R, compressed=False)
    (out / f"{name}.json").write_text(json.dumps(info, indent=1))
    print(name, json.dumps(info), flush=True)
