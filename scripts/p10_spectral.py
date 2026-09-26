"""P10: is P5's +16pp motion advantage a spectral-radius artefact? (see PROTOCOL_P10_spectral.md)

The v1 optic lobe is graded: x <- x + alpha * (g_gg * W_gg x + ... - x). Its recurrent gain is set by the
spectral radius rho of the graded->graded block W_gg. Rewiring keeps each neuron's total |input| but not rho,
and a reservoir's behaviour depends strongly on rho (fly-connectome-lab #74). This script
  1. measures rho(W_gg) on the GPU by power iteration for real and every control graph;
  2. with --scale, writes <graph>_rho.npz: the control with ONLY its W_gg block multiplied by
     rho_real / rho_control, so its optic-lobe recurrent gain matches real's (other blocks untouched).

usage: p10_spectral.py [--scale] GRAPH [GRAPH ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, frozen  # noqa: E402
from flyvl.extract import load_graph  # noqa: E402

OUT = connectome.DATA_ROOT / "runs" / "p10"
OUT.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def rho(M: sparse.csr_matrix, iters: int = 600, seed: int = 0) -> float:
    """Spectral radius by power iteration on the GPU: geometric mean growth over the last 200 steps."""
    M = M.tocsr()
    A = torch.sparse_csr_tensor(torch.as_tensor(M.indptr, dtype=torch.int64), torch.as_tensor(M.indices, dtype=torch.int64),
                                torch.as_tensor(M.data, dtype=torch.float32), size=M.shape).to("cuda")
    x = torch.randn(M.shape[0], 4, generator=torch.Generator().manual_seed(seed)).cuda()   # 4 starts
    x /= x.norm(dim=0)
    logs = []
    for k in range(iters):
        x = A @ x
        n = x.norm(dim=0)
        x /= n
        if k >= iters - 200:
            logs.append(torch.log(n))
    return float(torch.stack(logs).mean(0).max().exp())


if __name__ == "__main__":
    args = sys.argv[1:]
    scale = "--scale" in args
    graphs = [a for a in args if a != "--scale"]
    c = connectome.load()
    cfg, _, _ = frozen.load()
    g = np.flatnonzero(c.graded)
    path = OUT / "rho.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    W_real = load_graph(c, "real", cfg)
    res.setdefault("real", rho(W_real[g][:, g]))
    print(f"real                  rho(W_gg) {res['real']:.4f}", flush=True)
    for name in graphs:
        W = load_graph(c, name, cfg).tocsr()
        r = rho(W[g][:, g])
        res[name] = r
        print(f"{name:21s} rho(W_gg) {r:.4f}", flush=True)
        if scale:
            s = res["real"] / r
            coo = W.tocoo()
            gg = np.isin(coo.row, g) & np.isin(coo.col, g)
            data = coo.data.copy()
            data[gg] *= s
            R = sparse.csr_matrix((data, (coo.row, coo.col)), shape=W.shape, dtype=np.float32)
            R.sort_indices()
            check = rho(R[g][:, g])
            sparse.save_npz(connectome.DATA_ROOT / "graphs" / f"{name}_rho.npz", R, compressed=False)
            res[f"{name}_rho"] = check
            print(f"  -> {name}_rho scale {s:.3f}, rho now {check:.4f}", flush=True)
        path.write_text(json.dumps(res, indent=1))
