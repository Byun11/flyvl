"""Kernel timing on the real MaleCNS matrix (no training): W @ X and W.T @ X, original order vs locality order."""
import time, numpy as np, torch
from scipy import sparse
from pathlib import Path
S = Path.home() / "flyvl_data/connectome"
W = sparse.load_npz(S / "weights.npz").tocsr().astype(np.float32)
m = np.load(S / "brain.npz"); ct = m["cell_type"].astype(str); sc = m["superclass"].astype(str)
col = np.load(Path.home() / "flyvl_data/d5/input_columns.npy")
tid = np.unique(ct, return_inverse=True)[1]; sid = np.unique(sc, return_inverse=True)[1]
order = np.lexsort((tid, col[:, 2], col[:, 1], col[:, 0], sid))       # superclass, eye, column, type
def to_t(M):
    M = M.tocsr(); M.sort_indices()
    return torch.sparse_csr_tensor(torch.from_numpy(M.indptr.astype(np.int32)), torch.from_numpy(M.indices.astype(np.int32)),
                                   torch.from_numpy(M.data), size=M.shape).cuda()
for name, M in (("original", W), ("reordered", W[order][:, order])):
    A, AT = to_t(M), to_t(M.T)
    for w in (128, 256, 512):
        x = torch.randn(M.shape[0], w, device="cuda")
        for Mat, lab in ((A, "W"), (AT, "W.T")):
            for _ in range(2): y = Mat @ x
            torch.cuda.synchronize(); t = time.time()
            for _ in range(5): y = Mat @ x
            torch.cuda.synchronize(); ms = (time.time() - t) / 5 * 1e3
            print(f"{name:9s} {lab:3s} width {w:4d}: {ms:6.2f} ms  {ms / w * 1e3:5.1f} us/column", flush=True)
