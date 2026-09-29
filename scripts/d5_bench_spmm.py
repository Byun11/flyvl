"""Kernel timing only (no training): one CSR SpMM at MaleCNS size, n = 166,700, nnz = 25,582,938.
Two synthetic column layouts bracket the real graph: uniform random columns (worst cache behaviour) and
columns near the row index (best case, like a graph reordered for locality)."""
import time

import numpy as np
import torch

N, NNZ = 166_700, 25_582_938
g = np.random.default_rng(0)
deg = g.poisson(NNZ / N, N).astype(np.int64)
deg = (deg * NNZ / deg.sum()).astype(np.int64)
deg[: NNZ - deg.sum()] += 1
indptr = np.r_[0, np.cumsum(deg)]
rows = np.repeat(np.arange(N), deg)


def csr(cols):
    return torch.sparse_csr_tensor(torch.from_numpy(indptr.astype(np.int32)), torch.from_numpy(cols.astype(np.int32)),
                                   torch.ones(NNZ), size=(N, N)).cuda()


layouts = {"random": g.integers(0, N, NNZ),
           "local": np.clip(rows + g.integers(-2000, 2000, NNZ), 0, N - 1)}
for name, cols in layouts.items():
    order = np.lexsort((cols, rows))
    W = csr(cols[order])
    for w in (128, 256, 512, 1024, 2048):
        x = torch.randn(N, w, device="cuda")
        for _ in range(2):
            y = W @ x
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(5):
            y = W @ x
        torch.cuda.synchronize()
        ms = (time.time() - t) / 5 * 1e3
        print(f"{name:6s} width {w:5d}: {ms:7.2f} ms  {2 * NNZ * w / ms / 1e9:6.0f} GFLOP/s  "
              f"{ms / w * 1e3:6.1f} us per column   peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        del x, y
        torch.cuda.empty_cache()
