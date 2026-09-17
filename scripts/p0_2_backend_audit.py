"""Diagnostics only (not gates):
 (a) CPU-deterministic vs CUDA representation on the same 100 P0-2 images.
 (b) CUDA run-to-run variability: 20 images x 10 repeats, within-image vs between-image distances."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, data, masks, metrics  # noqa: E402
from flyvl.extract import Extractor, to_luma  # noqa: E402

RUNS = connectome.DATA_ROOT / "runs"
c = connectome.load()
M = masks.build(c)
views = ("all", "central_vnc", "visual_projection", "descending")
lif = ~c.graded


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# (a) CPU vs CUDA
Fc = np.load(RUNS / "p0_2_cuda_fast" / "features.npy").astype(np.float64)
Fd = np.load(RUNS / "p0_2_cpu_deterministic" / "features.npy").astype(np.float64)
comp = {}
for v in views:
    A, B = Fc[:, M[v]], Fd[:, M[v]]
    ca, cb = A - A.mean(0), B - B.mean(0)
    da, db = metrics.pairwise_cosine_distance(A), metrics.pairwise_cosine_distance(B)
    comp[v] = {
        "per_image_cosine_min": round(min(cos(A[i], B[i]) for i in range(len(A))), 6),
        "per_image_cosine_median": round(float(np.median([cos(A[i], B[i]) for i in range(len(A))])), 6),
        "per_image_centered_cosine_median": round(float(np.median([cos(ca[i], cb[i]) for i in range(len(A))])), 6),
        "relative_L2": float(np.linalg.norm(A - B) / np.linalg.norm(B)),
        "PR_cuda_cpu": [round(metrics.participation_ratio(A), 3), round(metrics.participation_ratio(B), 3)],
        "pairwise_dist_quantiles_cuda": np.round(np.percentile(da, [5, 25, 50, 75, 95]), 4).tolist(),
        "pairwise_dist_quantiles_cpu": np.round(np.percentile(db, [5, 25, 50, 75, 95]), 4).tolist(),
        "pairwise_dist_corr": round(float(np.corrcoef(da, db)[0, 1]), 6),
    }
comp["lif_elements_equal_frac"] = float((Fc[:, lif] == Fd[:, lif]).mean())

# (b) CUDA repeat variability
images, labels = data.cifar10(train=True)
idx = data.first_per_class(labels, 2)                     # 20 images, 2 per class (subset of P0-2's 100)
imgs = to_luma(images[idx]).to("cuda")
ex = Extractor(c, "real", backend="cuda_fast")
R = np.stack([ex.features(imgs)[0] for _ in range(10)]).astype(np.float64)   # (repeats, images, N)
audit = {}
for v in views:
    X = R[:, :, M[v]]
    within = [metrics.pairwise_cosine_distance(X[:, i]) for i in range(X.shape[1])]
    between = metrics.pairwise_cosine_distance(X[0])
    audit[v] = {"within_image_repeat_cos_dist_median": float(np.median(np.concatenate(within))),
                "within_image_repeat_cos_dist_max": float(np.max(np.concatenate(within))),
                "between_image_cos_dist_median": float(np.median(between)),
                "between_image_cos_dist_min": float(np.min(between))}
L = R[:, :, lif]
audit["lif_repeat_disagreement_vs_run0"] = float((L[1:] != L[:1]).mean())
audit["lif_units_ever_disagreeing_frac"] = float((L != L[:1]).any(0).mean())

out = {"cpu_vs_cuda": comp, "cuda_repeat_audit": audit}
(RUNS / "p0_2_backend_audit.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
