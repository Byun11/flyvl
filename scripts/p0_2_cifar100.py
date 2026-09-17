"""P0-2: CIFAR-10 train, first 10 per class (100 images). Gate defined in PROTOCOL.md before running."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, data, masks, metrics  # noqa: E402
from flyvl.extract import Extractor, to_luma  # noqa: E402

OUT = connectome.DATA_ROOT / "runs" / "p0_2"
OUT.mkdir(parents=True, exist_ok=True)

images, labels = data.cifar10(train=True)
idx = data.first_per_class(labels, 10)
imgs, y = images[idx], labels[idx]

c = connectome.load()
M = masks.build(c)
ex = Extractor(c, "real")


def extract():
    feats, tails = [], []
    for s in range(0, len(imgs), 25):
        f, t = ex.features(to_luma(imgs[s:s + 25]).to("cuda"))
        feats.append(f)
        tails.append(t)
    return np.concatenate(feats), np.concatenate(tails)


F1, T1 = extract()
F2, _ = extract()
np.save(OUT / "features.npy", F1)
np.save(OUT / "labels.npy", y)
np.save(OUT / "indices.npy", idx)

views = {k: M[k] for k in ("all", "no_photoreceptor", "no_ol_intrinsic", "central_vnc", "visual_projection",
                            "descending")}
report = {"n_images": len(idx), "views": {k: metrics.summary(F1[:, v]) for k, v in views.items()}}
report["reference"] = {
    "pixels_luma": metrics.summary(to_luma(imgs).reshape(len(imgs), -1).numpy()),
    "stim_only": metrics.summary(F1[:, ex.driven]),
}
central = report["views"]["central_vnc"]
lif = ~c.graded
report["repeat"] = {"bit_exact": bool(np.array_equal(F1, F2)),
                    "lif_frac_equal": float((F1[:, lif] == F2[:, lif]).mean()),
                    "graded_max_abs_diff": float(np.abs(F1[:, c.graded] - F2[:, c.graded]).max())}
gates = {
    "G1_lif_equal>=0.9999_and_graded_diff<=1e-4": report["repeat"]["lif_frac_equal"] >= 0.9999
                                                   and report["repeat"]["graded_max_abs_diff"] <= 1e-4,
    "G2_central_nonzero_frac>=0.95": central["frac_images_nonzero"] >= 0.95,
    "G3_tail_over40Hz_max<0.01": float(T1.max()) < 0.01,
    "G4a_central_participation_ratio>=3": central["participation_ratio"] >= 3,
    "G4b_central_cos_dist_median>=0.05": central["cos_dist_median"] >= 0.05,
}
report["tail_over40Hz_max"] = float(T1.max())
report["gates"] = gates
report["PASS"] = all(gates.values())
(OUT / "report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))

# report-only figure
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, X) in zip(axes, [("pixels", to_luma(imgs).reshape(len(imgs), -1).numpy()),
                                ("stim_only", F1[:, ex.driven]), ("central_vnc", F1[:, M["central_vnc"]])]):
    Xc = X.astype(np.float64) - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    P = U[:, :2] * S[:2]
    for k in range(10):
        ax.scatter(*P[y == k].T, s=14, label=data.CLASSES[k])
    ax.set_title(f"{name} PCA (report only)")
axes[-1].legend(fontsize=7, loc="best")
fig.tight_layout()
fig.savefig(OUT / "pca.png", dpi=110)
