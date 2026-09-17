"""P0-0: connectome/metadata loads correctly, counts match, normalization holds, masks are sane."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyvl import connectome, masks  # noqa: E402

c = connectome.load()
W = c.W
report = {"neurons": c.n, "edges": int(W.nnz), "weights_sha256": c.weights_sha256}
assert c.n == connectome.EXPECTED_NEURONS, c.n
assert W.nnz == connectome.EXPECTED_EDGES, W.nnz

row_abs = np.asarray(abs(W).sum(1)).ravel()
has_input = np.diff(W.indptr) > 0
report["row_abs_sum"] = {"max": float(row_abs.max()), "min_nonempty": float(row_abs[has_input].min()),
                         "frac_eq_1": float(np.mean(np.isclose(row_abs[has_input], 1.0, atol=1e-4))),
                         "no_input_neurons": int((~has_input).sum())}
report["frac_inhibitory_edges"] = float((W.data < 0).mean())

g = c.graded
report["graded"] = int(g.sum())
report["lif"] = int((~g).sum())
report["superclass"] = dict(collections.Counter(c.superclass.tolist()).most_common())

pr = c.photoreceptors
col = c.column[pr]
report["photoreceptors"] = {"total": len(pr), "with_column": int((col[:, 0] >= 0).sum()),
                            "R1-6": len(c.types(["R1-6"]))}
report["distinct_columns"] = {eye: int(len({tuple(x) for x in c.column[c.column[:, 0] == e][:, 1:]}))
                              for e, eye in enumerate("LR")}

m = masks.build(c)
report["masks"] = {k: int(len(v)) for k, v in m.items() if not k.startswith("_")}
hops = m["_hops"]
report["hops_from_R1-6"] = {int(h): int((hops == h).sum()) for h in np.unique(hops)}

for t in ["L1", "L2", "Mi1", "Tm3", "Mi4", "Mi9", "T4a", "T5a", "LC4", "LPLC2", "LC10a", "DNp01"]:
    idx = c.types([t])
    report.setdefault("hop_of_type", {})[t] = sorted(collections.Counter(hops[idx].tolist()).items())

print(json.dumps(report, indent=1, ensure_ascii=False, default=str))
out = connectome.DATA_ROOT / "runs" / "p0_0"
out.mkdir(parents=True, exist_ok=True)
(out / "report.json").write_text(json.dumps(report, indent=1, default=str))
np.save(out / "hops.npy", hops)
