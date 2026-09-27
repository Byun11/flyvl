"""E5g verdict: pick the best flyvis member and the best HR setting on TRAINING episodes, compare on test (paired)."""
import json
import sys

import numpy as np

c = sys.argv[1] if len(sys.argv) > 1 else "0.03"
r = json.load(open(rf"D:\flyvl_data\runs\e5\e5_c{c}_n0.06_s0_grid_e5g.json"))
arms = [k for k in r if k.startswith(("hrg_", "fly_m")) and not k.endswith(("_sem", "_episodes", "_train"))]
for k in sorted(arms, key=lambda k: r[k + "_train"]):
    print(f"  {k:12s} train {r[k + '_train']:.3f}  test {r[k]:.3f}")
fly = min((k for k in arms if k.startswith("fly_m")), key=lambda k: r[k + "_train"])
hr = min((k for k in arms if k.startswith("hrg_")), key=lambda k: r[k + "_train"])
a, b = np.array(r[fly + "_episodes"]), np.array(r[hr + "_episodes"])
d = b - a
flies = [r[k] for k in arms if k.startswith("fly_m")]
print(f"contrast {c}: chosen on train -> fly {fly} test {r[fly]:.3f} | HR {hr} test {r[hr]:.3f} | "
      f"paired {d.mean():+.3f} +- {d.std(ddof=1) / np.sqrt(len(d)):.3f}  t={d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.1f}  "
      f"fly better in {(d > 0).sum()}/{len(d)} | all members mean {np.mean(flies):.3f} (min {min(flies):.3f}, max {max(flies):.3f}) | no control {r['no_control']:.3f}")
