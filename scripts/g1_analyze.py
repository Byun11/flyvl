"""G1: does ONE untuned front end do well across tasks? Fixed fly (flyvis) vs HR carrying a single setting.

Per task, scores are normalised to [0, 1]: 0 = chance (accuracy tasks) or no control (arena slip),
1 = the best arm on that task. Compared on the mean over tasks:
  fly            the pretrained flyvis, never tuned
  hr_oracle1     the single HR setting with the best mean over ALL tasks (optimistic for HR)
  hr_loto        leave-one-task-out: the setting chosen on the other tasks, scored on the held-out task (fair)
  hr_per_task    HR tuned separately on every task (upper bound, needs per-task tuning)
"""
import json
from pathlib import Path

import numpy as np

R = Path(r"D:\flyvl_data\runs")
tasks = {}

for name in ("loc", "fg", "photon"):
    r = json.load(open(R / "mix" / f"g1_{name}.json"))
    s = {k.split(":")[0]: v for k, v in r.items() if k.endswith(":n2100")}
    tasks[name] = {"fly": s["flyvis_pooled"], **{k: v for k, v in s.items() if k.startswith("hrg_")}}
    tasks[name]["_chance"] = 1 / 16

for c, fly_file in (("0.03", "e5_c0.03_n0.06_s0_bank.json"), ("0.1", "e5_c0.1_n0.06.json")):
    g = json.load(open(R / "e5" / f"e5_c{c}_n0.06_s0_grid.json"))
    f = json.load(open(R / "e5" / fly_file))
    nc = g["no_control"]
    # arena: lower slip is better -> score = no_control - slip, so higher is better and 0 = no control
    tasks[f"arena{c}"] = {"fly": nc - f["flyvis"], **{k: nc - v for k, v in g.items() if k.startswith("hrg_") and
                                                       not k.endswith(("_sem", "_episodes"))}}
    tasks[f"arena{c}"]["_chance"] = 0.0

norm = {}
for t, d in tasks.items():
    lo = d.pop("_chance")
    hi = max(d.values())
    norm[t] = {k: (v - lo) / (hi - lo) for k, v in d.items()}

names = list(norm)
configs = sorted(k for k in norm[names[0]] if k.startswith("hrg_"))
print(f"{'task':10s} {'fly':>6s} " + " ".join(f"{c[4:]:>7s}" for c in configs))
for t in names:
    print(f"{t:10s} {norm[t]['fly']:6.2f} " + " ".join(f"{norm[t][c]:7.2f}" for c in configs))

fly = np.mean([norm[t]["fly"] for t in names])
per_cfg = {c: np.mean([norm[t][c] for t in names]) for c in configs}
oracle = max(per_cfg.values())
loto = []
for held in names:
    others = [t for t in names if t != held]
    pick = max(configs, key=lambda c: np.mean([norm[t][c] for t in others]))
    loto.append(norm[held][pick])
    print(f"  LOTO {held:10s}: HR setting chosen on the rest = {pick}, held-out score {norm[held][pick]:.2f} (fly {norm[held]['fly']:.2f})")
per_task = np.mean([max(norm[t][c] for c in configs) for t in names])
out = {"fly": fly, "hr_oracle1": oracle, "hr_oracle1_setting": max(per_cfg, key=per_cfg.get),
       "hr_loto": float(np.mean(loto)), "hr_per_task": per_task, "normalised": norm}
print(f"\nmean normalised score: fly {fly:.3f} | HR one setting (LOTO) {np.mean(loto):.3f} | "
      f"HR best single setting (oracle) {oracle:.3f} | HR tuned per task {per_task:.3f}")
(R / "g1_summary.json").write_text(json.dumps(out, indent=1))
