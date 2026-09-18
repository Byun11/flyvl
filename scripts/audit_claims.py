"""Audit every real-vs-control claim against its control-graph SD.

Thirteen corrections in this project came from the same failure: quoting a difference without checking
it against the spread of the control graphs that produced it. A difference of 1.3%p against a control
SD of 1.3%p says nothing, and looks identical in a table to one of 15.8%p against SD 2.5.

Rule used here: |difference| / SD(controls) >= 2 counts as resolved, below that is UNRESOLVED and must
not be written as a finding. Controls with fewer than 2 graphs are skipped - they cannot be judged.

usage: audit_claims.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome  # noqa: E402

RUNS = connectome.DATA_ROOT / "runs"
CONTROLS = {"matched": "matched_shuffle_s{i}", "optic-only": "shuffle_ol_s{i}",
            "central-only": "shuffle_central_s{i}", "global": "global_shuffle_s{i}"}
TASKS = [("MOTION", RUNS / "p5" / "motion_dir_hard.json", "{g}:{view}:K1024", "real:{view}:K1024"),
         ("CIFAR ", RUNS / "p7" / "drift_brain_300.json", "{g}:drift1.2:{view}", "real:drift1.2:{view}")]
VIEWS = ("visual_projection", "central_vnc")


def values(res, key_fmt, graph_fmt, view, seeds=range(4)):
    out = []
    for i in seeds:
        k = key_fmt.format(g=graph_fmt.format(i=i), view=view)
        if k in res:
            out.append(res[k]["mean"] * 100)
    return out


if __name__ == "__main__":
    print(f"{'claim':48s}{'diff':>8s}{'SD':>7s}{'n':>3s}{'ratio':>7s}  verdict")
    unresolved = 0
    for task, path, key_fmt, real_fmt in TASKS:
        if not path.exists():
            print(f"{task}: {path} missing")
            continue
        res = json.loads(path.read_text())["res"]
        for view in VIEWS:
            rk = real_fmt.format(view=view)
            if rk not in res:
                continue
            real = res[rk]["mean"] * 100
            for name, graph_fmt in CONTROLS.items():
                v = values(res, key_fmt, graph_fmt, view)
                if len(v) < 2:
                    print(f"{task} {view[:12]:13s} real vs {name:13s}"
                          f"{'':>22}  (n={len(v)}, cannot judge)")
                    continue
                d, sd = real - np.mean(v), np.std(v, ddof=1)
                ratio = abs(d) / sd if sd > 0 else float("inf")
                ok = ratio >= 2
                unresolved += not ok
                print(f"{task} {view[:12]:13s} real vs {name:13s}{d:+8.2f}{sd:7.2f}{len(v):3d}{ratio:7.1f}  "
                      f"{'OK' if ok else '*** UNRESOLVED ***'}")
    print(f"\n{unresolved} claim(s) below 2 SD - these must not be written as findings.")
