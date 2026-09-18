"""Render P5 result JSONs as markdown tables (rows = stimulus point, cols = graph).
usage: p5_table.py FILE.json [view]      view defaults to central_vnc
Keys are 'point:graph:view' for sweeps and 'graph:view' for the single-condition runs.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

path = Path(sys.argv[1])
VIEW = sys.argv[2] if len(sys.argv) > 2 else "central_vnc"
d = json.loads(path.read_text())
res = d["res"]

rows, graphs = [], []
table = {}
for key, v in res.items():
    parts = key.split(":")
    point, graph, view = (parts if len(parts) == 3 else ["-"] + parts)
    if view != VIEW and not (graph == "nobrain" and view == "photoreceptor"):
        continue
    label = "nobrain(eye)" if graph == "nobrain" else graph
    if point not in rows:
        rows.append(point)
    if label not in graphs:
        graphs.append(label)
    table[(point, label)] = v["mean"] * 100

chance = d.get("chance", 0.25) * 100
print(f"### {path.name} — view `{VIEW}`, chance {chance:.1f}%, "
      f"{len(d.get('seeds', [0, 1, 2]))} readout seeds, n={d.get('n')}\n")
head = ["condition"] + graphs + (["real − matched"] if "real" in graphs and
                                 any(g.startswith("matched") for g in graphs) else [])
print("| " + " | ".join(head) + " |")
print("|" + "---|" * len(head))
for r in rows:
    cells = [f"{table[(r, g)]:.2f}" if (r, g) in table else "–" for g in graphs]
    if len(head) > len(graphs) + 1:
        m = next(g for g in graphs if g.startswith("matched"))
        cells.append(f"**{table[(r, 'real')] - table[(r, m)]:+.2f}**"
                     if (r, "real") in table and (r, m) in table else "–")
    print("| " + " | ".join([r] + cells) + " |")
