"""D5 appendix B: structural coverage from the visual input neurons (graph structure only: no training, no labels).

Input neurons: L1 / R7 / R8 with a column in the MaleCNS optic-column table, plus L2 / L3 given the column of their
strongest connection (either direction, share of synapses) to a column-assigned L1 / R7 / R8 / R1-6. R1-6 inherit the
column of their strongest column-assigned postsynaptic partner, as in connectome._columns.
Per population (superclass labels as published):
  shortest path : hop count from the input set (BFS on the directed graph, up to 32)
  effective path: VIC_h = probability that a walk backwards along a neuron's inputs (each step picks a presynaptic
                  partner with probability = its share of the neuron's input synapses) reaches an input neuron within
                  h steps. Input neurons absorb; a walk that reaches a neuron without inputs stops (not visual).
Applies the T rule of appendix B and writes <FLYVL_DATA>/d5/coverage.json.
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome  # noqa: E402

SRC = connectome.CONNECTOME
OUT = connectome.DATA_ROOT / "d5"
H_MAX = 32
HOPS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
T_CANDIDATES = (2, 4, 8, 16)
INPUT_TYPES = ("L1", "L2", "L3", "R7", "R8")
LANDMARKS = ("Mi1", "Tm3", "Tm1", "Tm9", "T4a", "T5a", "LPi2-3", "LC4", "LPLC2", "LC10a", "DNp01", "DNa02")


def optic_columns(path):
    """body id -> (eye, h1, h2) for the L1 / R7 / R8 bodies in the table (same parsing as flybrain.build)."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    out = {}
    with zipfile.ZipFile(path) as z:
        strings = ["".join(e.itertext()) for e in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        for sheet in (1, 2):
            for row in ET.fromstring(z.read(f"xl/worksheets/sheet{sheet}.xml")).findall("s:sheetData/s:row", ns)[1:]:
                cells = {}
                for c in row:
                    v = c.find("s:v", ns)
                    if v is not None:
                        cells[re.sub(r"\d", "", c.attrib["r"])] = strings[int(v.text)] if c.get("t") == "s" else v.text
                m = re.fullmatch(r"ME_([LR])_col_(\d+)_(\d+)", cells.get("A", ""))
                if not m:
                    continue
                for col in ("B", "C", "E"):                     # L1, R7, R8
                    try:
                        body = int(cells.get(col, -1))
                    except ValueError:
                        continue
                    if body > 0:
                        out[body] = (0 if m.group(1) == "L" else 1, int(m.group(2)), int(m.group(3)))
    return out


def strongest(W, rows, known, column, both=True):
    """Column of each neuron in `rows` = column of its strongest column-assigned partner (|w| = share of input)."""
    col = np.full((len(rows), 3), -1, np.int32)
    Win = abs(W[rows][:, known]).tocsr()                    # known -> row (row's input share)
    Wout = abs(W[known][:, rows]).T.tocsr() if both else None   # row -> known (known's input share)
    for k in range(len(rows)):
        best, arg = 0.0, -1
        for M in (Win, Wout) if both else (Win,):
            a, b = M.indptr[k:k + 2]
            if b > a:
                j = a + np.argmax(M.data[a:b])
                if M.data[j] > best:
                    best, arg = M.data[j], M.indices[j]
        if arg >= 0:
            col[k] = column[known[arg]]
    return col


def main():
    meta = np.load(SRC / "brain.npz")
    W = sparse.load_npz(SRC / "weights.npz").tocsr().astype(np.float32)   # rows = post, cols = pre
    ids, ct, sc = meta["ids"], meta["cell_type"].astype(str), meta["superclass"].astype(str)
    n = len(ids)
    assert n == connectome.EXPECTED_NEURONS and W.nnz == connectome.EXPECTED_EDGES, (n, W.nnz)

    # ---- columns ----
    table = optic_columns(SRC / "optic-columns.xlsx")
    index = {int(b): i for i, b in enumerate(ids)}
    column = np.full((n, 3), -1, np.int32)
    for body, c in table.items():
        if body in index:
            column[index[body]] = c
    direct = np.flatnonzero(column[:, 0] >= 0)
    r16 = np.flatnonzero(ct == "R1-6")
    # R1-6: strongest column-assigned POSTsynaptic partner (connectome._columns rule)
    to_known = abs(W[direct][:, r16]).tocsc()
    for k, pr in enumerate(r16):
        a, b = to_known.indptr[k:k + 2]
        if b > a:
            column[pr] = column[direct[to_known.indices[a + np.argmax(to_known.data[a:b])]]]
    known = np.flatnonzero(column[:, 0] >= 0)
    l23 = np.flatnonzero(np.isin(ct, ("L2", "L3")))
    column[l23] = strongest(W, l23, known, column, both=True)

    info = {"neurons": int(n), "edges": int(W.nnz), "table_bodies": len(table)}
    per_type = {}
    for t in INPUT_TYPES + ("R1-6",):
        m = ct == t
        per_type[t] = {"cells": int(m.sum()), "with_column": int((column[m, 0] >= 0).sum()),
                       "left": int((column[m, 0] == 0).sum()), "right": int((column[m, 0] == 1).sum())}
    # one L2 / L3 per column? (a cartridge has exactly one of each)
    for t in ("L2", "L3"):
        c = column[(ct == t) & (column[:, 0] >= 0)]
        _, counts = np.unique(c, axis=0, return_counts=True)
        per_type[t]["columns_hit"] = int(len(counts))
        per_type[t]["columns_with_exactly_one"] = int((counts == 1).sum())
    info["inputs"] = per_type
    inp = np.flatnonzero(np.isin(ct, INPUT_TYPES) & (column[:, 0] >= 0))
    info["n_input_neurons"] = int(len(inp))
    info["columns_per_eye_L1"] = [int(((ct == "L1") & (column[:, 0] == e)).sum()) for e in (0, 1)]

    # ---- shortest path (BFS) ----
    A = (W != 0).astype(np.float32).tocsr()
    hop = np.full(n, -1, np.int32)
    hop[inp] = 0
    frontier = np.zeros(n, np.float32)
    frontier[inp] = 1
    for h in range(1, H_MAX + 1):
        new = ((A @ frontier) > 0) & (hop < 0)
        if not new.any():
            break
        hop[new] = h
        frontier = new.astype(np.float32)

    # ---- effective path (VIC) ----
    P = abs(W).tocsr()
    rowsum = np.asarray(P.sum(1)).ravel()
    is_in = np.zeros(n, bool)
    is_in[inp] = True
    b = np.asarray(P[:, inp].sum(1)).ravel()                 # one step straight into an input neuron
    keep = sparse.diags((~is_in).astype(np.float32))
    Pna = (keep @ P @ keep).tocsr()                           # walks continue only through non-input neurons
    b[is_in] = 0
    vic, v, acc = {}, b.copy(), b.copy()
    for h in range(1, H_MAX + 1):
        if h in HOPS:
            vic[h] = acc.copy()
        v = Pna @ v
        acc = acc + v
    ref = vic[H_MAX]

    pops = {
        "optic_lobe_intrinsic": (sc == "ol_intrinsic") & ~is_in,
        "visual_projection": np.isin(sc, ("visual_projection", "visual_projection_tbc")),
        "visual_centrifugal": sc == "visual_centrifugal",
        "central_brain_intrinsic": sc == "cb_intrinsic",
        "readout (all non-input)": ~is_in,
    }
    rep = {}
    for name, m in pops.items():
        w = ref[m]
        hm = hop[m]
        rep[name] = {
            "n": int(m.sum()),
            "vic32_gt_0.01": int((w > 0.01).sum()), "vic32_gt_0.05": int((w > 0.05).sum()),
            "unreachable_within_32": float((hm < 0).mean()),
            "median_hop_reachable": float(np.median(hm[hm >= 0])) if (hm >= 0).any() else None,
            "reach": {h: float(((hm >= 0) & (hm <= h)).mean()) for h in HOPS},
            "vic_mass_reached": {h: float(w[(hm >= 0) & (hm <= h)].sum() / max(w.sum(), 1e-12)) for h in HOPS},
            "vic_ratio": {h: float(vic[h][m].sum() / max(w.sum(), 1e-12)) for h in HOPS},
            "vic32_mean": float(w.mean()),
        }
    marks = {}
    for t in LANDMARKS:
        m = ct == t
        if m.any():
            hm = hop[m]
            marks[t] = {"cells": int(m.sum()), "median_hop": float(np.median(hm[hm >= 0])) if (hm >= 0).any() else None,
                        "vic32_median": float(np.median(ref[m]))}
    cb = rep["central_brain_intrinsic"]
    rule = {}
    for T in T_CANDIDATES:
        h = T // 2
        a = cb["vic_mass_reached"].get(h)
        r = cb["vic_ratio"].get(h)
        rule[T] = {"horizon_hops": h, "a_mass_within": a, "b_vic_ratio": r, "pass": bool(a >= 0.9 and r >= 0.9)}
    chosen = next((T for T in T_CANDIDATES if rule[T]["pass"]), None)
    conv = {name: float(vic[H_MAX][m].sum() / max(vic[24][m].sum(), 1e-12) - 1) for name, m in pops.items()}
    res = {"info": info, "populations": rep, "landmarks": marks, "rule": rule,
           "chosen_T": chosen if chosen is not None else 16, "rule_satisfied": chosen is not None,
           "vic_growth_24_to_32": conv, "rows_without_input": int((rowsum == 0).sum())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coverage.json").write_text(json.dumps(res, indent=1))
    np.save(OUT / "input_columns.npy", column)

    print(json.dumps(info, indent=1))
    print(f"\n{'population':26s} {'n':>7s}  " + "  ".join(f"h<={h:<2d}" for h in (2, 4, 8, 16)) + "   (reach | VIC-mass reached | VIC_h/VIC_32)")
    for name, r in rep.items():
        print(f"{name:26s} {r['n']:7d}  " + "  ".join(f"{r['reach'][h]:.2f}" for h in (2, 4, 8, 16)) + "  |  "
              + "  ".join(f"{r['vic_mass_reached'][h]:.2f}" for h in (2, 4, 8, 16)) + "  |  "
              + "  ".join(f"{r['vic_ratio'][h]:.2f}" for h in (2, 4, 8, 16))
              + f"   median hop {r['median_hop_reachable']}, unreachable {r['unreachable_within_32']:.3f}")
    print("\nlandmarks:", json.dumps(marks))
    print("\nrule (central brain, horizon T/2):", json.dumps(rule))
    print(f"chosen T = {res['chosen_T']} (rule satisfied: {res['rule_satisfied']}); VIC growth 24->32: {json.dumps(conv)}")


def long_horizon(h_max=1024):
    """Supplement (not part of the registered rule): run the VIC walk far past 32 hops to find where each population's
    visual-input mass actually saturates. Writes <FLYVL_DATA>/d5/coverage_long.json."""
    meta = np.load(SRC / "brain.npz")
    W = sparse.load_npz(SRC / "weights.npz").tocsr().astype(np.float32)
    ct, sc = meta["cell_type"].astype(str), meta["superclass"].astype(str)
    column = np.load(OUT / "input_columns.npy")
    inp = np.flatnonzero(np.isin(ct, INPUT_TYPES) & (column[:, 0] >= 0))
    n = W.shape[0]
    is_in = np.zeros(n, bool)
    is_in[inp] = True
    P = abs(W).tocsr()
    keep = sparse.diags((~is_in).astype(np.float32))
    Pna = (keep @ P @ keep).tocsr()
    b = np.asarray(P[:, inp].sum(1)).ravel()
    b[is_in] = 0
    pops = {"optic_lobe_intrinsic": (sc == "ol_intrinsic") & ~is_in,
            "visual_projection": np.isin(sc, ("visual_projection", "visual_projection_tbc")),
            "central_brain_intrinsic": sc == "cb_intrinsic", "readout (all non-input)": ~is_in}
    v, acc, curve = b.astype(np.float64), b.astype(np.float64), {}
    for h in range(1, h_max + 1):
        curve[h] = {k: float(acc[m].sum()) for k, m in pops.items()}
        v = Pna @ v
        acc = acc + v
    final = curve[h_max]
    res = {"h_max": h_max, "final_mass": final, "growth_last_half": {k: final[k] / curve[h_max // 2][k] - 1 for k in pops}}
    for q in (0.5, 0.9):
        res[f"h_{int(q * 100)}"] = {k: next(h for h in curve if curve[h][k] >= q * final[k]) for k in pops}
    res["fraction_at"] = {h: {k: curve[h][k] / final[k] for k in pops} for h in (4, 8, 16, 32, 64, 128, 256, 512)}
    (OUT / "coverage_long.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("h_max", "growth_last_half", "h_50", "h_90")}, indent=1))
    for h, f in res["fraction_at"].items():
        print(f"h={h:4d}  " + "  ".join(f"{k.split(' ')[0]} {x:.2f}" for k, x in f.items()))


if __name__ == "__main__":
    long_horizon() if "--long" in sys.argv else main()
