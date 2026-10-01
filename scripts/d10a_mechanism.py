"""D10-A (PROTOCOL_D10A_mechanism.md): which anatomical / topological property of the real wiring produces the two
confirmed topology effects, motion direction (D6 task, VPN readout) and looming (D8 task, descending-neuron readout).
One harness for everything (learned per-frame encoder -> frozen brain -> BatchNorm + linear readout, the D7/D8 recipe);
only the brain's neuron set, graph or dynamics change.
usage: d10a_mechanism.py build                      labels, circuits, ablation sets, A1/A3/A4 graphs (CPU + 1 GPU)
       d10a_mechanism.py build_a2                   A2 null graphs on each task's selected circuit (after the A1 verdict)
       d10a_mechanism.py queue PART                 print the run names of PART (a1 | a2 | a3 | a4)
       d10a_mechanism.py worker SLOT N_SLOTS PART..  run this slot's share of the queued runs (skips finished ones)
       d10a_mechanism.py verdict_a1 | verdict
"""
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flyvl import flygrapher as fg  # noqa: E402

OUT = fg.connectome.DATA_ROOT / "d10a"
GRAPHS = OUT / "graphs"
RUNS = OUT / ("smoke" if os.environ.get("D10A_SMOKE") else "runs")     # smoke: 1 epoch, code-path check only
for p in (OUT, GRAPHS, RUNS):
    p.mkdir(parents=True, exist_ok=True)
dev = "cuda"
SEEDS = (1, 2, 3)
DELTA = 0.01
VISUAL = ("ol_intrinsic", "ol_sensory", "visual_projection", "visual_projection_tbc", "visual_centrifugal")
VPN_SC = ("visual_projection", "visual_projection_tbc")
T4T5 = re.compile(r"^T[45]")
# anatomical stage (where a neuron receives its input), ordered input -> lamina -> medulla -> T4/T5 -> lobula/lobula
# plate -> VPN -> centrifugal -> central brain -> DN -> VNC
STAGES = ("input", "lamina", "medulla", "T4T5", "lobula_lp", "vpn", "vcn", "central", "dn", "vnc", "ol_other")
ORDER = {"input": 0, "lamina": 1, "medulla": 2, "T4T5": 3, "lobula_lp": 4, "ol_other": 4, "vpn": 5, "vcn": 6,
         "central": 7, "dn": 8, "vnc": 9}
LAMINA = re.compile(r"^(R1-6|R7R8_unclear|HBeyelet|R7|R8|L[1-5]($|_)|Lai|Lawf|LA)")
MEDULLA = re.compile(r"^(Mi|Tm|TmY|Dm|yDm|pDm|DmDRA|Pm|Sm|C2|C3|T1|T2|T2a|T3|CT1|Cm|MeMe|ME|aMe|MLt|Am|MeLp)")
LOBULA_LP = re.compile(r"^(Li|LPi|Y|Tlp|LMa|LMt|LLPt|LC|LO|LOP|LT|LPT|H1|PDt|LpMe|l)")


# ------------------------------------------------------------------ labels (D5 neuron order)
_LAB = {}


def labels(brain):
    """type, stage, Dale sign per neuron in the D5 order of flygrapher.load_malecns."""
    if _LAB:
        return SimpleNamespace(**_LAB)
    meta = np.load(fg.SRC / "brain.npz")
    ct, sc = meta["cell_type"].astype(str), meta["superclass"].astype(str)
    col = np.load(fg.D5 / "input_columns.npy")
    key = np.where(ct == "", np.char.add("untyped:", sc), ct)
    _, group = np.unique(key, return_inverse=True)
    _, sid = np.unique(sc, return_inverse=True)
    order = np.lexsort((group, col[:, 2], col[:, 1], col[:, 0], sid))
    ct, sc, key = ct[order], sc[order], key[order]
    assert (group[order] == brain.group).all() and (sc == brain.superclass).all()
    is_in = np.zeros(brain.n, bool)
    is_in[brain.inputs] = True
    stage = np.empty(brain.n, object)
    for i in range(brain.n):
        s, t = sc[i], ct[i]
        if is_in[i]:
            stage[i] = "input"
        elif s in VPN_SC:
            stage[i] = "vpn"
        elif s == "visual_centrifugal":
            stage[i] = "vcn"
        elif s.startswith("descending") or s == "sensory_descending" or s == "efferent_descending":
            stage[i] = "dn"
        elif s.startswith("vnc") or "ascending" in s:
            stage[i] = "vnc"
        elif s in ("ol_intrinsic", "ol_sensory"):
            if T4T5.match(t):
                stage[i] = "T4T5"
            elif LAMINA.match(t):
                stage[i] = "lamina"
            elif MEDULLA.match(t):
                stage[i] = "medulla"
            elif LOBULA_LP.match(t):
                stage[i] = "lobula_lp"
            else:
                stage[i] = "ol_other"
        else:
            stage[i] = "central"
    Wc = brain.W.tocsc()
    sign = np.zeros(brain.n, np.int8)
    has = np.diff(Wc.indptr) > 0
    first = Wc.data[Wc.indptr[:-1][has]]
    sign[has] = np.where(first > 0, 1, -1)
    _LAB.update(type=ct, key=key, superclass=sc, stage=stage.astype(str),
                stage_id=np.array([ORDER[s] for s in stage]), sign=sign, is_input=is_in)
    return SimpleNamespace(**_LAB)


def type_shares(brain, lab):
    """S[post_type, pre_type] = mean over neurons of post_type of the input share coming from pre_type."""
    types, tid = np.unique(lab.key, return_inverse=True)
    n_t = len(types)
    A = abs(brain.W).tocsr()
    Pre = sparse.csr_matrix((np.ones(brain.n), (np.arange(brain.n), tid)), shape=(brain.n, n_t))
    Post = sparse.csr_matrix((np.ones(brain.n), (tid, np.arange(brain.n))), shape=(n_t, brain.n))
    cnt = np.bincount(tid, minlength=n_t)
    S = (sparse.diags(1.0 / np.maximum(cnt, 1)) @ (Post @ (A @ Pre))).tocsr()
    return types, tid, cnt, S


# ------------------------------------------------------------------ circuits (registered rules, PROTOCOL §A1)
def circuits(brain):
    lab = labels(brain)
    types, tid, cnt, S = type_shares(brain, lab)
    ix = {t: i for i, t in enumerate(types)}
    sc_of_type = {t: lab.superclass[np.flatnonzero(tid == i)[0]] for i, t in enumerate(types)}
    stage_of_type = {t: lab.stage[np.flatnonzero(tid == i)[0]] for i, t in enumerate(types)}

    def pooled_in(tl):
        rows = [ix[t] for t in tl]
        w = cnt[rows] / cnt[rows].sum()
        return np.asarray((sparse.diags(w) @ S[rows]).sum(0)).ravel()

    def pooled_from(tl):
        return np.asarray(S[:, [ix[t] for t in tl]].sum(1)).ravel()
    t4 = [t for t in types if sc_of_type[t] == "ol_intrinsic" and t.startswith("T4")]
    t5 = [t for t in types if sc_of_type[t] == "ol_intrinsic" and t.startswith("T5")]
    tt = set(t4 + t5)
    input_types = set(np.unique(lab.key[lab.is_input]))
    u1 = {types[j] for v in (pooled_in(t4), pooled_in(t5)) for j in np.flatnonzero(v >= 0.02)} - tt
    u1_list = sorted(u1)
    u2 = set()
    for t in u1_list:
        v = S[ix[t]].toarray().ravel()
        u2 |= {types[j] for j in np.flatnonzero(v >= 0.05)}
    u2 = {t for t in u2 if stage_of_type[t] in ("lamina", "medulla")} - u1 - tt
    fr = pooled_from(sorted(tt))
    d1 = {types[j] for j in np.flatnonzero(fr >= 0.10) if sc_of_type[types[j]] in ("ol_intrinsic",) + VPN_SC + ("visual_centrifugal",)} - tt
    v1 = {t for t in d1 if sc_of_type[t] in VPN_SC}
    loom_up = {types[j] for v in (pooled_in(["LPLC2"]), pooled_in(["LC4"])) for j in np.flatnonzero(v >= 0.02)} - {"LPLC2", "LC4"}
    fl = pooled_from(["LPLC2", "LC4"])
    dn_loom = {types[j] for j in np.flatnonzero(fl >= 0.03) if sc_of_type[types[j]] == "descending_neuron"}

    def members(tset):
        return np.flatnonzero(np.isin(lab.key, sorted(tset)))
    inputs = brain.inputs
    sc = lab.superclass
    dn_all = np.flatnonzero(sc == "descending_neuron")
    vpn_all = np.flatnonzero(np.isin(sc, VPN_SC))
    visual = np.flatnonzero(np.isin(sc, VISUAL))
    tt_n, u1_n, u2_n, d1_n, v1_n = members(tt), members(u1), members(u2), members(d1), members(v1)
    loom_n, lp2, lc4 = members(loom_up), members({"LPLC2"}), members({"LC4"})
    dnl_n, gf = members(dn_loom), members({"DNp01"})
    U = lambda *a: np.unique(np.concatenate([np.asarray(x, np.int64) for x in a]))
    motion_up = U(inputs, u2_n, u1_n, tt_n)
    C = {
        "M0": (np.arange(brain.n), vpn_all, "whole CNS"),
        "M1": (visual, vpn_all, "visual system (optic lobe + VPN + VCN)"),
        "M2": (U(motion_up, d1_n), v1_n, "motion pathway: lamina/medulla upstream + T4/T5 inputs + T4/T5 + lobula-plate targets"),
        "M3": (U(inputs, u1_n, tt_n, v1_n), v1_n, "T4/T5 core: direct inputs + T4/T5 + VPN targets of T4/T5"),
        "M4": (U(inputs, u1_n, tt_n), tt_n, "T4/T5 readout: direct inputs + T4/T5"),
        "M5": (U(inputs, u1_n), u1_n, "medulla only: direct inputs of T4/T5 (no T4/T5)"),
        "M6": (U(inputs), U(inputs), "input neurons only"),
        "L0": (np.arange(brain.n), dn_all, "whole CNS"),
        "L1": (U(visual, np.flatnonzero(np.char.startswith(sc.astype(str), "descending_neuron"))), U(np.intersect1d(dn_all, dn_all)),
               "visual system + DNs (central brain removed)"),
        "L2": (U(motion_up, loom_n, lp2, lc4, dnl_n), dnl_n, "looming pathway: motion upstream + LPLC2/LC4 inputs + LPLC2 + LC4 + their DN targets"),
        "L3": (U(motion_up, loom_n, lp2, lc4, gf), gf, "giant-fiber pathway: as L2, DN = DNp01 only"),
        "L4": (U(motion_up, loom_n, lp2, lc4), U(lp2, lc4), "LPLC2/LC4 readout: as L2 without DNs"),
        "L5": (U(inputs, u1_n, tt_n), tt_n, "motion upstream, T4/T5 readout"),
    }
    sets = {"U1": sorted(u1), "U2": sorted(u2), "D1": sorted(d1), "V1": sorted(v1), "LOOM_UP": sorted(loom_up),
            "DN_LOOM": sorted(dn_loom), "T4T5": sorted(tt)}
    return C, sets


# ------------------------------------------------------------------ graphs
def sub(W, S):
    return W[S][:, S].tocsr()


def sub_brain(brain, S):
    """Inputs / patches of the circuit S (S contains every input neuron)."""
    loc = np.searchsorted(S, brain.inputs)
    assert (S[loc] == brain.inputs).all()
    return SimpleNamespace(inputs=loc, in_patch=brain.in_patch, n=len(S))


def rewire(W, block_fn, seed, max_iter=300):
    """flygrapher.rewire_blocks with the block given per connection: keep every connection's presynaptic neuron, weight
    and sign, permute postsynaptic ends among connections of the same block (in- and out-degree exact), then repair
    duplicates / self-loops by swapping posts inside the block. Rows are rescaled to the original sum of |w|."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    coo = W.tocoo()
    pre, post0, w = coo.col.astype(np.int64), coo.row.astype(np.int64), coo.data.copy()
    block = block_fn(pre, post0)
    o1 = np.argsort(block, kind="stable")
    o2 = np.lexsort((rng.random(len(pre)), block))
    post = post0.copy()
    post[o1] = post0[o2]
    sb = block[o1]
    starts = np.r_[0, np.flatnonzero(np.diff(sb)) + 1]
    lens = np.diff(np.r_[starts, len(sb)])
    bstart, bcount = np.empty(len(pre), np.int64), np.empty(len(pre), np.int64)
    bstart[o1] = np.repeat(starts, lens)
    bcount[o1] = np.repeat(lens, lens)

    def bad(post):
        key = pre * n + post
        srt = np.argsort(key, kind="stable")
        dup = np.zeros(len(key), bool)
        dup[srt[1:]] = key[srt[1:]] == key[srt[:-1]]
        return np.flatnonzero(dup | (pre == post))
    history, fallback = [], 0
    for it in range(max_iter + 100):
        b = bad(post)
        history.append(len(b))
        if len(b) == 0:
            break
        if it < max_iter:
            partner = o1[bstart[b] + (rng.random(len(b)) * bcount[b]).astype(np.int64)]
        else:
            fallback += 1
            partner = rng.integers(0, len(pre), len(b))
        ok = ~np.isin(partner, b) & (partner != b)
        _, first = np.unique(partner[ok], return_index=True)
        a, p = b[ok][first], partner[ok][first]
        post[a], post[p] = post[p].copy(), post[a].copy()
    R = sparse.csr_matrix((w, (post, pre)), shape=W.shape, dtype=np.float32)
    target = np.asarray(abs(W).sum(1)).ravel()
    have = np.asarray(abs(R).sum(1)).ravel()
    R = (sparse.diags(np.divide(target, have, out=np.zeros_like(target), where=have > 0).astype(np.float32)) @ R).tocsr()
    R.sort_indices()
    final_bad = len(bad(post))
    info = {"seed": seed, "blocks": int(len(starts)), "changed_fraction": float((post != post0).mean()),
            "fix_iterations": len(history), "bad_initial": history[0], "fallback_iterations": fallback,
            "edges_violating_block": int((block_fn(pre, post) != block).sum()), "unrepaired": int(final_bad),
            "nnz": int(R.nnz), "in_degree_exact": bool((np.bincount(post, minlength=n) == np.bincount(post0, minlength=n)).all())}
    return R, info


def tiles(W, sb):
    """Rewired-L tiles (8 x 8 over the 16 x 16 patches, + global + none) from the footprint of the graph W."""
    Fp = fg.footprint(W, sb, fg.T_STEPS, dev)
    t, n_t = fg.tiles_from_footprint(Fp, 8)
    pn = Fp / Fp.sum(1, keepdim=True).clamp_min(1e-30)
    r = torch.arange(fg.N_PATCH, device=dev) // fg.N_SIDE
    c = torch.arange(fg.N_PATCH, device=dev) % fg.N_SIDE
    cen = torch.stack([(pn * r).sum(1), (pn * c).sum(1)], 1).cpu().numpy()
    return t, n_t, cen


def key_block(gkey):
    G = int(gkey.max()) + 1
    return lambda pre, post: gkey[pre] * G + gkey[post]


def null_graph(W, kind, seed, brain, S, sb, tile_info=None):
    """Registered nulls (PROTOCOL §A2) on the circuit graph W (neurons S)."""
    lab = labels(brain)
    sign = (lab.sign[S] > 0).astype(np.int64)
    stage = lab.stage_id[S].astype(np.int64)
    _, typ = np.unique(lab.key[S], return_inverse=True)
    if tile_info is None:
        tile_info = tiles(W, sb)
    tile, n_t, cen = tile_info
    tile = tile.astype(np.int64)
    if kind == "rewired_l":
        return rewire(W, key_block(brain.scside[S].astype(np.int64) * n_t + tile), seed)
    if kind == "deg":
        return rewire(W, key_block(np.zeros(len(S), np.int64)), seed)
    if kind == "deg_sign":
        return rewire(W, key_block(sign), seed)
    if kind == "deg_sign_stage":
        return rewire(W, key_block(sign * 16 + stage), seed)
    if kind == "deg_sign_loc":
        return rewire(W, key_block(sign * n_t + tile), seed)
    if kind == "deg_sign_stage_loc":
        return rewire(W, key_block((sign * 16 + stage) * n_t + tile), seed)
    if kind == "type":
        return rewire(W, key_block(typ.astype(np.int64)), seed)
    if kind == "type_loc":
        return rewire(W, key_block(typ.astype(np.int64) * n_t + tile), seed)
    if kind == "deg_sign_dir":
        return rewire_mcmc(W, lambda pre, post: sign[pre] * 2 + sign[post],
                           lambda pre, post: np.sign(stage[post] - stage[pre]), seed)
    if kind == "sign_shuffle":
        g = np.random.default_rng(seed)
        Wc = W.tocsc()
        has = np.flatnonzero(np.diff(Wc.indptr) > 0)
        s = np.where(lab.sign[S] > 0, 1.0, -1.0)
        s_new = s.copy()
        s_new[has] = s[has][g.permutation(len(has))]
        R = abs(W).tocsr() @ sparse.diags(s_new.astype(np.float32))
        R = R.tocsr()
        R.sort_indices()
        return R, {"seed": seed, "neurons_permuted": int(len(has)), "sign_changed_neurons": int((s_new[has] != s[has]).sum()),
                   "inhibitory_edge_fraction_real": float((W.data < 0).mean()), "inhibitory_edge_fraction_new": float((R.data < 0).mean())}
    if kind in ("no_longrange", "rand_edges"):
        coo = W.tocoo()
        local = (tile < 64)
        elig = local[coo.row] & local[coo.col]
        dist = np.linalg.norm(cen[coo.row] - cen[coo.col], axis=1)
        lr = elig & (dist > 4.0)
        if kind == "no_longrange":
            drop = lr
        else:
            g = np.random.default_rng(seed)
            drop = np.zeros(len(lr), bool)
            drop[g.choice(np.flatnonzero(elig), int(lr.sum()), replace=False)] = True
        keep = ~drop
        R = sparse.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=W.shape, dtype=np.float32)
        R.sort_indices()
        return R, {"seed": seed, "edges_removed": int(drop.sum()), "edges_eligible": int(elig.sum()), "edges": int(W.nnz),
                   "abs_weight_removed_fraction": float(abs(coo.data[drop]).sum() / abs(coo.data).sum())}
    raise ValueError(kind)


def in_sorted(keys, q):
    i = np.searchsorted(keys, q)
    return (i < len(keys)) & (keys[np.minimum(i, len(keys) - 1)] == q)


def rewire_mcmc(W, cls_fn, rel_fn, seed, target=5.0, max_rounds=600):
    """Double-connection swaps starting from the real graph: (a->b, c->d) -> (a->d, c->b), accepted only when both
    connections are in the same class (cls_fn: pre sign x post sign), both keep their relational label (rel_fn: the
    direction of the connection along the anatomical stage order) and no duplicate or self-loop arises. In- and
    out-degree exact. Stops after target x (number of connections) accepted swaps. Rows rescaled to the original |w|."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    coo = W.tocoo()
    pre, post, w = coo.col.astype(np.int64), coo.row.astype(np.int64).copy(), coo.data.copy()
    post0 = post.copy()
    E = len(pre)
    cls, rel0 = cls_fn(pre, post), rel_fn(pre, post)
    keys = np.sort(pre * n + post)
    accepted, rounds = 0, 0
    while accepted < target * E and rounds < max_rounds:
        rounds += 1
        perm = rng.permutation(E)
        a, b = perm[:E // 2], perm[E // 2:2 * (E // 2)]
        ok = (cls[a] == cls[b]) & (pre[a] != post[b]) & (pre[b] != post[a])
        ok &= (rel_fn(pre[a], post[b]) == rel0[a]) & (rel_fn(pre[b], post[a]) == rel0[b])
        k1, k2 = pre[a] * n + post[b], pre[b] * n + post[a]
        ok &= ~in_sorted(keys, k1) & ~in_sorted(keys, k2)
        c = np.flatnonzero(ok)
        allk = np.concatenate([k1[c], k2[c]])
        _, inv, cnt = np.unique(allk, return_inverse=True, return_counts=True)
        c = c[(cnt[inv[:len(c)]] == 1) & (cnt[inv[len(c):]] == 1)]
        aa, bb = a[c], b[c]
        post[aa], post[bb] = post[bb].copy(), post[aa].copy()
        accepted += len(c)
        keys = np.sort(pre * n + post)
    assert (rel_fn(pre, post) == rel0).all() and (cls_fn(pre, post) == cls).all()
    R = sparse.csr_matrix((w, (post, pre)), shape=W.shape, dtype=np.float32)
    target_rs = np.asarray(abs(W).sum(1)).ravel()
    have = np.asarray(abs(R).sum(1)).ravel()
    R = (sparse.diags(np.divide(target_rs, have, out=np.zeros_like(target_rs), where=have > 0).astype(np.float32)) @ R).tocsr()
    R.sort_indices()
    return R, {"seed": seed, "method": "mcmc double swaps", "accepted_swaps": int(accepted), "rounds": rounds,
               "swaps_per_connection": accepted / E, "changed_fraction": float((post != post0).mean()), "nnz": int(R.nnz),
               "duplicates": int(E - len(np.unique(pre * n + post)))}


def save_graph(f, R, info):
    np.savez(f, data=R.data, indices=R.indices, indptr=R.indptr, shape=np.array(R.shape), info=np.array(info, dtype=object))


def load_graph(f):
    d = np.load(f, allow_pickle=True)
    return sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"])), d["info"].item()


# ------------------------------------------------------------------ A3 ablation sets, A4 dynamics
def ablation_sets(brain, sets):
    """Targeted sets (registered) and, per seed, a random set of the same size drawn from the same superclass(es),
    matched in (in + out) degree decile within each superclass."""
    lab = labels(brain)
    key = lab.key
    med = [t for t in sets["U1"] if lab.stage[np.flatnonzero(key == t)[0]] == "medulla"]
    target = {
        "T4T5": np.flatnonzero(np.isin(key, sets["T4T5"])),
        "MED": np.flatnonzero(np.isin(key, med)),
        "LPVPN": np.flatnonzero(np.isin(key, sets["V1"])),
        "LC4": np.flatnonzero(key == "LC4"),
        "LPLC2": np.flatnonzero(key == "LPLC2"),
        "GF": np.flatnonzero(key == "DNp01"),
    }
    deg = np.diff(brain.W.tocsr().indptr) + np.diff(brain.W.tocsc().indptr)
    rand = {}
    for name, X in target.items():
        for s in SEEDS:
            g = np.random.default_rng(1000 * s + len(name))
            pick = []
            for sc in np.unique(lab.superclass[X]):
                Xs = X[lab.superclass[X] == sc]
                pool = np.setdiff1d(np.flatnonzero((lab.superclass == sc) & ~lab.is_input), X)
                qs = np.quantile(deg[np.r_[pool, Xs]], np.linspace(0, 1, 11)[1:-1])
                bp, bx = np.digitize(deg[pool], qs), np.digitize(deg[Xs], qs)
                for b in np.unique(bx):
                    need = int((bx == b).sum())
                    cand = pool[bp == b]
                    if len(cand) < need:                         # borrow from the neighbouring deciles
                        cand = pool[np.argsort(np.abs(bp - b), kind="stable")[:max(need, len(cand))]]
                    pick.append(g.choice(cand, need, replace=False))
            rand[f"{name}|{s}"] = np.unique(np.concatenate(pick))
    sizes = {k: int(len(v)) for k, v in target.items()}
    return target, rand, sizes, med


def ablate(W, X):
    """Silence the neurons X: remove every connection to or from them."""
    keep = np.ones(W.shape[0], np.float32)
    keep[X] = 0
    D = sparse.diags(keep)
    R = (D @ W @ D).tocsr()
    R.eliminate_zeros()
    return R


def dynamics_graph(W, variant, brain):
    """A4 graph-level variants (applied identically to Real and Rewired-L)."""
    lab = labels(brain)
    if variant == "full" or variant == "reset":
        return W
    if variant == "nosign":
        return abs(W).tocsr()
    if variant == "ff":
        coo = W.tocoo()
        keep = lab.stage_id[coo.row] > lab.stage_id[coo.col]
        R = sparse.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=W.shape, dtype=np.float32)
        R.sort_indices()
        return R
    if variant == "rawgain":
        gain = raw_gain(brain)
        return (sparse.diags(gain.astype(np.float32)) @ W).tocsr()
    raise ValueError(variant)


_GAIN = {}


def raw_gain(brain):
    """Per-post factor turning input shares back into synapse counts with one global scale: real weights are
    count / (the neuron's total input), every row sums to 1 and share / (smallest share of the row) is an integer, so
    total input is proportional to 1 / (smallest share), assuming the smallest connection of every row has the same
    synapse count (approximation, PROTOCOL §A4). Normalised to mean 1 over neurons with input."""
    if "g" not in _GAIN:
        A = abs(brain.W).tocsr()
        has = np.diff(A.indptr) > 0
        mn = np.minimum.reduceat(A.data, A.indptr[:-1][has])
        tot = np.zeros(brain.n)
        tot[has] = 1.0 / mn
        _GAIN["g"] = np.where(has, tot / tot[has].mean(), 0.0)
    return _GAIN["g"]


# ------------------------------------------------------------------ model (the D7 / D8 harness with any graph)
class Core:
    """Frozen rate dynamics on W; drive enters the circuit's input neurons; readout = time mean ('mean') or the last
    frame's steps ('last') of f(V) of the readout neurons. alpha 0.5 (full) or 1 (A4 'reset')."""

    def __init__(self, W, inputs, ro, window, alpha=0.5):
        self.W, self.WT = fg.to_csr(W, dev), fg.to_csr(W.T.tocsr(), dev)
        self.n = W.shape[0]
        self.inputs = torch.as_tensor(inputs, device=dev)
        self.ro = torch.as_tensor(ro, device=dev)
        self.window, self.alpha = window, alpha

    def __call__(self, u):
        import d6_translation as d6
        FR, SPF = d6.FRAMES, d6.STEPS_PER_FRAME
        B = u.shape[2]
        V = torch.zeros(self.n, B, device=dev)
        acc = torch.zeros(len(self.ro), B, device=dev)
        for f in range(FR):
            for _ in range(SPF):
                drive = fg._SpMM.apply(self.W, self.WT, fg.act(V)).index_add(0, self.inputs, u[f])
                V = V + self.alpha * (drive - V)
                if self.window == "mean" or f == FR - 1:
                    acc = acc + fg.act(V)[self.ro]
        return acc / (FR * SPF if self.window == "mean" else SPF)


class Net(nn.Module):
    """d7.FlyNet / d8.FlyDN with the core given (same construction order, so the same initialisation per seed)."""

    def __init__(self, brain, core, n_cls):
        super().__init__()
        import d6_translation as d6
        import d7_screen as d7
        self.enc = d6.AIEncoder(brain)
        self.net = core
        self.readout = d7.Readout(len(core.ro), n_cls)

    def forward(self, x):
        return self.readout(self.net(self.enc(x)).T)


TASK = {"M": ("0", 4, "mean"), "L": ("looming", 2, "last")}


def task_data(task):
    import d7_screen as d7
    import d8_looming as d8
    return d7.task_data("0") if task == "M" else d8.task_data()


# ------------------------------------------------------------------ run specs
def queue(part):
    names = []
    if part == "a1":
        for c in ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "L0", "L1", "L2", "L3", "L4", "L5"):
            names += [f"a1-{c[0]}-{c}-{g}-s{s}" for g in ("real", "rewired_l") for s in SEEDS]
    elif part == "a2":
        sel = json.loads((OUT / "verdict_a1.json").read_text())["selected"]
        for task in ("M", "L"):
            c = sel[task]
            names += [f"a2-{task}-{c}-{k}-s{s}" for k in NULLS for s in SEEDS]
    elif part == "a3":
        for task, abls in (("M", ("T4T5", "MED", "LPVPN")), ("L", ("LC4", "LPLC2", "GF", "T4T5"))):
            c = f"{task}0"
            names += [f"a3-{task}-{c}-real-{x}{a}-s{s}" for a in abls for x in ("", "rand_") for s in SEEDS]
    elif part == "a4":
        for task in ("M", "L"):
            names += [f"a4-{task}-{task}0-{g}-{v}-s{s}" for v in VARIANTS for g in ("real", "rewired_l") for s in SEEDS]
    return names


NULLS = ("deg", "deg_sign", "deg_sign_dir", "deg_sign_stage", "deg_sign_loc", "deg_sign_stage_loc", "type", "type_loc",
         "sign_shuffle", "no_longrange", "rand_edges")
VARIANTS = ("ff", "reset", "nosign", "rawgain")
ALIAS = {"L5": "M4", "L0": "M0"}


def circuit_arrays():
    z = np.load(OUT / "circuits.npz")
    return z


def graph_for(brain, circ, kind, seed):
    """Real or control graph of circuit circ (neuron set S); whole-CNS Rewired-L = the cached D5 graph."""
    z = circuit_arrays()
    S = z[f"{circ}_S"]
    if circ in ("M0", "L0"):
        if kind == "real":
            return brain.W, {"kind": "real"}
        if kind == "rewired_l":
            return fg.build_graph(brain, "rewired_l", seed)
    if kind == "real":
        return sub(brain.W, S), {"kind": "real"}
    f = GRAPHS / f"{ALIAS.get(circ, circ)}_{kind}_s{seed}.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} (run build / build_a2 first)")
    return load_graph(f)


def run(name, brain, data_cache):
    out = RUNS / f"{name}.json"
    if out.exists():
        return
    import d7_screen as d7
    t0 = time.time()
    part, task, circ, graph = name.split("-")[:4]
    seed = int(name.split("-s")[-1])
    extra = name.split("-")[4] if part in ("a3", "a4") else None
    z = circuit_arrays()
    S, ro = z[f"{circ}_S"], z[f"{circ}_ro"]
    W, ginfo = graph_for(brain, circ, graph, seed)
    alpha = 0.5
    if part == "a3":
        abl = extra.replace("rand_", "")
        X = z[f"abl_{abl}"] if not extra.startswith("rand_") else z[f"rand_{abl}_{seed}"]
        W = ablate(W, X)
        ginfo = {**ginfo, "ablated": int(len(X))}
    if part == "a4":
        W = dynamics_graph(W, extra, brain)
        alpha = 1.0 if extra == "reset" else 0.5
    sb = sub_brain(brain, S)
    ro_loc = np.searchsorted(S, ro)
    _, n_cls, window = TASK[task]
    if task not in data_cache:
        data_cache[task] = task_data(task)
    d = data_cache[task]
    torch.manual_seed(seed)
    model = Net(brain, Core(W, sb.inputs, ro_loc, window, alpha), n_cls).to(dev)
    best, curve, pred = d7.fit(model, d, n_cls)
    y = d["test"][1]
    ok = (pred == y).int()
    rec = {"name": name, "part": part, "task": task, "circuit": circ, "graph": graph, "extra": extra, "seed": seed,
           "graph_info": ginfo, "neurons": int(len(S)), "edges": int(W.nnz), "readout": int(len(ro)), "val_best": best,
           "curve": curve, "test_pred": pred.tolist(), "minutes": (time.time() - t0) / 60}
    if task == "M":
        rec["acc"] = float(ok.float().mean())
    else:
        import d8_looming as d8
        kind = d["test"][2]
        pos = np.isin(np.asarray(kind), [d8.KINDS.index(k) for k in d8.POSITIVE])
        okn = ok.numpy()
        rec["acc"] = float(0.5 * (okn[pos].mean() + okn[~pos].mean()))
        rec["rate_by_kind"] = {k: float(pred[kind == j].float().mean()) for j, k in enumerate(d8.KINDS)}
    out.write_text(json.dumps(rec))
    print(f"{name}: {rec['acc']:.4f} (val {best:.3f}, {rec['neurons']:,} neurons, {rec['edges']:,} edges, "
          f"{rec['minutes']:.1f} min)", flush=True)


def worker(slot, n_slots, parts):
    if os.environ.get("D10A_SMOKE"):
        import d7_screen as d7
        d7.EPOCHS = 1
    names = [n for p in parts for n in queue(p)]
    mine = names[slot::n_slots]
    brain = fg.load_malecns()
    labels(brain)
    cache = {}
    for name in mine:
        run(name, brain, cache)
    print(f"WORKER {slot} DONE ({len(mine)} runs)", flush=True)


# ------------------------------------------------------------------ build
def build():
    t0 = time.time()
    brain = fg.load_malecns()
    lab = labels(brain)
    C, sets = circuits(brain)
    target, rand, abl_sizes, med = ablation_sets(brain, sets)
    arrays = {}
    meta = {"sets": {**sets, "MED": med}, "circuits": {}, "ablation_sizes": abl_sizes,
            "stage_counts": {s: int((lab.stage == s).sum()) for s in STAGES}}
    for c, (S, ro, desc) in C.items():
        arrays[f"{c}_S"], arrays[f"{c}_ro"] = S, ro
        Wc = brain.W if len(S) == brain.n else sub(brain.W, S)
        meta["circuits"][c] = {"desc": desc, "neurons": int(len(S)), "edges": int(Wc.nnz), "readout": int(len(ro)),
                               "types": int(len(np.unique(lab.key[S])))}
        print(c, meta["circuits"][c], flush=True)
    for k, v in target.items():
        arrays[f"abl_{k}"] = v
    for k, v in rand.items():
        name, s = k.split("|")
        arrays[f"rand_{name}_{s}"] = v
        meta.setdefault("random_sets", {})[k] = {"n": int(len(v)), "edges_touched": int(
            (np.isin(brain.W.tocoo().row, v) | np.isin(brain.W.tocoo().col, v)).sum())}
    for k, v in target.items():
        meta.setdefault("target_edges_touched", {})[k] = int((np.isin(brain.W.tocoo().row, v) | np.isin(brain.W.tocoo().col, v)).sum())
    np.savez(OUT / "circuits.npz", **arrays)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    for c, (S, ro, desc) in C.items():
        if c in ("M0", "L0") or c in ALIAS:
            continue
        Wc = sub(brain.W, S)
        sb = sub_brain(brain, S)
        ti = tiles(Wc, sb)
        for s in SEEDS:
            f = GRAPHS / f"{c}_rewired_l_s{s}.npz"
            if f.exists():
                continue
            R, info = null_graph(Wc, "rewired_l", s, brain, S, sb, ti)
            info["kind"] = "rewired_l"
            save_graph(f, R, info)
            print(c, "rewired_l", s, info, flush=True)
    for s in SEEDS:
        fg.build_graph(brain, "rewired_l", s)
    print(f"build done {(time.time() - t0) / 60:.1f} min", flush=True)


def build_a2():
    brain = fg.load_malecns()
    labels(brain)
    sel = json.loads((OUT / "verdict_a1.json").read_text())["selected"]
    z = circuit_arrays()
    for task in ("M", "L"):
        c = sel[task]
        S = z[f"{c}_S"]
        Wc = brain.W if len(S) == brain.n else sub(brain.W, S)
        sb = sub_brain(brain, S)
        ti = tiles(Wc, sb)
        for kind in NULLS:
            for s in SEEDS:
                f = GRAPHS / f"{ALIAS.get(c, c)}_{kind}_s{s}.npz"
                if f.exists():
                    continue
                t0 = time.time()
                R, info = null_graph(Wc, kind, s, brain, S, sb, ti)
                info["kind"], info["seconds"] = kind, time.time() - t0
                save_graph(f, R, info)
                print(task, c, kind, s, info, flush=True)


# ------------------------------------------------------------------ verdicts (PROTOCOL §5)
def test_meta(task):
    import d6_translation as d6
    import d8_looming as d8
    if task == "M":
        y = np.load(d6.OUT / "videos.npz")["test_y"]
        return y, None
    z = np.load(d8.OUT / "looming.npz")
    pos = np.isin(z["test_k"], [d8.KINDS.index(k) for k in d8.POSITIVE])
    return z["test_y"], (pos, z["test_k"])


def metric(task):
    y, extra = test_meta(task)
    if task == "M":
        return lambda p, idx: float((p[idx] == y[idx]).mean())
    pos = extra[0]

    def bal(p, idx):
        ok, q = p[idx] == y[idx], pos[idx]
        return float(0.5 * (ok[q].mean() + ok[~q].mean()))
    return bal


def preds(name):
    f = RUNS / f"{name}.json"
    return np.asarray(json.loads(f.read_text())["test_pred"]) if f.exists() else None


_BOOT = {}


def compare(task, A, B):
    """Paired comparison of two lists of per-seed predictions (seed-matched): per-seed metric differences, paired
    bootstrap (2,000 resamples of the test items, seed-mean difference), effect = all seeds > 0 and CI low > delta."""
    if any(a is None for a in A) or any(b is None for b in B):
        return None
    f = metric(task)
    n = len(A[0])
    if (task, n) not in _BOOT:
        g = np.random.default_rng(0)
        _BOOT[(task, n)] = [g.integers(0, n, n) for _ in range(2000)]
    full = np.arange(n)
    per = [f(a, full) - f(b, full) for a, b in zip(A, B)]
    dist = np.array([np.mean([f(a, bb) - f(b, bb) for a, b in zip(A, B)]) for bb in _BOOT[(task, n)]])
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return {"per_seed": [float(x) for x in per], "mean": float(np.mean(per)), "ci95": [float(lo), float(hi)],
            "effect": bool(all(x > 0 for x in per) and lo > DELTA)}


def accs(task, P):
    f = metric(task)
    return [None if p is None else f(p, np.arange(len(p))) for p in P]


def eff(c):
    return bool(c is not None and c["effect"])


FLOOR = {"M": 0.30, "L": 0.55}
CEIL = 0.97
M_CIRCS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
L_CIRCS = ("L0", "L1", "L2", "L3", "L4", "L5")
PATHWAY = {"M": ("M2", "M3", "M4", "M5", "M6"), "L": ("L2", "L3", "L4", "L5")}


def verdict_a1():
    meta = json.loads((OUT / "meta.json").read_text())
    rep = {"circuits": {}, "selected": {}, "reference_reproduced": {}}
    for task, cs in (("M", M_CIRCS), ("L", L_CIRCS)):
        for c in cs:
            R = [preds(f"a1-{task}-{c}-real-s{s}") for s in SEEDS]
            Wl = [preds(f"a1-{task}-{c}-rewired_l-s{s}") for s in SEEDS]
            cmp = compare(task, R, Wl)
            ar, aw = accs(task, R), accs(task, Wl)
            mr = np.mean(ar) if None not in ar else None
            mw = np.mean(aw) if None not in aw else None
            unin = None if mr is None or mw is None else bool((mr <= FLOOR[task] and mw <= FLOOR[task]) or (mr >= CEIL and mw >= CEIL))
            rep["circuits"][c] = {**meta["circuits"][c], "real": ar, "rewired_l": aw, "real_mean": mr, "rewired_l_mean": mw,
                                  "real-rewired_l": cmp, "uninformative": unin,
                                  "retains": bool(eff(cmp) and not unin)}
        ret = [c for c in cs if rep["circuits"][c]["retains"]]
        rep["reference_reproduced"][task] = rep["circuits"][cs[0]]["retains"]
        rep["selected"][task] = min(ret, key=lambda c: (rep["circuits"][c]["neurons"], rep["circuits"][c]["edges"])) if ret else cs[0]
    (OUT / "verdict_a1.json").write_text(json.dumps(rep, indent=1))
    for c, r in rep["circuits"].items():
        cm = r["real-rewired_l"]
        print(f"{c:3s} {r['neurons']:>7,} n {r['edges']:>10,} e ro {r['readout']:>5,} | real {r['real_mean']} rewired {r['rewired_l_mean']} | "
              f"{None if cm is None else (round(cm['mean'], 4), [round(x, 4) for x in cm['ci95']], cm['effect'])} retains {r['retains']}")
    print("selected", rep["selected"], "reference reproduced", rep["reference_reproduced"])
    return rep


def bypass_preds(task):
    import d7_screen as d7
    import d8_looming as d8
    if task == "M":
        return [np.asarray(json.loads((d7.OUT / f"task0_bypass_s{s}.json").read_text())["test_pred"]) for s in SEEDS]
    return [np.asarray(json.loads((d8.OUT / f"bypass_s{s}.json").read_text())["test_pred"]) for s in SEEDS]


def fa_linear_zoom(P):
    import d8_looming as d8
    _, (pos, kind) = test_meta("L")
    m = np.isin(kind, [d8.KINDS.index("linear"), d8.KINDS.index("zoom")])
    return [None if p is None else float((p[m] == 1).mean()) for p in P]


def verdict():
    a1 = verdict_a1()
    rep = {"a1": a1, "a2": {}, "a3": {}, "a4": {}, "labels": {}}
    for task in ("M", "L"):
        c0 = f"{task}0"
        real0 = [preds(f"a1-{task}-{c0}-real-s{s}") for s in SEEDS]
        rew0 = [preds(f"a1-{task}-{c0}-rewired_l-s{s}") for s in SEEDS]
        # A2
        c = a1["selected"][task]
        realS = [preds(f"a1-{task}-{c}-real-s{s}") for s in SEEDS]
        rewS = [preds(f"a1-{task}-{c}-rewired_l-s{s}") for s in SEEDS]
        a2 = {"circuit": c}
        for k in NULLS:
            N = [preds(f"a2-{task}-{c}-{k}-s{s}") for s in SEEDS]
            a2[k] = {"acc": accs(task, N), "real-null": compare(task, realS, N), "null-rewired_l": compare(task, N, rewS)}
        a2["deg_sign_vs_stage"] = compare(task, [preds(f"a2-{task}-{c}-deg_sign_stage-s{s}") for s in SEEDS],
                                          [preds(f"a2-{task}-{c}-deg_sign-s{s}") for s in SEEDS])
        a2["deg_sign_vs_dir"] = compare(task, [preds(f"a2-{task}-{c}-deg_sign_dir-s{s}") for s in SEEDS],
                                        [preds(f"a2-{task}-{c}-deg_sign-s{s}") for s in SEEDS])
        rep["a2"][task] = a2
        # A3
        a3 = {}
        for a in (("T4T5", "MED", "LPVPN") if task == "M" else ("LC4", "LPLC2", "GF", "T4T5")):
            T = [preds(f"a3-{task}-{c0}-real-{a}-s{s}") for s in SEEDS]
            Rn = [preds(f"a3-{task}-{c0}-real-rand_{a}-s{s}") for s in SEEDS]
            ct, cr = compare(task, T, rew0), compare(task, Rn, rew0)
            a3[a] = {"targeted": accs(task, T), "random": accs(task, Rn), "targeted-rewired_l": ct, "random-rewired_l": cr,
                     "real-targeted": compare(task, real0, T),
                     "mechanistic_positive": bool(ct is not None and cr is not None and not eff(ct) and eff(cr))}
            if task == "L":
                a3[a]["fa_linear_zoom"] = {"targeted": fa_linear_zoom(T), "random": fa_linear_zoom(Rn), "real": fa_linear_zoom(real0)}
        rep["a3"][task] = a3
        # A4
        byp = bypass_preds(task)
        a4 = {"full": {"real-rewired_l": compare(task, real0, rew0)}}
        for v in VARIANTS:
            Rv = [preds(f"a4-{task}-{c0}-real-{v}-s{s}") for s in SEEDS]
            Wv = [preds(f"a4-{task}-{c0}-rewired_l-{v}-s{s}") for s in SEEDS]
            top, rb, wb = compare(task, Rv, Wv), compare(task, Rv, byp), compare(task, Wv, byp)
            ar = accs(task, Rv)
            floor = None if None in ar else bool(np.mean(ar) <= FLOOR[task] + 0.05)
            if top is None:
                vv = None
            elif eff(top):
                vv = "TOPOLOGY"
            elif eff(rb) and eff(wb):
                vv = "GENERIC RESERVOIR"
            else:
                vv = "NO EFFECT"
            a4[v] = {"real": ar, "rewired_l": accs(task, Wv), "real-rewired_l": top, "real-bypass": rb, "rewired_l-bypass": wb,
                     "real_near_floor": floor, "verdict": vv}
        rep["a4"][task] = a4
        # labels (PROTOCOL §5.3)
        L = []
        if not a1["reference_reproduced"][task]:
            rep["labels"][task] = ["REFERENCE NOT REPRODUCED"]
            continue
        path = [x for x in PATHWAY[task] if a1["circuits"][x]["retains"]]
        mech = [a for a, r in a3.items() if r["mechanistic_positive"]]
        lr = a2.get("no_longrange", {})
        lr_pos = bool(lr and a2["rand_edges"]["null-rewired_l"] is not None and not eff(lr["null-rewired_l"]) and eff(a2["rand_edges"]["null-rewired_l"]))
        if path or mech or lr_pos:
            L.append({"SPECIFIC PATHWAY": {"retaining_circuits": path, "mechanistic_ablations": mech, "long_range_edges": lr_pos}})
        if eff(a2["type"]["real-null"]):
            L.append({"RETINOTOPIC LOCALITY": {"fine_arrangement_needed": eff(a2["type_loc"]["real-null"])}})
        st = (not eff(a2["deg_sign_stage"]["real-null"]) and eff(a2["deg_sign_vs_stage"])) or \
             (not eff(a2["deg_sign_dir"]["real-null"]) and eff(a2["deg_sign_vs_dir"]))
        if st:
            L.append("STAGE-ORDERED CONNECTIVITY")
        if eff(a2["sign_shuffle"]["real-null"]):
            L.append({"E/I OR SIGN STRUCTURE": {"a4_nosign": a4["nosign"]["verdict"]}})
        if a4["ff"]["verdict"] is not None and a4["ff"]["verdict"] != "TOPOLOGY" and a4["ff"]["real_near_floor"] is False:
            L.append({"RECURRENT MOTIF": {"a4_reset": a4["reset"]["verdict"]}})
        whole = any(a1["circuits"][x]["retains"] for x in (f"{task}0", f"{task}1"))
        if not path and not mech and whole:
            L.append("DISTRIBUTED WHOLE-CIRCUIT EFFECT")
        rep["labels"][task] = L or ["UNRESOLVED"]
    (OUT / "verdict.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep["labels"], indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "build_a2":
        build_a2()
    elif cmd == "queue":
        print("\n".join(queue(sys.argv[2])))
    elif cmd == "worker":
        worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:])
    elif cmd == "one":                                   # single named run (with D10A_SMOKE: 1 epoch, code check)
        if os.environ.get("D10A_SMOKE"):
            import d7_screen as d7
            d7.EPOCHS = 1
        b = fg.load_malecns()
        labels(b)
        cache = {}
        for nm in sys.argv[2:]:
            run(nm, b, cache)
    elif cmd == "verdict_a1":
        verdict_a1()
    elif cmd == "verdict":
        verdict()
