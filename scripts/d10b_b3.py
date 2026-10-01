"""D10-B part B3 / B4 (PROTOCOL_D10B_tsinghua.md): run only if B2 is REPLICATED or PARTIAL.
  B3a  the bar protocol on a rewired FlyWire connectome (Rewired-FW: degree-, sign-, superclass x side- and
       retinotopic-tile-preserving; each post keeps its total |input|)
  B3b  natural images (CIFAR-100 test, first 200) shown through the same OFF L1-L3 Poisson pipeline: 1 s gray, 3 s image;
       Real vs Rewired-FW vs Gabor vs Sobel; orientation information, localisation, sparsity, stability, CKA
usage: d10b_b3.py rewire | bars | images GRAPH | measures | b4
"""
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage, sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d10b_tsinghua as m  # noqa: E402

dev = "cuda"
N_IMG, GRAY_MS, IMG_MS, BASE_WIN_MS = 200, 1000.0, 3000.0, 500.0
CONDS = ("clean", "retest", "noise", "lowcon", "blur")
OUT = m.DATA / "b3"
OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ positions and Rewired-FW
def columnar_centres():
    """For every model neuron: synapse-weighted centroid (lattice x, y) of its upstream columnar inputs (the authors'
    31 columnar types, syn_count > 1) and the eye (hemisphere) giving the majority of those synapses; NaN if none."""
    z, _ = m.load_model()
    roots = z["roots"]
    conn = pd.read_csv(m.RUN / "flywire" / "connections_no_threshold.csv", usecols=["pre_root_id", "post_root_id", "syn_count"])
    cols = pd.read_csv(m.RUN / "flywire" / "column_assignment.csv").drop_duplicates("root_id", keep="last")
    cols = cols[cols["type"].isin(m.COLUMNAR)].set_index("root_id")
    f = conn[conn.pre_root_id.isin(cols.index) & (conn.syn_count > 1)]
    f = f.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"].sum()
    p, q = cols.p.reindex(f.pre_root_id).values, cols.q.reindex(f.pre_root_id).values
    right = (cols.hemisphere.reindex(f.pre_root_id).values == "right").astype(float)
    f = f.assign(x=(q - p) * np.sqrt(3) / 2 * f.syn_count, y=(p + q) / 2 * f.syn_count, r=right * f.syn_count)
    g = f.groupby("post_root_id")[["x", "y", "r", "syn_count"]].sum()
    ix = pd.Series(np.arange(len(roots)), index=roots)
    cen = np.full((len(roots), 2), np.nan)
    eye = np.full(len(roots), -1)
    k = ix[g.index].values
    cen[k] = np.stack([g.x / g.syn_count, g.y / g.syn_count], 1)
    eye[k] = (g.r / g.syn_count >= 0.5).astype(int).values
    return cen, eye


def rewired_fw(seed=1):
    f = OUT / f"rewired_fw_s{seed}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        return sparse.csc_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"])), d["info"].item()
    z, W = m.load_model()
    roots = z["roots"]
    cl = pd.read_csv(m.REF / "classification.csv.gz").drop_duplicates("root_id", keep="last").set_index("root_id")
    sc = cl.super_class.reindex(roots).fillna("unknown").values.astype(str)
    side = cl.side.reindex(roots).fillna("unknown").values.astype(str)
    _, scside = np.unique(np.char.add(np.char.add(sc, "|"), side), return_inverse=True)
    cen, eye = columnar_centres()
    tile = np.full(len(roots), 128)
    for e in (0, 1):
        mk = (eye == e) & ~np.isnan(cen[:, 0])
        lo, hi = cen[mk].min(0), cen[mk].max(0)
        tt = np.clip(((cen[mk] - lo) / (hi - lo + 1e-9) * 8).astype(int), 0, 7)
        tile[mk] = e * 64 + tt[:, 1] * 8 + tt[:, 0]
    gkey = (scside.astype(np.int64) * 129 + tile).astype(np.int64)
    G = int(gkey.max()) + 1
    R, info = rewire(W.tocsr(), lambda pre, post: gkey[pre] * G + gkey[post], seed)
    R = R.tocsc()
    np.savez(f, data=R.data, indices=R.indices, indptr=R.indptr, shape=np.array(R.shape), info=np.array(info, dtype=object))
    print("rewired_fw", info, flush=True)
    return R, info


def rewire(W, block_fn, seed, max_iter=300):
    """Same algorithm as D5-D10-A rewire_blocks: permute postsynaptic ends within blocks (pre, weight, sign kept; in-
    and out-degree exact), repair duplicates / self-loops inside the block, rescale rows to the original sum of |w|."""
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
    R = sparse.csr_matrix((w, (post, pre)), shape=W.shape, dtype=np.float64)
    target = np.asarray(abs(W).sum(1)).ravel()
    have = np.asarray(abs(R).sum(1)).ravel()
    R = (sparse.diags(np.divide(target, have, out=np.zeros_like(target), where=have > 0)) @ R).tocsr()
    info = {"seed": seed, "blocks": int(len(starts)), "changed_fraction": float((post != post0).mean()),
            "fix_iterations": len(history), "fallback_iterations": fallback,
            "edges_violating_block": int((block_fn(pre, post) != block).sum()), "unrepaired": int(len(bad(post))),
            "nnz": int(R.nnz), "in_degree_exact": bool((np.bincount(post, minlength=n) == np.bincount(post0, minlength=n)).all())}
    return R, info


# ------------------------------------------------------------------ images through the same OFF pipeline
def images():
    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(m.DATA.parent / "datasets" / "cifar100" / "test.parquet").slice(0, N_IMG).to_pydict()
    rgb = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB"), np.float64) / 255 for d in t["img"]])
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]          # (N, 32, 32)


def corrupt(L, cond, seed=0):
    g = np.random.default_rng(seed)
    if cond in ("clean", "retest"):
        return L
    if cond == "noise":
        return np.clip(L + g.normal(0, 0.1, L.shape), 0, 1)
    if cond == "lowcon":
        mu = L.mean((1, 2), keepdims=True)
        return mu + 0.3 * (L - mu)
    if cond == "blur":
        return np.stack([ndimage.gaussian_filter(x, 1.0, mode="reflect") for x in L])
    raise ValueError(cond)


def eye_frame():
    """Right-eye L columns in image coordinates: lattice (x, y) mapped onto the 32 x 32 image, aspect kept, centred;
    lattice +y = image up."""
    r = np.load(m.DATA / "configs_replica.npz")
    cols = pd.read_csv(m.RUN / "flywire" / "column_assignment.csv").drop_duplicates("root_id", keep="last").set_index("root_id")
    cells = r["cells"]
    has = np.array([c in cols.index for c in cells])
    p = np.array([cols.p[c] + 19 if h else np.nan for c, h in zip(cells, has)])
    q = np.array([cols.q[c] + 17 if h else np.nan for c, h in zip(cells, has)])
    x, y = (q - p) * np.sqrt(3) / 2, (p + q) / 2
    lo = np.array([np.nanmin(x), np.nanmin(y)])
    span = max(np.nanmax(x) - lo[0], np.nanmax(y) - lo[1])
    s = 31.0 / span
    cx = (x - lo[0]) * s + (31 - (np.nanmax(x) - lo[0]) * s) / 2
    cy = (y - lo[1]) * s + (31 - (np.nanmax(y) - lo[1]) * s) / 2
    col_img, row_img = cx, 31 - cy
    return has, row_img, col_img, s


def encode(L, has, row_img, col_img):
    """Level (1..10) per driven L cell: OFF contrast s = clip((mu - L)/(mu - L_1%), 0, 1), level = max(1, ceil(K s))."""
    r = np.load(m.DATA / "configs_replica.npz")
    z, _ = m.load_model()
    pos = pd.Series(np.arange(len(z["roots"])), index=z["roots"])
    tp = z["vtype"][pos[r["cells"]].values]
    K = np.select([tp == "L1", tp == "L2", tp == "L3"], [7, 5, 10], 0).astype(float)
    lev = np.ones((len(L), len(r["cells"])), np.int8)
    for i, img in enumerate(L):
        mu, lo = img.mean(), np.percentile(img, 1)
        val = ndimage.map_coordinates(img, [row_img[has], col_img[has]], order=1, mode="nearest")
        s = np.clip((mu - val) / max(mu - lo, 1e-6), 0, 1)
        lev[i, has] = np.maximum(1, np.ceil(K[has] * s)).astype(np.int8)
    return lev


def sim_images(graph="real", B=250):
    f = OUT / f"img_rates_{graph}.npz"
    if f.exists():
        return
    z, W0 = m.load_model()
    W = W0 if graph == "real" else rewired_fw(1)[0]
    r = np.load(m.DATA / "configs_replica.npz")
    pos = pd.Series(np.arange(len(z["roots"])), index=z["roots"])
    lif = m.LIF(W, pos[r["cells"]].values, pos[r["out"]].values)
    has, row_img, col_img, _ = eye_frame()
    L = images()
    gray = torch.ones((len(r["cells"]), 1), dtype=torch.int64)
    res, base = {}, {}
    for c in CONDS:
        lev = encode(corrupt(L, c, seed=7), has, row_img, col_img)
        out, bl = [], []
        for s in range(0, N_IMG, B):
            part = torch.as_tensor(lev[s:s + B].T.astype(np.int64))
            seed = (1000 if c == "retest" else 0) + s + CONDS.index(c) * 10_000
            cnt = run_phases(lif, [(gray.expand(-1, part.shape[1]), int(GRAY_MS / m.DT)), (part, int(IMG_MS / m.DT))],
                             seed, [(int((GRAY_MS - BASE_WIN_MS) / m.DT), int(GRAY_MS / m.DT)),
                                    (int(GRAY_MS / m.DT), int((GRAY_MS + IMG_MS) / m.DT))])
            bl.append((cnt[0].double() / (BASE_WIN_MS / 1000)).cpu().numpy())
            out.append((cnt[1].double() / (IMG_MS / 1000)).cpu().numpy())
        res[c], base[c] = np.concatenate(out, 1).astype(np.float32), np.concatenate(bl, 1).astype(np.float32)
        print(graph, c, "done", flush=True)
    np.savez(f, **{f"rate_{c}": v for c, v in res.items()}, **{f"base_{c}": v for c, v in base.items()})


@torch.no_grad()
def run_phases(lif, phases, seed, windows):
    """LIF.run with piecewise-constant input levels; spike counts of the out neurons inside each [start, end) window."""
    B, D = phases[0][0].shape[1], m.DELAY_STEPS
    gen = torch.Generator(device=dev).manual_seed(seed)
    v = torch.full((lif.n, B), m.V_REST, dtype=torch.float64, device=dev)
    g = torch.zeros_like(v)
    last = torch.full((lif.n, B), -10**8, dtype=torch.int32, device=dev)
    counts = [torch.zeros((len(lif.out), B), dtype=torch.int32, device=dev) for _ in windows]
    hist = torch.zeros((D, lif.n, B), dtype=torch.bool, device=dev)
    deliv = torch.zeros((D, lif.n, B), dtype=torch.float64, device=dev)
    tt = torch.zeros((), dtype=torch.int32, device=dev)
    sched = [(t0, t0 + n, lev) for (lev, n), t0 in zip(phases, np.cumsum([0] + [n for _, n in phases])[:-1])]
    total = sched[-1][1]
    for t in range(total):
        lev = next(l for a, b, l in sched if a <= t < b)
        p = (m.BASE_HZ * lev.to(torch.float64) * m.DT * 1e-3).to(dev)
        k = t % D
        tt.fill_(t)
        v, g, last, spk = m._step(v, g, last, deliv[k], tt, lif.ref, lif.a_m, lif.a_s, lif.c_g)
        hit = torch.rand(p.shape, generator=gen, device=dev, dtype=torch.float64) < p
        v[lif.drive] += m.KICK * (hit & ~spk[lif.drive])
        hist[k] = spk
        for c, (a, b) in zip(counts, windows):
            if a <= t < b:
                c += spk[lif.out]
        if k == D - 1:
            deliv.zero_()
            f = torch.nonzero(hist.view(-1)).squeeze(1)
            slot, rest = f // (lif.n * B), f % (lif.n * B)
            lif.deliver_block(deliv, slot, rest // B, rest % B, B)
    return counts


# ------------------------------------------------------------------ features and measures
def gabor_bank(L):
    """6 orientations x wavelengths 4, 8 px, even/odd -> energy sqrt(e^2 + o^2): (N, 12, 32, 32)."""
    out = []
    yy, xx = np.mgrid[-8:9, -8:9].astype(float)
    for lam in (4.0, 8.0):
        sig = 0.56 * lam
        for k in range(6):
            th = k * np.pi / 6
            xr = xx * np.cos(th) + yy * np.sin(th)
            env = np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2))
            ev, od = env * np.cos(2 * np.pi * xr / lam), env * np.sin(2 * np.pi * xr / lam)
            ev -= ev.mean()
            e = np.stack([ndimage.convolve(x - x.mean(), ev, mode="reflect") for x in L])
            o = np.stack([ndimage.convolve(x - x.mean(), od, mode="reflect") for x in L])
            out.append(np.sqrt(e ** 2 + o ** 2))
    return np.stack(out, 1)


def sobel_bank(L):
    gx = np.stack([ndimage.sobel(x, 1, mode="reflect") for x in L])
    gy = np.stack([ndimage.sobel(x, 0, mode="reflect") for x in L])
    return np.stack([gx, gy, np.hypot(gx, gy)], 1)


def structure_tensor(L, sigma=1.5):
    th, coh = [], []
    for x in L:
        gx, gy = ndimage.sobel(x, 1, mode="reflect"), ndimage.sobel(x, 0, mode="reflect")
        jxx, jyy, jxy = (ndimage.gaussian_filter(a, sigma) for a in (gx * gx, gy * gy, gx * gy))
        lam_d = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
        coh.append(lam_d / np.maximum(jxx + jyy, 1e-12))
        th.append(0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2)        # edge orientation (perpendicular to gradient)
    return np.stack(th), np.stack(coh)


def sample(maps, rows, cols):
    """maps (N, C, 32, 32) -> (N, P, C) at image positions (bilinear)."""
    N, C = maps.shape[:2]
    out = np.empty((N, len(rows), C))
    for i in range(N):
        for c in range(C):
            out[i, :, c] = ndimage.map_coordinates(maps[i, c], [rows, cols], order=1, mode="nearest")
    return out


def fly_features(rates, types, cen_img, pop, rows, cols, radius_px):
    """(N_out, N_img) rates -> (N_img, P, len(pop)): per evaluation position, mean rate of each population type over its
    neurons whose receptive-field centre lies within the radius."""
    F = np.zeros((rates.shape[1], len(rows), len(pop)))
    pts = np.stack([rows, cols], 1)
    for j, tp in enumerate(pop):
        mk = (types == tp) & ~np.isnan(cen_img[:, 0])
        if not mk.any():
            continue
        d = np.linalg.norm(cen_img[mk][None] - pts[:, None], axis=2)              # (P, n_tp)
        A = (d <= radius_px).astype(float)
        A /= np.maximum(A.sum(1, keepdims=True), 1)
        F[:, :, j] = (A @ rates[mk]).T
    return F


def ridge_r2(X, Y, groups, alpha=1.0, folds=5):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(folds).split(X, Y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        mdl = Ridge(alpha=alpha).fit((X[tr] - mu) / sd, Y[tr])
        pred[te] = mdl.predict((X[te] - mu) / sd)
    ss = ((Y - pred) ** 2).sum(0)
    st = ((Y - Y.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss / st))


def linear_cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return float(hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def treves_rolls(F):
    """F (N, P, C) -> mean over images of 1 - sparseness (higher = sparser)."""
    a = F.reshape(len(F), -1)
    a = np.maximum(a, 0)
    s = (a.mean(1) ** 2) / np.maximum((a ** 2).mean(1), 1e-30)
    n = a.shape[1]
    return float(np.mean((1 - s) / (1 - 1 / n)))


def map_corr(A, B):
    """Mean over images of the Pearson correlation between two (P x C) response maps."""
    out = []
    for a, b in zip(A.reshape(len(A), -1), B.reshape(len(B), -1)):
        if a.std() > 0 and b.std() > 0:
            out.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(out)) if out else None


def measures():
    z, _ = m.load_model()
    r = np.load(m.DATA / "configs_replica.npz")
    pos = pd.Series(np.arange(len(z["roots"])), index=z["roots"])
    types = z["vtype"][pos[r["out"]].values]
    b2 = json.loads((m.DATA / "analysis_real_s0.json").read_text())
    pop = sorted(set(["Dm3p", "Dm3q", "Dm3v", "TmY9q", "TmY9q__perp", "Dm15"]) | set(b2["max"]["selective_types"]))
    has, row_img, col_img, s = eye_frame()
    rows, cols = row_img[has], col_img[has]
    keep = np.unique(np.round(np.stack([rows, cols], 1), 3), axis=0, return_index=True)[1]
    rows, cols = rows[keep], cols[keep]                                          # one evaluation point per column
    cen, _ = columnar_centres()
    cen_out = cen[pos[r["out"]].values]
    lo_x = None
    # same lattice -> image transform as eye_frame (recomputed from the driven cells)
    cells = r["cells"]
    colsdf = pd.read_csv(m.RUN / "flywire" / "column_assignment.csv").drop_duplicates("root_id", keep="last").set_index("root_id")
    p = np.array([colsdf.p[c] + 19 if c in colsdf.index else np.nan for c in cells])
    q = np.array([colsdf.q[c] + 17 if c in colsdf.index else np.nan for c in cells])
    x, y = (q - p) * np.sqrt(3) / 2, (p + q) / 2
    lo = np.array([np.nanmin(x), np.nanmin(y)])
    span = max(np.nanmax(x) - lo[0], np.nanmax(y) - lo[1])
    sc = 31.0 / span
    offx, offy = (31 - (np.nanmax(x) - lo[0]) * sc) / 2, (31 - (np.nanmax(y) - lo[1]) * sc) / 2
    cen_img = np.stack([31 - ((cen_out[:, 1] - lo[1]) * sc + offy), (cen_out[:, 0] - lo[0]) * sc + offx], 1)
    radius = 2 * sc                                                              # 2 columns in image pixels
    L = images()
    th, coh = structure_tensor(L)
    T = sample(np.stack([np.cos(2 * th), np.sin(2 * th)], 1), rows, cols)        # (N, P, 2)
    C = sample(coh[:, None], rows, cols)[..., 0]
    E = sample(np.hypot(*[np.stack([ndimage.sobel(ndimage.gaussian_filter(x, 1.0), a) for x in L]) for a in (1, 0)])[:, None],
               rows, cols)[..., 0]
    reps = {}
    for g in ("real", "rewired"):
        f = OUT / f"img_rates_{g}.npz"
        if f.exists():
            d = np.load(f)
            reps[g] = {c: fly_features(d[f"rate_{c}"], types, cen_img, pop, rows, cols, radius) for c in CONDS}
    reps["gabor"] = {c: sample(gabor_bank(corrupt(L, c, 7)), rows, cols) for c in CONDS if c != "retest"}
    reps["sobel"] = {c: sample(sobel_bank(corrupt(L, c, 7)), rows, cols) for c in CONDS if c != "retest"}
    groups = np.repeat(np.arange(len(L)), len(rows))
    mask = (C > 0.3).ravel()
    res = {"population": pop, "positions": int(len(rows)), "radius_px": radius}
    for name, R in reps.items():
        Fc = R["clean"]
        X = Fc.reshape(-1, Fc.shape[-1])
        r2 = ridge_r2(X[mask], T.reshape(-1, 2)[mask], groups[mask])
        loc = map_corr(Fc.sum(-1, keepdims=True), E[..., None])
        res[name] = {"orientation_r2": r2, "localisation_r": loc, "sparsity": treves_rolls(Fc),
                     "stability": {c: map_corr(Fc, R[c]) for c in ("noise", "lowcon", "blur")}}
        if "retest" in R:
            res[name]["retest_r"] = map_corr(Fc, R["retest"])
    if "real" in reps:
        Xf = reps["real"]["clean"].reshape(-1, len(pop))
        res["cka"] = {"real_gabor": linear_cka(Xf, reps["gabor"]["clean"].reshape(-1, 12)),
                      "real_sobel": linear_cka(Xf, reps["sobel"]["clean"].reshape(-1, 3))}
        if "rewired" in reps:
            res["cka"]["real_rewired"] = linear_cka(Xf, reps["rewired"]["clean"].reshape(-1, len(pop)))
        Xg = reps["gabor"]["clean"].reshape(-1, 12)
        res["real_plus_gabor_r2"] = ridge_r2(np.concatenate([Xf, Xg], 1)[mask], T.reshape(-1, 2)[mask], groups[mask])
    (OUT / "measures.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return res


def b4():
    """PROTOCOL §B4 decision."""
    real = json.loads((m.DATA / "analysis_real_s0.json").read_text())["max"]
    rew = json.loads((m.DATA / "analysis_rewired_s0.json").read_text())["max"]
    me = json.loads((OUT / "measures.json").read_text())
    c1 = real["verdict"] == "REPLICATED"
    c2 = (me["real"].get("retest_r") or 0) >= 0.5
    frac = lambda a: np.mean([a["dm3"][t]["frac"] for t in ("Dm3p", "Dm3q", "Dm3v")])
    c3a = abs(frac(real) - frac(rew)) >= 0.30
    c3b = not all(rew["dm3"][t]["G2"] for t in ("Dm3p", "Dm3q", "Dm3v"))
    c3c = abs(me["real"]["orientation_r2"] - me["rewired"]["orientation_r2"]) >= 0.10
    c3 = c3a or c3b or c3c
    c4 = me["cka"]["real_gabor"] < 0.9 and me["cka"]["real_sobel"] < 0.9
    beyond = (sum(me["real"]["stability"][c] - me["gabor"]["stability"][c] >= 0.05 for c in ("noise", "lowcon", "blur")) >= 2
              or me["real_plus_gabor_r2"] - me["gabor"]["orientation_r2"] >= 0.05)
    if not c3:
        out = "STOP (Real ~ Rewired)"
    elif not c4 or not beyond:
        out = "BIOLOGICAL ORIENTATION FILTERING (no large FlyVL encoder)"
    else:
        out = "PROPOSE D11" if (c1 and c2) else "NO GO (conditions 1-2 not met)"
    rep = {"c1_replicated": c1, "c2_stable": c2, "c3": {"a": c3a, "b": c3b, "c": c3c}, "c4_not_gabor_sobel": c4,
           "beyond_gabor": bool(beyond), "decision": out}
    (OUT / "b4.json").write_text(json.dumps(rep, indent=1, default=bool))
    print(json.dumps(rep, indent=1, default=bool))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "rewire":
        rewired_fw(1)
    elif cmd == "bars":
        R, _ = rewired_fw(1)
        m.sim_extra("rewired", 0, W=R)
        m.sim("rewired", 0, None, B=256, W=R)
    elif cmd == "images":
        sim_images(sys.argv[2])
    elif cmd == "measures":
        measures()
    elif cmd == "b4":
        b4()
