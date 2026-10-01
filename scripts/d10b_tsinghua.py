"""D10-B (PROTOCOL_D10B_tsinghua.md): exact replication of Liew et al. (NeurIPS 2025), "Connectome-Based Modelling
Reveals Orientation Maps in the Drosophila Optic Lobe", oriented-bar experiment.

The authors' model (github.com/JNLiew/flylif_orientation_maps, Brian2) re-implemented for the GPU with the same
equations, parameters, schedule and inputs; validated against the authors' own Brian2 code on a subset of conditions.
  connectome  FlyWire FAFB v783 Codex connections_no_threshold: 138,639 neurons, w = sign(pre) x synapse count summed over
              neuropils; sign = argmax over the presynaptic neuron's summed nt_type (GABA, GLUT -> -), ties by first
              appearance in the file (the authors' dict order)
  neuron      dv/dt = (g - (v - V_rest)) / tau_m, dg/dt = -g / tau_s (exact integration, dt 0.1 ms); threshold v > -45 mV
              unless refractory (2.2 ms; 0 for driven L1-L3); reset v = -52 mV, g = 0; on a presynaptic spike, after
              1.8 ms, g_post += w * 1.5 mV
  schedule    per step (Brian2 default): state update -> threshold -> synaptic delivery and Poisson input -> reset
  input       right-eye L1/L2/L3: independent Poisson inputs of rate 20 Hz x level (level from the authors' configs),
              each input spike adds 35 mV to v
usage: d10b_tsinghua.py build                       connectome + groups + configs -> d10b/model.npz, configs.npz
       d10b_tsinghua.py sim NAME SEED [IDX..]       GPU simulation of configs (all if no IDX) -> d10b/rates/NAME_s{SEED}.npy
       d10b_tsinghua.py validate                    GPU vs the authors' Brian2 rates on the validation configs
       d10b_tsinghua.py analyze NAME SEED           max over phases -> authors' circular-Gaussian fits -> type table, gate
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

DATA = Path.home() / "flyvl_data" / "d10b"
REF = DATA / "ref" / "flywire_codex_783"
RUN = DATA / "authors_run"
RATES = DATA / "rates"
RATES.mkdir(parents=True, exist_ok=True)
dev = "cuda"

# authors' parameters (vis_fly/lif_simulation.py)
V_REST, V_RESET, V_TH = -52.0, -52.0, -45.0          # mV
TAU_M, TAU_S, DT = 20.0, 5.0, 0.1                    # ms
REF_STEPS, DELAY_STEPS = 22, 18                      # 2.2 ms, 1.8 ms
W_SYN, KICK = 1.5, (V_TH - V_RESET) * 5              # mV
SIM_MS, BASE_HZ = 3000.0, 20.0                       # rate = 200 Hz * level / 10
STEPS = int(round(SIM_MS / DT))
EXC = ("DA", "ACH", "SER", "OCT")
INH = ("GABA", "GLUT")


# ------------------------------------------------------------------ connectome (the authors' process_csv semantics)
def build_connectome():
    f = DATA / "model.npz"
    if f.exists():
        return
    t0 = time.time()
    c = pd.read_csv(RUN / "flywire" / "connections_no_threshold.csv",
                    dtype={"pre_root_id": np.int64, "post_root_id": np.int64, "syn_count": np.int64, "nt_type": str})
    # neuron index = order of first appearance, pre before post within a row (Connectome.add_synapse)
    inter = np.empty(2 * len(c), np.int64)
    inter[0::2], inter[1::2] = c.pre_root_id.values, c.post_root_id.values
    uniq, first = np.unique(inter, return_index=True)
    order = np.argsort(first, kind="stable")
    roots = uniq[order]
    pos = np.empty(len(uniq), np.int64)
    pos[order] = np.arange(len(uniq))
    pre = pos[np.searchsorted(uniq, c.pre_root_id.values)]
    post = pos[np.searchsorted(uniq, c.post_root_id.values)]
    # transmitter per presynaptic neuron: max summed syn_count, ties -> first-encountered transmitter
    nt = c.nt_type.fillna("").values.astype(str)
    key = pd.DataFrame({"pre": pre, "nt": nt, "w": c.syn_count.values, "row": np.arange(len(c))})
    agg = key.groupby(["pre", "nt"], sort=False).agg(w=("w", "sum"), first=("row", "min")).reset_index()
    agg = agg.sort_values(["pre", "w", "first"], ascending=[True, False, True])
    best = agg.drop_duplicates("pre", keep="first").set_index("pre")["nt"]
    inhib = np.zeros(len(roots), bool)
    inhib[best.index.values] = np.isin(best.values, INH)
    n_ties = int((agg.groupby("pre")["w"].apply(lambda s: (s == s.max()).sum() > 1)).sum())
    sign = np.where(inhib[pre], -1.0, 1.0)
    W = sparse.csr_matrix((sign * c.syn_count.values, (post, pre)), shape=(len(roots), len(roots)))  # sums neuropils
    W.sum_duplicates()
    W = W.tocsc()                                                       # column j = outputs of presynaptic j
    vt = pd.read_csv(RUN / "flywire" / "visual_neuron_types.csv")
    cols = pd.read_csv(RUN / "flywire" / "column_assignment.csv")
    vt_idx = pd.Series(np.arange(len(roots)), index=roots)
    vtype = np.full(len(roots), "", object)
    vside = np.full(len(roots), "", object)
    m = vt.root_id.isin(vt_idx.index)
    vtype[vt_idx[vt.root_id[m]].values] = vt.type[m].values
    vside[vt_idx[vt.root_id[m]].values] = vt.side[m].values
    hex_pq = np.full((len(roots), 2), -10**6, np.int64)
    col_id = np.full(len(roots), -1, np.int64)
    m = cols.root_id.isin(vt_idx.index)
    hex_pq[vt_idx[cols.root_id[m]].values] = np.stack([cols.p[m].values + 19, cols.q[m].values + 17], 1)
    col_id[vt_idx[cols.root_id[m]].values] = cols.column_id[m].values
    np.savez(f, roots=roots, data=W.data, indices=W.indices, indptr=W.indptr, inhib=inhib, vtype=vtype.astype(str),
             vside=vside.astype(str), hex_pq=hex_pq, col_id=col_id)
    print(f"connectome: {len(roots):,} neurons, {W.nnz:,} connections, {int(np.abs(W.data).sum()):,} synapses, "
          f"{int(inhib.sum()):,} inhibitory, {n_ties} transmitter ties ({time.time() - t0:.0f} s)", flush=True)


def load_model():
    z = np.load(DATA / "model.npz", allow_pickle=True)
    W = sparse.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(len(z["roots"]),) * 2)
    return z, W


# ------------------------------------------------------------------ configs
def replica_configs():
    """tools/gen_config.py re-done with the same float64 arithmetic: 225 angles (0.8 deg) x ~40 phases (0.2 column);
    10 parallel OFF lines 8 columns apart; an L cell with column at perpendicular distance d < 1 from a line gets level
    ceil(K sqrt(1 - d^2)) (K = 7, 5, 10 for L1, L2, L3), every other right L1-L3 cell of the >= 5-synapse graph gets
    level 1. Checked config by config against the authors' JSON files (validate_configs)."""
    f = DATA / "configs_replica.npz"
    if f.exists():
        return
    conn = pd.read_csv(RUN / "flywire" / "connections.csv", usecols=["pre_root_id", "post_root_id"])
    inter = np.empty(2 * len(conn), np.int64)
    inter[0::2], inter[1::2] = conn.pre_root_id.values, conn.post_root_id.values
    uniq, first = np.unique(inter, return_index=True)
    roots_f = uniq[np.argsort(first, kind="stable")]                   # filtered-graph neuron order
    vt = pd.read_csv(RUN / "flywire" / "visual_neuron_types.csv").drop_duplicates("root_id", keep="last").set_index("root_id")
    cols = pd.read_csv(RUN / "flywire" / "column_assignment.csv").drop_duplicates("root_id", keep="last").set_index("root_id")
    in_f = vt.index.intersection(roots_f)
    pos = pd.Series(np.arange(len(roots_f)), index=roots_f)

    def group(tp):
        ids = [r for r in roots_f[np.sort(pos[in_f].values)] if (tp is None or vt.type[r] == tp) and vt.side[r] == "right"]
        return np.array(ids, np.int64)
    out = group(None)
    groups = {t: group(t) for t in ("L1", "L2", "L3")}
    K = {"L1": 7, "L2": 5, "L3": 10}
    cells = np.concatenate([groups[t] for t in ("L1", "L2", "L3")])
    has = np.array([r in cols.index for r in cells])
    p = np.array([cols.p[r] + 19 if h else 0 for r, h in zip(cells, has)], np.float64)
    q = np.array([cols.q[r] + 17 if h else 0 for r, h in zip(cells, has)], np.float64)
    x = (q - p) * np.sqrt(3) / 2
    y = (p + q) * 0.5
    Kc = np.concatenate([np.full(len(groups[t]), K[t]) for t in ("L1", "L2", "L3")]).astype(np.float64)
    names, angles, bs, levels = [], [], [], []
    for angle in np.arange(0, 180, 0.8):
        k = math.tan((90 - angle) * math.pi / 180) if angle > 1e-5 else None
        db = 0.2 * math.sqrt(k * k + 1) if k is not None else 0.2
        for b in np.arange(-200 * db, -160 * db, db):
            lev = np.ones(len(cells), np.int8)
            for i in range(10):
                bi = b + i * db * 40
                d = np.abs(x - bi) if k is None else np.abs(k * x + bi - y) / np.sqrt(k * k + 1)
                hit = has & (d < 1)
                lev[hit] = np.ceil(Kc[hit] * np.sqrt(1 - d[hit] * d[hit])).astype(np.int8)
            names.append("angle_{}_b_{:.1f}".format(angle, b))
            angles.append(angle)
            bs.append(b)
            levels.append(lev)
    np.savez(f, names=np.array(names), angle=np.array(angles), b=np.array(bs), cells=cells, levels=np.array(levels),
             out=out)
    print(f"replica: {len(names)} configs, {len(cells)} driven L cells ({int(has.sum())} with a column), {len(out)} out neurons")


def validate_configs():
    """Every authors' JSON config present must equal the replica (same name, same driven set, same levels, same outs)."""
    r = np.load(DATA / "configs_replica.npz")
    exps = json.loads((RUN / "exp" / "experiments.json").read_text()) if (RUN / "exp" / "experiments.json").exists() else None
    files = sorted((RUN / "exp" / "cfs").glob("*.json"))
    ix = {n: i for i, n in enumerate(r["names"])}
    cell_pos = {c: i for i, c in enumerate(r["cells"])}
    out_set = set(r["out"].tolist())
    bad = 0
    for fjs in files:
        cfg = json.loads(fjs.read_text())
        i = ix.get(cfg["experiment_name"])
        if i is None:
            bad += 1
            print("missing in replica:", cfg["experiment_name"])
            continue
        lev = np.zeros(len(r["cells"]), np.int8)
        for nm, f in zip(cfg["stimulate_neurons"], cfg["stimulate_frates"]):
            lev[cell_pos[int(nm)]] = f
        ok = (lev == r["levels"][i]).all() and set(map(int, cfg["out_neurons"])) == out_set
        bad += int(not ok)
    print(f"checked {len(files)} authors' configs against the replica: {bad} mismatches; "
          f"experiments.json {'present (' + str(len(exps)) + ')' if exps else 'not yet written'}; replica {len(r['names'])}")
    return bad


def build_configs():
    """Pack the authors' JSON configs (exp/cfs, in exp/experiments.json order) into arrays: level (1..10) of every
    driven L cell per config; the out-neuron list (identical in every config)."""
    f = DATA / "configs.npz"
    if f.exists():
        return
    z, _ = load_model()
    idx = pd.Series(np.arange(len(z["roots"])), index=z["roots"])
    exps = json.loads((RUN / "exp" / "experiments.json").read_text())
    names, angles, bs, out = [], [], [], None
    drive = None
    for k, p in enumerate(exps):
        cfg = json.loads((RUN / p).read_text() if not Path(p).is_absolute() else Path(p).read_text())
        nm = cfg["experiment_name"]
        a, b = nm.split("_b_")
        names.append(nm)
        angles.append(float(a.replace("angle_", "")))
        bs.append(float(b))
        stim = idx[np.array(cfg["stimulate_neurons"], np.int64)].values
        lev = np.array(cfg["stimulate_frates"], np.int8)
        if drive is None:
            drive_cells = np.sort(stim)
            drive = np.zeros((len(exps), len(drive_cells)), np.int8)
            out = idx[np.array(cfg["out_neurons"], np.int64)].values
        assert len(stim) == len(drive_cells) and set(stim) == set(drive_cells)
        drive[k, np.searchsorted(drive_cells, stim)] = lev
        assert cfg["simulation_time"] == 3000 and cfg["firing_rate"] == 200 and not cfg["silenced_neurons"]
        if k % 1000 == 0:
            print(k, nm, flush=True)
    np.savez(f, names=np.array(names), angle=np.array(angles), b=np.array(bs), drive_cells=drive_cells, drive=drive,
             out=out)
    print(f"{len(names)} configs, {len(drive_cells)} driven L cells, {len(out)} out neurons", flush=True)


# ------------------------------------------------------------------ GPU simulator
@torch.compile(dynamic=False)
def _step(v, g, last, deliv_k, t, ref, a_m: float, a_s: float, c_g: float):
    """Brian2 order within one step: refractoriness + exact update (groups), threshold, synaptic delivery, reset."""
    not_ref = (t - last) >= ref
    v = V_REST + (v - V_REST) * a_m + g * c_g
    g = g * a_s + deliv_k
    spk = (v > V_TH) & not_ref
    last = torch.where(spk, t, last)
    v = torch.where(spk, torch.full_like(v, V_RESET), v)
    g = torch.where(spk, torch.zeros_like(g), g)
    return v, g, last, spk


class LIF:
    """Batched replica of the authors' Brian2 network (float64). Columns of the batch = independent simulations."""

    def __init__(self, W, drive_cells, out):
        self.n = W.shape[0]
        self.indptr = torch.as_tensor(W.indptr.astype(np.int64), device=dev)
        self.indices = torch.as_tensor(W.indices.astype(np.int64), device=dev)
        self.data = torch.as_tensor(W.data.astype(np.float64) * W_SYN, device=dev)
        self.drive = torch.as_tensor(drive_cells, device=dev)
        self.out = torch.as_tensor(out, device=dev)
        ref = torch.full((self.n,), REF_STEPS, dtype=torch.int32, device=dev)
        ref[self.drive] = 0
        self.ref = ref[:, None]
        self.a_m, self.a_s = math.exp(-DT / TAU_M), math.exp(-DT / TAU_S)
        self.c_g = TAU_S / (TAU_S - TAU_M) * (self.a_s - self.a_m)    # v gain from g over one exact step

    def deliver(self, g, spk_flat, B):
        """g[post, b] += w[post, pre] * W_SYN for every spike (pre, b) in spk_flat (flat index pre * B + b)."""
        if spk_flat.numel() == 0:
            return
        pre, b = spk_flat // B, spk_flat % B
        start, cnt = self.indptr[pre], self.indptr[pre + 1] - self.indptr[pre]
        tot = int(cnt.sum())
        if tot == 0:
            return
        chunk = 60_000_000
        if tot > chunk:                                                  # split very large deliveries
            cs = torch.cumsum(cnt, 0)
            cut = torch.searchsorted(cs, torch.arange(chunk, tot, chunk, device=dev))
            for s, e in zip(torch.cat([torch.zeros(1, dtype=torch.long, device=dev), cut]).tolist(),
                            torch.cat([cut, torch.tensor([len(cnt)], device=dev)]).tolist()):
                self.deliver(g, spk_flat[s:e], B)
            return
        rep = torch.repeat_interleave(torch.arange(len(pre), device=dev), cnt)
        off = torch.arange(tot, device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
        e = start[rep] + off
        g.view(-1).index_add_(0, self.indices[e] * B + b[rep], self.data[e])

    @torch.no_grad()
    def run(self, levels, seed, steps=STEPS):
        """levels (n_drive, B) int (1..10) -> spike counts of the out neurons (n_out, B).
        The delay is exactly 18 steps, so the spikes of a block of 18 steps are all delivered in the next block: they
        are expanded once per block into per-step delivery slices. Per step, one fused kernel does Brian2's state
        update, threshold, synaptic delivery and reset; the Poisson kick is then added to the driven L cells that did
        not spike this step (in Brian2 it is added before the reset, which overwrites it for cells that spiked)."""
        B, D = levels.shape[1], DELAY_STEPS
        gen = torch.Generator(device=dev).manual_seed(seed)
        p = (BASE_HZ * levels.to(torch.float64) * DT * 1e-3).to(dev)
        v = torch.full((self.n, B), V_REST, dtype=torch.float64, device=dev)
        g = torch.zeros_like(v)
        last = torch.full((self.n, B), -10**8, dtype=torch.int32, device=dev)
        count = torch.zeros((len(self.out), B), dtype=torch.int32, device=dev)
        hist = torch.zeros((D, self.n, B), dtype=torch.bool, device=dev)          # spikes of the current block
        deliv = torch.zeros((D, self.n, B), dtype=torch.float64, device=dev)      # slot k: input arriving at step = k mod D
        tt = torch.zeros((), dtype=torch.int32, device=dev)
        for t in range(steps):
            k = t % D
            tt.fill_(t)
            v, g, last, spk = _step(v, g, last, deliv[k], tt, self.ref, self.a_m, self.a_s, self.c_g)
            hit = torch.rand(p.shape, generator=gen, device=dev, dtype=torch.float64) < p
            v[self.drive] += KICK * (hit & ~spk[self.drive])                  # PoissonInput (N = 1)
            hist[k] = spk
            if k == D - 1:                                                 # block done: count, schedule deliveries
                count += hist[:, self.out].sum(0, dtype=torch.int32)
                deliv.zero_()
                f = torch.nonzero(hist.view(-1)).squeeze(1)                # flat (slot, pre, b)
                slot, rest = f // (self.n * B), f % (self.n * B)
                self.deliver_block(deliv, slot, rest // B, rest % B, B)
        r = steps % D
        if r:
            count += hist[:r, self.out].sum(0, dtype=torch.int32)
        return count

    def deliver_block(self, deliv, slot, pre, b, B, chunk=40_000_000):
        if pre.numel() == 0:
            return
        start, cnt = self.indptr[pre], self.indptr[pre + 1] - self.indptr[pre]
        cs = torch.cumsum(cnt, 0)
        tot = int(cs[-1])
        cuts = torch.searchsorted(cs, torch.arange(chunk, tot, chunk, device=dev), right=True).tolist() if tot > chunk else []
        bounds = [0] + cuts + [len(pre)]
        for s, e in zip(bounds[:-1], bounds[1:]):
            if e <= s:
                continue
            c = cnt[s:e]
            n_e = int(c.sum())
            rep = torch.repeat_interleave(torch.arange(s, e, device=dev), c, output_size=n_e)
            off = torch.arange(n_e, device=dev) - torch.repeat_interleave(torch.cumsum(c, 0) - c, c, output_size=n_e)
            ed = start[rep] + off
            deliv.view(-1).index_add_(0, (slot[rep] * self.n + self.indices[ed]) * B + b[rep], self.data[ed])


def sim(name, seed, which=None, B=512, W=None, steps=STEPS):
    z, W0 = load_model()
    W = W0 if W is None else W
    cf = np.load(DATA / "configs_replica.npz")
    pos = pd.Series(np.arange(len(z["roots"])), index=z["roots"])
    cells, out = pos[cf["cells"]].values, pos[cf["out"]].values
    out_f = RATES / f"{name}_s{seed}.npy"
    idx = np.arange(len(cf["names"])) if which is None else np.asarray(which)
    rates = np.full((len(out), len(cf["names"])), np.nan, np.float32)
    if out_f.exists():
        rates = np.load(out_f)
    todo = [i for i in idx if np.isnan(rates[0, i])]
    lif = LIF(W, cells, out)
    for s in range(0, len(todo), B):
        t0 = time.time()
        part = todo[s:s + B]
        lev = torch.as_tensor(cf["levels"][part].T.astype(np.int64))
        cnt = lif.run(lev, seed * 100_003 + int(part[0]), steps)
        rates[:, part] = (cnt.double() / (steps * DT / 1000)).cpu().numpy()
        np.save(out_f, rates)
        print(f"{name} s{seed}: {s + len(part)}/{len(todo)} configs ({time.time() - t0:.0f} s for {len(part)})", flush=True)


# ------------------------------------------------------------------ validation against the authors' Brian2 code
VAL_ANGLES = (0.0, 45.6, 90.4, 135.2)


def val_configs():
    r = np.load(DATA / "configs_replica.npz")
    ang = np.round(r["angle"], 4)
    idx = []
    for a in VAL_ANGLES:
        first = np.flatnonzero(np.isclose(ang, a))
        idx += [int(first[0])] + ([int(first[20])] if a in (0.0, 90.4) else [])
    return sorted(idx)


def write_val_jsons():
    """The validation configs in the authors' JSON format (identical to tools/gen_config.py output) for main_single.py."""
    r = np.load(DATA / "configs_replica.npz")
    d = RUN / "exp_val" / "cfs"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in val_configs():
        cfg = {"experiment_name": str(r["names"][i]), "experiment_directory": "./exp_val/",
               "stimulate_neurons": [str(c) for c in r["cells"]], "stimulate_frates": [int(x) for x in r["levels"][i]],
               "out_neurons": [str(c) for c in r["out"]], "simulation_time": 3000, "firing_rate": 200, "silenced_neurons": []}
        f = d / f"{cfg['experiment_name']}.json"
        f.write_text(json.dumps(cfg))
        paths.append(str(f))
    print("\n".join(paths))


def validate():
    """Per validation config: Brian2 (authors' code) rates vs GPU rates (seed 0), with GPU seed 0 vs seed 1 as the
    Poisson-noise reference."""
    r = np.load(DATA / "configs_replica.npz")
    out = r["out"]
    g0, g1 = np.load(RATES / "val_s0.npy"), np.load(RATES / "val_s1.npy")
    rep = {}
    for i in val_configs():
        nm = str(r["names"][i])
        f = RUN / "exp_val" / "res" / f"{nm}.csv"
        if not f.exists():
            rep[nm] = None
            continue
        b2 = pd.read_csv(f).set_index("neuron_name")["firing_rate"].reindex(out).values
        a, c = g0[:, i], g1[:, i]
        rep[nm] = {"brian2_mean_hz": float(np.nanmean(b2)), "gpu_s0_mean_hz": float(a.mean()), "gpu_s1_mean_hz": float(c.mean()),
                   "r_brian2_gpu": float(np.corrcoef(b2, a)[0, 1]), "r_gpu_gpu": float(np.corrcoef(a, c)[0, 1]),
                   "active_brian2": int((b2 > 0).sum()), "active_gpu": int((a > 0).sum())}
        print(nm, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rep[nm].items()}), flush=True)
    (DATA / "validation.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build_connectome()
        build_configs()
    elif cmd == "build_connectome":
        build_connectome()
    elif cmd == "sim":
        sim(sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4:]] or None)
    elif cmd == "val_jsons":
        write_val_jsons()
    elif cmd == "val_gpu":
        for sd in (0, 1):
            sim("val", sd, val_configs(), B=8)
    elif cmd == "validate":
        validate()

