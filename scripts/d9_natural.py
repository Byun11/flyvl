"""D9 (PROTOCOL_D9_naturalimage.md): natural images shown to the fixed visual connectome in a fly-compatible way.
usage: d9_natural.py stage0
       d9_natural.py features GRAPH SEED INPUT        GRAPH real | rewired_l | random;  INPUT A | B | C | D
       d9_natural.py train REP SEED INPUT            REP rgb real rewired_l random gabor sobel rgb+real rgb+gabor rgb+rewired_l
       d9_natural.py select | verdict | diagnostics
"""
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from flyvl import flygrapher as fg  # noqa: E402
import d6_translation as d6  # noqa: E402

OUT = fg.connectome.DATA_ROOT / "d9"
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
PIX, STEPS, GRID = 32, 16, 16
ORI = np.deg2rad(np.arange(0, 180, 30))
DM3 = ("Dm3p", "Dm3q", "Dm3v")
POP = ("Dm3p", "Dm3q", "Dm3v", "TmY9q", "TmY9q__perp", "Mi1", "Mi4", "Mi9", "Tm3", "Tm1", "Tm2", "Tm4", "Tm9", "Tm20")


# ------------------------------------------------------------------ fixed brain
class StaticBrain:
    """Frozen D6 dynamics; a static luminance image is shown through D6's fixed retina for 16 steps; returns the
    time-mean f(V) of the readout neurons (all neurons of the POP types)."""

    def __init__(self, brain, kind, seed):
        W, self.info = fg.build_graph(brain, kind, seed)
        self.W, self.WT = fg.to_csr(W, dev), fg.to_csr(W.T, dev)
        self.n, self.brain = W.shape[0], brain
        self.inputs = torch.as_tensor(brain.inputs, device=dev)
        ct = load_types(brain)
        self.ro = np.flatnonzero(np.isin(ct, POP))
        self.ro_type = ct[self.ro]
        self.ro_t = torch.as_tensor(self.ro, device=dev)
        Fp = fg.footprint(W, brain, STEPS, dev)[self.ro_t]
        pn = Fp / Fp.sum(1, keepdim=True).clamp_min(1e-30)
        r = torch.arange(fg.N_PATCH, device=dev) // fg.N_SIDE
        c = torch.arange(fg.N_PATCH, device=dev) % fg.N_SIDE
        cy, cx = (pn * r).sum(1), (pn * c).sum(1)
        self.cell = (cy.round().clamp(0, GRID - 1) * GRID + cx.round().clamp(0, GRID - 1)).long()
        self.has_fp = (Fp.sum(1) > 0)

    @torch.no_grad()
    def __call__(self, img):
        """img (B, PIX, PIX) luminance in [0, 1] (0.5 = mean) -> (n_readout, B)."""
        u = d6.retina_drive(img[:, None].expand(-1, d6.FRAMES, -1, -1).contiguous(), self.brain)[0]
        B = img.shape[0]
        V = torch.zeros(self.n, B, device=dev)
        acc = torch.zeros(len(self.ro), B, device=dev)
        for _ in range(STEPS):
            drive = torch.sparse.mm(self.W, fg.act(V)).index_add(0, self.inputs, u)
            V = V + 0.5 * (drive - V)
            acc = acc + fg.act(V)[self.ro_t]
        return acc / STEPS

    def to_map(self, resp):
        """(n_readout, B) -> (B, GRID*GRID, len(POP)) mean response per retinotopic cell and cell type."""
        B = resp.shape[1]
        out = torch.zeros(B, GRID * GRID, len(POP), device=dev)
        cnt = torch.zeros(GRID * GRID, len(POP), device=dev)
        for j, t in enumerate(POP):
            m = torch.as_tensor((self.ro_type == t), device=dev) & self.has_fp
            idx = self.cell[m]
            out[:, :, j].index_add_(1, idx, resp[m].T)
            cnt[:, j].index_add_(0, idx, torch.ones(len(idx), device=dev))
        return out / cnt.clamp_min(1)[None]


_TYPES = {}


def load_types(brain):
    """Cell types in the D5 neuron order (flygrapher.load_malecns reorders neurons)."""
    if "ct" not in _TYPES:
        meta = np.load(fg.SRC / "brain.npz")
        ct, sc = meta["cell_type"].astype(str), meta["superclass"].astype(str)
        col = np.load(fg.D5 / "input_columns.npy")
        key = np.where(ct == "", np.char.add("untyped:", sc), ct)
        _, group = np.unique(key, return_inverse=True)
        _, sid = np.unique(sc, return_inverse=True)
        order = np.lexsort((group, col[:, 2], col[:, 1], col[:, 0], sid))
        _TYPES["ct"] = ct[order]
        assert (group[order] == brain.group).all()
    return _TYPES["ct"]


def input_A(L):
    z = (L - L.mean((-2, -1), keepdim=True)) / L.std((-2, -1), keepdim=True).clamp_min(1e-6)
    return 0.5 + 0.15 * z


# ------------------------------------------------------------------ stage 0
def stage0_stimuli():
    yy, xx = torch.meshgrid(torch.arange(PIX) - PIX / 2 + 0.5, torch.arange(PIX) - PIX / 2 + 0.5, indexing="ij")
    grat, glab = [], []
    for k, th in enumerate(ORI):
        proj = xx * math.cos(th) + yy * math.sin(th)
        for ph in (0, 0.25, 0.5, 0.75):
            for P in (8, 16):
                for c in (0.1, 0.3):
                    grat.append(0.5 + c * torch.sign(torch.sin(2 * math.pi * (proj / P + ph)) + 1e-9))
                    glab.append(k)
    bars, blab = [], []
    for k, th in enumerate(ORI):
        proj = xx * math.cos(th) + yy * math.sin(th)
        for di, d in enumerate((-8, -4, 0, 4, 8)):
            for w in (2, 4):
                for pol in (-0.3, 0.3):
                    bars.append(0.5 + pol * ((proj - d).abs() < w / 2).float())
                    blab.append((k, di))
    return torch.stack(grat), np.array(glab), torch.stack(bars), np.array(blab)


def tuning(resp, lab):
    """resp (n, n_stim), lab orientation index per stimulus -> r (n, 6), OSI (n,), preferred angle (n,) in [0, pi)."""
    r = np.stack([np.abs(resp[:, lab == k]).mean(1) for k in range(len(ORI))], 1)
    z = (r * np.exp(2j * ORI)[None]).sum(1)
    osi = np.abs(z) / np.maximum(r.sum(1), 1e-30)
    return r, osi, (np.angle(z) / 2) % np.pi


def stage0():
    brain = fg.load_malecns()
    G, glab, Bs, blab = stage0_stimuli()
    rep = {}
    for kind, seed in (("real", 0), ("rewired_l", 1), ("rewired_l", 2), ("rewired_l", 3)):
        net = StaticBrain(brain, kind, seed)
        rg = net(input_A(G).to(dev)).cpu().numpy()
        rb = net(input_A(Bs).to(dev)).cpu().numpy()
        res = {}
        for pop in DM3 + ("TmY9q", "TmY9q__perp"):
            m = net.ro_type == pop
            r, osi, pref = tuning(rg[m], glab)
            circ = np.exp(2j * pref)
            k_pref = np.rint(pref / (np.pi / 6)).astype(int) % 6
            sel = []
            for i, kp in enumerate(k_pref):
                v = np.array([np.abs(rb[m][i, [j for j, (kk, dd) in enumerate(blab) if kk == kp and dd == di]]).mean() for di in range(5)])
                sel.append(v.max() / max(v.mean(), 1e-30))
            res[pop] = {"n": int(m.sum()), "osi_median": float(np.median(osi)), "osi_iqr": [float(np.percentile(osi, 25)), float(np.percentile(osi, 75))],
                        "pref_circ_mean_deg": float(np.rad2deg(np.angle(circ.mean()) / 2) % 180), "pref_concentration": float(np.abs(circ.mean())),
                        "spatial_frac_max_over_mean_ge2": float(np.mean(np.array(sel) >= 2)), "mean_abs_resp": float(np.abs(rg[m]).mean())}
        allm = np.isin(net.ro_type, DM3)
        _, osi_all, _ = tuning(rg[allm], glab)
        res["Dm3_all_osi_median"] = float(np.median(osi_all))
        rep[f"{kind}_s{seed}"] = res
        print(kind, seed, json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in res.items()}), flush=True)
    real = rep["real_s0"]
    dm3 = [real[p] for p in DM3]
    means = [d["pref_circ_mean_deg"] for d in dm3]
    conc = [d["pref_concentration"] for d in dm3]
    best = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            if conc[i] >= 0.3 and conc[j] >= 0.3:
                dd = abs(means[i] - means[j]) % 180
                best = max(best, min(dd, 180 - dd))
    c1 = real["Dm3_all_osi_median"] >= 0.15
    c2 = best >= 30
    sp = np.mean([d["spatial_frac_max_over_mean_ge2"] for d in dm3])
    c3 = sp >= 0.5
    rep["gate"] = {"c1_osi_median>=0.15": bool(c1), "c2_subtype_pref_diff_deg": best, "c2": bool(c2),
                   "c3_spatial_fraction": float(sp), "c3": bool(c3), "pass": bool(c1 and c2 and c3)}
    (OUT / "stage0.json").write_text(json.dumps(rep, indent=1))
    print("STAGE0 GATE", json.dumps(rep["gate"]), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "stage0":
        stage0()
