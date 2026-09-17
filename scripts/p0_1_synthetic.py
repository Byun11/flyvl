"""P0-1: label-free synthetic sanity + parameter selection. Nothing here looks at CIFAR.

For each candidate config: warm up on gray, branch into synthetic stimuli (blank is branch 0),
measure stability, optic-lobe propagation, T4/T5 direction selectivity, looming lateralization,
central evoked response (P0-A eye route), and the detector-injection diagnostic (P0-B).
The selection rule is fixed below before running; the winner is frozen to configs/sim_frozen.json.
"""
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome, masks  # noqa: E402
from flyvl.sim import Sim, SimConfig  # noqa: E402
from flyvl.stimulus import EyeConfig, Retina, synthetic  # noqa: E402

WARM, T = 100, 50
DIRS = ["ftb", "btf", "up", "down"]
EXPECTED = {"a": "ftb", "b": "btf", "c": "up", "d": "down"}
STIMULI = (["blank", "flash", "loom_L", "loom_R"] + [f"grating_{d}" for d in DIRS]
           + [f"on_edge_{d}" for d in DIRS] + [f"off_edge_{d}" for d in DIRS])
OUT = connectome.DATA_ROOT / "runs" / f"p0_1_kc{sys.argv[1] if len(sys.argv) > 1 else '1.0'}"

c = connectome.load()
M = masks.build(c)
r16 = c.types(["R1-6"])
driven = r16[c.column[r16, 0] >= 0]
KC_SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
sim = Sim(c, SimConfig(noise="off", cut_input_to_nonvisual_sensory=True, kc_kc_scale=KC_SCALE), driven=driven)
eye_cfg = EyeConfig()
retina = Retina(c, driven, eye_cfg, SimConfig().dt)
central = torch.as_tensor(sim.local[M["central_vnc"]], device="cuda")     # all central_vnc are LIF
ol_side = {s: c.side for s in "LR"}


def g_idx(types, side=None):
    idx = c.types(types, side)
    return idx[c.graded[idx]], idx[~c.graded[idx]]


def warm_snapshot():
    st = sim.zero_state(1)
    rates = []
    for _ in range(WARM):
        st = sim.step(st)
        rates.append(st.s.mean().item() / sim.cfg.dt)
    return st, rates


def run_branches(snapshot, stimuli, inject_fn=None):
    B = len(stimuli)
    st = snapshot.expand(B)
    retina.reset(B)
    sum_x = torch.zeros(sim.Ng, B, device="cuda")
    sum_abs = torch.zeros(sim.Ng, B, device="cuda")      # time-summed |x - x_blank|
    sum_pos = torch.zeros(sim.Ng, B, device="cuda")      # time-summed relu(x - x_blank)
    sum_s = torch.zeros(sim.Nl, B, device="cuda")
    tail_s = torch.zeros(sim.Nl, B, device="cuda")
    max_x = 0.0
    for k in range(T):
        t = k * sim.cfg.dt
        lum = torch.stack([synthetic(s if s in STIMULI else "blank", retina, t) for s in stimuli], 1)
        st = sim.step(st, retina.transduce(lum), inject_fn(k) if inject_fn else ())
        d = st.x - st.x[:, :1]
        sum_x += st.x
        sum_abs += d.abs()
        sum_pos += d.clamp(min=0)
        sum_s += st.s
        if k >= T - 10:
            tail_s += st.s
        max_x = max(max_x, st.x.abs().max().item())
    rate = sum_s / (T * sim.cfg.dt)                                  # Hz
    dx = (sum_x - sum_x[:, :1]) / T                                  # evoked, vs blank branch
    ds = sum_s - sum_s[:, :1]                                        # evoked spike counts
    return dict(dx=dx, ds=ds, rate=rate, max_x=max_x, abs_dx=sum_abs / T, pos_dx=sum_pos / T,
                tail_rate=tail_s / (10 * sim.cfg.dt))


def type_mean_dx(res, t, side=None, key="pos_dx"):
    gi, _ = g_idx([t], side)
    return res[key][torch.as_tensor(sim.local[gi], device="cuda")].mean(0).cpu().numpy() if len(gi) else None


def type_ds(res, t, side=None):
    _, li = g_idx([t], side)
    return res["ds"][torch.as_tensor(sim.local[li], device="cuda")].float().mean(0).cpu().numpy() / (T * sim.cfg.dt)


def evaluate(cfg: SimConfig) -> dict:
    sim.set_config(cfg)
    snap, warm_rates = warm_snapshot()
    res = run_branches(snap, STIMULI)
    col = {s: i for i, s in enumerate(STIMULI)}
    blank_rate = res["rate"][:, 0]
    out = {
        "warm_rate_hz_first_last": [round(warm_rates[10], 3), round(warm_rates[-1], 3)],
        "blank_mean_rate_hz": round(blank_rate.mean().item(), 3),
        "frac_lif_over_40hz_blank": round((blank_rate > 40).float().mean().item(), 5),
        "graded_max_abs_x": round(res["max_x"], 3),
    }
    out["frac_lif_over_40hz_stim_tail_max"] = round((res["tail_rate"] > 40).float().mean(0).max().item(), 5)
    out["stable"] = bool(np.isfinite(res["max_x"]) and res["max_x"] < 50 and out["frac_lif_over_40hz_blank"] < 0.01
                         and out["frac_lif_over_40hz_stim_tail_max"] < 0.01
                         and abs(warm_rates[-1] - warm_rates[-20]) < 0.5 * max(warm_rates[-20], 0.1) + 0.5)

    # optic-lobe relay: |dx| per type for the ftb grating
    g = col["grating_ftb"]
    out["relay_abs_dx_grating"] = {t: round(res["abs_dx"][torch.as_tensor(sim.local[g_idx([t])[0]],
                                   device="cuda"), g].abs().mean().item(), 5)
                                   for t in ["R1-6", "L1", "L2", "L3", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2",
                                             "T4a", "T5a", "LPi1-2"] if len(g_idx([t])[0])}

    # direction selectivity: mean positive dx per subtype, per eye, over the 4 directions
    ds_ok, ds_table = 0, {}
    for kind, stim in (("T4", "on_edge"), ("T5", "off_edge")):
        for sub, want in EXPECTED.items():
            for side in "LR":
                m = type_mean_dx(res, kind + sub, side)
                if m is None:
                    continue
                resp = {d: float(m[col[f"{stim}_{d}"]]) for d in DIRS}
                best = max(resp, key=resp.get)
                opp = {"ftb": "btf", "btf": "ftb", "up": "down", "down": "up"}[best]
                a, b = max(resp[best], 0), max(resp[opp], 0)
                dsi = (a - b) / (a + b) if a + b > 1e-9 else 0.0
                ds_table[f"{kind}{sub}_{side}"] = {"pref": best, "dsi": round(dsi, 3), **{d: round(v, 5) for d, v in resp.items()}}
                ds_ok += int(best == want and dsi > 0.1)
    out["ds_table"] = ds_table
    out["ds_correct"] = ds_ok
    out["ds_correct_if_front_flipped"] = sum(
        int(v["pref"] == {"a": "btf", "b": "ftb", "c": "up", "d": "down"}[k[2]] and v["dsi"] > 0.1)
        for k, v in ds_table.items())

    # looming: LPLC2 / LC4 rate change, ipsi vs contra
    loom = {}
    for t in ["LPLC2", "LC4", "DNp01"]:
        for side in "LR":
            r = type_ds(res, t, side)
            loom[f"{t}_{side}"] = {"loom_L": round(float(r[col["loom_L"]]), 3), "loom_R": round(float(r[col["loom_R"]]), 3)}
    out["loom_dHz"] = loom

    # central evoked response (P0-A)
    ds_c = res["ds"][central].abs()
    out["central_frac_units_changed"] = {s: round((ds_c[:, i] >= 1).float().mean().item(), 5)
                                         for s, i in col.items() if s != "blank"}
    out["central_mean_abs_dcount"] = {s: round(ds_c[:, i].float().mean().item(), 5) for s, i in col.items() if s != "blank"}
    moving = [s for s in STIMULI if "grating" in s or "edge" in s]
    out["central_frac_changed_moving_mean"] = round(float(np.mean([out["central_frac_units_changed"][s] for s in moving])), 5)
    return out


def detector_diagnostic(cfg: SimConfig) -> dict:
    """P0-B: bypass the optic lobe and inject into visual projection neurons on the left."""
    sim.set_config(cfg)
    snap, _ = warm_snapshot()
    names = ["blank", "LC4+LPLC2_L", "LC10a_L"]
    groups = {1: c.types(["LC4", "LPLC2"], "L"), 2: c.types(["LC10a"], "L")}

    def inject(k):
        return [(idx, torch.tensor([0.8 if b == i else 0.0 for b in range(3)], device="cuda"))
                for i, idx in groups.items()]

    res = run_branches(snap, names, inject)
    out = {}
    for t in ["DNp01", "DNa02", "LPLC2", "LC4"]:
        for side in "LR":
            r = type_ds(res, t, side)
            out[f"{t}_{side}"] = {n: round(float(r[i]), 3) for i, n in enumerate(names)}
    ds_c = res["ds"][central].abs()
    out["central_frac_units_changed"] = {n: round((ds_c[:, i] >= 1).float().mean().item(), 5) for i, n in enumerate(names)}
    return out


# ---- candidate grid (fixed before running) ----
# frozen noise was ruled out before the grid: at rest 6-11% of LIF units (mostly cb_intrinsic)
# lock above 40 Hz even with the sensory cut. So noise off + sensory cut are fixed, explicit flags.
# Grid v1 (72 configs, kc_kc_scale=1): saturation only mattered for g_gg=4 (diverges without it), the
# Mi4/Mi9/Tm9 delay changed nothing, and every unstable run was a Kenyon-cell (KC-KC) runaway.
# Grid v2: saturation fixed at 1.0, no delay, run once per kc_kc_scale (argv[1]).
BASE = SimConfig(noise="off", cut_input_to_nonvisual_sensory=True, kc_kc_scale=KC_SCALE, saturation=1.0, g_sg=0.5)
GRID = [replace(BASE, g_gg=g_gg, g_gs=g_gs) for g_gg, g_gs in itertools.product([1.5, 2.0, 3.0, 4.0], [3.0, 10.0, 30.0])]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for i, cfg in enumerate(GRID):
        r = evaluate(cfg)
        row = {"g_gg": cfg.g_gg, "g_gs": cfg.g_gs, "g_sg": cfg.g_sg, "saturation": cfg.saturation,
               "delay": bool(cfg.tau_graded_by_type), **r}
        results.append(row)
        print(f"[{i + 1}/{len(GRID)}] gg={cfg.g_gg} gs={cfg.g_gs} sg={cfg.g_sg} sat={cfg.saturation} delay={row['delay']} "
              f"stable={r['stable']} maxx={r['graded_max_abs_x']} blankHz={r['blank_mean_rate_hz']} "
              f"DS={r['ds_correct']}/{r['ds_correct_if_front_flipped']} central={r['central_frac_changed_moving_mean']} "
              f"T4a={r['relay_abs_dx_grating'].get('T4a')} tail40={r['frac_lif_over_40hz_stim_tail_max']}",
              flush=True)
    (OUT / "grid.json").write_text(json.dumps(results, indent=1))

    # selection rule: stable; front_sign from whichever DS count is larger; then max DS correct,
    # tie-break by central propagation on moving stimuli.
    stable = [(cfg, r) for cfg, r in zip(GRID, results) if r["stable"]]
    if not stable:
        print("NO STABLE CONFIG")
        sys.exit(1)
    score = lambda cr: (max(cr[1]["ds_correct"], cr[1]["ds_correct_if_front_flipped"]),
                        cr[1]["central_frac_changed_moving_mean"])
    cfg, r = max(stable, key=score)
    front_sign = 1 if r["ds_correct"] >= r["ds_correct_if_front_flipped"] else -1
    diag = detector_diagnostic(cfg)
    frozen = {"sim": sim.spec() | cfg.spec(), "eye": replace(eye_cfg, front_sign=front_sign).spec(),
              "driven": "R1-6 with column", "warm_steps": WARM, "stim_steps": T,
              "selection": {"rule": "stable -> max DS correct (either front sign) -> max central frac changed (moving)",
                            "metrics": r, "detector_diagnostic": diag}}
    (OUT / "selected.json").write_text(json.dumps(frozen, indent=1))
    print(json.dumps({k: r[k] for k in r if k not in ("ds_table",)}, indent=1))
    print("DETECTOR DIAGNOSTIC", json.dumps(diag, indent=1))
