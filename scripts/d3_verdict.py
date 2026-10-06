"""D3 Stage 1 verdict exactly as registered in PROTOCOL_D3_swarm.md.

Per contrast and CEM seed, each family (flyvis members, hr settings, framediff settings) is represented by the
setting with the lowest TRAIN tracking error. Paired t over the 128 test episodes: t = mean(d) / (sd(d) / sqrt(n)),
d = err(classical) - err(flyvis), so t > 0 means the fly tracks better.
  fly advantage: at some contrast, t >= 2 against BOTH hr and framediff in ALL seeds, and seed-mean reading
                 accuracy above both
  classical advantage: at every contrast some classical family has t <= -2 in all seeds
  otherwise: tie
usage: d3_verdict.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome  # noqa: E402

OUT = connectome.DATA_ROOT / "runs" / "d3"
CONTRASTS, SEEDS, FAMS = (0.1, 0.05), (0, 1, 2), ("flyvis", "hr", "framediff")


def load(c, s, fam):
    recs = [json.loads(f.read_text()) for f in OUT.glob(f"c{c:g}_s{s}_{fam}_*.json")]
    return {r["setting"]: r for r in recs}


def paired_t(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def main():
    rep, lines = {}, []
    fly_adv, cls_adv = [], []
    for c in CONTRASTS:
        rows, t_hr, t_fd, read = [], [], [], {f: [] for f in FAMS}
        for s in SEEDS:
            pick = {}
            for fam in FAMS:
                recs = load(c, s, fam)
                assert recs, f"missing c{c:g} s{s} {fam}"
                pick[fam] = min(recs.values(), key=lambda r: r["train_err"])
                read[fam].append(float(np.mean(pick[fam]["read"])))
            e = {f: pick[f]["test_err_episodes"] for f in FAMS}
            t_hr.append(paired_t(e["hr"], e["flyvis"]))
            t_fd.append(paired_t(e["framediff"], e["flyvis"]))
            rows.append({"seed": s, **{f"{f}_choice": pick[f]["setting"] for f in FAMS},
                         **{f"{f}_train": pick[f]["train_err"] for f in FAMS},
                         **{f"{f}_test": float(np.mean(e[f])) for f in FAMS},
                         **{f"{f}_read": read[f][-1] for f in FAMS},
                         "t_vs_hr": t_hr[-1], "t_vs_framediff": t_fd[-1]})
        base = {}
        for mode in ("static", "random", "oracle"):
            f = OUT / f"c{c:g}_s0_base_{mode}.json"
            if f.exists():
                r = json.loads(f.read_text())
                base[mode] = {"test": float(np.mean(r["test_err_episodes"])), "read": float(np.mean(r.get("read", [np.nan])))}
        mread = {f: float(np.mean(read[f])) for f in FAMS}
        adv = (all(t >= 2 for t in t_hr) and all(t >= 2 for t in t_fd)
               and mread["flyvis"] > mread["hr"] and mread["flyvis"] > mread["framediff"])
        cls = all(t <= -2 for t in t_hr) or all(t <= -2 for t in t_fd)
        fly_adv.append(adv)
        cls_adv.append(cls)
        rep[f"c{c:g}"] = {"seeds": rows, "read_mean": mread, "base": base, "fly_advantage": adv, "classical_advantage": cls}
        lines.append(f"contrast {c:g}")
        for r in rows:
            lines.append(f"  seed {r['seed']}: test err fly {r['flyvis_test']:.2f} ({r['flyvis_choice']}) | "
                         f"hr {r['hr_test']:.2f} ({r['hr_choice']}) | fd {r['framediff_test']:.2f} ({r['framediff_choice']}) | "
                         f"t vs hr {r['t_vs_hr']:+.2f} vs fd {r['t_vs_framediff']:+.2f}")
        lines.append(f"  read (seed mean): fly {mread['flyvis']:.3f} hr {mread['hr']:.3f} fd {mread['framediff']:.3f} | "
                     + " ".join(f"{k} err {v['test']:.1f} read {v['read']:.3f}" for k, v in base.items()))
    verdict = "fly advantage" if any(fly_adv) else ("classical advantage" if all(cls_adv) else "tie")
    rep["verdict"] = verdict
    lines.append(f"VERDICT: {verdict}")
    (OUT / "verdict_stage1.json").write_text(json.dumps(rep, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
