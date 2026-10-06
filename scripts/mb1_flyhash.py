"""MB-1 (PROTOCOL_MB1_flyhash.md): the MaleCNS mushroom-body PN->KC wiring used as a FlyHash for LLM embeddings.
Real wiring vs degree-preserving shuffle (primary), uniform FlyHash, dense WTA and SimHash; nearest-neighbour mAP.
usage: mb1_flyhash.py embed | run | verdict
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome  # noqa: E402
from flyvl import flygrapher as fg  # noqa: E402

OUT = connectome.DATA_ROOT / "mb1"
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
N_DB, N_Q, TOP_FRAC, MIN_SYN = 20000, 1000, 0.02, 3
KS, PRIMARY_K, DELTA = (16, 32, 64, 128), 32, 0.01
ASSIGN_SEEDS, CTRL_SEEDS = range(5), range(5)


# ------------------------------------------------------------------ data: AG News through the LLM of InternVL3-1B
def embed():
    f = OUT / "embeddings.npz"
    if f.exists():
        return
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoProcessor, InternVLForConditionalGeneration
    texts = {}
    for split, n in (("train", N_DB), ("test", N_Q)):
        p = hf_hub_download("fancyzhx/ag_news", f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
        texts[split] = pq.read_table(p).column("text").to_pylist()[:n]
    mid = "OpenGVLab/InternVL3-1B-hf"
    tok = AutoProcessor.from_pretrained(mid).tokenizer
    vlm = InternVLForConditionalGeneration.from_pretrained(mid, dtype=torch.bfloat16).to(dev).eval()
    lm = vlm.model.language_model if hasattr(vlm.model, "language_model") else vlm.language_model
    out = {}
    for split, tx in texts.items():
        embs = []
        with torch.no_grad():
            for s in range(0, len(tx), 128):
                b = tok(tx[s:s + 128], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
                h = lm(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state.float()
                m = b["attention_mask"][..., None].float()
                e = (h * m).sum(1) / m.sum(1)
                embs.append(torch.nn.functional.normalize(e, dim=1).cpu())
        out[split] = torch.cat(embs).numpy()
    np.savez(f, db=out["train"], q=out["test"])
    print("embeddings", {k: v.shape for k, v in out.items()})


# ------------------------------------------------------------------ wiring
def mb_wiring():
    """Binary PN -> KC adjacency (KC x PN) from MaleCNS: >= MIN_SYN synapses, synapse counts recovered from the
    input-normalised weights (share / the row's smallest share = 1 synapse)."""
    meta = np.load(fg.SRC / "brain.npz")
    W = sparse.load_npz(fg.SRC / "weights.npz").tocsr()
    ct, side = meta["cell_type"].astype(str), meta["side"].astype(str)
    kc = np.flatnonzero(np.char.startswith(ct, "KC"))
    pn_all = np.flatnonzero(np.array(["PN" in t for t in ct]))
    Wk = abs(W[kc]).tocsr()
    counts = Wk.copy()
    for r in range(Wk.shape[0]):
        a, b = Wk.indptr[r:r + 2]
        if b > a:
            counts.data[a:b] = np.rint(Wk.data[a:b] / Wk.data[a:b].min())
    C = counts[:, pn_all].tocsc()
    pn = pn_all[np.asarray(C.getnnz(0)).ravel() > 0]
    A = (counts[:, pn] >= MIN_SYN).astype(np.float32).tocsr()
    info = {"KC": int(len(kc)), "PN": int(len(pn)), "edges": int(A.nnz),
            "kc_in_degree_mean": float(A.getnnz(1).mean()), "kc_in_degree_zero": int((A.getnnz(1) == 0).sum()),
            "pn_out_degree_mean": float(A.getnnz(0).mean())}
    return A, side[kc], side[pn], info


def shuffle(A, kc_side, pn_side, seed):
    """Degree-preserving bipartite rewiring within (PN side, KC side) blocks (PN out- and KC in-degree exact)."""
    nk, npn = A.shape
    n = nk + npn
    coo = A.tocoo()
    W = sparse.csr_matrix((np.ones(A.nnz, np.float32), (coo.row, nk + coo.col)), shape=(n, n))   # rows = KC (post)
    sides = np.r_[kc_side, pn_side]
    _, sid = np.unique(sides, return_inverse=True)
    gkey = sid * 2 + np.r_[np.zeros(nk, int), np.ones(npn, int)]
    R, info = fg.rewire_blocks(W, gkey.astype(np.int64), seed)
    R = R[:nk][:, nk:].tocsr()
    R.data[:] = 1.0
    return R, info


def uniform(A, seed):
    """FlyHash: each KC samples its real in-degree of PNs uniformly at random."""
    g = np.random.default_rng(seed)
    deg = A.getnnz(1)
    rows = np.repeat(np.arange(A.shape[0]), deg)
    cols = np.concatenate([g.choice(A.shape[1], d, replace=False) for d in deg])
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=A.shape)


# ------------------------------------------------------------------ codes and evaluation
def wta(X, M, k):
    act = X @ M.T
    idx = act.topk(k, dim=1).indices
    return torch.zeros_like(act).scatter_(1, idx, 1.0)


def mean_ap(score, gt_idx, tie):
    """score (Q, N_DB); gt_idx (Q, n_rel) true neighbours; tie (N_DB,) fixed tie-break. Returns AP per query."""
    order = torch.argsort(score + 1e-6 * tie, dim=1, descending=True)
    rel = torch.zeros_like(score, dtype=torch.bool).scatter_(1, gt_idx, True)
    hit = torch.gather(rel, 1, order).float()
    prec = hit.cumsum(1) / torch.arange(1, score.shape[1] + 1, device=score.device)
    return (prec * hit).sum(1) / hit.sum(1)


def run():
    e = np.load(OUT / "embeddings.npz")
    db, q = torch.as_tensor(e["db"], device=dev), torch.as_tensor(e["q"], device=dev)
    n_rel = int(TOP_FRAC * len(db))
    gt = (q @ db.T).topk(n_rel, dim=1).indices
    A, kc_side, pn_side, info = mb_wiring()
    npn = A.shape[1]
    mu = db.mean(0)
    U, S, V = torch.linalg.svd(db - mu, full_matrices=False)
    P = V[:npn].T                                                          # 896 -> 300 principal directions
    Xdb, Xq = (db - mu) @ P, (q - mu) @ P
    tie = torch.as_tensor(np.random.default_rng(123).random(len(db)), device=dev, dtype=torch.float32)
    ref = mean_ap(torch.nn.functional.normalize(Xq, dim=1) @ torch.nn.functional.normalize(Xdb, dim=1).T, gt, tie)
    res = {"info": info, "pca_cosine_mAP": float(ref.mean()), "ap": {}}
    wirings = {"real": [("real", A)]}
    wirings["shuffle"] = []
    for s in CTRL_SEEDS:
        R, rinfo = shuffle(A, kc_side, pn_side, s)
        wirings["shuffle"].append((f"s{s}", R))
        res.setdefault("shuffle_info", []).append(rinfo)
    wirings["uniform"] = [(f"s{s}", uniform(A, s)) for s in CTRL_SEEDS]
    for a in ASSIGN_SEEDS:
        perm = torch.as_tensor(np.random.default_rng(1000 + a).permutation(npn), device=dev)
        xd, xq = Xdb[:, perm], Xq[:, perm]
        for method, lst in wirings.items():
            for tag, Mx in lst:
                M = torch.as_tensor(Mx.toarray(), device=dev)
                for k in KS:
                    cq, cd = wta(xq, M, k), wta(xd, M, k)
                    res["ap"][f"{method}|{tag}|a{a}|k{k}"] = mean_ap(cq @ cd.T, gt, tie).cpu().tolist()
        for s in CTRL_SEEDS:
            gen = torch.Generator(device=dev).manual_seed(5000 + s)
            M = torch.randn(A.shape[0], npn, generator=gen, device=dev)
            for k in KS:
                cq, cd = wta(xq, M, k), wta(xd, M, k)
                res["ap"][f"dense|s{s}|a{a}|k{k}"] = mean_ap(cq @ cd.T, gt, tie).cpu().tolist()
            for k in KS:
                H = torch.randn(k, npn, generator=gen, device=dev)
                bq, bd = torch.sign(xq @ H.T), torch.sign(xd @ H.T)
                res["ap"][f"simhash|s{s}|a{a}|k{k}"] = mean_ap((k + bq @ bd.T) / 2, gt, tie).cpu().tolist()
        print(f"assignment {a} done", flush=True)
    (OUT / "results.json").write_text(json.dumps(res))
    print(json.dumps(info))


def verdict():
    r = json.loads((OUT / "results.json").read_text())
    ap = {k: np.asarray(v) for k, v in r["ap"].items()}
    summ = {"pca_cosine_mAP": r["pca_cosine_mAP"], "info": r["info"]}
    for k in KS:
        row = {}
        for m in ("real", "shuffle", "uniform", "dense", "simhash"):
            vals = [v.mean() for key, v in ap.items() if key.startswith(m + "|") and key.endswith(f"|k{k}")]
            row[m] = float(np.mean(vals))
        summ[f"k{k}"] = row
    pairs = [(a, s) for a in ASSIGN_SEEDS for s in CTRL_SEEDS]
    diffs = [ap[f"real|real|a{a}|k{PRIMARY_K}"].mean() - ap[f"shuffle|s{s}|a{a}|k{PRIMARY_K}"].mean() for a, s in pairs]
    perq = np.mean([ap[f"real|real|a{a}|k{PRIMARY_K}"] - ap[f"shuffle|s{s}|a{a}|k{PRIMARY_K}"] for a, s in pairs], 0)
    g = np.random.default_rng(0)
    boot = [perq[g.integers(0, len(perq), len(perq))].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    summ["primary"] = {"k": PRIMARY_K, "real_minus_shuffle_pairs": [float(x) for x in diffs], "mean": float(np.mean(diffs)),
                       "ci95": [float(lo), float(hi)], "min_pair": float(min(diffs))}
    summ["verdict"] = "TOPOLOGY EFFECT" if (min(diffs) > DELTA and lo > DELTA) else "NO TOPOLOGY EFFECT"
    (OUT / "verdict.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    {"embed": embed, "run": run, "verdict": verdict}[sys.argv[1]]()
