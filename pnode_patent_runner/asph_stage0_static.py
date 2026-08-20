"""asph_stage0_static.py — Stage-0 falsification gate for the ASPH-Flow spec.

ASPH-Flow proposes: text features -> SVD top-2 latent positions q_i -> unit-circle
projection -> cosine decoder P(link) = sigmoid(kappa * cos(q_u, q_v)), with SDE
dynamics on top. Stage 0 evaluates the NO-DYNAMICS ablation: initial positions
only, exact decoder geometry, on the leak-free Task B protocol that defined the
relatedness ceiling (task_b_static_ceiling.py). If the static geometry is far
below the ceiling, dynamics would have to add an implausible margin (velocity
attribution showed the continuous-drift signal is ~0 on this data).

Scorers (all training-free, deterministic):
  asph_s1_2d      : SVD-2D of content embeddings -> S^1 -> cosine   (spec decoder)
  svd2d_eucl      : same 2D, score = -||q_u - q_c||                 (what S^1 discards)
  content_384_cos : cosine in full 384-d MiniLM space               (content ceiling)
  relatedness     : train-only co-occurrence to portfolio           (the 0.213 bar)
  popularity      : #firms holding c in train                       (relatedness-free)

Firm feature h_u = mean content embedding of its TRAIN portfolio (leak-free w.r.t.
the time split; the per-CPC title sample in cpc_content_*.npz is not year-filtered,
same accepted caveat as the content-entry diagnostic).

Run:  python pnode_patent_runner/asph_stage0_static.py --granularity maingroup
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/nakamuraroi/kumagai")


def coarsen(code: str, level: str) -> str:
    if level == "subclass":
        m = re.match(r"^[A-Z]\d{2}[A-Z]", code)
        return m.group(0) if m else code
    if level == "maingroup":
        return code.split("/")[0]
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--granularity", default="maingroup",
                    choices=["subgroup", "maingroup", "subclass"])
    ap.add_argument("--test-start", type=int, default=2019)
    ap.add_argument("--test-end", type=int, default=2023)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    args = ap.parse_args()

    df = pd.read_csv(ROOT / f"data/processed/bipartite_{args.domain}_firm.csv",
                     dtype={"u": str, "i": str})
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    df["c"] = df["i"].map(lambda x: coarsen(x, args.granularity))
    df = df[["year", "u", "c"]].drop_duplicates()

    train = df[df.year < args.test_start]
    test = df[(df.year >= args.test_start) & (df.year <= args.test_end)]
    all_ipcs = sorted(df.c.unique())
    print(f"domain={args.domain} granularity={args.granularity} "
          f"codes={len(all_ipcs)} | train<{args.test_start} rows={len(train):,} "
          f"test {args.test_start}-{args.test_end} rows={len(test):,}")

    # ---- content embeddings, coarsened by mean over member subgroups ----
    z = np.load(ROOT / f"data/processed/cpc_content_{args.domain}.npz",
                allow_pickle=True)
    sub_codes = z["codes"].tolist()
    sub_emb = z["emb"]
    groups = defaultdict(list)
    for k, sc in enumerate(sub_codes):
        groups[coarsen(sc, args.granularity)].append(k)
    cvec = {}
    for c in all_ipcs:
        if groups.get(c):
            cvec[c] = sub_emb[groups[c]].mean(0)
    print(f"content vectors for {len(cvec)}/{len(all_ipcs)} codes")

    # ---- portfolios (train years only) and test first-entries ----
    pre = defaultdict(set)
    for u, g in train.groupby("u"):
        pre[u] = set(g.c)
    test_first = defaultdict(set)
    for u, g in test.groupby("u"):
        new = set(g.c) - pre.get(u, set())
        if new:
            test_first[u] = new
    firms = [u for u in test_first if pre.get(u)]
    print(f"evaluable firms: {len(firms):,}")

    # ---- firm feature = mean content vector of train portfolio ----
    dim = sub_emb.shape[1]
    fvec = {}
    for u in firms:
        vs = [cvec[c] for c in pre[u] if c in cvec]
        if vs:
            fvec[u] = np.mean(vs, 0)

    # ---- SVD top-2 (PCA, deterministic) fit on all nodes, spec section 2.2 ----
    tech_order = [c for c in all_ipcs if c in cvec]
    firm_order = [u for u in firms if u in fvec]
    X = np.vstack([np.stack([cvec[c] for c in tech_order]),
                   np.stack([fvec[u] for u in firm_order])])
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Q = Xc @ Vt[:2].T                                   # (n_nodes, 2)
    q2 = {}
    for k, c in enumerate(tech_order):
        q2[("t", c)] = Q[k]
    for k, u in enumerate(firm_order):
        q2[("f", u)] = Q[len(tech_order) + k]
    var2 = np.linalg.svd(Xc, compute_uv=False)[:2] ** 2
    tot = (Xc ** 2).sum()
    print(f"SVD-2D explained variance: {var2.sum() / tot:.3f}")

    def unit(v, eps=1e-8):
        return v / (np.linalg.norm(v) + eps)

    # ---- co-occurrence graph from train (for relatedness/popularity bars) ----
    cooc = defaultdict(lambda: defaultdict(int))
    for (u, y), g in train.groupby(["u", "year"]):
        cl = list(set(g.c))
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                cooc[cl[a]][cl[b]] += 1
                cooc[cl[b]][cl[a]] += 1
    pop = defaultdict(int)
    for u, s in pre.items():
        for c in s:
            pop[c] += 1

    # ---- scorers: score(u) -> dict {code: score} ----
    def score_asph(u):
        if u not in fvec:
            return {}
        qu = unit(q2[("f", u)])
        return {c: float(qu @ unit(q2[("t", c)])) for c in tech_order}

    def score_svd2d_eucl(u):
        if u not in fvec:
            return {}
        qu = q2[("f", u)]
        return {c: -float(np.linalg.norm(qu - q2[("t", c)])) for c in tech_order}

    def score_content_full(u):
        if u not in fvec:
            return {}
        hu = unit(fvec[u])
        return {c: float(hu @ unit(cvec[c])) for c in tech_order}

    def score_relatedness(u):
        sc = defaultdict(float)
        for inc in pre[u]:
            for c, w in cooc[inc].items():
                sc[c] += w
        return sc

    methods = {
        "asph_s1_2d": score_asph,
        "svd2d_eucl": score_svd2d_eucl,
        "content_384_cos": score_content_full,
        "relatedness": score_relatedness,
        "popularity": lambda u: dict(pop),
    }

    results = {}
    for mname, fn in methods.items():
        hits = {k: 0 for k in args.ks}
        rr, n_pairs = 0.0, 0
        for u in firms:
            sc = fn(u)
            cand = [(c, sc.get(c, 0.0)) for c in all_ipcs if c not in pre[u]]
            cand.sort(key=lambda x: x[1], reverse=True)
            rank = {c: r for r, (c, _) in enumerate(cand, 1)}
            for true_c in test_first[u]:
                if true_c in rank:
                    r = rank[true_c]
                    rr += 1.0 / r
                    for k in args.ks:
                        if r <= k:
                            hits[k] += 1
                    n_pairs += 1
        res = {f"hit@{k}": hits[k] / n_pairs for k in args.ks}
        res["mrr"] = rr / n_pairs
        res["n_pairs"] = n_pairs
        results[mname] = res
        line = "  ".join(f"Hit@{k}={res[f'hit@{k}']:.3f}" for k in args.ks)
        print(f"{mname:16s}: {line}  MRR={res['mrr']:.3f}  (n_pairs={n_pairs:,})")

    out_dir = ROOT / "pnode_patent_runner/outputs/asph_stage0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.domain}_{args.granularity}.json"
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
