"""asph_stage0b_blend.py — Gate 0b/0c for the ASPH-Flow REDESIGN.

The redesigned spec (docs/ASPH_FLOW_REDESIGN.md) delegates prediction to the
proven signals and reserves dynamics-free machinery for uncertainty. This script
measures, leak-free, the two numbers the redesign needs:

Gate 0b (Layer 1, prediction): does blending content into relatedness beat the
relatedness bar (MRR 0.213)? Blend weight beta is selected on a VALIDATION
transition (portfolio <2017 -> entries 2017-2018), then applied once to the
test transition (portfolio <2019 -> entries 2019-2023).
  z-blend : score = (1-beta) * z(relatedness) + beta * z(content_384_cos)
  rrf     : reciprocal-rank fusion, 1/(60+rank_rel) + 1/(60+rank_cont)

Gate 0c (Layer 2, uncertainty): conformal rank-coverage. Nonconformity of a true
entry = its rank under the scorer. r*(alpha) = (1-alpha) quantile of validation
ranks (with (n+1) correction). Report empirical coverage of top-r* on test.
This is the Task B instance of the "only conformal transfers" claim (C8).

Run:  python pnode_patent_runner/asph_stage0b_blend.py --granularity maingroup
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


def build_period(df, cvec, cutoff_year, entry_start, entry_end):
    """Leak-free artifacts for one transition: portfolio/cooc from years <cutoff,
    first-entries within [entry_start, entry_end]."""
    train = df[df.year < cutoff_year]
    test = df[(df.year >= entry_start) & (df.year <= entry_end)]
    pre = {u: set(g.c) for u, g in train.groupby("u")}
    cooc = defaultdict(lambda: defaultdict(int))
    for (u, y), g in train.groupby(["u", "year"]):
        cl = list(set(g.c))
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                cooc[cl[a]][cl[b]] += 1
                cooc[cl[b]][cl[a]] += 1
    test_first = {}
    for u, g in test.groupby("u"):
        new = set(g.c) - pre.get(u, set())
        if new:
            test_first[u] = new
    firms = [u for u in test_first if pre.get(u)]
    fvec = {}
    for u in firms:
        vs = [cvec[c] for c in pre[u] if c in cvec]
        if vs:
            fvec[u] = np.mean(vs, 0)
    return pre, cooc, test_first, firms, fvec


def zscore(d):
    v = np.array(list(d.values()), dtype=float)
    mu, sd = v.mean(), v.std()
    if sd < 1e-12:
        return {k: 0.0 for k in d}
    return {k: (x - mu) / sd for k, x in d.items()}


def eval_ranks(firms, pre, test_first, all_ipcs, score_fn):
    """Return list of ranks of true entries (one per (firm, entry) pair)."""
    ranks = []
    for u in firms:
        sc = score_fn(u)
        cand = [(c, sc.get(c, 0.0)) for c in all_ipcs if c not in pre[u]]
        cand.sort(key=lambda x: x[1], reverse=True)
        rank = {c: r for r, (c, _) in enumerate(cand, 1)}
        for true_c in test_first[u]:
            if true_c in rank:
                ranks.append(rank[true_c])
    return ranks


def metrics(ranks, ks=(5, 10, 20)):
    r = np.array(ranks, dtype=float)
    out = {f"hit@{k}": float((r <= k).mean()) for k in ks}
    out["mrr"] = float((1.0 / r).mean())
    out["n_pairs"] = len(ranks)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--granularity", default="maingroup",
                    choices=["subgroup", "maingroup", "subclass"])
    ap.add_argument("--val-cutoff", type=int, default=2017)
    ap.add_argument("--val-end", type=int, default=2018)
    ap.add_argument("--test-start", type=int, default=2019)
    ap.add_argument("--test-end", type=int, default=2023)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2])
    args = ap.parse_args()

    df = pd.read_csv(ROOT / f"data/processed/bipartite_{args.domain}_firm.csv",
                     dtype={"u": str, "i": str})
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    df["c"] = df["i"].map(lambda x: coarsen(x, args.granularity))
    df = df[["year", "u", "c"]].drop_duplicates()
    all_ipcs = sorted(df.c.unique())

    z = np.load(ROOT / f"data/processed/cpc_content_{args.domain}.npz",
                allow_pickle=True)
    groups = defaultdict(list)
    for k, sc in enumerate(z["codes"].tolist()):
        groups[coarsen(sc, args.granularity)].append(k)
    cvec = {c: z["emb"][idx].mean(0) for c, idx in groups.items()}
    cunit = {c: v / (np.linalg.norm(v) + 1e-8) for c, v in cvec.items()}

    def make_scorers(pre, cooc, fvec):
        def s_rel(u):
            sc = defaultdict(float)
            for inc in pre[u]:
                for c, w in cooc[inc].items():
                    sc[c] += w
            return {c: sc.get(c, 0.0) for c in all_ipcs}

        def s_cont(u):
            if u not in fvec:
                return {c: 0.0 for c in all_ipcs}
            hu = fvec[u] / (np.linalg.norm(fvec[u]) + 1e-8)
            return {c: float(hu @ cunit[c]) if c in cunit else 0.0
                    for c in all_ipcs}

        def s_blend(u, beta):
            zr, zc = zscore(s_rel(u)), zscore(s_cont(u))
            return {c: (1 - beta) * zr[c] + beta * zc[c] for c in all_ipcs}

        def s_rrf(u):
            out = defaultdict(float)
            for s in (s_rel(u), s_cont(u)):
                order = sorted(s.items(), key=lambda x: x[1], reverse=True)
                for r, (c, _) in enumerate(order, 1):
                    out[c] += 1.0 / (60 + r)
            return out

        return s_rel, s_cont, s_blend, s_rrf

    # ---------- validation transition: <val_cutoff -> [val_cutoff, val_end] ----------
    pre_v, cooc_v, tf_v, firms_v, fvec_v = build_period(
        df, cvec, args.val_cutoff, args.val_cutoff, args.val_end)
    s_rel_v, s_cont_v, s_blend_v, s_rrf_v = make_scorers(pre_v, cooc_v, fvec_v)
    print(f"[val]  portfolio<{args.val_cutoff}, entries {args.val_cutoff}-{args.val_end}, "
          f"firms={len(firms_v):,}")

    betas = [round(b, 2) for b in np.arange(0.0, 1.01, 0.05)]
    val_curve = {}
    for b in betas:
        m = metrics(eval_ranks(firms_v, pre_v, tf_v, all_ipcs,
                               lambda u, b=b: s_blend_v(u, b)))
        val_curve[b] = m["mrr"]
    beta_star = max(val_curve, key=val_curve.get)
    print("  beta sweep (val MRR): " +
          "  ".join(f"{b:.2f}:{val_curve[b]:.3f}" for b in betas[::2]))
    print(f"  beta* = {beta_star}  (val MRR {val_curve[beta_star]:.3f}; "
          f"beta=0 -> {val_curve[0.0]:.3f})")

    # conformal calibration ranks on validation (relatedness and blend*)
    ranks_v_rel = eval_ranks(firms_v, pre_v, tf_v, all_ipcs, s_rel_v)
    ranks_v_bl = eval_ranks(firms_v, pre_v, tf_v, all_ipcs,
                            lambda u: s_blend_v(u, beta_star))

    def r_star(ranks, alpha):
        n = len(ranks)
        q = math.ceil((n + 1) * (1 - alpha)) / n
        return int(np.quantile(np.array(ranks), min(q, 1.0), method="higher"))

    # ---------- test transition: <test_start -> [test_start, test_end] ----------
    pre_t, cooc_t, tf_t, firms_t, fvec_t = build_period(
        df, cvec, args.test_start, args.test_start, args.test_end)
    s_rel_t, s_cont_t, s_blend_t, s_rrf_t = make_scorers(pre_t, cooc_t, fvec_t)
    print(f"[test] portfolio<{args.test_start}, entries {args.test_start}-{args.test_end}, "
          f"firms={len(firms_t):,}")

    results = {}
    ranks_t = {}
    for name, fn in [
        ("relatedness", s_rel_t),
        ("content_384_cos", s_cont_t),
        (f"blend_beta{beta_star}", lambda u: s_blend_t(u, beta_star)),
        ("rrf", s_rrf_t),
    ]:
        ranks = eval_ranks(firms_t, pre_t, tf_t, all_ipcs, fn)
        ranks_t[name] = ranks
        results[name] = metrics(ranks)
        m = results[name]
        print(f"  {name:20s}: Hit@5={m['hit@5']:.3f}  Hit@10={m['hit@10']:.3f}  "
              f"Hit@20={m['hit@20']:.3f}  MRR={m['mrr']:.3f}  (n={m['n_pairs']:,})")

    # ---------- Gate 0c: conformal rank coverage, calibrated on val ----------
    conformal = {}
    print("[conformal] r* from validation ranks -> empirical coverage on test")
    for alpha in args.alphas:
        for name, rv, rt in [("relatedness", ranks_v_rel, ranks_t["relatedness"]),
                             (f"blend_beta{beta_star}", ranks_v_bl,
                              ranks_t[f"blend_beta{beta_star}"])]:
            rs = r_star(rv, alpha)
            cov = float((np.array(rt) <= rs).mean())
            conformal[f"{name}_alpha{alpha}"] = {"r_star": rs, "coverage": cov,
                                                 "target": 1 - alpha}
            print(f"  alpha={alpha:.2f} {name:20s}: r*={rs:4d}  "
                  f"coverage={cov:.3f} (target {1-alpha:.2f})")

    out_dir = ROOT / "pnode_patent_runner/outputs/asph_stage0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.domain}_{args.granularity}_gate0bc.json"
    out.write_text(json.dumps({
        "args": vars(args), "beta_star": beta_star,
        "val_curve": {str(k): v for k, v in val_curve.items()},
        "test": results, "conformal": conformal}, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
