"""task_b_static_ceiling.py — the make-or-break number before building any model.

Task B (technology entry prediction), TIME-SPLIT, STATIC baselines only.
For each firm, given its portfolio built from years < TEST_START, rank all
not-yet-entered IPCs and check whether the IPCs it ACTUALLY first-enters during
the test window are ranked high. This is the relatedness ceiling that PNODE /
plain-ODE / GRU / JODIE must beat to justify themselves.

Baselines:
  - relatedness : score(c) = Σ_{i in portfolio} cooc(c, i)   (co-occurrence to portfolio)
  - adamic_adar : score(c) = Σ_{i in portfolio} Σ_{w in N(c)∩N(i)} 1/log(deg w)
  - popularity  : score(c) = #firms that entered c during train (relatedness-free)

Co-occurrence graph is built ONLY from years < TEST_START (leak-free).
Granularity is configurable (subgroup / maingroup / subclass).

Run:  python pnode_patent_runner/task_b_static_ceiling.py --granularity subgroup
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/home/nakamuraroi/kumagai")


def coarsen(code: str, level: str) -> str:
    if level == "subclass":          # E21B7/20 -> E21B
        m = re.match(r"^[A-Z]\d{2}[A-Z]", code)
        return m.group(0) if m else code
    if level == "maingroup":         # E21B7/20 -> E21B7
        return code.split("/")[0]
    return code                      # subgroup (full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--granularity", default="subgroup",
                    choices=["subgroup", "maingroup", "subclass"])
    ap.add_argument("--test-start", type=int, default=2019)
    ap.add_argument("--test-end", type=int, default=2023)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--skip-adamic", action="store_true",
                    help="skip adamic_adar (its 2-hop pass is O(deg^2), slow at subgroup)")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / f"data/processed/bipartite_{args.domain}_firm.csv",
                     dtype={"u": str, "i": str})
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    df["c"] = df["i"].map(lambda x: coarsen(x, args.granularity))
    df = df[["year", "u", "c"]].drop_duplicates()

    train = df[df.year < args.test_start]
    test = df[(df.year >= args.test_start) & (df.year <= args.test_end)]
    print(f"domain={args.domain} granularity={args.granularity} "
          f"IPCs={df.c.nunique()} | train<{args.test_start} rows={len(train):,} "
          f"test {args.test_start}-{args.test_end} rows={len(test):,}")

    # ---- co-occurrence graph from TRAIN only (same firm-year co-occurrence) ----
    cooc = defaultdict(lambda: defaultdict(int))
    deg = defaultdict(int)
    fy_train = defaultdict(set)
    for (u, y), g in train.groupby(["u", "year"]):
        cl = list(g.c)
        fy_train[(u, y)] = set(cl)
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                cooc[cl[a]][cl[b]] += 1
                cooc[cl[b]][cl[a]] += 1
    for c in cooc:
        deg[c] = len(cooc[c])

    all_ipcs = sorted(df.c.unique())

    # ---- portfolios (years < test) and test entries (first appearance in window) ----
    pre = defaultdict(set)
    for (u, y), s in fy_train.items():
        pre[u] |= s
    test_first = defaultdict(set)
    for u, g in test.groupby("u"):
        new = set(g.c) - pre.get(u, set())
        if new:
            test_first[u] = new

    # popularity (relatedness-free): #distinct firms entering c in train
    pop = defaultdict(int)
    for u, s in pre.items():
        for c in s:
            pop[c] += 1

    def score_relatedness(portfolio):
        sc = defaultdict(float)
        for inc in portfolio:
            for c, w in cooc[inc].items():
                sc[c] += w
        return sc

    def score_adamic(portfolio):
        sc = defaultdict(float)
        for inc in portfolio:
            for w, _ in cooc[inc].items():          # w is neighbour of incumbent
                inv = 1.0 / math.log(deg[w] + 1e-9) if deg[w] > 1 else 0.0
                for c, _ in cooc[w].items():         # c two hops from incumbent
                    sc[c] += inv
        return sc

    methods = {"relatedness": score_relatedness}
    if not args.skip_adamic:
        methods["adamic_aar"] = score_adamic
    methods["popularity"] = lambda port: {c: pop[c] for c in all_ipcs}

    firms = [u for u in test_first if pre.get(u)]
    print(f"evaluable firms (portfolio>0 & >=1 test entry): {len(firms):,}")

    for mname, fn in methods.items():
        hits = {k: 0 for k in args.ks}
        rr = 0.0
        n_pairs = 0
        for u in firms:
            port = pre[u]
            sc = fn(port)
            # candidates = not-yet-entered IPCs
            cand = [(c, sc.get(c, 0.0)) for c in all_ipcs if c not in port]
            cand.sort(key=lambda x: x[1], reverse=True)
            rank = {c: r for r, (c, _) in enumerate(cand, 1)}
            for true_c in test_first[u]:
                if true_c in rank:                  # IPC seen at this granularity
                    r = rank[true_c]
                    rr += 1.0 / r
                    for k in args.ks:
                        if r <= k:
                            hits[k] += 1
                    n_pairs += 1
        line = "  ".join(f"Hit@{k}={hits[k] / n_pairs:.3f}" for k in args.ks)
        print(f"{mname:12s}: {line}  MRR={rr / n_pairs:.3f}  (n_pairs={n_pairs:,})")


if __name__ == "__main__":
    main()
