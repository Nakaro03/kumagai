"""diagnose_convergence_signal.py — does CPC-CPC technology convergence have
predictive signal?

This is a CHEAP go/no-go BEFORE building any Neural SDE. We test whether the
standard temporal-link-prediction signal exists: can the structure of the
CPC co-occurrence graph up to year Y predict which NEW CPC pairs start
co-occurring in year Y+1?

CPC co-occurrence: two CPC codes a,b are linked in year Y if the same inventor
filed patents in both a and b during year Y (inventor bridges the two areas).

Diagnostics reported:
  1. Convergence link prediction AUC/AP using simple topological scores
     (Common Neighbors, Adamic-Adar, Jaccard, Preferential Attachment) on the
     graph up to year Y, evaluated on genuinely NEW edges at Y+1.
     AUC >> 0.5  => signal exists => a Neural SDE on convergence is worth building.
  2. Persistence-style autocorrelation of the per-CPC activity growth, to
     contrast: confirms why growth-rate ranking (X5's target) was unpredictable.

Run (main env, has pandas/sklearn):
  python pnode_patent_runner/diagnose_convergence_signal.py --domain energy
"""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("/home/nakamuraroi/kumagai")
CSV = {
    "energy":       "data/processed/bipartite_energy.csv",
    "construction": "data/processed/bipartite_construction.csv",
    "computing":    "data/processed/bipartite_computing.csv",
    "semiconductor":"data/processed/bipartite_semiconductor.csv",
    "agrifood":     "data/processed/bipartite_agrifood.csv",
    "pharma":       "data/processed/bipartite_pharma.csv",
}


def yearly_cooc_edges(df: pd.DataFrame, year: int) -> set:
    """Set of frozenset({a,b}) CPC pairs co-filed by the same inventor in `year`."""
    sub = df[df.year == year]
    edges = set()
    # group by inventor: CPC codes that inventor touched this year
    for _, codes in sub.groupby("u")["i"]:
        uniq = sorted(set(codes))
        if len(uniq) < 2:
            continue
        for a, b in itertools.combinations(uniq, 2):
            edges.add(frozenset((a, b)))
    return edges


def build_adj(edges: set) -> dict:
    adj = defaultdict(set)
    for e in edges:
        a, b = tuple(e)
        adj[a].add(b); adj[b].add(a)
    return adj


def score_pairs(pairs, adj):
    """Return dict of score arrays for several topological predictors."""
    cn, aa, jac, pa = [], [], [], []
    deg = {n: len(adj[n]) for n in adj}
    for e in pairs:
        a, b = tuple(e)
        na, nb = adj.get(a, set()), adj.get(b, set())
        common = na & nb
        cn.append(len(common))
        aa.append(sum(1.0 / np.log(len(adj[c]) + 1e-9) for c in common if len(adj[c]) > 1))
        union = len(na | nb)
        jac.append(len(common) / union if union else 0.0)
        pa.append(deg.get(a, 0) * deg.get(b, 0))
    return {"CommonNeighbors": np.array(cn, float),
            "AdamicAdar": np.array(aa, float),
            "Jaccard": np.array(jac, float),
            "PrefAttach": np.array(pa, float)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="energy")
    ap.add_argument("--year-start", type=int, default=2010)
    ap.add_argument("--year-end", type=int, default=2021)
    ap.add_argument("--neg-ratio", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(ROOT / CSV[args.domain])
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    df = df[(df.year >= args.year_start - 5) & (df.year <= args.year_end)]
    print(f"domain={args.domain} rows={len(df)} "
          f"unique CPC={df.i.nunique()} inventors={df.u.nunique()}")

    all_codes = sorted(df.i.unique())

    # cumulative co-occurrence graph up to each year
    print("\nConvergence link prediction (predict NEW co-occurring CPC pairs at Y+1):")
    print(f"{'Y->Y+1':<10} {'#new':<7} {'CN':<7} {'AA':<7} {'Jac':<7} {'PA':<7}  (AUC)")
    aucs = defaultdict(list)
    cum_edges = set()
    # warm up cumulative graph
    for y in range(args.year_start - 5, args.year_start):
        cum_edges |= yearly_cooc_edges(df, y)

    for y in range(args.year_start, args.year_end):
        adj = build_adj(cum_edges)
        next_edges = yearly_cooc_edges(df, y + 1)
        new_edges = [e for e in next_edges if e not in cum_edges]
        # candidates must be between codes already present in the graph
        present = set(adj.keys())
        pos = [e for e in new_edges if set(e) <= present]
        if len(pos) < 10:
            cum_edges |= next_edges
            continue
        # negatives: random non-edges among present codes
        present_list = list(present)
        neg = set()
        target_neg = len(pos) * args.neg_ratio
        attempts = 0
        while len(neg) < target_neg and attempts < target_neg * 50:
            a, b = rng.choice(present_list, 2, replace=False)
            e = frozenset((a, b))
            if e not in cum_edges and e not in next_edges:
                neg.add(e)
            attempts += 1
        pairs = pos + list(neg)
        labels = np.array([1] * len(pos) + [0] * len(neg))
        scores = score_pairs(pairs, adj)
        row = f"{y}->{y+1:<5} {len(pos):<7}"
        for name in ["CommonNeighbors", "AdamicAdar", "Jaccard", "PrefAttach"]:
            s = scores[name]
            try:
                auc = roc_auc_score(labels, s)
            except ValueError:
                auc = float("nan")
            aucs[name].append(auc)
            short = {"CommonNeighbors":"CN","AdamicAdar":"AA","Jaccard":"Jac","PrefAttach":"PA"}[name]
            row += f" {auc:<7.3f}"
        print(row)
        cum_edges |= next_edges

    print("\nMean AUC across years:")
    for name in ["CommonNeighbors", "AdamicAdar", "Jaccard", "PrefAttach"]:
        vals = [a for a in aucs[name] if a == a]
        if vals:
            print(f"  {name:18s} = {np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)} years)")

    best = max((np.mean([a for a in aucs[n] if a==a]) if any(x==x for x in aucs[n]) else 0)
               for n in aucs)
    print("\n" + "=" * 60)
    if best >= 0.70:
        print(f"  VERDICT: STRONG signal (best mean AUC={best:.3f} >= 0.70)")
        print("  => Convergence prediction is learnable. Build the Neural SDE.")
    elif best >= 0.60:
        print(f"  VERDICT: MODERATE signal (best mean AUC={best:.3f}).")
        print("  => Worth building; expect modest gains over topological baselines.")
    else:
        print(f"  VERDICT: WEAK signal (best mean AUC={best:.3f} < 0.60).")
        print("  => Even convergence is hard here. Reconsider before building.")
    print("=" * 60)


if __name__ == "__main__":
    main()
