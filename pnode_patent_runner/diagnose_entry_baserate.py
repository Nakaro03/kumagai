"""diagnose_entry_baserate.py — firm technology-entry base-rate diagnostics.

Answers the professor's question: is the task trivial because firms don't move
(hypothesis A), or do firms move but only into *related* fields (hypothesis B)?

Input : data/processed/bipartite_{domain}_firm.csv  (ts, u=firm, i=cpc)
Output: prints 5 numbers + writes data/processed/entry_baserate_{domain}.json

Metrics
  1. Portfolio persistence : mean year-over-year Jaccard of a firm's CPC set.
  2. Stickiness           : P(cpc in portfolio_{t} | cpc in portfolio_{t-1}).
  3. Entry rate           : mean # of *new* CPCs a firm adds per active year.
  4. Relatedness of entry : among entries, fraction 1-hop related to the firm's
                            prior portfolio in the CPC co-occurrence graph
                            (= "principle of relatedness"). The complement is
                            the non-trivial "jump" rate.
  5. No-change ceiling    : accuracy of predicting next-year portfolio = this
                            year's portfolio (the trivial baseline's top score).

Run:  python pnode_patent_runner/diagnose_entry_baserate.py --domain construction
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/nakamuraroi/kumagai")


def _cooc_edges(neigh, cset):
    """Add same-firm-year co-occurrence edges among cset into neigh."""
    cl = list(cset)
    for a in range(len(cl)):
        for b in range(a + 1, len(cl)):
            neigh[cl[a]].add(cl[b])
            neigh[cl[b]].add(cl[a])


def relatedness_of_entries(fy, allc, min_firm_years, leak_free):
    """Fraction of new firm-CPC entries that are 1-hop related to the firm's
    prior portfolio, plus a random-CPC null.

    leak_free=False : co-occurrence graph built over ALL firm-years. The entry's
        own year creates c<->incumbent edges, inflating the related rate.
    leak_free=True  : the graph for testing entries in year t uses only
        co-occurrences from years STRICTLY BEFORE t, so an entered CPC's
        neighbours are defined by other firms / earlier years — no self-edge.
    """
    if not leak_free:
        neigh = defaultdict(set)
        for ydict in fy.values():
            for cset in ydict.values():
                _cooc_edges(neigh, cset)
        return _score_entries(fy, allc, min_firm_years, lambda _y: neigh)

    # leak-free: grow the graph year by year; test year t against years < t.
    year_sets = defaultdict(list)
    for ydict in fy.values():
        for y, cset in ydict.items():
            year_sets[y].append(cset)
    years_sorted = sorted(year_sets)
    neigh = defaultdict(set)
    state = {"next_idx": 0}

    def graph_before(y):
        # add every year strictly less than y that isn't in the graph yet
        while state["next_idx"] < len(years_sorted) and years_sorted[state["next_idx"]] < y:
            for cset in year_sets[years_sorted[state["next_idx"]]]:
                _cooc_edges(neigh, cset)
            state["next_idx"] += 1
        return neigh

    return _score_entries(fy, allc, min_firm_years, graph_before, ordered=True)


def _score_entries(fy, allc, min_firm_years, graph_for_year, ordered=False):
    """Walk entries and tally related / null hits. When ordered=True, entries are
    visited in global year order so graph_for_year can grow monotonically."""
    # collect (year, incumbents, new_cpc) entries
    items = []
    for ydict in fy.values():
        years = sorted(ydict)
        if len(years) < min_firm_years:
            continue
        cumulative = set()
        for k, y in enumerate(years):
            cur = ydict[y]
            if k > 0:
                for c in cur - cumulative:
                    items.append((y, frozenset(cumulative), c))
            cumulative |= cur
    if ordered:
        items.sort(key=lambda x: x[0])

    e_rel = e_tot = n_rel = n_tot = 0
    for y, cum, c in items:
        neigh = graph_for_year(y)
        e_tot += 1
        if cum & neigh[c]:
            e_rel += 1
        n_tot += 1
        if cum & neigh[random.choice(allc)]:
            n_rel += 1
    return e_rel, e_tot, n_rel, n_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--min-firm-years", type=int, default=2,
                    help="only firms active in >= this many distinct years")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random null")
    ap.add_argument("--leak-free", action="store_true",
                    help="build the relatedness graph only from years < entry year "
                         "(removes the self-edge created by the entry itself)")
    args = ap.parse_args()
    random.seed(args.seed)

    path = ROOT / f"data/processed/bipartite_{args.domain}_firm.csv"
    df = pd.read_csv(path, dtype={"u": str, "i": str})
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    df = df[["year", "u", "i"]].drop_duplicates()
    print(f"loaded {path.name}: {len(df):,} (year,firm,cpc) rows, "
          f"{df.u.nunique():,} firms, {df.i.nunique():,} CPCs, "
          f"years {df.year.min()}-{df.year.max()}")

    # firm -> {year -> set(cpc)}
    fy = defaultdict(dict)
    for (u, y), g in df.groupby(["u", "year"]):
        fy[u][y] = set(g.i)

    allc = list(df.i.unique())  # for the random null

    # ---- mode-independent metrics: persistence, stickiness, no-change ceiling ----
    jacc, stick_num, stick_den = [], 0, 0
    entries_per_year, n_active_years = 0, 0
    nochange_correct, nochange_total = 0, 0
    for u, ydict in fy.items():
        years = sorted(ydict)
        if len(years) < args.min_firm_years:
            continue
        cumulative = set()
        for k, y in enumerate(years):
            cur = ydict[y]
            if k > 0:
                prev = ydict[years[k - 1]]
                union = prev | cur
                if union:
                    jacc.append(len(prev & cur) / len(union))   # 1. persistence
                stick_num += len(prev & cur)                    # 2. stickiness
                stick_den += len(prev)
                nochange_correct += len(prev & cur)             # 5. no-change ceiling
                nochange_total += len(union)
                entries_per_year += len(cur - cumulative)       # 3. entry rate
                n_active_years += 1
            cumulative |= cur

    # 4. relatedness of entry (with random null) — leak-free or full graph
    entry_related, entry_total, null_related, null_total = relatedness_of_entries(
        fy, allc, args.min_firm_years, leak_free=args.leak_free)

    res = {
        "domain": args.domain,
        "relatedness_mode": "leak_free" if args.leak_free else "full_graph",
        "n_firms_kept": sum(1 for u in fy if len(fy[u]) >= args.min_firm_years),
        "portfolio_persistence_jaccard": round(float(np.mean(jacc)), 4) if jacc else None,
        "stickiness_pct": round(100 * stick_num / stick_den, 2) if stick_den else None,
        "mean_entries_per_firm_year": round(entries_per_year / n_active_years, 3) if n_active_years else None,
        "entry_related_pct": round(100 * entry_related / entry_total, 2) if entry_total else None,
        "entry_jump_pct": round(100 * (entry_total - entry_related) / entry_total, 2) if entry_total else None,
        "null_related_pct": round(100 * null_related / null_total, 2) if null_total else None,
        "relatedness_lift_pts": round(100 * (entry_related / entry_total - null_related / null_total), 2)
        if entry_total and null_total else None,
        "nochange_ceiling_pct": round(100 * nochange_correct / nochange_total, 2) if nochange_total else None,
        "n_entries_total": entry_total,
    }

    print(f"\n=== ENTRY BASE-RATE DIAGNOSTICS (relatedness: {res['relatedness_mode']}) ===")
    print(f"firms kept (>= {args.min_firm_years} active years) : {res['n_firms_kept']:,}")
    print(f"1. portfolio persistence (Jaccard) : {res['portfolio_persistence_jaccard']}")
    print(f"2. stickiness  P(keep CPC)         : {res['stickiness_pct']} %")
    print(f"3. mean new CPCs / firm-year       : {res['mean_entries_per_firm_year']}")
    print(f"4. entries 1-hop RELATED           : {res['entry_related_pct']} %   "
          f"(n={res['n_entries_total']:,})")
    print(f"   entries non-trivial JUMP        : {res['entry_jump_pct']} %")
    print(f"   random-null related            : {res['null_related_pct']} %   "
          f"(lift {res['relatedness_lift_pts']:+} pts)")
    print(f"5. no-change baseline ceiling      : {res['nochange_ceiling_pct']} %")

    suffix = "_leakfree" if args.leak_free else ""
    out = ROOT / f"data/processed/entry_baserate_{args.domain}{suffix}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")

    # interpretation hint
    print("\n--- reading ---")
    print("High persistence + high stickiness + most entries RELATED => hypothesis B:")
    print("  firms DO move, but into related fields; the predictable part is trivial,")
    print("  the JUMP part is the rare, idiosyncratic, unpredictable residual.")


if __name__ == "__main__":
    main()
