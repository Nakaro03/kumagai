"""diagnose_convergence_timing.py — Phase-0 go/no-go for PISDE-Converge.

The binary "will pair a,b ever converge?" signal is already confirmed strong
(diagnose_convergence_signal.py: Adamic-Adar AUC 0.83-0.89). That alone does
NOT justify a Neural SDE — random forests / temporal heterogeneous GNNs already
hit AUC>0.90 in the 2024-2025 literature. PISDE-Converge's value must come from
outputs those methods cannot produce. This script measures whether the DATA
supports those outputs, BEFORE any SDE is built. Three questions:

  Q1. TIMING signal (the survival target).
      Do topological scores at anchor year Y0 predict *when* a not-yet-converged
      pair first converges? Measured by Harrell's C-index over (time-to-event,
      event/censored). C-index > 0.5 => timing is learnable => a continuous-time
      hazard/SDE has a target. C-index ~ 0.5 => only eventual occurrence is
      predictable, not its timing => weaken the time-to-convergence claim.

  Q2. HARD-NEGATIVE headroom (the EdgeBank/TGB-Seq lesson).
      How much does binary AUC drop when negatives are structurally plausible
      (pairs sharing >=1 common neighbor, i.e. AA>0) instead of random? A large
      drop means simple topology is exploiting easy negatives and a learned model
      has real room to win. A small drop means topology already solves the hard
      case and a heavier model is hard to justify.

  Q3. INDUCTIVE opportunity.
      What fraction of true future convergences involve a CPC code NOT present in
      the graph at Y0? Pure topological scores structurally score these 0 (no
      neighbors), so this fraction is the headroom an embedding-based SDE could
      capture and baselines cannot.

Run (main env: pandas 2.0 / sklearn 1.3):
  python pnode_patent_runner/diagnose_convergence_timing.py --domain energy
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# reuse the validated helpers from the binary diagnostic
from diagnose_convergence_signal import (
    ROOT,
    CSV,
    yearly_cooc_edges,
    build_adj,
    score_pairs,
)

SCORE_NAMES = ["CommonNeighbors", "AdamicAdar", "Jaccard", "PrefAttach"]


def first_convergence_year(df: pd.DataFrame, years: range) -> dict:
    """Map frozenset({a,b}) -> first year the pair co-occurs (over `years`)."""
    first: dict = {}
    for y in years:
        for e in yearly_cooc_edges(df, y):
            if e not in first:
                first[e] = y
    return first


def c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's concordance index with right-censoring (vectorized).

    risk : higher => predicted to converge sooner.
    time : time-to-event (events) or censoring time (event==0).
    A comparable pair (i,j) requires the earlier one (i) to be an event and j to
    survive strictly past it (or tie in time while being censored).
    """
    risk = np.asarray(risk, float)
    time = np.asarray(time, float)
    event = np.asarray(event)
    ev = np.flatnonzero(event == 1)
    if ev.size == 0:
        return float("nan")
    ti = time[ev][:, None]                    # (E,1)
    ri = risk[ev][:, None]
    comp = (time[None, :] > ti) | ((time[None, :] == ti) & (event[None, :] == 0))
    den = comp.sum()
    if den == 0:
        return float("nan")
    conc = (ri > risk[None, :]) & comp
    ties = (ri == risk[None, :]) & comp
    return float((conc.sum() + 0.5 * ties.sum()) / den)


def sample_present_nonedges(present_list, cum_edges, n, rng, require_common=False, adj=None):
    """Sample non-co-occurring pairs among present nodes.

    require_common=True restricts to HARD negatives: pairs with >=1 common
    neighbor (so a topological score would rank them high).
    """
    out = set()
    attempts = 0
    cap = n * 200
    while len(out) < n and attempts < cap:
        a, b = rng.choice(present_list, 2, replace=False)
        e = frozenset((a, b))
        attempts += 1
        if e in cum_edges or e in out:
            continue
        if require_common:
            if len(adj.get(a, set()) & adj.get(b, set())) < 1:
                continue
        out.add(e)
    return list(out)


def run_domain(df: pd.DataFrame, args, rng) -> dict:
    yend = args.year_end
    # ---- Q1: timing / survival, averaged over anchor years ----
    cindex = defaultdict(list)
    induct_frac = []
    auc_easy = defaultdict(list)
    auc_hard = defaultdict(list)

    for y0 in args.anchors:
        # cumulative graph up to and including y0
        cum_edges = set()
        for y in range(args.year_start - 5, y0 + 1):
            cum_edges |= yearly_cooc_edges(df, y)
        adj = build_adj(cum_edges)
        present = set(adj.keys())
        present_list = list(present)
        if len(present_list) < 20:
            continue

        # first convergence year for pairs that converge AFTER y0
        future_first = first_convergence_year(df, range(y0 + 1, yend + 1))
        new_future = {e: yr for e, yr in future_first.items() if e not in cum_edges}

        # Q3: inductive fraction — convergence touching a node absent at y0
        if new_future:
            induct = sum(1 for e in new_future if not (set(e) <= present))
            induct_frac.append(induct / len(new_future))

        # positives for survival: convergences between PRESENT nodes (topology-scorable)
        pos = [(e, yr) for e, yr in new_future.items() if set(e) <= present]
        if len(pos) < 10:
            continue
        if len(pos) > args.max_pos:                       # subsample for tractable C-index
            idx = rng.choice(len(pos), args.max_pos, replace=False)
            pos = [pos[k] for k in idx]
        n_pos = len(pos)

        # censored sample: present non-edges never converging by yend
        cens = sample_present_nonedges(present_list, set(future_first) | cum_edges,
                                       n_pos * args.neg_ratio, rng)
        surv_pairs = [e for e, _ in pos] + cens
        time = np.array([yr - y0 for _, yr in pos] + [yend - y0 + 1] * len(cens), float)
        event = np.array([1] * n_pos + [0] * len(cens))
        sc = score_pairs(surv_pairs, adj)
        for name in SCORE_NAMES:
            cindex[name].append(c_index(sc[name], time, event))

        # ---- Q2: next-year binary AUC, easy vs hard negatives ----
        next_edges = yearly_cooc_edges(df, y0 + 1)
        bin_pos = [e for e in next_edges if e not in cum_edges and set(e) <= present]
        if len(bin_pos) > args.max_pos:
            bin_pos = [bin_pos[k] for k in rng.choice(len(bin_pos), args.max_pos, replace=False)]
        if len(bin_pos) >= 10:
            neg_easy = sample_present_nonedges(present_list, cum_edges | next_edges,
                                               len(bin_pos) * args.neg_ratio, rng)
            neg_hard = sample_present_nonedges(present_list, cum_edges | next_edges,
                                               len(bin_pos) * args.neg_ratio, rng,
                                               require_common=True, adj=adj)
            for neg, store in [(neg_easy, auc_easy), (neg_hard, auc_hard)]:
                if len(neg) < 10:
                    continue
                pairs = bin_pos + neg
                labels = np.array([1] * len(bin_pos) + [0] * len(neg))
                s = score_pairs(pairs, adj)
                for name in SCORE_NAMES:
                    try:
                        store[name].append(roc_auc_score(labels, s[name]))
                    except ValueError:
                        pass

    return {"cindex": cindex, "auc_easy": auc_easy, "auc_hard": auc_hard,
            "induct_frac": induct_frac}


def mean_std(vals):
    vals = [v for v in vals if v == v]
    return (np.mean(vals), np.std(vals), len(vals)) if vals else (float("nan"), float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="energy")
    ap.add_argument("--year-start", type=int, default=2010)
    ap.add_argument("--year-end", type=int, default=2021)
    ap.add_argument("--anchors", type=int, nargs="+", default=[2013, 2015, 2017])
    ap.add_argument("--neg-ratio", type=int, default=5)
    ap.add_argument("--max-pos", type=int, default=1500,
                    help="cap positives per anchor (keeps C-index O(E*N) tractable)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(ROOT / CSV[args.domain])
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    df = df[(df.year >= args.year_start - 5) & (df.year <= args.year_end)]
    print(f"domain={args.domain} rows={len(df)} unique CPC={df.i.nunique()} "
          f"inventors={df.u.nunique()} anchors={args.anchors}")

    r = run_domain(df, args, rng)

    print("\nQ1. TIMING signal — survival C-index (>0.5 => when-it-converges is learnable):")
    best_c = 0.0
    for name in SCORE_NAMES:
        m, s, n = mean_std(r["cindex"][name])
        best_c = max(best_c, m if m == m else 0)
        print(f"  {name:18s} C-index = {m:.3f} ± {s:.3f}  (n={n} anchors)")

    print("\nQ2. HARD-NEGATIVE headroom — next-year AUC (easy vs common-neighbor negatives):")
    max_drop = 0.0
    for name in SCORE_NAMES:
        me, se, ne = mean_std(r["auc_easy"][name])
        mh, sh, nh = mean_std(r["auc_hard"][name])
        drop = (me - mh) if (me == me and mh == mh) else float("nan")
        if drop == drop:
            max_drop = max(max_drop, drop)
        print(f"  {name:18s} easy={me:.3f}  hard={mh:.3f}  drop={drop:+.3f}")

    mf, sf, nf = mean_std(r["induct_frac"])
    print(f"\nQ3. INDUCTIVE opportunity — fraction of future convergences touching a "
          f"node absent at Y0:\n  {mf:.3f} ± {sf:.3f}  (n={nf} anchors) "
          f"[topological baselines score these 0 => embedding-SDE headroom]")

    print("\n" + "=" * 64)
    timing_ok = best_c >= 0.60
    headroom_ok = max_drop >= 0.10 or mf >= 0.20
    print(f"  Q1 timing C-index best = {best_c:.3f}  -> {'OK' if timing_ok else 'WEAK'}")
    print(f"  Q2 max hard-neg AUC drop = {max_drop:+.3f};  Q3 inductive frac = {mf:.3f}")
    print(f"  Q2/Q3 learned-model headroom -> {'OK' if headroom_ok else 'THIN'}")
    if timing_ok and headroom_ok:
        print("  VERDICT: GO. Timing is learnable AND topology leaves headroom.")
        print("           Build PISDE-Converge (survival + Phi landscape + uncertainty).")
    elif timing_ok or headroom_ok:
        print("  VERDICT: PARTIAL. One axis fires — applied paper scope, not full SDE claim.")
    else:
        print("  VERDICT: NO-GO on the SDE-specific value. Convergence is binary-only and")
        print("           topology already solves the hard case; prefer descriptive/neg-results.")
    print("=" * 64)


if __name__ == "__main__":
    main()
