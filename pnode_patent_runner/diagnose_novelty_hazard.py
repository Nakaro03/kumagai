"""diagnose_novelty_hazard.py — can we self-predict JUMPS without proximity?

Every signal we tested (structure AA, semantic content, human-aware expertise)
scored ~0.50 on genuinely novel/inductive events, because all are PROXIMITY to
the current state and a jump is by definition far from it. This script tests two
NON-proximity signals on exactly those jumps:

  BROKERAGE (where is ripe): for a candidate convergence (i,j) with NO common
    neighbour (AA==0), count BRIDGE inventors = inventors active in i's graph
    neighbourhood AND in j's neighbourhood. Even with no direct i-j link, such
    inventors are structurally positioned to bridge the gap (structural-hole
    filling, Burt). Orthogonal to AA (which needs a shared neighbour).
  PRECURSOR (when): endpoints heating up before the jump — activity ACCELERATION
    (2nd difference) and rising VARIANCE of the recent per-year activity series
    of i and j (critical-slowing-down style early-warning, Scheffer 2009).

Target = NEW co-occurring CPC pairs at Y+1 that are JUMPS (AA==0, endpoints share
no neighbour). Hard negatives = present AA==0 pairs that do NOT converge.

  brokerage / precursor AUC > 0.5 (where AA==0.5) => novelty is self-predictable
  at the propensity level via non-proximity structure => the "right form".
  ~0.5 => jumps are irreducibly unpredictable from the trend data itself.

Run:  python pnode_patent_runner/diagnose_novelty_hazard.py --domain construction
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from diagnose_convergence_signal import ROOT, CSV, yearly_cooc_edges, build_adj


def build_cands(df, Y, rng, max_pos=800, neg_ratio=5, nbr_cap=300):
    cum = set()
    for y in range(Y - 8, Y + 1):
        cum |= yearly_cooc_edges(df, y)
    adj = build_adj(cum)
    present = set(adj)
    sub = df[df.year <= Y]
    inv_of = sub.groupby("i")["u"].agg(set).to_dict()           # inventors per CPC up to Y
    act = (sub.groupby(["i", "year"])["u"].nunique()
              .unstack(fill_value=0))                            # CPC x year activity
    years = sorted(act.columns)

    def aa(i, j):
        return len(adj.get(i, set()) & adj.get(j, set()))

    nxt = yearly_cooc_edges(df, Y + 1)
    newe = [e for e in nxt if e not in cum]
    jumps = [e for e in newe if set(e) <= present and aa(*tuple(e)) == 0]
    rng.shuffle(jumps)
    pos = jumps[:max_pos]
    if len(pos) < 20:
        return None

    # hard negatives: present AA==0 pairs, both currently active, not converging
    active_now = [c for c in present if act.loc[c, Y] > 0] if Y in act.columns else list(present)
    negs = set()
    tries = 0
    target = len(pos) * neg_ratio
    while len(negs) < target and tries < target * 80:
        a, b = rng.choice(active_now, 2, replace=False)
        e = frozenset((a, b))
        tries += 1
        if e not in cum and e not in nxt and aa(a, b) == 0:
            negs.add(e)
    pairs = pos + list(negs)
    labels = [1] * len(pos) + [0] * len(negs)

    # inv_near(c): inventors active in c's graph neighbourhood (cap neighbours)
    endpoints = {c for e in pairs for c in e}
    inv_near = {}
    for c in endpoints:
        nbrs = list(adj.get(c, set()))
        if len(nbrs) > nbr_cap:
            nbrs = [nbrs[k] for k in rng.choice(len(nbrs), nbr_cap, replace=False)]
        s = set()
        for cc in nbrs:
            s |= inv_of.get(cc, set())
        inv_near[c] = s

    def accel(c):
        if c not in act.index:
            return 0.0, 0.0
        ys = [y for y in years if y <= Y][-4:]
        v = np.array([act.loc[c, y] for y in ys], float)
        ac = (v[-1] - 2 * v[-2] + v[-3]) if len(v) >= 3 else 0.0
        va = v.var() if len(v) >= 2 else 0.0
        return float(ac), float(va)

    # 2-hop CPC neighbours (neighbours-of-neighbours), to test whether brokerage
    # is just higher-order PROXIMITY rather than inventor-mediated bridging
    n2 = {}
    for c in endpoints:
        s = set()
        nbrs = list(adj.get(c, set()))
        if len(nbrs) > nbr_cap:
            nbrs = [nbrs[k] for k in rng.choice(len(nbrs), nbr_cap, replace=False)]
        for cc in nbrs:
            s |= adj.get(cc, set())
        n2[c] = s

    rows = []
    for e, lab in zip(pairs, labels):
        i, j = tuple(e)
        bridge = len(inv_near.get(i, set()) & inv_near.get(j, set()))
        twohop = len(n2.get(i, set()) & n2.get(j, set()))          # higher-order proximity
        pa_inv = len(inv_of.get(i, set())) * len(inv_of.get(j, set()))  # popularity (pref-attach)
        pa_deg = len(adj.get(i, set())) * len(adj.get(j, set()))    # CPC-degree pref-attach
        ai, vi = accel(i); aj, vj = accel(j)
        rows.append((lab, aa(i, j), bridge, ai + aj, vi + vj, twohop, pa_inv, pa_deg))
    return np.array(rows, float)


def auc(y, s):
    try:
        return roc_auc_score(y, s)
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--train-year", type=int, default=2015)
    ap.add_argument("--test-year", type=int, default=2018)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(ROOT / CSV[args.domain])
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    print(f"domain={args.domain} rows={len(df)} train={args.train_year} test={args.test_year}")

    tr = build_cands(df, args.train_year, rng)
    te = build_cands(df, args.test_year, rng)
    if tr is None or te is None:
        print("too few jump positives — abort"); return

    y = te[:, 0]
    print(f"\nJUMP test set: n={len(te)} pos(AA==0 new convergences)={int(y.sum())}")
    print("\nAUC on JUMPS (proximity is ~chance here by construction):")
    print(f"  AA (1-hop proximity)        = {auc(y, te[:,1]):.3f}")
    print(f"  2-hop proximity (CPC NoN)   = {auc(y, te[:,5]):.3f}")
    print(f"  pref-attach (inventor pop)  = {auc(y, te[:,6]):.3f}")
    print(f"  pref-attach (CPC degree)    = {auc(y, te[:,7]):.3f}")
    print(f"  PRECURSOR accel             = {auc(y, te[:,3]):.3f}")
    print(f"  PRECURSOR variance          = {auc(y, te[:,4]):.3f}")
    print(f"  BROKERAGE (bridge invs)     = {auc(y, te[:,2]):.3f}   <-- the test")

    # does brokerage survive CONTROLLING for popularity + 2-hop? LR with all
    # confounds vs LR confounds-only; gain = brokerage's unique contribution.
    conf = [5, 6, 7]              # 2-hop, pa_inv, pa_deg
    def fuse(cols):
        sc = StandardScaler().fit(tr[:, cols]);
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(tr[:, cols]), tr[:, 0])
        return auc(y, clf.predict_proba(sc.transform(te[:, cols]))[:, 1])
    a_conf = fuse(conf)
    a_all = fuse(conf + [2])
    print(f"\n  confounds only (2hop+pop)   = {a_conf:.3f}")
    print(f"  confounds + brokerage       = {a_all:.3f}   (brokerage unique gain {a_all-a_conf:+.3f})")
    best = auc(y, te[:, 2])
    uniq = a_all - a_conf
    print("\n" + "=" * 64)
    print(f"  brokerage AUC={best:.3f}; unique gain over popularity/2-hop confounds={uniq:+.3f}")
    if best >= 0.62 and uniq >= 0.03:
        print("  VERDICT: brokerage predicts jumps AND survives the popularity/2-hop")
        print("           controls => genuine structural-hole signal. The 'right form'.")
    elif best >= 0.60 and uniq >= 0.01:
        print("  VERDICT: brokerage helps but is PARTLY popularity/2-hop. Real but weaker.")
    else:
        print("  VERDICT: brokerage's apparent signal is mostly popularity/2-hop proximity")
        print("           => not genuine brokerage. Jumps stay effectively unpredictable.")
    print("=" * 64)


if __name__ == "__main__":
    main()
