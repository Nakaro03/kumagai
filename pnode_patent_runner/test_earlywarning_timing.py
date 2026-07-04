"""test_earlywarning_timing.py — does FINER (monthly) granularity reveal a WHEN
signal that yearly resolution could not?

The precursor test failed at YEARLY resolution (accel/variance AUC ~0.5), but that
test was underpowered — you cannot estimate variance/autocorrelation TRENDS from
3-4 yearly points. Critical-slowing-down early-warning signals (Scheffer 2009;
Dakos generic EWS) need DENSE series. Here we test, at MONTHLY resolution, whether
EWS predict the TIMING of a technology EMERGENCE (a CPC's activity takeoff):

  For a CPC that emerges at month T (sustained >=3x activity jump), do EWS computed
  on the trailing window ending LEAD months BEFORE T separate it from active-but-
  stable control windows? EWS = Kendall-tau TREND of rolling variance and rolling
  lag-1 autocorrelation (rising = critical slowing down), plus their means.

Crucially we CONTROL for the current activity level + slope (a system about to take
off may already be rising): the question is whether EWS add UNIQUE signal over
level/slope. If yes -> a legitimate Neural-ODE/SDE niche (WHEN-hazard) that needs
fine granularity. If no -> even WHEN is not predictable; the limit stands.

Run:  python pnode_patent_runner/test_earlywarning_timing.py --domain construction
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/nakamuraroi/kumagai")
CSV = {"energy": "data/processed/bipartite_energy.csv",
       "construction": "data/processed/bipartite_construction.csv",
       "computing": "data/processed/bipartite_computing.csv"}


def ews_features(x, H=18):
    """x: trailing monthly activity series. Return (level, slope, tau_var, tau_ac1,
    mean_var, mean_ac1)."""
    W = len(x)
    level = float(x[-12:].mean())
    slope = float(np.polyfit(np.arange(W), x, 1)[0])
    detr = x - uniform_filter1d(x.astype(float), size=6, mode="nearest")
    vs, acs = [], []
    for k in range(0, W - H + 1):
        sub = detr[k:k + H]
        v = sub.var()
        vs.append(v)
        if v > 1e-9 and len(sub) > 2:
            a0, a1 = sub[:-1], sub[1:]
            sd = a0.std() * a1.std()
            acs.append(float(((a0 - a0.mean()) * (a1 - a1.mean())).mean() / sd) if sd > 1e-9 else 0.0)
        else:
            acs.append(0.0)
    vs, acs = np.array(vs), np.array(acs)
    tv = kendalltau(np.arange(len(vs)), vs).statistic if len(vs) > 3 else 0.0
    ta = kendalltau(np.arange(len(acs)), acs).statistic if len(acs) > 3 else 0.0
    return (level, slope, float(np.nan_to_num(tv)), float(np.nan_to_num(ta)),
            float(vs.mean()), float(acs.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--y0", type=int, default=2005)
    ap.add_argument("--y1", type=int, default=2021)
    ap.add_argument("--W", type=int, default=36)
    ap.add_argument("--lead", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(ROOT / CSV[args.domain])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df[(df.ts.dt.year >= args.y0) & (df.ts.dt.year <= args.y1)]
    ym = df.ts.dt.year * 12 + df.ts.dt.month - 1
    m0 = ym.min(); df = df.assign(m=(ym - m0).astype(int))
    M = int(df.m.max()) + 1
    print(f"domain={args.domain} months={M} CPC={df.i.nunique()} W={args.W} lead={args.lead}")

    act = (df.groupby(["i", "m"])["u"].nunique().unstack(fill_value=0)
              .reindex(columns=range(M), fill_value=0))
    codes = act.index.to_list()
    A = act.to_numpy().astype(float)

    W, LEAD = args.W, args.lead
    pos, neg = [], []
    for r in range(A.shape[0]):
        a = A[r]
        # emergence: first T with sustained >=3x jump, prior weak-but-present activity
        Tc = None
        for t in range(W + LEAD, M - 12):
            pre = a[t - 12:t].mean(); fut = a[t:t + 12].mean()
            if fut >= 3.0 and fut >= 3 * max(pre, 1.0) and a[t - W:t].mean() > 0.2:
                Tc = t; break
        if Tc is not None:
            m = Tc - LEAD
            pos.append(ews_features(a[m - W:m]))
        # negatives: active-but-stable windows (no takeoff ahead)
        cand = [t for t in range(W, M - 18)
                if a[t - W:t].mean() > 0.2 and a[t:t + 12].mean() < 2 * max(a[t - 12:t].mean(), 1.0)]
        if cand:
            for t in rng.choice(cand, size=min(2, len(cand)), replace=False):
                neg.append(ews_features(a[t - W:t]))

    pos, neg = np.array(pos), np.array(neg)
    if len(pos) < 20:
        print(f"only {len(pos)} emergence positives — abort"); return
    # balance negatives ~5x
    if len(neg) > 5 * len(pos):
        neg = neg[rng.choice(len(neg), 5 * len(pos), replace=False)]
    X = np.vstack([pos, neg]); y = np.array([1] * len(pos) + [0] * len(neg))
    names = ["level", "slope", "tau_var", "tau_ac1", "mean_var", "mean_ac1"]
    print(f"\nemergence positives={len(pos)}  stable negatives={len(neg)}")
    print("\nSingle-feature AUC (predict emergence imminent in <=lead months):")
    for k, nm in enumerate(names):
        try: a_ = roc_auc_score(y, X[:, k])
        except ValueError: a_ = float("nan")
        print(f"  {nm:10s} = {a_:.3f}")

    def fuse(cols):
        idx = rng.permutation(len(y)); cut = len(y) // 2
        tr, te = idx[:cut], idx[cut:]
        sc = StandardScaler().fit(X[tr][:, cols])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr][:, cols]), y[tr])
        return roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te][:, cols]))[:, 1])
    base = fuse([0, 1])               # level + slope (confound)
    ews = fuse([2, 3, 4, 5])          # EWS only
    full = fuse([0, 1, 2, 3, 4, 5])   # level+slope+EWS
    print("\nFusion AUC (random split):")
    print(f"  level+slope (confound)   = {base:.3f}")
    print(f"  EWS only                 = {ews:.3f}")
    print(f"  level+slope + EWS        = {full:.3f}   (EWS unique gain {full-base:+.3f})")

    print("\n" + "=" * 64)
    uniq = full - base
    if ews >= 0.60 and uniq >= 0.03:
        print(f"  VERDICT: EWS predict emergence TIMING (EWS AUC {ews:.3f}) and add over")
        print(f"           level/slope ({uniq:+.3f}) => a real WHEN-hazard niche for")
        print(f"           continuous-time SDE at fine granularity. NEW positive claim.")
    elif ews >= 0.58 or uniq >= 0.02:
        print(f"  VERDICT: WEAK EWS signal — promising, needs hardening (seeds/domains).")
    else:
        print(f"  VERDICT: EWS add ~nothing over level/slope => even the WHEN is not")
        print(f"           predictable beyond 'it's already rising'. Limit holds.")
    print("=" * 64)


if __name__ == "__main__":
    main()
