"""recommender_inventor.py — tech-expansion recommender + calibration (v2).

Productizes the entry engine into a RANKING recommender with honest confidence.
For each actor (inventor; firm later) rank candidate CPCs (not in portfolio) by a
fusion of:
  relatedness : sum_{j in portfolio} AdamicAdar(t,j)   (inventor-specific)
  momentum    : recent activity level of t              (global, WHEN signal)
  human-aware : cos(actor, t) in inventor x CPC bipartite embedding
  content     : cos(content[t], portfolio content centroid)

v2 fixes (vs v1 where naive fusion HURT precision@k and calibration was broken):
  (FIX 1) train with HARD negatives = inventor-specific plausible non-entries
          (relatedness>0, co-occur with portfolio) instead of random negatives,
          so the model must discriminate within the relevant set, not rank
          popular-vs-random (which let global popularity dominate top-k).
  (FIX 2) ISOTONIC calibration on a held-out inventor split, mapping raw scores
          to probabilities at the TRUE deployment base rate -> usable confidence.

Reports: precision@k (full vs relatedness-only), ranking AUC, ECE before/after
calibration, and reliability buckets (the honest confidence flags).

Run:  python pnode_patent_runner/recommender_inventor.py --domain construction
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from diagnose_convergence_signal import ROOT, CSV
from diagnose_entry_humanaware import cooc_graph, bipartite_embed


def build_world(df, Y, cidx, emb, n_c):
    cooc, deg = cooc_graph(df, Y)
    rows, cols, vals = [], [], []
    for a, nbrs in cooc.items():
        if a not in cidx:
            continue
        ia = cidx[a]
        for b, w in nbrs.items():
            if b in cidx:
                rows.append(ia); cols.append(cidx[b]); vals.append(w / np.log(deg[b] + 2))
    W = csr_matrix((vals, (rows, cols)), shape=(n_c, n_c))
    momentum = np.zeros(n_c)
    for c, n in Counter(df[df.year == Y]["i"]).items():
        if c in cidx:
            momentum[cidx[c]] = n
    uidx, Uemb, Cemb = bipartite_embed(df, Y, cidx)
    prior = df[df.year <= Y].groupby("u")["i"].agg(set)
    nextf = df[df.year == Y + 1].groupby("u")["i"].agg(set)
    have = (emb != 0).any(1)
    return dict(W=W, momentum=momentum, uidx=uidx, Uemb=Uemb, Cemb=Cemb,
                prior=prior, nextf=nextf, have=have)


def actor_scores(u, w, emb, cidx, n_c):
    Su = [c for c in w["prior"][u] if c in cidx and w["have"][cidx[c]]]
    cols = [cidx[c] for c in Su]
    rel = np.asarray(w["W"][:, cols].sum(1)).ravel() if cols else np.zeros(n_c)
    human = w["Uemb"][w["uidx"][u]] @ w["Cemb"].T if u in w["uidx"] else np.zeros(n_c)
    cen = emb[cols].mean(0); cen /= (np.linalg.norm(cen) + 1e-9)
    content = emb @ cen
    X = np.column_stack([rel, w["momentum"], human, content])
    return X, set(cols), Su


def actors_with_entries(w, cidx):
    return [u for u in w["prior"].index if u in w["nextf"].index
            and sum(1 for c in w["prior"][u] if c in cidx) >= 3]


def train_lr(df, Yt, cidx, emb, n_c, rng, cols, max_inv=1200, neg_ratio=8):
    """FIX 1: train on HARD negatives (relatedness>0, plausible) per inventor."""
    w = build_world(df, Yt, cidx, emb, n_c)
    invs = actors_with_entries(w, cidx); rng.shuffle(invs)
    Xs, ys = [], []
    for u in invs[:max_inv]:
        X, owned, Su = actor_scores(u, w, emb, cidx, n_c)
        if len(Su) < 3:
            continue
        new = [cidx[c] for c in (w["nextf"][u] - w["prior"][u]) if c in cidx and w["have"][cidx[c]]]
        if not new:
            continue
        newset = set(new)
        hard = [t for t in range(n_c) if t not in owned and t not in newset and X[t, 0] > 0]
        if len(hard) < 2:
            continue
        negs = list(rng.choice(hard, min(len(new) * neg_ratio, len(hard)), replace=False))
        idx = new + negs
        Xs.append(X[idx][:, cols]); ys.append(np.array([1] * len(new) + [0] * len(negs)))
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)
    return sc, clf


def collect(df, Yte, cidx, emb, n_c, rng, sc, clf, cols, max_inv, ks=(5, 10, 20)):
    """Return per-inventor precision@k (full & relatedness-only) and (rawscore,label)
    over ALL candidates, with an inventor split for calibration."""
    w = build_world(df, Yte, cidx, emb, n_c)
    invs = actors_with_entries(w, cidx); rng.shuffle(invs); invs = invs[:max_inv]
    split = len(invs) // 2
    calib_s, calib_y = [], []
    eval_s, eval_y = [], []
    prec = {k: [] for k in ks}; prec_rel = {k: [] for k in ks}
    for n, u in enumerate(invs):
        X, owned, Su = actor_scores(u, w, emb, cidx, n_c)
        if len(Su) < 3:
            continue
        new = {cidx[c] for c in (w["nextf"][u] - w["prior"][u]) if c in cidx and w["have"][cidx[c]]}
        if not new:
            continue
        cand = np.array([t for t in range(n_c) if t not in owned])
        raw = clf.decision_function(sc.transform(X[cand][:, cols]))
        lab = np.array([1 if t in new else 0 for t in cand])
        if n < split:
            calib_s.append(raw); calib_y.append(lab)
        else:
            eval_s.append(raw); eval_y.append(lab)
            order = cand[np.argsort(-raw)]
            rel_order = cand[np.argsort(-X[cand][:, 0])]
            for k in ks:
                prec[k].append(np.mean([1 if t in new else 0 for t in order[:k]]))
                prec_rel[k].append(np.mean([1 if t in new else 0 for t in rel_order[:k]]))
    return (np.concatenate(calib_s), np.concatenate(calib_y),
            np.concatenate(eval_s), np.concatenate(eval_y), prec, prec_rel)


def ece_buckets(P, Y, nb=10):
    edges = np.quantile(P, np.linspace(0, 1, nb + 1)); edges[-1] += 1e-9
    rows, ece = [], 0.0
    for i in range(nb):
        m = (P >= edges[i]) & (P < edges[i + 1])
        if m.sum() < 10:
            continue
        conf, acc, frac = P[m].mean(), Y[m].mean(), m.mean()
        ece += frac * abs(conf - acc); rows.append((conf, acc, int(m.sum())))
    return rows, ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--train-year", type=int, default=2015)
    ap.add_argument("--test-year", type=int, default=2018)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(ROOT / CSV[args.domain]); df["year"] = pd.to_datetime(df["ts"]).dt.year
    z = np.load(ROOT / f"data/processed/cpc_content_{args.domain}.npz", allow_pickle=True)
    codes = list(z["codes"]); emb = z["emb"]; cidx = {c: k for k, c in enumerate(codes)}
    n_c = len(codes)
    fnames = ["relatedness", "momentum", "human-aware", "content"]
    print(f"domain={args.domain} CPC={n_c} train={args.train_year} test={args.test_year}  "
          f"[v2: hard-neg training + isotonic calibration]")

    sc, clf = train_lr(df, args.train_year, cidx, emb, n_c, rng, cols=[0, 1, 2, 3])
    print("\nFusion weights (standardized):",
          {fnames[i]: round(float(clf.coef_[0][i]), 2) for i in range(4)})

    cs, cy, es, ey, prec, prec_rel = collect(df, args.test_year, cidx, emb, n_c, rng,
                                             sc, clf, cols=[0, 1, 2, 3], max_inv=800)

    print("\n(2)+(4) precision@k on held-out test (full fusion vs relatedness-only):")
    for k in sorted(prec):
        f, r = np.mean(prec[k]), np.mean(prec_rel[k])
        print(f"  P@{k:<3d} full={f:.3f}  relatedness-only={r:.3f}  uplift={f - r:+.3f}")
    print(f"\n  ranking AUC (full) = {roc_auc_score(ey, es):.3f}  (base rate {ey.mean():.4f})")

    # FIX 2: isotonic calibration fit on calib split, applied to eval split
    iso = IsotonicRegression(out_of_bounds="clip").fit(cs, cy)
    raw_rows, raw_ece = ece_buckets(1 / (1 + np.exp(-es)), ey)   # uncalibrated sigmoid
    cal_rows, cal_ece = ece_buckets(iso.predict(es), ey)
    print(f"\n(3) Calibration ECE: uncalibrated {raw_ece:.3f} -> isotonic {cal_ece:.3f}")
    print("  isotonic reliability buckets (predicted vs realized hit-rate):")
    print(f"  {'pred':>7} {'actual':>7} {'n':>8}")
    for conf, acc, n in cal_rows:
        print(f"  {conf:7.4f} {acc:7.4f} {n:8d}")
    print("\n  -> top bucket realized rate = the 'high-confidence' flag; calibrated so")
    print("     predicted prob ~ realized rate => honest confidence for decisions.")


if __name__ == "__main__":
    main()
