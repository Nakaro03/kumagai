"""recommender_firm.py — FIRM-level technology-expansion recommender (decision-realistic).

Upgrades the inventor prototype to the decision unit (FIRM = assignee) and to a
decision-realistic granularity/horizon:
  - unit       : assignee_id (firm), from bipartite_{domain}_firm.csv
  - technology : CPC SUBCLASS (e.g. E02F) not subgroup -> ~hundreds of candidates
  - horizon    : enter within the next H years (not exactly +1)
Same engine as recommender_inventor.py (relatedness x momentum x human x content,
hard-negative training, isotonic calibration), reporting precision@k (full vs
relatedness-only) + calibrated confidence buckets.

Run:  python pnode_patent_runner/recommender_firm.py --domain construction
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from diagnose_convergence_signal import ROOT
from diagnose_entry_humanaware import cooc_graph, bipartite_embed

SUBCLASS = re.compile(r"^([A-HY]\d{2}[A-Z])")


def coarsen(c, level="group"):
    c = str(c)
    if level == "subgroup":
        return c
    if level == "group":            # part before '/', e.g. E02F5/106 -> E02F5
        return c.split("/")[0] or None
    m = SUBCLASS.match(c); return m.group(1) if m else None   # subclass e.g. E02F


def subclass(c):
    return coarsen(c, LEVEL)


LEVEL = "group"


def build_world(df, Y, cidx, emb, n_c, H):
    cooc, deg = cooc_graph(df, Y)
    rows, cols, vals = [], [], []
    for a, nbrs in cooc.items():
        if a not in cidx:
            continue
        for b, w in nbrs.items():
            if b in cidx:
                rows.append(cidx[a]); cols.append(cidx[b]); vals.append(w / np.log(deg[b] + 2))
    W = csr_matrix((vals, (rows, cols)), shape=(n_c, n_c))
    momentum = np.zeros(n_c)
    for c, n in Counter(df[df.year == Y]["i"]).items():
        if c in cidx:
            momentum[cidx[c]] = n
    uidx, Uemb, Cemb = bipartite_embed(df, Y, cidx)
    prior = df[df.year <= Y].groupby("u")["i"].agg(set)
    nextf = df[(df.year > Y) & (df.year <= Y + H)].groupby("u")["i"].agg(set)
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
    return np.column_stack([rel, w["momentum"], human, content]), set(cols), Su


def actors(w, cidx, minp=2):
    return [u for u in w["prior"].index if u in w["nextf"].index
            and sum(1 for c in w["prior"][u] if c in cidx) >= minp]


def train_lr(df, Yt, cidx, emb, n_c, H, rng, cols, max_inv=2000, neg_ratio=8):
    w = build_world(df, Yt, cidx, emb, n_c, H)
    invs = actors(w, cidx); rng.shuffle(invs)
    Xs, ys = [], []
    for u in invs[:max_inv]:
        X, owned, Su = actor_scores(u, w, emb, cidx, n_c)
        if len(Su) < 2:
            continue
        new = [cidx[c] for c in (w["nextf"][u] - w["prior"][u]) if c in cidx and w["have"][cidx[c]]]
        if not new:
            continue
        ns = set(new)
        hard = [t for t in range(n_c) if t not in owned and t not in ns and X[t, 0] > 0]
        if len(hard) < 2:
            continue
        negs = list(rng.choice(hard, min(len(new) * neg_ratio, len(hard)), replace=False))
        idx = new + negs
        Xs.append(X[idx][:, cols]); ys.append(np.array([1] * len(new) + [0] * len(negs)))
    Xtr = np.vstack(Xs); ytr = np.concatenate(ys)
    sc = StandardScaler().fit(Xtr)
    return sc, LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)


def collect(df, Yte, cidx, emb, n_c, H, rng, sc, clf, cols, max_inv, ks=(5, 10, 20)):
    w = build_world(df, Yte, cidx, emb, n_c, H)
    invs = actors(w, cidx); rng.shuffle(invs); invs = invs[:max_inv]
    sp = len(invs) // 2
    cs, cy, es, ey = [], [], [], []
    prec = {k: [] for k in ks}; prec_rel = {k: [] for k in ks}
    for n, u in enumerate(invs):
        X, owned, Su = actor_scores(u, w, emb, cidx, n_c)
        if len(Su) < 2:
            continue
        new = {cidx[c] for c in (w["nextf"][u] - w["prior"][u]) if c in cidx and w["have"][cidx[c]]}
        if not new:
            continue
        cand = np.array([t for t in range(n_c) if t not in owned])
        raw = clf.decision_function(sc.transform(X[cand][:, cols]))
        lab = np.array([1 if t in new else 0 for t in cand])
        if n < sp:
            cs.append(raw); cy.append(lab)
        else:
            es.append(raw); ey.append(lab)
            order = cand[np.argsort(-raw)]; rel_order = cand[np.argsort(-X[cand][:, 0])]
            for k in ks:
                prec[k].append(np.mean([1 if t in new else 0 for t in order[:k]]))
                prec_rel[k].append(np.mean([1 if t in new else 0 for t in rel_order[:k]]))
    return (np.concatenate(cs), np.concatenate(cy), np.concatenate(es), np.concatenate(ey),
            prec, prec_rel)


def ece_buckets(P, Y, nb=8):
    edges = np.quantile(P, np.linspace(0, 1, nb + 1)); edges[-1] += 1e-9
    rows, ece = [], 0.0
    for i in range(nb):
        m = (P >= edges[i]) & (P < edges[i + 1])
        if m.sum() < 10:
            continue
        ece += m.mean() * abs(P[m].mean() - Y[m].mean()); rows.append((P[m].mean(), Y[m].mean(), int(m.sum())))
    return rows, ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--train-year", type=int, default=2012)
    ap.add_argument("--test-year", type=int, default=2015)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--y0", type=int, default=2000)
    ap.add_argument("--y1", type=int, default=2020)
    ap.add_argument("--level", default="group", choices=["subgroup", "group", "subclass"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    global LEVEL; LEVEL = args.level

    df = pd.read_csv(ROOT / f"data/processed/bipartite_{args.domain}_firm.csv")
    df["year"] = pd.to_datetime(df["ts"], errors="coerce").dt.year
    df = df.dropna(subset=["year"]); df["year"] = df["year"].astype(int)
    df = df[(df.year >= args.y0) & (df.year <= args.y1)]
    df["i"] = df["i"].map(subclass)
    df = df.dropna(subset=["i"]).drop_duplicates(["u", "i", "year"])

    # subclass content = mean of subgroup content vectors
    z = np.load(ROOT / f"data/processed/cpc_content_{args.domain}.npz", allow_pickle=True)
    sub_emb = {}
    for c, v in zip(list(z["codes"]), z["emb"]):
        s = subclass(c)
        if s:
            sub_emb.setdefault(s, []).append(v)
    codes = sorted(set(df.i) & set(sub_emb))
    emb = np.array([np.mean(sub_emb[s], 0) for s in codes])
    cidx = {c: k for k, c in enumerate(codes)}; n_c = len(codes)
    df = df[df.i.isin(cidx)]
    print(f"domain={args.domain} FIRM-level subclass  firms={df.u.nunique()} "
          f"subclasses={n_c} train={args.train_year} test={args.test_year} horizon={args.horizon}y")

    sc, clf = train_lr(df, args.train_year, cidx, emb, n_c, args.horizon, rng, cols=[0, 1, 2, 3])
    fn = ["relatedness", "momentum", "human", "content"]
    print("\nfusion weights:", {fn[i]: round(float(clf.coef_[0][i]), 2) for i in range(4)})

    cs, cy, es, ey, prec, prec_rel = collect(df, args.test_year, cidx, emb, n_c, args.horizon,
                                             rng, sc, clf, cols=[0, 1, 2, 3], max_inv=1500)
    print(f"\nprecision@k (firm enters subclass within {args.horizon}y) — full vs relatedness-only:")
    for k in sorted(prec):
        f, r = np.mean(prec[k]), np.mean(prec_rel[k])
        print(f"  P@{k:<3d} full={f:.3f}  relatedness-only={r:.3f}  uplift={f - r:+.3f}")
    print(f"  ranking AUC (full) = {roc_auc_score(ey, es):.3f}  (base rate {ey.mean():.4f})")

    iso = IsotonicRegression(out_of_bounds="clip").fit(cs, cy)
    _, raw_ece = ece_buckets(1 / (1 + np.exp(-es)), ey)
    rows, cal_ece = ece_buckets(iso.predict(es), ey)
    print(f"\ncalibration ECE: uncalibrated {raw_ece:.3f} -> isotonic {cal_ece:.3f}")
    print(f"  {'pred':>7} {'actual':>7} {'n':>8}")
    for c, a, n in rows:
        print(f"  {c:7.3f} {a:7.3f} {n:8d}")


if __name__ == "__main__":
    main()
