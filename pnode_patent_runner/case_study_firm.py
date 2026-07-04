"""case_study_firm.py — qualitative firm cases for the tech-expansion recommender.

For real (named) firms, show: current CPC-group portfolio -> top-K recommended
technology groups (with isotonic-calibrated confidence) -> which ones the firm
ACTUALLY entered within the horizon (hit/miss). Makes the recommender concrete
for a professor / decision-maker beyond the aggregate P@k.

Run:  python pnode_patent_runner/case_study_firm.py --domain construction
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from diagnose_convergence_signal import ROOT
import recommender_firm as R

# rough CPC-class glossary (construction-relevant) for readability
GLOSS = {
    "E01": "roads/railways/bridges", "E02": "hydraulic eng./foundations/earthworks",
    "E03": "water supply/sewerage", "E04": "building", "E05": "locks/fittings",
    "E06": "doors/windows", "E21": "earth/rock drilling, mining",
    "B28": "working cement/clay", "B66": "hoisting/lifting", "C04": "cements/concrete",
    "B65": "conveying/packaging", "F16": "machine elements", "B60": "vehicles",
    "G01": "measuring/testing", "H02": "electric power", "B23": "machine tools",
    "B32": "layered products", "C09": "coatings/adhesives", "E03F": "sewerage",
}


def label(g):
    return GLOSS.get(g[:3], "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--train-year", type=int, default=2012)
    ap.add_argument("--base-year", type=int, default=2015)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--n-firms", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    R.LEVEL = "group"

    df = pd.read_csv(ROOT / f"data/processed/bipartite_{args.domain}_firm.csv")
    df["year"] = pd.to_datetime(df["ts"], errors="coerce").dt.year
    df = df.dropna(subset=["year"]); df["year"] = df["year"].astype(int)
    df = df[(df.year >= 2000) & (df.year <= 2020)]
    df["i"] = df["i"].map(R.coarsen)
    df = df.dropna(subset=["i"]).drop_duplicates(["u", "i", "year"])

    z = np.load(ROOT / f"data/processed/cpc_content_{args.domain}.npz", allow_pickle=True)
    sub = {}
    for c, v in zip(list(z["codes"]), z["emb"]):
        s = R.coarsen(c)
        if s:
            sub.setdefault(s, []).append(v)
    codes = sorted(set(df.i) & set(sub))
    emb = np.array([np.mean(sub[s], 0) for s in codes])
    cidx = {c: k for k, c in enumerate(codes)}; n_c = len(codes); inv_c = {k: c for c, k in cidx.items()}
    df = df[df.i.isin(cidx)]
    names = pd.read_csv(ROOT / f"data/processed/{args.domain}_firm_names.csv").drop_duplicates("u").set_index("u")["org"].to_dict()

    sc, clf = R.train_lr(df, args.train_year, cidx, emb, n_c, args.horizon, rng, cols=[0, 1, 2, 3])
    w = R.build_world(df, args.base_year, cidx, emb, n_c, args.horizon)

    # fit isotonic calibration on a sample of firms
    invs = R.actors(w, cidx); rng.shuffle(invs)
    cs, cy = [], []
    for u in invs[:600]:
        X, owned, Su = R.actor_scores(u, w, emb, cidx, n_c)
        new = {cidx[c] for c in (w["nextf"][u] - w["prior"][u]) if c in cidx and w["have"][cidx[c]]}
        if len(Su) < 2 or not new:
            continue
        cand = np.array([t for t in range(n_c) if t not in owned])
        cs.append(clf.decision_function(sc.transform(X[cand][:, [0, 1, 2, 3]])))
        cy.append(np.array([1 if t in new else 0 for t in cand]))
    iso = IsotonicRegression(out_of_bounds="clip").fit(np.concatenate(cs), np.concatenate(cy))

    # pick illustrative named firms: portfolio 5-25 groups, >=3 future entries, has name
    elig = []
    for u in invs:
        if u not in names or not isinstance(names[u], str) or len(names[u]) < 3:
            continue
        port = [c for c in w["prior"][u] if c in cidx]
        new = [c for c in (w["nextf"][u] - w["prior"][u]) if c in cidx and w["have"][cidx[c]]]
        if 5 <= len(port) <= 25 and len(new) >= 3:
            elig.append(u)
    rng.shuffle(elig)
    print(f"=== {args.domain} firm tech-expansion case studies "
          f"(base {args.base_year}, enter within {args.horizon}y) ===")
    print(f"(eligible named firms: {len(elig)}; showing {args.n_firms})\n")

    for u in elig[:args.n_firms]:
        X, owned, Su = R.actor_scores(u, w, emb, cidx, n_c)
        new = {cidx[c] for c in (w["nextf"][u] - w["prior"][u]) if c in cidx and w["have"][cidx[c]]}
        cand = np.array([t for t in range(n_c) if t not in owned])
        conf = iso.predict(clf.decision_function(sc.transform(X[cand][:, [0, 1, 2, 3]])))
        order = cand[np.argsort(-conf)]
        confm = {t: c for t, c in zip(cand, conf)}
        port = sorted([inv_c[cidx[c]] for c in Su])
        print(f"■ {names[u][:60]}")
        pstr = ", ".join(f"{g}({label(g)})" if label(g) else g for g in port[:8])
        print(f"  portfolio({len(Su)}): {pstr}{' ...' if len(Su) > 8 else ''}")
        hits = 0
        print(f"  top-{args.topk} recommendations (✓=actually entered within {args.horizon}y):")
        for r, t in enumerate(order[:args.topk], 1):
            g = inv_c[t]; hit = "✓" if t in new else " "
            if t in new:
                hits += 1
            lab = f" — {label(g)}" if label(g) else ""
            print(f"    {hit} {r:2d}. {g:<10} conf={confm[t]*100:4.1f}%{lab}")
        print(f"  => hits in top-{args.topk}: {hits}/{args.topk}  "
              f"(actual new entries total: {len(new)})\n")


if __name__ == "__main__":
    main()
