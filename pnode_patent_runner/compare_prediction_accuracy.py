"""compare_prediction_accuracy.py — head-to-head comparison of recommender variants.

Reports AUC, P@5/10/20, lift over base rate, ECE (where applicable) for:
  - Random
  - Popularity (momentum only)
  - Adamic-Adar relatedness (the strong baseline we keep finding hard to beat)
  - Content (MiniLM patent-title embedding)
  - Human-aware (PPMI+SVD bipartite embedding)
  - Fusion LR (relatedness + momentum + human + content, hard-neg trained)
  - Fusion LR + isotonic calibration (for ECE)
Single domain firm x CPC-group recommender setting (construction, 3y horizon).
Produces a clean table + bar chart.

Run:  python pnode_patent_runner/compare_prediction_accuracy.py --domain construction
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from diagnose_convergence_signal import ROOT
import recommender_firm as R


def setup(args):
    df = pd.read_csv(ROOT / f"data/processed/bipartite_{args.domain}_firm.csv")
    df["year"] = pd.to_datetime(df["ts"], errors="coerce").dt.year
    df = df.dropna(subset=["year"]); df["year"] = df["year"].astype(int)
    df = df[(df.year >= 2000) & (df.year <= 2020)]
    df["i"] = df["i"].map(R.coarsen); df = df.dropna(subset=["i"]).drop_duplicates(["u", "i", "year"])
    z = np.load(ROOT / f"data/processed/cpc_content_{args.domain}.npz", allow_pickle=True)
    sub = {}
    for c, v in zip(list(z["codes"]), z["emb"]):
        s = R.coarsen(c)
        if s:
            sub.setdefault(s, []).append(v)
    codes = sorted(set(df.i) & set(sub))
    emb = np.array([np.mean(sub[s], 0) for s in codes])
    cidx = {c: k for k, c in enumerate(codes)}; inv_c = {k: c for c, k in cidx.items()}
    df = df[df.i.isin(cidx)]
    return df, emb, codes, cidx, inv_c, len(codes)


def ece(P, Y, nb=8):
    edges = np.quantile(P, np.linspace(0, 1, nb + 1)); edges[-1] += 1e-9
    e = 0.0
    for i in range(nb):
        m = (P >= edges[i]) & (P < edges[i + 1])
        if m.sum() < 10:
            continue
        e += m.mean() * abs(P[m].mean() - Y[m].mean())
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--train-year", type=int, default=2012)
    ap.add_argument("--test-year", type=int, default=2015)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--n-eval", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="compare_prediction_accuracy.png")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    R.LEVEL = "group"

    df, emb, codes, cidx, inv_c, n_c = setup(args)
    print(f"firms={df.u.nunique()} CPC-groups={n_c}")
    sc, clf = R.train_lr(df, args.train_year, cidx, emb, n_c, args.horizon, rng, cols=[0, 1, 2, 3])
    w = R.build_world(df, args.test_year, cidx, emb, n_c, args.horizon)
    invs = R.actors(w, cidx); rng.shuffle(invs); invs = invs[:args.n_eval]
    sp = len(invs) // 2

    # collect per-firm candidate features + label
    cs, cy = [], []   # calibration split (for isotonic on fusion LR)
    eval_rows = []
    for n, u in enumerate(invs):
        X, owned, Su = R.actor_scores(u, w, emb, cidx, n_c)
        new = {cidx[c] for c in (w["nextf"][u] - w["prior"][u]) if c in cidx and w["have"][cidx[c]]}
        if len(Su) < 2 or not new:
            continue
        cand = np.array([t for t in range(n_c) if t not in owned])
        feats = X[cand][:, [0, 1, 2, 3]]
        lab = np.array([1 if t in new else 0 for t in cand])
        fus_raw = clf.decision_function(sc.transform(feats))
        if n < sp:
            cs.append(fus_raw); cy.append(lab)
        else:
            eval_rows.append((u, cand, feats, lab, fus_raw, new))
    iso = IsotonicRegression(out_of_bounds="clip").fit(np.concatenate(cs), np.concatenate(cy))

    base = float(np.mean([y.mean() for u, c, f, y, r, n in eval_rows]))
    print(f"base rate ≈ {base:.4f}, eval firms = {len(eval_rows)}")

    # methods: each returns score per candidate (higher = better)
    rng_s = np.random.default_rng(args.seed + 1)
    methods = {
        "Random": lambda f, r: rng_s.random(len(f)),
        "Popularity (momentum)": lambda f, r: f[:, 1],
        "Adamic-Adar (relatedness)": lambda f, r: f[:, 0],
        "Content (MiniLM)": lambda f, r: f[:, 3],
        "Human-aware (SVD)": lambda f, r: f[:, 2],
        "Fusion LR (hard-neg)": lambda f, r: r,
        "Fusion + isotonic cal.": lambda f, r: iso.predict(r),
    }

    ks = [5, 10, 20]
    results = {}
    for nm, fn in methods.items():
        all_s, all_y = [], []
        prec = {k: [] for k in ks}
        for u, cand, feats, lab, raw, new in eval_rows:
            s = fn(feats, raw)
            order = np.argsort(-s)
            for k in ks:
                prec[k].append(np.mean([1 if cand[i] in new else 0 for i in order[:k]]))
            all_s.append(s); all_y.append(lab)
        S = np.concatenate(all_s); Y = np.concatenate(all_y)
        try:
            auc = roc_auc_score(Y, S)
        except ValueError:
            auc = float("nan")
        e = ece(S, Y) if nm == "Fusion + isotonic cal." else None
        results[nm] = {"AUC": auc, **{f"P@{k}": np.mean(prec[k]) for k in ks},
                       "Lift@5": np.mean(prec[5]) / base, "ECE": e}

    # ---- print table
    print("\n" + "=" * 90)
    print(f"{'method':30s}  {'AUC':>6}  {'P@5':>6}  {'P@10':>6}  {'P@20':>6}  {'Lift@5':>7}  {'ECE':>7}")
    print("-" * 90)
    for nm, r in results.items():
        ece_s = f"{r['ECE']:.3f}" if r["ECE"] is not None else "  —  "
        print(f"{nm:30s}  {r['AUC']:6.3f}  {r['P@5']:6.3f}  {r['P@10']:6.3f}  "
              f"{r['P@20']:6.3f}  {r['Lift@5']:7.1f}×  {ece_s:>7s}")
    print("=" * 90)
    print(f"base rate = {base:.4f}  (random P@anything ≈ base)")

    # ---- bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = list(results.keys())
    colors = ["lightgrey", "tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:red", "darkred"]
    for ax, metric, ttl in zip(axes, ["AUC", "P@5", "Lift@5"],
                                ["AUC (higher = better)",
                                 "Precision@5 (top-5 hit fraction)",
                                 "Lift@5 = P@5 / base rate"]):
        vals = [results[n][metric] for n in names]
        bars = ax.barh(names, vals, color=colors, edgecolor="k")
        for b, v in zip(bars, vals):
            label = f"{v:.3f}" if metric != "Lift@5" else f"{v:.1f}×"
            ax.text(v, b.get_y() + b.get_height() / 2, " " + label, va="center", fontsize=9)
        ax.set_title(ttl, fontsize=11)
        ax.invert_yaxis()
        if metric == "AUC":
            ax.axvline(0.5, color="red", linestyle=":", alpha=0.6, label="chance (AUC=0.5)")
            ax.legend(fontsize=8)
        elif metric == "P@5":
            ax.axvline(base, color="red", linestyle=":", alpha=0.6, label=f"base rate {base:.3f}")
            ax.legend(fontsize=8)
        elif metric == "Lift@5":
            ax.axvline(1.0, color="red", linestyle=":", alpha=0.6, label="no lift (=1×)")
            ax.legend(fontsize=8)

    plt.suptitle(f"Prediction accuracy comparison — {args.domain} firm × CPC-group, "
                 f"train {args.train_year} → test {args.test_year} (+{args.horizon}y), "
                 f"base rate {base:.3f}", fontsize=12, y=1.02)
    plt.tight_layout()
    out = ROOT / "pnode_patent_runner" / args.out
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
