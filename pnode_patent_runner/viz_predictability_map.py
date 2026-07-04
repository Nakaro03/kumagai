"""viz_predictability_map.py — Pattern-3 visualization: predictability boundary
on a 2D latent landscape of CPC technology groups.

We project the joint firm x CPC bipartite embedding (PPMI-SVD) to 2D via UMAP,
then for each CPC overlay:
  (A) mean CALIBRATED CONFIDENCE the recommender assigned to it across firms
  (B) the REALIZED hit rate (when this CPC was in a firm's top-K, did the firm
      actually enter within the horizon)
  (C) calibration RESIDUAL = (A) - (B)  (signed; ≈0 = honest)

Pattern-3 idea: where (B) collapses to the base rate, predictions are
proximity-bound (speculative); where (B) stays high, predictions are reliable.
The MAP itself shows where the boundary lies — the proximity-bound limit
visualized as a feature, not a footnote.

We also overlay a sample firm: portfolio (★) and top recommendation arrows,
colored by hit/miss.

Run:  python pnode_patent_runner/viz_predictability_map.py --domain construction
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.isotonic import IsotonicRegression

from diagnose_convergence_signal import ROOT
import recommender_firm as R

GLOSS = {
    "E01": "roads/bridges", "E02": "hydraulic/foundations",
    "E03": "water/sewerage", "E04": "building", "E05": "locks/fittings",
    "E06": "doors/windows", "E21": "earth/rock drilling",
    "B28": "cement/clay", "B66": "hoisting", "C04": "cements/concrete",
    "F16": "machine elements", "B23": "machine tools",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--train-year", type=int, default=2012)
    ap.add_argument("--test-year", type=int, default=2015)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--min-rec", type=int, default=20)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--example-firm-portfolio-min", type=int, default=6)
    ap.add_argument("--out", default="viz_predictability_map.png")
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
    print(f"firms={df.u.nunique()} CPC-groups={n_c}")

    sc, clf = R.train_lr(df, args.train_year, cidx, emb, n_c, args.horizon, rng, cols=[0, 1, 2, 3])
    w = R.build_world(df, args.test_year, cidx, emb, n_c, args.horizon)

    # bipartite CPC embeddings (rich joint structure) -> UMAP 2D
    print("UMAP to 2D ...")
    Cemb = w["Cemb"]
    xy = umap.UMAP(n_neighbors=15, min_dist=0.30, n_components=2,
                   random_state=args.seed).fit_transform(Cemb)

    # fit isotonic on first half of eval firms, gather per-CPC stats on second half
    invs = R.actors(w, cidx); rng.shuffle(invs); invs = invs[:args.n_eval]
    sp = len(invs) // 2
    cs, cy = [], []
    eval_data = []
    for n, u in enumerate(invs):
        X, owned, Su = R.actor_scores(u, w, emb, cidx, n_c)
        new = {cidx[c] for c in (w["nextf"][u] - w["prior"][u]) if c in cidx and w["have"][cidx[c]]}
        if len(Su) < 2 or not new:
            continue
        cand = np.array([t for t in range(n_c) if t not in owned])
        raw = clf.decision_function(sc.transform(X[cand][:, [0, 1, 2, 3]]))
        lab = np.array([1 if t in new else 0 for t in cand])
        if n < sp:
            cs.append(raw); cy.append(lab)
        else:
            eval_data.append((u, X, owned, Su, new, cand, raw, lab))
    iso = IsotonicRegression(out_of_bounds="clip").fit(np.concatenate(cs), np.concatenate(cy))

    stats = defaultdict(lambda: {"n_rec": 0, "sum_conf": 0, "n_hit": 0, "activity": 0})
    for u, X, owned, Su, new, cand, raw, lab in eval_data:
        cal = iso.predict(raw)
        order = np.argsort(-cal)[:args.topk]
        for j in order:
            t = cand[j]
            st = stats[t]
            st["n_rec"] += 1; st["sum_conf"] += cal[j]
            if t in new:
                st["n_hit"] += 1
    for t in range(n_c):
        stats[t]["activity"] = float(w["momentum"][t])

    base = float(np.mean([y.mean() for y in cy]))
    print(f"base rate (cy) ≈ {base:.4f}")

    rec = []
    for t in range(n_c):
        s = stats[t]
        if s["n_rec"] >= args.min_rec:
            rec.append((t, s["n_rec"], s["sum_conf"] / s["n_rec"],
                        s["n_hit"] / s["n_rec"], s["activity"]))
    print(f"CPCs with >= {args.min_rec} recommendations: {len(rec)}")
    rec_arr = np.array([(t, n, c, h, a) for t, n, c, h, a in rec])
    tidx = rec_arr[:, 0].astype(int)
    nrec = rec_arr[:, 1]; conf = rec_arr[:, 2]; hit = rec_arr[:, 3]; act = rec_arr[:, 4]
    pts = xy[tidx]
    grey = xy[[t for t in range(n_c) if stats[t]["n_rec"] < args.min_rec]]

    # pick an example focused firm: portfolio >= example-firm-portfolio-min,
    # tightly clustered (small std of portfolio xy)
    best_u, best_var = None, np.inf
    names = {}
    nm_path = ROOT / f"data/processed/{args.domain}_firm_names.csv"
    if nm_path.exists():
        names = pd.read_csv(nm_path).drop_duplicates("u").set_index("u")["org"].to_dict()
    for u, X, owned, Su, new, cand, raw, lab in eval_data:
        if len(Su) < args.example_firm_portfolio_min:
            continue
        port_t = [cidx[c] for c in Su if c in cidx]
        var = xy[port_t].std(0).sum()
        if var < best_var and isinstance(names.get(u, ""), str) and len(names.get(u, "")) > 3:
            best_var = var; best_u = u
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    cmaps = ["viridis", "RdYlGn", "coolwarm"]
    titles = ["(A) mean calibrated CONFIDENCE", "(B) realized HIT RATE (predictability)",
              "(C) calibration RESIDUAL = conf − hit (≈0 honest)"]
    vals = [conf, hit, conf - hit]
    vlims = [(0, max(0.06, conf.max())), (0, max(0.06, hit.max())),
             (-max(abs((conf - hit)).max(), 0.05), max(abs((conf - hit)).max(), 0.05))]
    for ax, cmap, title, v, (lo, hi) in zip(axes, cmaps, titles, vals, vlims):
        ax.scatter(grey[:, 0], grey[:, 1], s=8, c="lightgrey", alpha=0.4,
                   label=f"rarely recommended (n<{args.min_rec})")
        sc_ = ax.scatter(pts[:, 0], pts[:, 1], c=v, cmap=cmap, vmin=lo, vmax=hi,
                         s=8 + 30 * np.sqrt(np.clip(act / (act.max() + 1e-9), 0, 1)),
                         edgecolors="k", linewidths=0.3)
        fig.colorbar(sc_, ax=ax, shrink=0.85)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        # labels for high-value points (top 8 by metric magnitude)
        order = np.argsort(-np.abs(v - (base if cmap != "coolwarm" else 0)))[:8]
        for k in order:
            g = inv_c[int(tidx[k])]
            ax.annotate(g, (pts[k, 0], pts[k, 1]), fontsize=7, alpha=0.8)

    # overlay example firm on panel B
    if best_u is not None:
        port_t = [cidx[c] for c in w["prior"][best_u] if c in cidx]
        port_xy = xy[port_t]
        new_set = {cidx[c] for c in (w["nextf"][best_u] - w["prior"][best_u])
                   if c in cidx and w["have"][cidx[c]]}
        X, owned, Su = R.actor_scores(best_u, w, emb, cidx, n_c)
        cand = np.array([t for t in range(n_c) if t not in owned])
        cal = iso.predict(clf.decision_function(sc.transform(X[cand][:, [0, 1, 2, 3]])))
        top = cand[np.argsort(-cal)[:8]]
        center = port_xy.mean(0)
        for t in top:
            tx, ty = xy[t]
            color = "darkgreen" if t in new_set else "darkred"
            axes[1].annotate("", xy=(tx, ty), xytext=center,
                             arrowprops=dict(arrowstyle="->", color=color, alpha=0.7, lw=1.3))
        axes[1].scatter(port_xy[:, 0], port_xy[:, 1], marker="*", c="black", s=140, zorder=5,
                        label=f"★ portfolio: {names.get(best_u, '?')[:30]}")
        axes[1].legend(loc="best", fontsize=8)

    plt.suptitle(f"Predictability map ({args.domain}, FIRM x CPC-group, train {args.train_year}, "
                 f"horizon {args.horizon}y, base rate {base:.4f})", fontsize=12)
    plt.tight_layout()
    out_path = ROOT / "pnode_patent_runner" / args.out
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved -> {out_path}")
    print(f"\nlegend: A = where the model is CONFIDENT; B = where predictions actually pay off")
    print(f"        (B ≫ base rate {base:.4f} = predictable; B ≈ base = proximity-bound zone)")
    print(f"        C = honesty of confidence (green=well-calibrated, red=overconfident)")


if __name__ == "__main__":
    main()
