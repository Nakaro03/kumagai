"""viz_predictability_map_v2.py — Pattern-3 visualization, v2 (refined).

Critique of v1 (viz_predictability_map.py): horizontal 3-panel layout too
shallow; predictability boundary not visible; scatter too sparse. v2 fixes by:
  - 2x2 layout with the MAP as the hero panel
  - KDE-smoothed heatmap BACKGROUND of realized hit rate -> zones, not dots
  - explicit BOUNDARY CONTOUR at hit_rate = base_rate * margin (proximity-bound
    line, visualised)
  - more recommended CPCs (topk=50, min_rec=10)
  - two example firms (focused vs diversified) overlaid for comparison
  - a calibration scatter (pred vs actual) replaces the residual panel

Run:  python pnode_patent_runner/viz_predictability_map_v2.py --domain construction
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy.stats import gaussian_kde
from sklearn.isotonic import IsotonicRegression

from diagnose_convergence_signal import ROOT
import recommender_firm as R


def setup(args, rng):
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
    return df, emb, codes, cidx, inv_c, n_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--train-year", type=int, default=2012)
    ap.add_argument("--test-year", type=int, default=2015)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--min-rec", type=int, default=10)
    ap.add_argument("--n-eval", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="viz_predictability_map_v2.png")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    R.LEVEL = "group"

    df, emb, codes, cidx, inv_c, n_c = setup(args, rng)
    print(f"firms={df.u.nunique()} CPC-groups={n_c}")

    sc, clf = R.train_lr(df, args.train_year, cidx, emb, n_c, args.horizon, rng, cols=[0, 1, 2, 3])
    w = R.build_world(df, args.test_year, cidx, emb, n_c, args.horizon)

    print("UMAP to 2D ...")
    xy = umap.UMAP(n_neighbors=20, min_dist=0.30, n_components=2,
                   random_state=args.seed).fit_transform(w["Cemb"])

    invs = R.actors(w, cidx); rng.shuffle(invs); invs = invs[:args.n_eval]
    sp = len(invs) // 2
    cs, cy = [], []; eval_data = []
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

    stats = defaultdict(lambda: {"n_rec": 0, "sum_conf": 0, "n_hit": 0})
    for u, X, owned, Su, new, cand, raw, lab in eval_data:
        cal = iso.predict(raw)
        order = np.argsort(-cal)[:args.topk]
        for j in order:
            t = cand[j]
            stats[t]["n_rec"] += 1; stats[t]["sum_conf"] += cal[j]
            if t in new:
                stats[t]["n_hit"] += 1
    base = float(np.mean([y.mean() for y in cy]))
    print(f"base rate ≈ {base:.4f}")

    rows = [(t, s["n_rec"], s["sum_conf"] / s["n_rec"], s["n_hit"] / s["n_rec"])
            for t, s in stats.items() if s["n_rec"] >= args.min_rec]
    tidx = np.array([r[0] for r in rows]); nrec = np.array([r[1] for r in rows])
    conf = np.array([r[2] for r in rows]); hit = np.array([r[3] for r in rows])
    pts = xy[tidx]; grey = xy[[t for t in range(n_c) if stats[t]["n_rec"] < args.min_rec]]
    print(f"colored CPCs: {len(rows)} ; grey CPCs: {len(grey)}")

    # KDE-smoothed heatmap of hit-rate over a grid
    grid_n = 120
    x_min, x_max = xy[:, 0].min() - 0.5, xy[:, 0].max() + 0.5
    y_min, y_max = xy[:, 1].min() - 0.5, xy[:, 1].max() + 0.5
    gx = np.linspace(x_min, x_max, grid_n); gy = np.linspace(y_min, y_max, grid_n)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.vstack([GX.ravel(), GY.ravel()])
    if len(pts) >= 3:
        # weighted KDE: each point's contribution proportional to hit ; another for n_rec
        kde_hit = gaussian_kde(pts.T, bw_method=0.20, weights=hit * nrec)(grid).reshape(grid_n, grid_n)
        kde_n = gaussian_kde(pts.T, bw_method=0.20, weights=nrec)(grid).reshape(grid_n, grid_n)
        heat = np.divide(kde_hit, kde_n + 1e-12, out=np.zeros_like(kde_hit), where=kde_n > 0)
        # mask where total density is too low (no support)
        mask = kde_n < (kde_n.max() * 0.03)
        heat = np.ma.array(heat, mask=mask)
    else:
        heat = None

    # find an interesting example firm: focused (low std of portfolio xy) with hits
    def find_firm(focused=True):
        best_u, best_score = None, np.inf if focused else -np.inf
        for u, X, owned, Su, new, cand, raw, lab in eval_data:
            if len(Su) < 6:
                continue
            port_t = [cidx[c] for c in Su if c in cidx]
            var = xy[port_t].std(0).sum()
            if focused and var < best_score:
                best_score = var; best_u = u
            elif not focused and var > best_score:
                best_score = var; best_u = u
        return best_u
    nm_path = ROOT / f"data/processed/{args.domain}_firm_names.csv"
    names = pd.read_csv(nm_path).drop_duplicates("u").set_index("u")["org"].to_dict() if nm_path.exists() else {}
    u_focus = find_firm(True); u_diverse = find_firm(False)

    def overlay(ax, u, color_main, label_prefix):
        port_t = [cidx[c] for c in w["prior"][u] if c in cidx]
        new_set = {cidx[c] for c in (w["nextf"][u] - w["prior"][u]) if c in cidx and w["have"][cidx[c]]}
        X, owned, Su = R.actor_scores(u, w, emb, cidx, n_c)
        cand = np.array([t for t in range(n_c) if t not in owned])
        cal = iso.predict(clf.decision_function(sc.transform(X[cand][:, [0, 1, 2, 3]])))
        top = cand[np.argsort(-cal)[:6]]
        center = xy[port_t].mean(0)
        ax.scatter(xy[port_t, 0], xy[port_t, 1], marker="*", c=color_main, s=180, zorder=6,
                   edgecolors="white", linewidths=1.0, label=f"★ {label_prefix}: {str(names.get(u, '?'))[:32]}")
        for t in top:
            cc = "limegreen" if t in new_set else "crimson"
            ax.annotate("", xy=(xy[t, 0], xy[t, 1]), xytext=center,
                        arrowprops=dict(arrowstyle="->", color=cc, alpha=0.9, lw=1.8))

    fig = plt.figure(figsize=(18, 12))
    # hero panel: predictability map (KDE) + boundary contour
    ax0 = fig.add_subplot(2, 2, (1, 2))
    if heat is not None:
        im = ax0.imshow(heat, extent=(x_min, x_max, y_min, y_max), origin="lower",
                        cmap="RdYlGn", vmin=0, vmax=max(0.08, hit.max()), aspect="auto", alpha=0.85)
        # proximity-bound boundary: hit_rate = 2 * base rate (= where model adds value)
        cs_ = ax0.contour(GX, GY, heat.filled(0), levels=[2 * base, 4 * base],
                          colors=["black", "darkred"], linestyles=["--", "-"], linewidths=[1.2, 1.6])
        ax0.clabel(cs_, fmt={2 * base: f"2×base ({2*base:.3f})",
                             4 * base: f"4×base ({4*base:.3f})"}, fontsize=8)
        fig.colorbar(im, ax=ax0, shrink=0.7, label="realized hit rate (predictability)")
    ax0.scatter(grey[:, 0], grey[:, 1], s=5, c="lightgrey", alpha=0.5)
    ax0.scatter(pts[:, 0], pts[:, 1], s=10 + nrec / 4, c="black", alpha=0.35, zorder=3)
    if u_focus:
        overlay(ax0, u_focus, "navy", "focused firm")
    if u_diverse:
        overlay(ax0, u_diverse, "purple", "diversified firm")
    ax0.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax0.set_title(f"Predictability map — green = where recommendations actually work (>>base {base:.3f}); "
                  f"red = proximity-bound (≈chance); dashed = boundary",
                  fontsize=11)
    ax0.set_xticks([]); ax0.set_yticks([])

    # secondary: scatter of conf vs hit (calibration health)
    ax1 = fig.add_subplot(2, 2, 3)
    ax1.scatter(conf, hit, s=10 + nrec / 4, alpha=0.6, c="steelblue", edgecolors="k", linewidths=0.3)
    lim = max(conf.max(), hit.max()) * 1.05
    ax1.plot([0, lim], [0, lim], "k--", alpha=0.5, label="perfect calibration")
    ax1.axhline(base, color="red", alpha=0.5, linestyle=":", label=f"base rate {base:.3f}")
    ax1.set_xlabel("mean calibrated confidence per CPC")
    ax1.set_ylabel("realized hit rate per CPC")
    ax1.set_title("(B) per-CPC calibration health  (point above red = beats base rate)")
    ax1.legend(fontsize=8); ax1.set_xlim(0, lim); ax1.set_ylim(0, lim)

    # third: distribution of hit rates (predictable vs not)
    ax2 = fig.add_subplot(2, 2, 4)
    ax2.hist(hit, bins=20, color="seagreen", alpha=0.75, edgecolor="black")
    ax2.axvline(base, color="red", linestyle=":", label=f"base rate {base:.3f}")
    ax2.axvline(2 * base, color="black", linestyle="--", label=f"2×base = predictable threshold")
    ax2.set_xlabel("realized hit rate per CPC"); ax2.set_ylabel("# CPC groups")
    ax2.set_title("(C) distribution of predictability across CPC groups")
    ax2.legend(fontsize=8)

    plt.suptitle(f"Pattern-3 predictability map ({args.domain}, FIRM × CPC-group, "
                 f"train {args.train_year}, horizon {args.horizon}y)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = ROOT / "pnode_patent_runner" / args.out
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
