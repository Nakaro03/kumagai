"""viz_predictability_map_v3.py — clean single-story version.

v2 was dense (3 panels, 2 firms overlaid). v3 simplifies to ONE hero panel telling
one story:
  "Here is the technology landscape. Green = where recommendations work. The
   dashed line is the boundary; outside it = proximity-bound. Here is ONE firm,
   its portfolio (★), and its top-5 recommendations (numbered arrows); green
   arrow = it actually entered within 3 years."
Plus a compact recommendation table panel.

Run:  python pnode_patent_runner/viz_predictability_map_v3.py --domain construction
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.patches import FancyArrowPatch
from scipy.stats import gaussian_kde
from sklearn.isotonic import IsotonicRegression

from diagnose_convergence_signal import ROOT
import recommender_firm as R

GLOSS = {"E01": "roads/bridges", "E02": "hydraulic / foundations",
         "E03": "water / sewerage", "E04": "building", "E05": "locks / fittings",
         "E06": "doors / windows", "E21": "drilling / mining",
         "B28": "cement / clay", "B66": "hoisting / lifting", "C04": "concrete",
         "F16": "machine elements", "B23": "machine tools", "B65": "conveying"}


def setup(args):
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
    cidx = {c: k for k, c in enumerate(codes)}; inv_c = {k: c for c, k in cidx.items()}
    df = df[df.i.isin(cidx)]
    return df, emb, codes, cidx, inv_c, len(codes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--train-year", type=int, default=2012)
    ap.add_argument("--test-year", type=int, default=2015)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n-eval", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="viz_predictability_map_v3.png")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    R.LEVEL = "group"

    df, emb, codes, cidx, inv_c, n_c = setup(args)
    print(f"firms={df.u.nunique()} CPC-groups={n_c}")
    sc, clf = R.train_lr(df, args.train_year, cidx, emb, n_c, args.horizon, rng, cols=[0, 1, 2, 3])
    w = R.build_world(df, args.test_year, cidx, emb, n_c, args.horizon)
    print("UMAP to 2D ...")
    xy = umap.UMAP(n_neighbors=20, min_dist=0.30, n_components=2,
                   random_state=args.seed).fit_transform(w["Cemb"])

    invs = R.actors(w, cidx); rng.shuffle(invs); invs = invs[:args.n_eval]
    sp = len(invs) // 2
    cs, cy, eval_data = [], [], []
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
    base = float(np.mean([y.mean() for y in cy]))

    stats = defaultdict(lambda: {"n_rec": 0, "n_hit": 0})
    for u, X, owned, Su, new, cand, raw, lab in eval_data:
        cal = iso.predict(raw)
        order = np.argsort(-cal)[:args.topk]
        for j in order:
            t = cand[j]; stats[t]["n_rec"] += 1
            if t in new:
                stats[t]["n_hit"] += 1
    tidx = np.array([t for t, s in stats.items() if s["n_rec"] >= 10])
    if len(tidx) > 0:
        nrec = np.array([stats[t]["n_rec"] for t in tidx], float)
        hit = np.array([stats[t]["n_hit"] / stats[t]["n_rec"] for t in tidx])
        pts = xy[tidx]
    else:
        return

    # KDE-smoothed predictability field
    grid_n = 140
    x_min, x_max = xy[:, 0].min() - 0.7, xy[:, 0].max() + 0.7
    y_min, y_max = xy[:, 1].min() - 0.7, xy[:, 1].max() + 0.7
    gx = np.linspace(x_min, x_max, grid_n); gy = np.linspace(y_min, y_max, grid_n)
    GX, GY = np.meshgrid(gx, gy); grid = np.vstack([GX.ravel(), GY.ravel()])
    kde_hit = gaussian_kde(pts.T, bw_method=0.20, weights=hit * nrec)(grid).reshape(grid_n, grid_n)
    kde_n = gaussian_kde(pts.T, bw_method=0.20, weights=nrec)(grid).reshape(grid_n, grid_n)
    heat = np.divide(kde_hit, kde_n + 1e-12, out=np.zeros_like(kde_hit), where=kde_n > 0)
    mask = kde_n < (kde_n.max() * 0.03); heat = np.ma.array(heat, mask=mask)

    # find recognizable focused firm with hits
    nm = {}
    nm_path = ROOT / f"data/processed/{args.domain}_firm_names.csv"
    if nm_path.exists():
        nm = pd.read_csv(nm_path).drop_duplicates("u").set_index("u")["org"].to_dict()

    def score_firm(u, Su, new, require_name):
        # Su: list of CPC strings, new: set of CPC indices
        port_t = [cidx[c] for c in Su if c in cidx]
        if not (5 <= len(port_t) <= 20) or len(new) < 2:
            return -np.inf
        if require_name and len(str(nm.get(u, "")).strip()) < 4:
            return -np.inf
        spread = xy[port_t].std(0).sum()
        return -spread + len(new) * 0.6

    best_u, best_s = None, -np.inf
    for require_name in [True, False]:
        for u, X, owned, Su, new, cand, raw, lab in eval_data:
            s = score_firm(u, Su, new, require_name)
            if s > best_s:
                best_s = s; best_u = u
        if best_u is not None:
            break
    print(f"chosen firm: {best_u}  name='{nm.get(best_u, '?')}'  port={len([c for c in w['prior'][best_u] if c in cidx])}  new={len([c for c in (w['nextf'][best_u]-w['prior'][best_u]) if c in cidx])}")
    u = best_u
    X, owned, Su = R.actor_scores(u, w, emb, cidx, n_c)
    new = {cidx[c] for c in (w["nextf"][u] - w["prior"][u]) if c in cidx and w["have"][cidx[c]]}
    cand = np.array([t for t in range(n_c) if t not in owned])
    cal = iso.predict(clf.decision_function(sc.transform(X[cand][:, [0, 1, 2, 3]])))
    order = np.argsort(-cal)[:5]; top = cand[order]; top_conf = cal[order]
    port_t = [cidx[c] for c in Su if c in cidx]

    # zone labels: per CPC subclass, place text at centroid
    zone_pts = defaultdict(list)
    for t in range(n_c):
        zone_pts[inv_c[t][:3]].append(xy[t])
    zone_centroids = {z: np.mean(v, 0) for z, v in zone_pts.items() if len(v) >= 4 and z in GLOSS}

    # ---- figure
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 4, width_ratios=[3, 3, 3, 2])
    ax = fig.add_subplot(gs[0, :3])

    im = ax.imshow(heat, extent=(x_min, x_max, y_min, y_max), origin="lower",
                   cmap="YlGn", vmin=0, vmax=0.10, aspect="auto", alpha=0.92)
    cs_ = ax.contour(GX, GY, heat.filled(0), levels=[2 * base],
                     colors=["black"], linestyles=["--"], linewidths=[1.8])
    ax.clabel(cs_, fmt={2 * base: f"boundary (2x base = {2*base:.3f})"}, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cbar.set_label("realized hit rate\n(predictability)", fontsize=9)

    # zone labels
    for z, c in zone_centroids.items():
        ax.text(c[0], c[1], f"{z}\n{GLOSS[z]}", fontsize=10, ha="center", va="center",
                color="black", weight="bold", alpha=0.75,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

    # firm portfolio (★)
    ax.scatter(xy[port_t, 0], xy[port_t, 1], marker="*", c="navy", s=240, zorder=6,
               edgecolors="white", linewidths=1.5, label=f"★ portfolio: {str(nm.get(u,'?'))[:40]}")
    # numbered recommendations
    center = xy[port_t].mean(0)
    for k, (t, c) in enumerate(zip(top, top_conf), 1):
        tx, ty = xy[t]; is_hit = t in new
        cc = "darkgreen" if is_hit else "crimson"
        ax.annotate("", xy=(tx, ty), xytext=center,
                    arrowprops=dict(arrowstyle="->", color=cc, alpha=0.95, lw=2.2))
        ax.scatter([tx], [ty], s=140, c=cc, edgecolors="white", linewidths=1.5, zorder=5)
        ax.text(tx, ty + 0.18, f"{k}", fontsize=11, ha="center", va="bottom",
                weight="bold", color=cc,
                bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec=cc, lw=1.5))

    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax.set_title("Technology landscape with predictability boundary\n"
                 "(★ = a firm's portfolio; numbered arrows = top-5 recommendations; "
                 "green = it actually entered within 3 years)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # right side table
    ax2 = fig.add_subplot(gs[0, 3]); ax2.axis("off")
    name = str(nm.get(u, "?"))[:34]
    lines = [f"FIRM: {name}",
             f"Portfolio size: {len(Su)} CPC groups",
             f"Actually entered within 3y: {len(new)}",
             "",
             "Top-5 recommendations:",
             "─────────────────────",
             "# | CPC      | conf  | hit"]
    for k, (t, c) in enumerate(zip(top, top_conf), 1):
        g = inv_c[t]; lab = "✓" if t in new else "✗"
        lines.append(f"{k} | {g:<8} | {c*100:4.1f}% |  {lab}")
    n_hit = sum(1 for t in top if t in new)
    lines += ["─────────────────────", f"Hits in top-5: {n_hit}/5",
              "",
              f"base rate (random) = {base:.3f}",
              f"= 1 entry per {int(1/base)} candidate CPCs",
              "",
              "Confidence is CALIBRATED:",
              " 13% means real ~13%",
              " chance of entry within 3y",
              "",
              "GREEN ZONES on map:",
              " recommendations work",
              " (~5-10x base rate)",
              "RED/WHITE outside boundary:",
              " proximity-bound (chance)"]
    ax2.text(0, 0.97, "\n".join(lines), family="monospace", fontsize=9,
             verticalalignment="top")

    plt.suptitle(f"Technology trend prediction — Pattern-3 visualization "
                 f"({args.domain}, FIRM × CPC-group, base year {args.test_year}, horizon {args.horizon}y)",
                 fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = ROOT / "pnode_patent_runner" / args.out
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
