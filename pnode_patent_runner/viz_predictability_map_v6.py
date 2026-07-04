"""viz_predictability_map_v6.py — publication-style (Nature-like) past vs
predicted-future flow on a technology predictability landscape.

Design principles:
  - sans-serif (Arial), tight clean layout, DPI 300
  - 2-panel side-by-side: (a) past observed flow, (b) predicted future flow
  - shared colorbar, shared scale, letter labels
  - fewer + bolder arrows (grid 14, K=10) for legibility
  - colorblind-safe palette: viridis-ish green for predictability; navy/crimson
    for past/predicted arrows; black dashed contour for proximity-bound boundary

Run:  python pnode_patent_runner/viz_predictability_map_v6.py --domain construction
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy.stats import gaussian_kde
from sklearn.isotonic import IsotonicRegression

from diagnose_convergence_signal import ROOT
import recommender_firm as R

# ---- publication style ----
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 300,
    "axes.linewidth": 0.9, "axes.edgecolor": "#222",
    "axes.spines.right": False, "axes.spines.top": False,
})

GLOSS = {"E01": "roads / bridges", "E02": "hydraulic / foundations",
         "E03": "water / sewerage", "E04": "building", "E05": "locks / fittings",
         "E06": "doors / windows", "E21": "drilling / mining",
         "B28": "cement / clay", "B66": "hoisting", "C04": "concrete",
         "F16": "machine elements", "B23": "machine tools", "B65": "conveying"}


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


def bin_field(flow_xy, flow_uv, x_min, x_max, y_min, y_max, grid_v=14, min_count=12):
    bx = np.linspace(x_min, x_max, grid_v + 1); by = np.linspace(y_min, y_max, grid_v + 1)
    ix = np.clip(np.searchsorted(bx, flow_xy[:, 0]) - 1, 0, grid_v - 1)
    iy = np.clip(np.searchsorted(by, flow_xy[:, 1]) - 1, 0, grid_v - 1)
    U = np.zeros((grid_v, grid_v)); V = np.zeros((grid_v, grid_v)); C = np.zeros((grid_v, grid_v))
    for k in range(len(flow_xy)):
        U[iy[k], ix[k]] += flow_uv[k, 0]; V[iy[k], ix[k]] += flow_uv[k, 1]; C[iy[k], ix[k]] += 1
    cx = (bx[:-1] + bx[1:]) / 2; cy_ = (by[:-1] + by[1:]) / 2
    CGX, CGY = np.meshgrid(cx, cy_)
    U = np.divide(U, C, out=np.zeros_like(U), where=C > 0)
    V = np.divide(V, C, out=np.zeros_like(V), where=C > 0)
    nr = np.sqrt(U ** 2 + V ** 2); nr[nr < 1e-9] = 1; U /= nr; V /= nr
    op = np.clip(C / max(C.max(), 1) * 1.3, 0, 1); op[C < min_count] = 0
    return CGX, CGY, U, V, op


def panel(ax, heat_args, field, color, letter, title, base, zones, x_lims, y_lims):
    heat, x_min, x_max, y_min, y_max, GX, GY = heat_args
    im = ax.imshow(heat, extent=(x_min, x_max, y_min, y_max), origin="lower",
                   cmap="YlGn", vmin=0, vmax=0.13, aspect="auto",
                   interpolation="bilinear", alpha=0.62)            # lighter background
    cs_ = ax.contour(GX, GY, heat.filled(0), levels=[2 * base],
                     colors=["#111"], linestyles=["--"], linewidths=[1.6])
    CGX_, CGY_, U_, V_, op_ = field
    grid_v = CGX_.shape[0]
    for iy_ in range(grid_v):
        for ix_ in range(grid_v):
            if op_[iy_, ix_] > 0.06:
                ax.quiver(CGX_[iy_, ix_], CGY_[iy_, ix_], U_[iy_, ix_], V_[iy_, ix_],
                          angles="xy", scale_units="xy", scale=1.15,
                          color=color, alpha=min(op_[iy_, ix_] * 1.1 + 0.1, 1.0),
                          width=0.0085, headwidth=4.2, headlength=5.5, headaxislength=5,
                          edgecolor="white", linewidth=0.4)
    # stagger overlapping labels manually: E04 and E02 share the building/foundations area
    label_offset = {"E02": (0, -0.6), "E04": (0, 0.6)}
    for z, c in zones.items():
        dx, dy = label_offset.get(z, (0, 0))
        ax.text(c[0] + dx, c[1] + dy, f"{z}  {GLOSS[z]}", fontsize=10, ha="center", va="center",
                weight="bold", color="#111", alpha=0.95,
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#333", lw=0.8, alpha=0.96))
    ax.set_title(title, pad=10, fontsize=11.5)
    ax.text(0.01, 0.98, letter, transform=ax.transAxes, fontsize=15, weight="bold",
            va="top", ha="left")
    ax.set_xlim(*x_lims); ax.set_ylim(*y_lims)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#444"); s.set_linewidth(0.8)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--train-year", type=int, default=2012)
    ap.add_argument("--test-year", type=int, default=2015)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--pred-topk", type=int, default=10)
    ap.add_argument("--n-eval", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="viz_predictability_map_v6.png")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed); R.LEVEL = "group"

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
        for j in np.argsort(-cal)[:args.topk]:
            t = cand[j]; stats[t]["n_rec"] += 1
            if t in new:
                stats[t]["n_hit"] += 1
    tidx = np.array([t for t, s in stats.items() if s["n_rec"] >= 10])
    nrec = np.array([stats[t]["n_rec"] for t in tidx], float)
    hit = np.array([stats[t]["n_hit"] / stats[t]["n_rec"] for t in tidx])
    pts = xy[tidx]
    x_min, x_max = xy[:, 0].min() - 0.7, xy[:, 0].max() + 0.7
    y_min, y_max = xy[:, 1].min() - 0.7, xy[:, 1].max() + 0.7
    grid_n = 100
    gx = np.linspace(x_min, x_max, grid_n); gy = np.linspace(y_min, y_max, grid_n)
    GX, GY = np.meshgrid(gx, gy); grid = np.vstack([GX.ravel(), GY.ravel()])
    kde_hit = gaussian_kde(pts.T, bw_method=0.20, weights=hit * nrec)(grid).reshape(grid_n, grid_n)
    kde_n = gaussian_kde(pts.T, bw_method=0.20, weights=nrec)(grid).reshape(grid_n, grid_n)
    heat = np.divide(kde_hit, kde_n + 1e-12, out=np.zeros_like(kde_hit), where=kde_n > 0)
    heat = np.ma.array(heat, mask=kde_n < (kde_n.max() * 0.03))

    # build past + predicted flows
    past_xy, past_uv, pred_xy, pred_uv = [], [], [], []
    for u, X, owned, Su, new, cand, raw, lab in eval_data:
        port_t = [cidx[c] for c in Su if c in cidx]
        if not port_t:
            continue
        center = xy[port_t].mean(0)
        for new_t in new:
            target = xy[new_t]
            for p in xy[port_t]:
                v = target - p; nv = np.linalg.norm(v)
                if nv > 1e-6:
                    past_xy.append((p + target) / 2); past_uv.append(v / nv)
        top = cand[np.argsort(-raw)[:args.pred_topk]]
        for t in top:
            v = xy[t] - center; nv = np.linalg.norm(v)
            if nv > 1e-6:
                pred_xy.append((center + xy[t]) / 2); pred_uv.append(v / nv)
    past_xy = np.array(past_xy); past_uv = np.array(past_uv)
    pred_xy = np.array(pred_xy); pred_uv = np.array(pred_uv)
    print(f"past flows={len(past_xy)}, predicted flows={len(pred_xy)} (K={args.pred_topk})")
    past_field = bin_field(past_xy, past_uv, x_min, x_max, y_min, y_max, grid_v=14, min_count=10)
    pred_field = bin_field(pred_xy, pred_uv, x_min, x_max, y_min, y_max, grid_v=14, min_count=8)

    zone_pts = defaultdict(list)
    for t in range(n_c):
        zone_pts[inv_c[t][:3]].append(xy[t])
    zones = {z: np.mean(v, 0) for z, v in zone_pts.items() if len(v) >= 4 and z in GLOSS}

    fig = plt.figure(figsize=(15, 7.2))
    gs = fig.add_gridspec(1, 2, wspace=0.04, left=0.04, right=0.92, top=0.92, bottom=0.10)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    heat_args = (heat, x_min, x_max, y_min, y_max, GX, GY)
    im = panel(ax_a, heat_args, past_field, "#0050B3", "a",   # vivid blue
               f"Past observed direction (year {args.test_year} → +{args.horizon}y)",
               base, zones, (x_min, x_max), (y_min, y_max))
    panel(ax_b, heat_args, pred_field, "#D62728", "b",         # vivid red
          f"Predicted future direction (model top-{args.pred_topk})",
          base, zones, (x_min, x_max), (y_min, y_max))

    cax = fig.add_axes([0.93, 0.18, 0.012, 0.62])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Predictability\n(realised hit rate)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.text(0.5, 0.04,
             f"Green = where recommendations actually work (>>base rate {base:.3f}); "
             "dashed line = boundary (2× base). Arrows: opacity ∝ data density "
             "(faded = direction unreliable). Past flows = observed firm migrations; "
             "predicted = recommender top-K from each firm's portfolio centroid.",
             ha="center", va="bottom", fontsize=9, color="#333")
    fig.suptitle(f"Technology trend flow on a predictability landscape "
                 f"({args.domain}, firm × CPC-group)",
                 fontsize=13, y=0.98)
    out = ROOT / "pnode_patent_runner" / args.out
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
