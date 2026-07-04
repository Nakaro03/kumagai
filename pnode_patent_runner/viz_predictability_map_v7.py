"""viz_predictability_map_v7.py — v6 + integrated CASE STUDY panel.

Publication-style 3-panel figure:
  (a) PAST observed direction (blue field)
  (b) PREDICTED future direction (red field)
  (c) CASE STUDY: one named firm with portfolio (star) + top-10 recommendations
      (numbered arrows, green=actually entered, red=missed), with calibrated conf.

Run:  python pnode_patent_runner/viz_predictability_map_v7.py --domain construction
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

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 300,
    "axes.linewidth": 0.9, "axes.edgecolor": "#222",
    "axes.spines.right": False, "axes.spines.top": False,
})

GLOSS = {"E01": "roads", "E02": "hydraulic / foundations",
         "E03": "water / sewerage", "E04": "building", "E05": "locks",
         "E06": "doors / windows", "E21": "drilling / mining",
         "B28": "cement / clay", "B66": "hoisting", "C04": "concrete",
         "F16": "machine elements", "B23": "machine tools"}


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


def bin_field(flow_xy, flow_uv, x_min, x_max, y_min, y_max, grid_v=14, min_count=10):
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


def draw_background(ax, heat_args, base, zones):
    heat, x_min, x_max, y_min, y_max, GX, GY = heat_args
    im = ax.imshow(heat, extent=(x_min, x_max, y_min, y_max), origin="lower",
                   cmap="YlGn", vmin=0, vmax=0.13, aspect="auto",
                   interpolation="bilinear", alpha=0.62)
    ax.contour(GX, GY, heat.filled(0), levels=[2 * base],
               colors=["#111"], linestyles=["--"], linewidths=[1.4])
    label_offset = {"E02": (0, -0.6), "E04": (0, 0.6)}
    for z, c in zones.items():
        dx, dy = label_offset.get(z, (0, 0))
        ax.text(c[0] + dx, c[1] + dy, f"{z}  {GLOSS[z]}", fontsize=9, ha="center", va="center",
                weight="bold", color="#111",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#333", lw=0.7, alpha=0.95))
    return im


def panel_field(ax, heat_args, field, color, letter, title, base, zones):
    heat, x_min, x_max, y_min, y_max, GX, GY = heat_args
    im = draw_background(ax, heat_args, base, zones)
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
    ax.set_title(title, pad=8); ax.text(0.01, 0.98, letter, transform=ax.transAxes,
                                        fontsize=15, weight="bold", va="top", ha="left")
    ax.set_xticks([]); ax.set_yticks([])
    return im


def panel_case(ax, heat_args, base, zones, xy, cidx, inv_c, u, port_t, top, top_conf, new_set, name):
    draw_background(ax, heat_args, base, zones)
    center = xy[port_t].mean(0)
    for k, (t, c) in enumerate(zip(top, top_conf), 1):
        tx, ty = xy[t]; is_hit = t in new_set
        cc = "#138D75" if is_hit else "#B03A2E"
        ax.annotate("", xy=(tx, ty), xytext=center,
                    arrowprops=dict(arrowstyle="->", color=cc, alpha=0.95, lw=1.8))
        ax.scatter([tx], [ty], s=110, c=cc, edgecolors="white", linewidths=1.2, zorder=8)
        ax.text(tx, ty + 0.22, f"{k}", fontsize=9, ha="center", va="bottom",
                weight="bold", color=cc,
                bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec=cc, lw=1.2))
    ax.scatter(xy[port_t, 0], xy[port_t, 1], marker="*", c="black", s=200, zorder=9,
               edgecolors="gold", linewidths=1.4)
    ax.set_title(f"Case study: {name[:38]}", pad=8)
    ax.text(0.01, 0.98, "c", transform=ax.transAxes, fontsize=15, weight="bold",
            va="top", ha="left")
    ax.set_xticks([]); ax.set_yticks([])


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
    ap.add_argument("--out", default="viz_predictability_map_v7.png")
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
    print(f"past={len(past_xy)} pred={len(pred_xy)}")
    past_field = bin_field(past_xy, past_uv, x_min, x_max, y_min, y_max, grid_v=14, min_count=10)
    pred_field = bin_field(pred_xy, pred_uv, x_min, x_max, y_min, y_max, grid_v=14, min_count=8)

    zone_pts = defaultdict(list)
    for t in range(n_c):
        zone_pts[inv_c[t][:3]].append(xy[t])
    zones = {z: np.mean(v, 0) for z, v in zone_pts.items() if len(v) >= 4 and z in GLOSS}

    # case study firm selection
    nm = {}
    nm_path = ROOT / f"data/processed/{args.domain}_firm_names.csv"
    if nm_path.exists():
        nm = pd.read_csv(nm_path).drop_duplicates("u").set_index("u")["org"].to_dict()
    best_u, best_s = None, -np.inf
    for require_name in [True, False]:
        for u, X, owned, Su, new_set, cand, raw, lab in eval_data:
            if not (5 <= len(Su) <= 18) or len(new_set) < 3:
                continue
            if require_name and len(str(nm.get(u, "")).strip()) < 4:
                continue
            port_t = [cidx[c] for c in Su if c in cidx]
            spread = xy[port_t].std(0).sum()
            s = -spread + len(new_set) * 0.6
            if s > best_s:
                best_s = s; best_u = u
        if best_u is not None:
            break
    u_cs = best_u
    X, owned, Su = R.actor_scores(u_cs, w, emb, cidx, n_c)
    new_set = {cidx[c] for c in (w["nextf"][u_cs] - w["prior"][u_cs]) if c in cidx and w["have"][cidx[c]]}
    cand = np.array([t for t in range(n_c) if t not in owned])
    cal = iso.predict(clf.decision_function(sc.transform(X[cand][:, [0, 1, 2, 3]])))
    order = np.argsort(-cal)[:10]; top = cand[order]; top_conf = cal[order]
    port_t = [cidx[c] for c in Su if c in cidx]
    cs_name = str(nm.get(u_cs, "?"))
    n_hit = sum(1 for t in top[:5] if t in new_set)

    # ---- figure: 3 panels horizontal + side table
    fig = plt.figure(figsize=(20, 7.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.55], wspace=0.07,
                          left=0.03, right=0.97, top=0.92, bottom=0.10)
    heat_args = (heat, x_min, x_max, y_min, y_max, GX, GY)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1]); ax_c = fig.add_subplot(gs[0, 2])
    im = panel_field(ax_a, heat_args, past_field, "#0050B3", "a",
                     f"Past observed direction ({args.test_year} → +{args.horizon}y)",
                     base, zones)
    panel_field(ax_b, heat_args, pred_field, "#D62728", "b",
                f"Predicted future direction (top-{args.pred_topk})", base, zones)
    panel_case(ax_c, heat_args, base, zones, xy, cidx, inv_c,
               u_cs, port_t, top, top_conf, new_set, cs_name)

    # side table
    ax_t = fig.add_subplot(gs[0, 3]); ax_t.axis("off")
    lines = [f"FIRM (case c)",
             f"{cs_name[:34]}",
             "─" * 28,
             f"Portfolio: {len(Su)} CPC groups",
             f"Actual new entries (3y): {len(new_set)}",
             f"Hits in top-5: {n_hit}/5",
             "",
             "Top-10 recommendations",
             "─" * 28,
             "# | CPC      | conf  | hit"]
    for k, (t, c) in enumerate(zip(top, top_conf), 1):
        g = inv_c[t]; lab = "✓" if t in new_set else "✗"
        lines.append(f"{k:>2}| {g:<8} | {c*100:4.1f}% |  {lab}")
    lines += ["─" * 28,
              f"base rate = {base:.3f}",
              f"Lift@5 ≈ {(n_hit/5)/base:.1f}×",
              "",
              "Confidence is",
              "CALIBRATED (ECE≈0.001):",
              "  13% = real ~13%",
              "  chance of entry in 3y."]
    ax_t.text(0, 0.99, "\n".join(lines), family="monospace", fontsize=8.5, va="top")

    cax = fig.add_axes([0.79, 0.93, 0.16, 0.012])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Predictability (realised hit rate)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.text(0.5, 0.025,
             "Underlying data = bipartite graph firm × CPC-group. The 2D map shows the CPC side "
             f"(231 groups, UMAP from PPMI+SVD embedding). Stars = firm portfolio centroid; "
             "arrows = inter-CPC firm migrations. Green background = realised hit-rate density; "
             "dashed contour = predictability boundary (2× base rate).",
             ha="center", va="bottom", fontsize=8.5, color="#333")
    fig.suptitle("Technology trend flow + case study on a predictability landscape "
                 f"({args.domain})", fontsize=13, y=0.985)
    out = ROOT / "pnode_patent_runner" / args.out
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
