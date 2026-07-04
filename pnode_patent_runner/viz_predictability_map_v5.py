"""viz_predictability_map_v5.py — Past vs PREDICTED FUTURE direction fields.

v4 showed empirical past flows (where firms have moved). v5 adds a second field:
PREDICTED future direction from the trained recommender (where the model says
firms WILL move). Overlaying both reveals: (a) where past and predicted agree =
direction we trust, (b) where they disagree = forecast novelty (or model error).

  blue arrows  = empirical past direction (observed Y -> Y+H)
  red arrows   = model-PREDICTED future direction (model's top-K recommendations
                 averaged per portfolio-centroid cell)
Opacity ~ density-of-evidence in each cell (epistemic-like reliability).

Run:  python pnode_patent_runner/viz_predictability_map_v5.py --domain construction
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

GLOSS = {"E01": "roads/bridges", "E02": "hydraulic/foundations",
         "E03": "water/sewerage", "E04": "building", "E05": "locks/fittings",
         "E06": "doors/windows", "E21": "drilling/mining",
         "B28": "cement/clay", "B66": "hoisting", "C04": "concrete",
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


def bin_field(flow_xy, flow_uv, x_min, x_max, y_min, y_max, grid_v=20, min_count=8):
    bx = np.linspace(x_min, x_max, grid_v + 1); by = np.linspace(y_min, y_max, grid_v + 1)
    ix = np.clip(np.searchsorted(bx, flow_xy[:, 0]) - 1, 0, grid_v - 1)
    iy = np.clip(np.searchsorted(by, flow_xy[:, 1]) - 1, 0, grid_v - 1)
    U = np.zeros((grid_v, grid_v)); V = np.zeros((grid_v, grid_v)); C = np.zeros((grid_v, grid_v))
    for k in range(len(flow_xy)):
        U[iy[k], ix[k]] += flow_uv[k, 0]; V[iy[k], ix[k]] += flow_uv[k, 1]
        C[iy[k], ix[k]] += 1
    cx = (bx[:-1] + bx[1:]) / 2; cy_ = (by[:-1] + by[1:]) / 2
    CGX, CGY = np.meshgrid(cx, cy_)
    U = np.divide(U, C, out=np.zeros_like(U), where=C > 0)
    V = np.divide(V, C, out=np.zeros_like(V), where=C > 0)
    nr = np.sqrt(U ** 2 + V ** 2); nr[nr < 1e-9] = 1
    U /= nr; V /= nr
    op = np.clip(C / max(C.max(), 1) * 1.3, 0, 1); op[C < min_count] = 0
    return CGX, CGY, U, V, op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--train-year", type=int, default=2012)
    ap.add_argument("--test-year", type=int, default=2015)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--pred-topk", type=int, default=5)
    ap.add_argument("--n-eval", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="viz_predictability_map_v5.png")
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

    # predictability KDE
    stats = defaultdict(lambda: {"n_rec": 0, "n_hit": 0})
    for u, X, owned, Su, new, cand, raw, lab in eval_data:
        cal = iso.predict(raw)
        for j in np.argsort(-cal)[:args.topk]:
            t = cand[j]; stats[t]["n_rec"] += 1
            if t in new:
                stats[t]["n_hit"] += 1
    tidx = np.array([t for t, s in stats.items() if s["n_rec"] >= 10])
    if len(tidx) == 0:
        return
    nrec = np.array([stats[t]["n_rec"] for t in tidx], float)
    hit = np.array([stats[t]["n_hit"] / stats[t]["n_rec"] for t in tidx])
    pts = xy[tidx]
    grid_n = 100
    x_min, x_max = xy[:, 0].min() - 0.7, xy[:, 0].max() + 0.7
    y_min, y_max = xy[:, 1].min() - 0.7, xy[:, 1].max() + 0.7
    gx = np.linspace(x_min, x_max, grid_n); gy = np.linspace(y_min, y_max, grid_n)
    GX, GY = np.meshgrid(gx, gy); grid = np.vstack([GX.ravel(), GY.ravel()])
    kde_hit = gaussian_kde(pts.T, bw_method=0.20, weights=hit * nrec)(grid).reshape(grid_n, grid_n)
    kde_n = gaussian_kde(pts.T, bw_method=0.20, weights=nrec)(grid).reshape(grid_n, grid_n)
    heat = np.divide(kde_hit, kde_n + 1e-12, out=np.zeros_like(kde_hit), where=kde_n > 0)
    heat = np.ma.array(heat, mask=kde_n < (kde_n.max() * 0.03))

    # ---- PAST field: portfolio CPC -> actually newly-entered CPC
    past_xy, past_uv = [], []
    # ---- PREDICTED field: portfolio centroid -> model's top-K recommended CPC
    pred_xy, pred_uv = [], []
    for u, X, owned, Su, new, cand, raw, lab in eval_data:
        port_t = [cidx[c] for c in Su if c in cidx]
        if not port_t:
            continue
        center = xy[port_t].mean(0)
        # past
        for new_t in new:
            target = xy[new_t]
            for p in xy[port_t]:
                v = target - p; nv = np.linalg.norm(v)
                if nv > 1e-6:
                    past_xy.append((p + target) / 2); past_uv.append(v / nv)
        # predicted (model's top-K from THIS firm's portfolio)
        top = cand[np.argsort(-raw)[:args.pred_topk]]
        for t in top:
            v = xy[t] - center; nv = np.linalg.norm(v)
            if nv > 1e-6:
                pred_xy.append((center + xy[t]) / 2); pred_uv.append(v / nv)
    past_xy = np.array(past_xy); past_uv = np.array(past_uv)
    pred_xy = np.array(pred_xy); pred_uv = np.array(pred_uv)
    print(f"past flows={len(past_xy)}, predicted flows={len(pred_xy)}")
    CGXp, CGYp, Up, Vp, opP = bin_field(past_xy, past_uv, x_min, x_max, y_min, y_max, grid_v=20)
    CGXf, CGYf, Uf, Vf, opF = bin_field(pred_xy, pred_uv, x_min, x_max, y_min, y_max, grid_v=20)

    # zone labels
    zone_pts = defaultdict(list)
    for t in range(n_c):
        zone_pts[inv_c[t][:3]].append(xy[t])
    zone_centroids = {z: np.mean(v, 0) for z, v in zone_pts.items()
                      if len(v) >= 4 and z in GLOSS}

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for ax, (CGX_, CGY_, U_, V_, op_, color, title) in zip(axes, [
        (CGXp, CGYp, Up, Vp, opP, "darkblue",
         f"PAST direction (observed firm migrations Y={args.test_year} -> Y+{args.horizon})"),
        (CGXf, CGYf, Uf, Vf, opF, "crimson",
         f"PREDICTED FUTURE direction (model's top-{args.pred_topk} per firm-centroid)"),
    ]):
        im = ax.imshow(heat, extent=(x_min, x_max, y_min, y_max), origin="lower",
                       cmap="YlGn", vmin=0, vmax=0.10, aspect="auto", alpha=0.82)
        cs_ = ax.contour(GX, GY, heat.filled(0), levels=[2 * base],
                         colors=["black"], linestyles=["--"], linewidths=[1.6])
        ax.clabel(cs_, fmt={2 * base: f"boundary (2×base={2*base:.3f})"}, fontsize=8)
        grid_v = CGX_.shape[0]
        for iy_ in range(grid_v):
            for ix_ in range(grid_v):
                if op_[iy_, ix_] > 0.05:
                    ax.quiver(CGX_[iy_, ix_], CGY_[iy_, ix_], U_[iy_, ix_], V_[iy_, ix_],
                              angles="xy", scale_units="xy", scale=1.6,
                              color=color, alpha=op_[iy_, ix_] * 0.9,
                              width=0.0035, headwidth=4, headlength=5)
        for z, c in zone_centroids.items():
            ax.text(c[0], c[1], f"{z}\n{GLOSS[z]}", fontsize=10, ha="center", va="center",
                    weight="bold", alpha=0.85,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.85))
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, label="predictability\n(realized hit rate)")
    plt.suptitle(f"Past vs Predicted-Future technology flow ({args.domain}, "
                 f"FIRM × CPC-group, base {args.test_year}, +{args.horizon}y)", fontsize=13)
    out = ROOT / "pnode_patent_runner" / args.out
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
