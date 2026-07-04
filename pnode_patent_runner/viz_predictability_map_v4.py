"""viz_predictability_map_v4.py — Pattern-3 + LEARNED VECTOR FIELD overlay.

LGNSDE / X3-clean style: a learned drift vector field on the latent landscape that
visualizes where technologies are HEADING. Combined with the predictability map,
the result is:
  - background heatmap = WHERE predictions work (green) vs proximity-bound (red)
  - arrows = WHICH WAY technology is flowing (firm-migration field)
  - arrow opacity = field reliability (low data density -> faded = honest about
                    where we DON'T know the direction)

We do NOT train a full LGNSDE; we derive the field empirically from observed firm
transitions: every (existing CPC) -> (newly entered CPC) pair for every firm
contributes a flow vector. Field at each grid point = density-weighted mean of
nearby flows. Faded arrows = low-data zones (epistemic-like uncertainty).

Run:  python pnode_patent_runner/viz_predictability_map_v4.py --domain construction
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
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="viz_predictability_map_v4.png")
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

    # eval loop (predictability stats)
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

    # predictability KDE (as v3)
    stats = defaultdict(lambda: {"n_rec": 0, "n_hit": 0})
    for u, X, owned, Su, new, cand, raw, lab in eval_data:
        cal = iso.predict(raw)
        for j in np.argsort(-cal)[:args.topk]:
            t = cand[j]; stats[t]["n_rec"] += 1
            if t in new:
                stats[t]["n_hit"] += 1
    tidx = np.array([t for t, s in stats.items() if s["n_rec"] >= 10])
    if len(tidx) == 0:
        print("no CPCs with enough recs"); return
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
    mask = kde_n < (kde_n.max() * 0.03); heat = np.ma.array(heat, mask=mask)

    # ---- VECTOR FIELD from observed firm transitions ----
    # For each eval firm: for each (old portfolio CPC) -> (newly entered CPC),
    # produce a flow vector at the midpoint
    flow_xy, flow_uv, flow_w = [], [], []
    for u, X, owned, Su, new_set, cand, raw, lab in eval_data:
        port = [cidx[c] for c in Su if c in cidx]
        port_xy = xy[port]
        if not port or not new_set:
            continue
        for new_t in new_set:
            target = xy[new_t]
            # connect each portfolio item to the new entry (firm "migration")
            for p in port_xy:
                mid = (p + target) / 2
                vec = target - p
                nv = np.linalg.norm(vec)
                if nv < 1e-6:
                    continue
                flow_xy.append(mid); flow_uv.append(vec / nv); flow_w.append(1.0)
    flow_xy = np.array(flow_xy); flow_uv = np.array(flow_uv); flow_w = np.array(flow_w)
    print(f"flow vectors: {len(flow_xy)}")

    # bin into grid_v x grid_v cells, average direction + count
    grid_v = 22
    bx = np.linspace(x_min, x_max, grid_v + 1)
    by = np.linspace(y_min, y_max, grid_v + 1)
    ix = np.clip(np.searchsorted(bx, flow_xy[:, 0]) - 1, 0, grid_v - 1)
    iy = np.clip(np.searchsorted(by, flow_xy[:, 1]) - 1, 0, grid_v - 1)
    U = np.zeros((grid_v, grid_v)); V = np.zeros((grid_v, grid_v)); C = np.zeros((grid_v, grid_v))
    for k in range(len(flow_xy)):
        U[iy[k], ix[k]] += flow_uv[k, 0] * flow_w[k]
        V[iy[k], ix[k]] += flow_uv[k, 1] * flow_w[k]
        C[iy[k], ix[k]] += flow_w[k]
    cx = (bx[:-1] + bx[1:]) / 2; cy_ = (by[:-1] + by[1:]) / 2
    CGX, CGY = np.meshgrid(cx, cy_)
    valid = C > 8
    U = np.divide(U, C, out=np.zeros_like(U), where=C > 0)
    V = np.divide(V, C, out=np.zeros_like(V), where=C > 0)
    norm = np.sqrt(U ** 2 + V ** 2)
    norm[norm < 1e-9] = 1
    U /= norm; V /= norm
    # opacity: more flows = clearer arrow (epistemic-like: low density = unreliable)
    op = np.clip(C / C.max() * 1.3, 0.0, 1.0)
    op[~valid] = 0

    # pick a firm
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
    u = best_u
    X, owned, Su = R.actor_scores(u, w, emb, cidx, n_c)
    new_set = {cidx[c] for c in (w["nextf"][u] - w["prior"][u]) if c in cidx and w["have"][cidx[c]]}
    cand = np.array([t for t in range(n_c) if t not in owned])
    cal = iso.predict(clf.decision_function(sc.transform(X[cand][:, [0, 1, 2, 3]])))
    order = np.argsort(-cal)[:5]; top = cand[order]; top_conf = cal[order]
    port_t = [cidx[c] for c in Su if c in cidx]

    # zone labels
    zone_pts = defaultdict(list)
    for t in range(n_c):
        zone_pts[inv_c[t][:3]].append(xy[t])
    zone_centroids = {z: np.mean(v, 0) for z, v in zone_pts.items()
                      if len(v) >= 4 and z in GLOSS}

    # ---- figure
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 4, width_ratios=[3, 3, 3, 2])
    ax = fig.add_subplot(gs[0, :3])

    im = ax.imshow(heat, extent=(x_min, x_max, y_min, y_max), origin="lower",
                   cmap="YlGn", vmin=0, vmax=0.10, aspect="auto", alpha=0.85)
    cs_ = ax.contour(GX, GY, heat.filled(0), levels=[2 * base],
                     colors=["black"], linestyles=["--"], linewidths=[1.6])
    ax.clabel(cs_, fmt={2 * base: f"boundary 2×base={2*base:.3f}"}, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01, label="predictability\n(realized hit rate)")

    # vector field — arrows colored by opacity (= reliability)
    for iy_ in range(grid_v):
        for ix_ in range(grid_v):
            if op[iy_, ix_] > 0.05:
                ax.quiver(CGX[iy_, ix_], CGY[iy_, ix_], U[iy_, ix_], V[iy_, ix_],
                          angles="xy", scale_units="xy", scale=1.8,
                          color="darkblue", alpha=op[iy_, ix_] * 0.85,
                          width=0.0030, headwidth=4, headlength=5)

    for z, c in zone_centroids.items():
        ax.text(c[0], c[1], f"{z}\n{GLOSS[z]}", fontsize=10, ha="center", va="center",
                weight="bold", alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.85))

    # firm overlay
    ax.scatter(xy[port_t, 0], xy[port_t, 1], marker="*", c="black", s=260, zorder=8,
               edgecolors="yellow", linewidths=1.8,
               label=f"★ portfolio: {str(nm.get(u,'?'))[:38]}")
    center = xy[port_t].mean(0)
    for k, (t, c) in enumerate(zip(top, top_conf), 1):
        tx, ty = xy[t]; is_hit = t in new_set
        cc = "limegreen" if is_hit else "crimson"
        ax.annotate("", xy=(tx, ty), xytext=center,
                    arrowprops=dict(arrowstyle="->", color=cc, alpha=0.95, lw=2.5))
        ax.scatter([tx], [ty], s=170, c=cc, edgecolors="white", linewidths=1.5, zorder=9)
        ax.text(tx, ty + 0.22, f"{k}", fontsize=12, ha="center", va="bottom",
                weight="bold", color=cc,
                bbox=dict(boxstyle="circle,pad=0.20", fc="white", ec=cc, lw=1.8))

    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax.set_title("Predictability map + learned vector field\n"
                 "(arrows = firm-migration direction; opacity = reliability of the field)",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # right panel: rec table
    ax2 = fig.add_subplot(gs[0, 3]); ax2.axis("off")
    name = str(nm.get(u, "?"))[:34]
    lines = [f"FIRM: {name}",
             f"Portfolio size: {len(Su)} groups",
             f"Actually entered (3y): {len(new_set)}",
             "",
             "Top-5 recommendations:",
             "─────────────────────",
             "# | CPC      | conf  | hit"]
    for k, (t, c) in enumerate(zip(top, top_conf), 1):
        lab = "✓" if t in new_set else "✗"
        lines.append(f"{k} | {inv_c[t]:<8} | {c*100:4.1f}% |  {lab}")
    n_hit = sum(1 for t in top if t in new_set)
    lines += ["─────────────────────",
              f"Hits in top-5: {n_hit}/5", "",
              f"base rate ≈ {base:.3f}",
              f"= 1 per {int(1/base)} candidates",
              "",
              "VECTOR FIELD (blue):",
              "  empirical firm-migration",
              "  direction at each zone.",
              "  Faded = low-data area =",
              "  direction unreliable.",
              "",
              "PREDICTABILITY (green):",
              "  green = recommendations",
              "  work; outside dashed = ",
              "  proximity-bound chance."]
    ax2.text(0, 0.97, "\n".join(lines), family="monospace", fontsize=9, va="top")

    plt.suptitle(f"Technology trend prediction with vector field & predictability "
                 f"({args.domain}, FIRM × CPC-group, base {args.test_year}, +{args.horizon}y)",
                 fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = ROOT / "pnode_patent_runner" / args.out
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
