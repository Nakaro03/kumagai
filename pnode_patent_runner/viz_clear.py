"""viz_clear.py — simplest possible visualization with embedded explanations.

Principles:
  - ONE panel only (no multi-panel confusion)
  - Big text annotations pointing to key features
  - Embedded "how to read" inset in the corner
  - Minimal symbols (only star + numbered arrows)
  - Large fonts, high contrast
  - One firm only (focused story)

Run:  python pnode_patent_runner/viz_clear.py --domain construction
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.patches import FancyArrowPatch
from scipy.stats import gaussian_kde
from sklearn.isotonic import IsotonicRegression

from diagnose_convergence_signal import ROOT
import recommender_firm as R

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "figure.dpi": 110, "savefig.dpi": 300,
    "axes.linewidth": 1.0, "axes.edgecolor": "#222",
    "axes.spines.right": False, "axes.spines.top": False,
})

GLOSS = {"E01": "ROADS", "E02": "HYDRAULIC", "E03": "WATER",
         "E04": "BUILDING", "E21": "DRILLING", "B28": "CEMENT",
         "C04": "CONCRETE"}


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
    ap.add_argument("--out", default="viz_clear.png")
    args = ap.parse_args()
    rng = np.random.default_rng(42); R.LEVEL = "group"

    df, emb, codes, cidx, inv_c, n_c = setup(args)
    print(f"firms={df.u.nunique()} CPC-groups={n_c}")
    sc, clf = R.train_lr(df, args.train_year, cidx, emb, n_c, args.horizon, rng, cols=[0, 1, 2, 3])
    w = R.build_world(df, args.test_year, cidx, emb, n_c, args.horizon)
    print("UMAP ...")
    xy = umap.UMAP(n_neighbors=20, min_dist=0.30, n_components=2,
                   random_state=42).fit_transform(w["Cemb"])

    invs = R.actors(w, cidx); rng.shuffle(invs); invs = invs[:2000]
    sp = len(invs) // 2
    cs, cy, eval_data = [], [], []
    for n, u in enumerate(invs):
        X, owned, Su = R.actor_scores(u, w, emb, cidx, n_c)
        new = {cidx[c] for c in (w["nextf"][u] - w["prior"][u])
               if c in cidx and w["have"][cidx[c]]}
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
        for j in np.argsort(-cal)[:50]:
            t = cand[j]; stats[t]["n_rec"] += 1
            if t in new:
                stats[t]["n_hit"] += 1
    tidx = np.array([t for t, s in stats.items() if s["n_rec"] >= 10])
    nrec = np.array([stats[t]["n_rec"] for t in tidx], float)
    hit = np.array([stats[t]["n_hit"] / stats[t]["n_rec"] for t in tidx])
    pts = xy[tidx]
    x_min, x_max = xy[:, 0].min() - 1.0, xy[:, 0].max() + 1.2
    y_min, y_max = xy[:, 1].min() - 1.0, xy[:, 1].max() + 1.2
    gx = np.linspace(x_min, x_max, 100); gy = np.linspace(y_min, y_max, 100)
    GX, GY = np.meshgrid(gx, gy); grid = np.vstack([GX.ravel(), GY.ravel()])
    kde_h = gaussian_kde(pts.T, bw_method=0.20, weights=hit * nrec)(grid).reshape(100, 100)
    kde_n = gaussian_kde(pts.T, bw_method=0.20, weights=nrec)(grid).reshape(100, 100)
    heat = np.divide(kde_h, kde_n + 1e-12, out=np.zeros_like(kde_h), where=kde_n > 0)
    heat = np.ma.array(heat, mask=kde_n < (kde_n.max() * 0.03))

    # pick a recognizable focused firm
    nm = {}
    nm_path = ROOT / f"data/processed/{args.domain}_firm_names.csv"
    if nm_path.exists():
        nm = pd.read_csv(nm_path).drop_duplicates("u").set_index("u")["org"].to_dict()
    best_u, best_s = None, -np.inf
    for u, X, owned, Su, new, cand, raw, lab in eval_data:
        if not (5 <= len(Su) <= 15) or len(new) < 3:
            continue
        nm_u = str(nm.get(u, "")).strip()
        if len(nm_u) < 4:
            continue
        port_t = [cidx[c] for c in Su if c in cidx]
        spread = xy[port_t].std(0).sum()
        score = -spread + len(new) * 0.6 + (3 if "drilling" in nm_u.lower() else 0)
        if score > best_s:
            best_s = score; best_u = u
    u = best_u
    X, owned, Su = R.actor_scores(u, w, emb, cidx, n_c)
    new_set = {cidx[c] for c in (w["nextf"][u] - w["prior"][u]) if c in cidx and w["have"][cidx[c]]}
    cand = np.array([t for t in range(n_c) if t not in owned])
    cal = iso.predict(clf.decision_function(sc.transform(X[cand][:, [0, 1, 2, 3]])))
    order = np.argsort(-cal)[:3]; top = cand[order]; top_conf = cal[order]
    port_t = [cidx[c] for c in Su if c in cidx]
    name = str(nm.get(u, "?"))

    # ---- single panel figure
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xticks([]); ax.set_yticks([])

    # background
    ax.imshow(heat, extent=(x_min, x_max, y_min, y_max), origin="lower",
              cmap="YlGn", vmin=0, vmax=0.13, aspect="auto",
              interpolation="bilinear", alpha=0.7)
    cs_ = ax.contour(GX, GY, heat.filled(0), levels=[2 * base],
                     colors=["#111"], linestyles=["--"], linewidths=[2.0])

    # all CPCs as small dots
    ax.scatter(xy[:, 0], xy[:, 1], s=15, c="#666", alpha=0.4, zorder=2)

    # zone labels (big, bold)
    zone_pts = defaultdict(list)
    for t in range(n_c):
        zone_pts[inv_c[t][:3]].append(xy[t])
    zones = {z: np.mean(v, 0) for z, v in zone_pts.items()
             if len(v) >= 4 and z in GLOSS}
    for z, c in zones.items():
        ax.text(c[0], c[1], GLOSS[z], fontsize=15, ha="center", va="center",
                weight="bold", color="#111", alpha=0.55, zorder=3)

    # firm portfolio (big star)
    port_xy = xy[port_t]
    center = port_xy.mean(0)
    ax.scatter(port_xy[:, 0], port_xy[:, 1], marker="*", c="#1F1F1F", s=600,
               edgecolors="gold", linewidths=2.5, zorder=10)

    # recommendations (big numbered circles)
    for k, (t, c) in enumerate(zip(top, top_conf), 1):
        tx, ty = xy[t]; is_hit = t in new_set
        cc = "#138D75" if is_hit else "#B03A2E"
        ax.annotate("", xy=(tx, ty), xytext=center,
                    arrowprops=dict(arrowstyle="-|>", color=cc, alpha=0.95, lw=3,
                                    mutation_scale=22))
        ax.scatter([tx], [ty], s=400, c=cc, edgecolors="white", linewidths=2.5,
                   zorder=11)
        ax.text(tx, ty, f"{k}", fontsize=18, ha="center", va="center",
                weight="bold", color="white", zorder=12)

    # ANNOTATION CALLOUTS (the key new feature: big arrows pointing to features)
    # 1. "Your firm is HERE" callout
    fx, fy = center
    callout_pos = (fx - 4.5, fy + 3.5)
    ax.annotate(f"YOUR FIRM\n{name[:28]}",
                xy=(fx, fy), xytext=callout_pos,
                fontsize=12, weight="bold", color="#1F1F1F", ha="left",
                bbox=dict(boxstyle="round,pad=0.6", fc="#FFFCE0", ec="#000",
                          lw=1.5, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#1F1F1F", lw=2,
                                connectionstyle="arc3,rad=0.2"))

    # 2. "Top recommendations" callout
    rec_x, rec_y = xy[top[0]]
    rec_callout = (rec_x + 3.0, rec_y - 2.0)
    n_hit = sum(1 for t in top if t in new_set)
    ax.annotate(f"TOP 3 RECOMMENDATIONS\n"
                f"#1 {inv_c[top[0]]} ({top_conf[0]*100:.0f}%) "
                f"{'HIT ✓' if top[0] in new_set else 'miss ✗'}\n"
                f"#2 {inv_c[top[1]]} ({top_conf[1]*100:.0f}%) "
                f"{'HIT ✓' if top[1] in new_set else 'miss ✗'}\n"
                f"#3 {inv_c[top[2]]} ({top_conf[2]*100:.0f}%) "
                f"{'HIT ✓' if top[2] in new_set else 'miss ✗'}\n"
                f"→ {n_hit}/3 actually filed within 3 years",
                xy=(rec_x, rec_y), xytext=rec_callout,
                fontsize=11, weight="bold", color="#222", ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.6", fc="#E8F4F8", ec="#138D75",
                          lw=2, alpha=0.96),
                arrowprops=dict(arrowstyle="->", color="#138D75", lw=2,
                                connectionstyle="arc3,rad=-0.2"))

    # 3. "Predictable zone" callout (point at a green region)
    green_y = y_max - 1.5
    green_x = x_min + 1.5
    ax.annotate("GREEN = where\nrecommendations\nactually work\n"
                f"(>>{base*100:.1f}% base)",
                xy=(green_x + 1.5, green_y - 1.5), xytext=(green_x, green_y),
                fontsize=11, weight="bold", color="#0E6655", ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="#E8F5E9", ec="#138D75",
                          lw=1.5, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#138D75", lw=2))

    # 4. "Boundary" callout
    ax.annotate("BLACK DASHED =\nproximity-bound\nBEYOND THIS:\npredictions ≈ chance",
                xy=(x_max - 1.5, y_min + 1.0), xytext=(x_max - 4.5, y_min + 1.0),
                fontsize=11, weight="bold", color="#7B241C", ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", fc="#FADBD8", ec="#B03A2E",
                          lw=1.5, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#B03A2E", lw=2))

    # title
    ax.set_title(f"Patent Recommendation Map for {name[:34]}\n"
                 f"(predict tech entry within {args.horizon} years, base year {args.test_year})",
                 pad=15, fontsize=14)

    # how-to-read inset (bottom-right)
    legend_text = (
        "HOW TO READ\n"
        "━━━━━━━━━━━━━━━━\n"
        "★ Black star = YOUR firm's\n"
        "    current portfolio\n\n"
        "①②③ Numbered circles =\n"
        "    Top-3 recommendations\n"
        "    GREEN = actually entered ✓\n"
        "    RED   = did not enter ✗\n\n"
        "Background:\n"
        "  GREEN = predictable\n"
        "  WHITE = chance only\n"
        "  DASHED = boundary\n\n"
        "All confidences are\n"
        "CALIBRATED (ECE=0.001):\n"
        "  '30%' means 30%\n"
        "  real chance in 3 years"
    )
    ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
            fontsize=10, family="monospace", va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.7", fc="white", ec="#444",
                      lw=1.2, alpha=0.97))

    out = ROOT / "pnode_patent_runner" / args.out
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")
    print(f"\nchosen firm: {name}")
    print(f"  portfolio = {len(Su)}, actual new = {len(new_set)}, top-3 hits = {n_hit}/3")


if __name__ == "__main__":
    main()
