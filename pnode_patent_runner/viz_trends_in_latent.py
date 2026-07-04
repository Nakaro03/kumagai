"""viz_trends_in_latent.py — TIME-EVOLUTION of technology trends in latent space.

So far we've shown snapshots. Here we show the latent technology landscape
EVOLVING over time:
  - Same 2D latent space (UMAP fixed on cumulative graph)
  - Multiple year snapshots: 2005, 2010, 2015, 2020
  - Bubble size = activity in that year (notable technologies)
  - Color = trend (heating up red / cooling blue) relative to 5y earlier
  - Right panel: "trajectory" view = where technologies grew vs declined over
    the 15-year span, with summary of rising and falling technologies.

Run:  python pnode_patent_runner/viz_trends_in_latent.py --domain construction
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from diagnose_convergence_signal import ROOT
import recommender_firm as R

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 300,
    "axes.linewidth": 0.9, "axes.edgecolor": "#222",
    "axes.spines.right": False, "axes.spines.top": False,
})

GLOSS = {"E01": "roads", "E02": "hydraulic", "E03": "water",
         "E04": "building", "E05": "locks", "E06": "doors",
         "E21": "drilling", "B28": "cement", "B66": "hoisting",
         "C04": "concrete", "F16": "machine elements", "B23": "machine tools"}


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


def activity(df, year, cidx, n_c, window=1):
    """unique firms per CPC in [year-window+1, year]."""
    sub = df[(df.year > year - window) & (df.year <= year)]
    out = np.zeros(n_c)
    for c, n in sub.groupby("i")["u"].nunique().items():
        if c in cidx:
            out[cidx[c]] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--years", type=int, nargs="+", default=[2005, 2010, 2015, 2020])
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="viz_trends_in_latent.png")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed); R.LEVEL = "group"

    df, emb, codes, cidx, inv_c, n_c = setup(args)
    print(f"firms={df.u.nunique()} CPC-groups={n_c}")

    # build a CUMULATIVE bipartite at last year → fix UMAP space
    w = R.build_world(df, max(args.years), cidx, emb, n_c, 1)
    print("UMAP to 2D (fixed) ...")
    xy = umap.UMAP(n_neighbors=20, min_dist=0.30, n_components=2,
                   random_state=args.seed).fit_transform(w["Cemb"])

    # activity per year + trend (vs window-shifted)
    acts = {y: activity(df, y, cidx, n_c, args.window) for y in args.years}
    trend = {y: np.log((acts[y] + 1) / (activity(df, y - 5, cidx, n_c, args.window) + 1))
             for y in args.years}

    x_min, x_max = xy[:, 0].min() - 0.8, xy[:, 0].max() + 0.8
    y_min, y_max = xy[:, 1].min() - 0.8, xy[:, 1].max() + 0.8

    # zone labels (from last-year zone clusters)
    zone_pts = defaultdict(list)
    for t in range(n_c):
        zone_pts[inv_c[t][:3]].append(xy[t])
    zones = {z: np.mean(v, 0) for z, v in zone_pts.items() if len(v) >= 4 and z in GLOSS}

    # ---- figure ----
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.9],
                          height_ratios=[1, 0.05], wspace=0.06, hspace=0.18,
                          left=0.03, right=0.97, top=0.91, bottom=0.05)
    axes = [fig.add_subplot(gs[0, k]) for k in range(4)]
    ax_summary = fig.add_subplot(gs[0, 4])

    max_act = max(a.max() for a in acts.values())
    for k, (ax, y) in enumerate(zip(axes, args.years)):
        norm_act = acts[y] / (max_act + 1e-9)
        sizes = 8 + 360 * np.sqrt(np.clip(norm_act, 0, 1))
        sc_ = ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=trend[y],
                         cmap="RdYlBu_r", vmin=-1.5, vmax=1.5,
                         edgecolors="#222", linewidths=0.4, alpha=0.85)
        # label hot tech in that year
        for t in np.argsort(-acts[y])[:6]:
            ax.annotate(inv_c[t], (xy[t, 0], xy[t, 1]), fontsize=8.5, weight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="#a8323f", lw=0.7, alpha=0.9))
        for z, c in zones.items():
            ax.text(c[0], c[1] - 0.7, z, fontsize=8.5, ha="center", va="center",
                    color="#444", alpha=0.65, weight="bold")
        ax.set_title(f"year {y}\n({acts[y].sum():.0f} firm-CPC pairs in {args.window}y window)",
                     pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.text(0.02, 0.97, chr(ord("a") + k), transform=ax.transAxes,
                fontsize=14, weight="bold", va="top")
        if k == 3:
            cax = fig.add_axes([0.59, 0.965, 0.10, 0.012])
            cb = fig.colorbar(sc_, cax=cax, orientation="horizontal")
            cb.set_label("trend (log ratio vs 5y earlier)", fontsize=8.5)
            cb.ax.tick_params(labelsize=8)

    # ---- summary panel: TOP RISING / TOP FALLING over the whole span ----
    long_trend = np.log((acts[args.years[-1]] + 1) / (acts[args.years[0]] + 1))
    activity_avg = np.mean(list(acts.values()), axis=0)
    # require non-trivial activity
    valid = activity_avg >= 5
    rising = np.argsort(np.where(valid, long_trend, -np.inf))[::-1][:12]
    falling = np.argsort(np.where(valid, long_trend, np.inf))[:8]

    # summary plot: scatter all CPCs with long-trend color
    sizes = 6 + 200 * np.sqrt(np.clip(activity_avg / (activity_avg.max() + 1e-9), 0, 1))
    sc2 = ax_summary.scatter(xy[:, 0], xy[:, 1], s=sizes, c=long_trend,
                             cmap="RdYlBu_r", vmin=-2, vmax=2,
                             edgecolors="#222", linewidths=0.4, alpha=0.7)
    # mark top rising with red star
    for t in rising[:5]:
        ax_summary.scatter([xy[t, 0]], [xy[t, 1]], marker="^", c="#B03A2E",
                           s=130, edgecolors="white", linewidths=1.3, zorder=8)
        ax_summary.annotate(inv_c[t], (xy[t, 0], xy[t, 1] + 0.3), fontsize=8.5,
                            weight="bold", ha="center", color="#7B241C",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="#7B241C", lw=0.7, alpha=0.95))
    for t in falling[:3]:
        ax_summary.scatter([xy[t, 0]], [xy[t, 1]], marker="v", c="#1F4E79",
                           s=120, edgecolors="white", linewidths=1.3, zorder=8)
        ax_summary.annotate(inv_c[t], (xy[t, 0], xy[t, 1] - 0.3), fontsize=8.5,
                            weight="bold", ha="center", color="#1F4E79",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="#1F4E79", lw=0.7, alpha=0.95))
    ax_summary.set_title(f"15-yr trend summary\n"
                         f"({args.years[0]} → {args.years[-1]})", pad=6)
    ax_summary.text(0.02, 0.97, "e", transform=ax_summary.transAxes,
                    fontsize=14, weight="bold", va="top")
    ax_summary.set_xticks([]); ax_summary.set_yticks([])
    ax_summary.set_xlim(x_min, x_max); ax_summary.set_ylim(y_min, y_max)

    # right side text: top rising/falling lists
    fig.text(0.815, 0.86,
             "TOP-12 RISING TECHNOLOGIES\n"
             f"({args.years[0]} → {args.years[-1]}, log ratio)\n"
             "─" * 26, fontsize=9, va="top", family="monospace", color="#7B241C")
    rising_lines = "\n".join(
        f"▲ {inv_c[t]:<10} +{long_trend[t]:.2f}  "
        f"({GLOSS.get(inv_c[t][:3], '')[:14]})" for t in rising)
    fig.text(0.815, 0.83, rising_lines, fontsize=8.5, va="top",
             family="monospace", color="#7B241C")
    fig.text(0.815, 0.40,
             "TOP-8 FALLING TECHNOLOGIES\n"
             "─" * 26, fontsize=9, va="top", family="monospace", color="#1F4E79")
    falling_lines = "\n".join(
        f"▽ {inv_c[t]:<10} {long_trend[t]:+.2f}  "
        f"({GLOSS.get(inv_c[t][:3], '')[:14]})" for t in falling)
    fig.text(0.815, 0.37, falling_lines, fontsize=8.5, va="top",
             family="monospace", color="#1F4E79")

    fig.suptitle("Technology trends in latent space — multi-year evolution "
                 f"({args.domain}, firm × CPC group)", fontsize=13, y=0.96)
    fig.text(0.5, 0.015,
             "Latent space (2D UMAP) is fixed across years; bubble size = activity in window; "
             "color = log activity ratio vs 5y earlier (red = heating up, blue = cooling). "
             "Panel (e) summarises the 15-year trajectory: ▲ rising, ▽ falling.",
             ha="center", va="bottom", fontsize=9, color="#333")
    out = ROOT / "pnode_patent_runner" / args.out
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
