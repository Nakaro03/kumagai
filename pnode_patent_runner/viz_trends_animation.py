"""viz_trends_animation.py — animated GIF of technology trends in latent space.

One frame per year. Same fixed latent space (UMAP); bubble size = activity
in that year's window; color = log-ratio vs 5y earlier (red heating, blue
cooling). Year label prominent in corner.

Run:  python pnode_patent_runner/viz_trends_animation.py --domain construction
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.animation import FuncAnimation, PillowWriter

from diagnose_convergence_signal import ROOT
import recommender_firm as R

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "figure.dpi": 100, "savefig.dpi": 130,
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
    df = df[(df.year >= 1995) & (df.year <= 2020)]
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


def activity(df, year, cidx, n_c, window=3):
    sub = df[(df.year > year - window) & (df.year <= year)]
    out = np.zeros(n_c)
    for c, n in sub.groupby("i")["u"].nunique().items():
        if c in cidx:
            out[cidx[c]] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--y0", type=int, default=2001)
    ap.add_argument("--y1", type=int, default=2020)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="viz_trends_animation.gif")
    args = ap.parse_args()
    R.LEVEL = "group"

    df, emb, codes, cidx, inv_c, n_c = setup(args)
    print(f"firms={df.u.nunique()} CPC-groups={n_c}")

    w = R.build_world(df, args.y1, cidx, emb, n_c, 1)
    print("UMAP to 2D (fixed) ...")
    xy = umap.UMAP(n_neighbors=20, min_dist=0.30, n_components=2,
                   random_state=args.seed).fit_transform(w["Cemb"])

    years = list(range(args.y0, args.y1 + 1))
    acts = {y: activity(df, y, cidx, n_c, args.window) for y in years}
    trend = {y: np.log((acts[y] + 1) / (activity(df, y - 5, cidx, n_c, args.window) + 1))
             for y in years}
    max_act = max(a.max() for a in acts.values())

    x_min, x_max = xy[:, 0].min() - 0.8, xy[:, 0].max() + 0.8
    y_min, y_max = xy[:, 1].min() - 0.8, xy[:, 1].max() + 0.8
    zone_pts = defaultdict(list)
    for t in range(n_c):
        zone_pts[inv_c[t][:3]].append(xy[t])
    zones = {z: np.mean(v, 0) for z, v in zone_pts.items() if len(v) >= 4 and z in GLOSS}

    # ---- figure (one panel, will be redrawn per frame) ----
    fig, ax = plt.subplots(figsize=(11, 8))

    def draw_frame(i):
        ax.clear()
        y = years[i]
        norm_act = acts[y] / (max_act + 1e-9)
        sizes = 10 + 420 * np.sqrt(np.clip(norm_act, 0, 1))
        sc_ = ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=trend[y],
                         cmap="RdYlBu_r", vmin=-1.5, vmax=1.5,
                         edgecolors="#222", linewidths=0.5, alpha=0.88)
        # label top 5 by activity in this year
        for t in np.argsort(-acts[y])[:5]:
            ax.annotate(inv_c[t], (xy[t, 0], xy[t, 1]), fontsize=10, weight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec="#a8323f", lw=1.0, alpha=0.92))
        # zone labels
        for z, c in zones.items():
            ax.text(c[0], c[1] - 0.7, z, fontsize=9.5, ha="center", va="center",
                    color="#444", alpha=0.7, weight="bold")
        # year big in corner
        ax.text(0.97, 0.96, f"{y}", transform=ax.transAxes, fontsize=42,
                weight="bold", va="top", ha="right", color="#1F3A52", alpha=0.85)
        ax.text(0.97, 0.84, f"{acts[y].sum():.0f} firm-CPC pairs\n({args.window}y window)",
                transform=ax.transAxes, fontsize=10, va="top", ha="right", color="#444")
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Technology trends in latent space — yearly evolution\n"
                     f"({args.domain}, firm × CPC group)", pad=10, fontsize=12)

    # one-time colorbar
    sm = mpl.cm.ScalarMappable(cmap="RdYlBu_r", norm=mpl.colors.Normalize(-1.5, 1.5))
    cax = fig.add_axes([0.92, 0.18, 0.013, 0.4])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("trend (log ratio vs 5y earlier)", fontsize=10)

    print(f"animating {len(years)} frames @ {args.fps} fps ...")
    anim = FuncAnimation(fig, draw_frame, frames=len(years), interval=1000 // args.fps)
    out = ROOT / "pnode_patent_runner" / args.out
    writer = PillowWriter(fps=args.fps)
    anim.save(out, writer=writer, dpi=110)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
