"""
Weather-map style landscape — 技術地形図 + 人流ベクトル + 発明者軌跡.

天気図メタファ:
  - 等高線 (contour)  = Φ 場 (成長ポテンシャル)
  - 矢印 (quiver)     = SDE drift -∇Φ (人/論文/特許が流れる方向)
  - 発明者軌跡         = 実際の流れの観測 (彗星のしっぽ)
"""
from __future__ import annotations

import os, sys, glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

for cand in ["Noto Sans CJK JP", "IPAGothic", "TakaoGothic"]:
    if cand in [f.name for f in font_manager.fontManager.ttflist]:
        plt.rcParams["font.family"] = cand
        break

from src.model import ForwardSDE
from types import SimpleNamespace

DATA_PT = "data/PNode_Patent_Energy_X1_top50/alltime/fate_train.pt"
BIPARTITE = "data/processed/bipartite_energy.csv"
ROOT = "RESULTS/PNode_Patent_Energy_X1_top50"
TAG_SUFFIX = "-x1_v1.0_g0.1_b0.01"
SEED = 42
YEAR_BASE = 2013
LAST_T = 11
TIME_RANGE = list(range(2013, 2022))
TOP_INVENTORS = 12        # focus on top 12 for clarity
GRID_RES = 80             # grid density for contour/quiver

CPC_LABELS = {
    "Y02A": "気候適応", "Y02B": "建物省エネ", "Y02C": "GHG 削減",
    "Y02D": "ICT 省エネ", "Y02E": "クリーンエネルギー", "Y02P": "製造省エネ",
    "Y02T": "交通省エネ", "Y02W": "廃棄物",
}


def main():
    pat = f"{ROOT}/*{TAG_SUFFIX}/seed_{SEED}/alltime"
    cands = [Path(p) for p in glob.glob(pat) if "_lx" not in Path(p).parent.name]
    out_dir = cands[0]
    out_png = Path("RESULTS/weather_map_patent_energy.png")

    print(f"Loading model + data...")
    data = torch.load(DATA_PT, weights_only=False)
    xp = data["xp"]
    y = data["y"]
    topic_names = data["topic_names"]
    centroids = data["centroids"]
    n_topics = data["n_topics"]
    n_T = len(y)
    topic_to_idx = {tn: j for j, tn in enumerate(topic_names)}

    config = SimpleNamespace(**torch.load(out_dir / "config.pt", weights_only=False))
    config.x_dim = xp[0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardSDE(config).to(device)
    ckpts = sorted(glob.glob(str(out_dir / "train.epoch_*.pt")))
    ckpt = ckpts[-1] if ckpts else str(out_dir / "train.best.pt")
    state = torch.load(ckpt, weights_only=False, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Inventor data
    print("Loading bipartite...")
    df = pd.read_csv(BIPARTITE)
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    topic_set = set(topic_names)
    df_filt = df[df["i"].isin(topic_set)]
    df_filt = df_filt[(df_filt["year"] >= TIME_RANGE[0]) & (df_filt["year"] <= TIME_RANGE[-1])]
    multi_yr = df_filt.groupby("u")["year"].nunique()
    df_active = df_filt[df_filt["u"].isin(multi_yr[multi_yr >= 5].index)]
    top = df_active["u"].value_counts().head(TOP_INVENTORS).index
    df_top = df_active[df_active["u"].isin(top)]

    # Build inventor trajectories
    inventor_emb = {}
    for inv in top:
        inventor_emb[inv] = {}
        sub = df_top[df_top["u"] == inv]
        for yr in TIME_RANGE:
            sub_yr = sub[sub["year"] == yr]
            if len(sub_yr) == 0: continue
            counts = sub_yr["i"].value_counts()
            total = counts.sum()
            t_idx = yr - YEAR_BASE
            if not (0 <= t_idx < n_T): continue
            cents_t = centroids[t_idx].numpy()
            w_vec = np.zeros(n_topics)
            for cpc, n in counts.items():
                j = topic_to_idx.get(cpc)
                if j is not None:
                    w_vec[j] = n / total
            e = (w_vec[:, None] * cents_t).sum(axis=0)
            if np.linalg.norm(e) > 1e-6:
                inventor_emb[inv][yr] = e
    inventor_emb = {k: v for k, v in inventor_emb.items() if len(v) >= 4}
    print(f"  {len(inventor_emb)} inventors with ≥4 years")

    # UMAP fit on subsample of obs + centroids + inventor embeddings
    print("UMAP fitting...")
    import umap
    rng = np.random.RandomState(42)
    x_all_pts = [v.numpy()[rng.choice(len(v), min(800, len(v)), replace=False)] for v in xp]
    x_all = np.concatenate(x_all_pts)
    cent_all = np.concatenate([centroids[t].numpy() for t in range(n_T)])
    inv_all = np.array([e for ev in inventor_emb.values() for e in ev.values()])
    big = np.concatenate([x_all, cent_all, inv_all])
    um = umap.UMAP(n_components=2, n_neighbors=30, random_state=42, transform_seed=42)
    big_2d = um.fit_transform(big)

    n_x = len(x_all)
    n_c = len(cent_all)
    x_2d = big_2d[:n_x]
    cent_2d_all = big_2d[n_x:n_x+n_c]
    inv_2d_flat = big_2d[n_x+n_c:]

    # Map inventor → 2D + year
    inv_2d = {}
    iflat = 0
    for inv, ev in inventor_emb.items():
        inv_2d[inv] = {}
        for yr in sorted(ev.keys()):
            inv_2d[inv][yr] = inv_2d_flat[iflat]
            iflat += 1

    # Compute Φ at all observed points for last year
    print(f"Evaluating Φ at t=last...")
    x_all_dev = torch.tensor(x_all, dtype=torch.float32, device=device)
    t_col = torch.full((x_all_dev.shape[0], 1), float(y[LAST_T]), device=device)
    with torch.enable_grad():
        xt = torch.cat([x_all_dev, t_col], dim=1).requires_grad_()
        phi_obs = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()

    # We want "growing = peak", so invert sign for visualization (height = -Φ)
    height_obs = -phi_obs

    # Grid in UMAP 2D
    pad = 0.5
    x_min, x_max = x_2d[:, 0].min() - pad, x_2d[:, 0].max() + pad
    y_min, y_max = x_2d[:, 1].min() - pad, x_2d[:, 1].max() + pad
    gx = np.linspace(x_min, x_max, GRID_RES)
    gy = np.linspace(y_min, y_max, GRID_RES)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

    # Interpolate height field on grid (linear + nearest-fallback)
    print(f"Interpolating Φ on {GRID_RES}x{GRID_RES} grid...")
    interp_lin = LinearNDInterpolator(x_2d, height_obs)
    interp_nn = NearestNDInterpolator(x_2d, height_obs)
    H = interp_lin(grid_pts)
    nan_mask = np.isnan(H)
    H[nan_mask] = interp_nn(grid_pts[nan_mask])
    H = H.reshape(GX.shape)

    # Compute gradient (∇ height = -∇Φ) → "wind toward peaks"
    dy_grid = (gy[1] - gy[0])
    dx_grid = (gx[1] - gx[0])
    Hy, Hx = np.gradient(H, dy_grid, dx_grid)
    # Wind direction is along ∇H (uphill, toward growth peaks)
    # Normalize for visual clarity
    M = np.sqrt(Hx**2 + Hy**2)
    M_safe = M + 1e-8

    # Subsample for quiver
    step = max(1, GRID_RES // 22)
    QX = GX[::step, ::step]
    QY = GY[::step, ::step]
    QU = Hx[::step, ::step]
    QV = Hy[::step, ::step]
    QM = M[::step, ::step]

    # ───────────────────────────── FIGURE ─────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 10))

    # Stronger contrast for "weather map" feel — emphasis on peaks vs valleys
    # Use percentiles to clip outliers and enhance dynamic range
    H_lo = np.percentile(H, 3)
    H_hi = np.percentile(H, 97)
    H_clip = np.clip(H, H_lo, H_hi)

    # Atmospheric pressure-like cmap: deep blue (low pressure / valley) → white → red (high)
    # But we want "growth = peak" so high values should be warm
    cmap = matplotlib.colormaps["RdBu_r"]   # blue=cold/decline → red=hot/growing
    levels = np.linspace(H_lo, H_hi, 40)
    cf = ax.contourf(GX, GY, H_clip, levels=levels, cmap=cmap, alpha=0.92, extend="both")

    # Isolines emphasized
    cs = ax.contour(GX, GY, H_clip, levels=15, colors="black", linewidths=0.7, alpha=0.55)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f", colors="#222")

    # Wind / drift quiver — bigger, more dramatic
    # Normalize length for clearer direction visualization
    M_norm = np.maximum(QM, 1e-8)
    QU_n = QU / M_norm * np.sqrt(QM)   # length proportional to sqrt(strength)
    QV_n = QV / M_norm * np.sqrt(QM)
    q = ax.quiver(QX, QY, QU_n, QV_n, QM, cmap="Greys",
                  alpha=0.85, scale_units="xy", scale=2.2,
                  width=0.004, headwidth=4.5, headlength=6,
                  edgecolor="black", linewidth=0.3)

    # Topic centroids (last year) — circular markers
    cent_2d_last = cent_2d_all[(LAST_T) * n_topics:(LAST_T+1) * n_topics]
    growth_last = data["growth"][LAST_T].numpy()
    # Mark by growth
    g_max = max(abs(growth_last.max()), abs(growth_last.min())) + 1e-6
    sc = ax.scatter(cent_2d_last[:, 0], cent_2d_last[:, 1], c=growth_last,
                    cmap="RdYlGn", vmin=-g_max, vmax=g_max,
                    s=120, edgecolors="black", linewidth=1.0, zorder=5)

    # Label top growing + top declining topics
    order = np.argsort(-growth_last)
    for i in order[:3]:
        cat = CPC_LABELS.get(topic_names[i][:4], "")
        ax.annotate(f"↑{topic_names[i]}\n({cat})", cent_2d_last[i],
                    xytext=(10, 8), textcoords="offset points", fontsize=8,
                    fontweight="bold", color="#073",
                    bbox=dict(facecolor="white", edgecolor="#073", lw=1,
                              alpha=0.9, boxstyle="round,pad=0.25"))
    for i in order[-3:]:
        cat = CPC_LABELS.get(topic_names[i][:4], "")
        ax.annotate(f"↓{topic_names[i]}\n({cat})", cent_2d_last[i],
                    xytext=(10, -22), textcoords="offset points", fontsize=8,
                    color="#a00",
                    bbox=dict(facecolor="white", edgecolor="#a00", lw=1,
                              alpha=0.9, boxstyle="round,pad=0.25"))

    # Inventor trajectories (comet style with year-based color)
    for ci, (inv, traj) in enumerate(inv_2d.items()):
        yrs = sorted(traj.keys())
        pts = np.array([traj[yr] for yr in yrs])
        # comet: rainbow tail by year
        for i in range(len(pts) - 1):
            frac = i / max(1, len(pts) - 1)
            c = plt.cm.cool(frac)
            ax.plot(pts[i:i+2, 0], pts[i:i+2, 1], "-", color=c,
                    lw=1.5 + 2.5 * frac,
                    alpha=0.75 + 0.2 * frac, zorder=6,
                    solid_capstyle="round",
                    path_effects=None)
        # head circle (current = end)
        ax.scatter(pts[-1, 0], pts[-1, 1], color="#003c80", s=80,
                   edgecolors="white", linewidths=1.5, zorder=7)
        # name
        short = inv.replace("fl:", "").replace("ln:", "").replace("_", "")[:14]
        ax.annotate(short, (pts[-1, 0], pts[-1, 1]),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=7, color="#003c80", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="#003c80",
                              lw=0.6, alpha=0.9, boxstyle="round,pad=0.2"))

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    # Colorbars
    cbar = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.01)
    cbar.set_label("成長ポテンシャル (赤 = 山頂 = 成長中, 青 = 谷 = 衰退)", fontsize=10)
    cbar2 = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.025)
    cbar2.set_label("topic 実成長率 g (緑 = 高成長)", fontsize=10)

    # Legend for trajectory color
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=plt.cm.cool(0.0), lw=3, label="2013 (起点)"),
        Line2D([0], [0], color=plt.cm.cool(0.5), lw=3, label="~2017"),
        Line2D([0], [0], color=plt.cm.cool(1.0), lw=3, label="2021 (最新)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#003c80",
               markersize=10, markeredgecolor="white", label="発明者 現在地"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9,
              frameon=True, facecolor="white", edgecolor="#888")

    # Title & legend
    ax.set_title(f"技術地形 天気図 — Patent Energy CPC Y02, year {YEAR_BASE+LAST_T}\n"
                 "赤い山 = 成長技術ゾーン / 矢印 = 技術空間の流れ (-∇Φ) / 軌跡 = Top {} 発明者の DNA 移動 (2013→2021)".format(len(inv_2d)),
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
