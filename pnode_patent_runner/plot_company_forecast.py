"""
企業 (仮想クラスタ) × 技術 × 時系列予測 — PI-SDE rollout で未来を可視化。

1. 発明者を CPC pattern で k-means クラスタリング → "仮想企業" 6 つ
2. 各企業の e_org(t) を 2013-2021 から計算 (履歴)
3. PI-SDE drift `-∇Φ(x, t)` を使って 2022-2030 を SDE rollout 予測
4. 天気図地形 + 企業軌跡 (履歴=実線, 予測=点線)
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
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from sklearn.cluster import KMeans

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
LAST_TRAIN_T = 11        # model trained up to t=11 (year 2024)
PREDICT_END_T = 17       # extrapolate to t=17 (year 2030)
TOP_INVENTORS = 200      # use top 200 inventors for clustering
N_VIRTUAL_ORGS = 6       # k-means clusters
GRID_RES = 80

CPC_LABELS = {
    "Y02A": "気候適応", "Y02B": "建物省エネ", "Y02C": "GHG 削減",
    "Y02D": "ICT 省エネ", "Y02E": "クリーンエネ", "Y02P": "製造省エネ",
    "Y02T": "交通省エネ", "Y02W": "廃棄物",
}


def main():
    pat = f"{ROOT}/*{TAG_SUFFIX}/seed_{SEED}/alltime"
    cands = [Path(p) for p in glob.glob(pat) if "_lx" not in Path(p).parent.name]
    out_dir = cands[0]
    out_png = Path("RESULTS/company_forecast_patent_energy.png")

    print("Loading model + data...")
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

    # Load bipartite, get top inventors
    print("Loading bipartite...")
    df = pd.read_csv(BIPARTITE)
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    topic_set = set(topic_names)
    df_filt = df[df["i"].isin(topic_set)]
    df_filt = df_filt[(df_filt["year"] >= YEAR_BASE) & (df_filt["year"] <= 2021)]
    multi_yr = df_filt.groupby("u")["year"].nunique()
    df_active = df_filt[df_filt["u"].isin(multi_yr[multi_yr >= 5].index)]
    top_inv = df_active["u"].value_counts().head(TOP_INVENTORS).index
    df_top = df_active[df_active["u"].isin(top_inv)]
    print(f"  Top {TOP_INVENTORS} inventors selected")

    # Build inventor × CPC count matrix (over all years) for clustering
    inv_cpc = pd.crosstab(df_top["u"], df_top["i"])
    inv_cpc = inv_cpc.reindex(columns=topic_names, fill_value=0)
    # Normalize per inventor (their CPC distribution)
    row_sum = inv_cpc.sum(axis=1).values
    cpc_dist = inv_cpc.values / np.maximum(row_sum[:, None], 1)

    # K-means cluster into virtual orgs
    print(f"K-means {N_VIRTUAL_ORGS} virtual orgs on CPC patterns...")
    km = KMeans(n_clusters=N_VIRTUAL_ORGS, random_state=42, n_init=10)
    org_labels = km.fit_predict(cpc_dist)

    # Find dominant CPC family for each org → label
    org_info = {}
    inv_to_org = dict(zip(inv_cpc.index, org_labels))
    for c in range(N_VIRTUAL_ORGS):
        members = inv_cpc.index[org_labels == c]
        center = km.cluster_centers_[c]
        # top 3 CPC for this org
        top3_idx = np.argsort(-center)[:3]
        top3_codes = [topic_names[i] for i in top3_idx]
        top3_families = list({CPC_LABELS.get(c[:4], "?") for c in top3_codes})
        org_info[c] = {
            "members": list(members),
            "top_cpc": top3_codes,
            "label": "+".join(top3_families[:2]),
            "size": len(members),
        }
        print(f"  Org {c}: {len(members)} members, dominant = {top3_codes} ({top3_families})")

    # Per-year e_org(t) for each virtual org (historical 2013-2021)
    org_traj = {c: {} for c in range(N_VIRTUAL_ORGS)}
    for yr in range(YEAR_BASE, 2022):
        t_idx = yr - YEAR_BASE
        if t_idx >= n_T: continue
        cents_t = centroids[t_idx].numpy()
        sub_yr = df_top[df_top["year"] == yr]
        sub_yr = sub_yr.assign(org=sub_yr["u"].map(inv_to_org))
        for c in range(N_VIRTUAL_ORGS):
            sub_c = sub_yr[sub_yr["org"] == c]
            if len(sub_c) == 0: continue
            counts = sub_c["i"].value_counts()
            total = counts.sum()
            w = np.zeros(n_topics)
            for cpc, n in counts.items():
                j = topic_to_idx.get(cpc)
                if j is not None:
                    w[j] = n / total
            e = (w[:, None] * cents_t).sum(axis=0)
            if np.linalg.norm(e) > 1e-6:
                org_traj[c][yr] = e

    # SDE rollout for future prediction (2022 → 2030)
    print(f"SDE rollout for future prediction (year 2022 → {YEAR_BASE+PREDICT_END_T})...")
    n_sde_steps = 50          # sub-steps per year
    dt = 1.0 / n_sde_steps
    sigma = 0.02              # small noise for visualization
    for c, traj in org_traj.items():
        if not traj: continue
        last_year = max(traj.keys())
        e_curr = torch.tensor(traj[last_year], dtype=torch.float32, device=device)
        t_curr = float(last_year - YEAR_BASE)
        for yr in range(last_year + 1, YEAR_BASE + PREDICT_END_T + 1):
            for _ in range(n_sde_steps):
                with torch.enable_grad():
                    xt = torch.cat([e_curr.unsqueeze(0),
                                    torch.tensor([[t_curr]], device=device)], dim=1).requires_grad_()
                    phi = model._func._pot(xt).squeeze()
                    grad = torch.autograd.grad(phi, xt)[0].squeeze()[:-1]   # ∇_x Φ
                drift = -grad
                e_curr = e_curr + dt * drift + sigma * np.sqrt(dt) * torch.randn_like(e_curr)
                t_curr += dt
            org_traj[c][yr] = e_curr.detach().cpu().numpy()

    # UMAP fit on observation subsample + centroids + ALL trajectory points (historical + predicted)
    print("UMAP fitting...")
    import umap
    rng = np.random.RandomState(42)
    x_all_pts = [v.numpy()[rng.choice(len(v), min(800, len(v)), replace=False)] for v in xp]
    x_all = np.concatenate(x_all_pts)
    cent_all = np.concatenate([centroids[t].numpy() for t in range(n_T)])
    org_pts = []
    org_ptr = {c: [] for c in range(N_VIRTUAL_ORGS)}
    for c, traj in org_traj.items():
        for yr in sorted(traj.keys()):
            org_pts.append(traj[yr])
            org_ptr[c].append(yr)
    org_pts = np.array(org_pts)
    big = np.concatenate([x_all, cent_all, org_pts])
    um = umap.UMAP(n_components=2, n_neighbors=30, random_state=42, transform_seed=42)
    big_2d = um.fit_transform(big)

    n_x = len(x_all)
    n_c = len(cent_all)
    x_2d = big_2d[:n_x]
    cent_2d_all = big_2d[n_x:n_x+n_c]
    org_2d_flat = big_2d[n_x+n_c:]

    # Map back to org → year → 2D
    org_2d = {c: {} for c in range(N_VIRTUAL_ORGS)}
    ptr = 0
    for c in range(N_VIRTUAL_ORGS):
        for yr in org_ptr[c]:
            org_2d[c][yr] = org_2d_flat[ptr]
            ptr += 1

    # Compute Φ field at year 2025 (mid future)
    eval_year_t = 12      # t=12 = 2025
    print(f"Evaluating Φ heatmap at t={eval_year_t} ({YEAR_BASE+eval_year_t})...")
    x_all_dev = torch.tensor(x_all, dtype=torch.float32, device=device)
    t_col = torch.full((x_all_dev.shape[0], 1), float(eval_year_t), device=device)
    with torch.enable_grad():
        xt = torch.cat([x_all_dev, t_col], dim=1).requires_grad_()
        phi_obs = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()
    height_obs = -phi_obs

    # Interpolate on grid
    pad = 0.5
    x_min, x_max = x_2d[:, 0].min() - pad, x_2d[:, 0].max() + pad
    y_min, y_max = x_2d[:, 1].min() - pad, x_2d[:, 1].max() + pad
    gx = np.linspace(x_min, x_max, GRID_RES)
    gy = np.linspace(y_min, y_max, GRID_RES)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    interp_lin = LinearNDInterpolator(x_2d, height_obs)
    interp_nn = NearestNDInterpolator(x_2d, height_obs)
    H = interp_lin(grid_pts)
    nan = np.isnan(H); H[nan] = interp_nn(grid_pts[nan])
    H = H.reshape(GX.shape)

    # Quiver from gradient
    dy_grid = gy[1] - gy[0]; dx_grid = gx[1] - gx[0]
    Hy, Hx = np.gradient(H, dy_grid, dx_grid)
    step = max(1, GRID_RES // 22)
    QX = GX[::step, ::step]; QY = GY[::step, ::step]
    QU = Hx[::step, ::step]; QV = Hy[::step, ::step]
    QM = np.sqrt(QU**2 + QV**2)
    QU_n = QU / np.maximum(QM, 1e-8) * np.sqrt(QM)
    QV_n = QV / np.maximum(QM, 1e-8) * np.sqrt(QM)

    # ═══════════════════════════ FIGURE ═══════════════════════════
    fig, ax = plt.subplots(figsize=(17, 11))

    # Topography
    H_lo, H_hi = np.percentile(H, 3), np.percentile(H, 97)
    H_clip = np.clip(H, H_lo, H_hi)
    cmap = matplotlib.colormaps["RdBu_r"]
    levels = np.linspace(H_lo, H_hi, 40)
    cf = ax.contourf(GX, GY, H_clip, levels=levels, cmap=cmap, alpha=0.9, extend="both")
    cs = ax.contour(GX, GY, H_clip, levels=15, colors="black", linewidths=0.7, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f", colors="#222")

    # Wind
    ax.quiver(QX, QY, QU_n, QV_n, alpha=0.7, color="#222", scale_units="xy", scale=2.5,
              width=0.0035, headwidth=4.5, headlength=6)

    # Topic centroids at year 2025 (t=12)
    if eval_year_t < n_T:
        cent_2d_t = cent_2d_all[eval_year_t * n_topics:(eval_year_t+1) * n_topics]
        g_last = data["growth"][LAST_TRAIN_T].numpy()   # use last observed growth
    else:
        cent_2d_t = cent_2d_all[(n_T-1) * n_topics:n_T * n_topics]
        g_last = data["growth"][LAST_TRAIN_T].numpy()
    g_max = max(abs(g_last.max()), abs(g_last.min())) + 1e-6
    sc = ax.scatter(cent_2d_t[:, 0], cent_2d_t[:, 1], c=g_last, cmap="RdYlGn",
                    vmin=-g_max, vmax=g_max, s=110, edgecolors="black", linewidth=1.0, zorder=5)

    # Annotate top growth + top decline
    order = np.argsort(-g_last)
    for i in order[:3]:
        cat = CPC_LABELS.get(topic_names[i][:4], "")
        ax.annotate(f"★ {topic_names[i]}\n({cat})", cent_2d_t[i],
                    xytext=(10, 8), textcoords="offset points", fontsize=8.5,
                    fontweight="bold", color="#073",
                    bbox=dict(facecolor="white", edgecolor="#073", lw=1,
                              alpha=0.92, boxstyle="round,pad=0.3"))

    # Organization trajectories: solid (historical) → dashed (predicted)
    org_colors = plt.cm.Set1(np.linspace(0, 1, N_VIRTUAL_ORGS))
    legend_lines = []
    for c in range(N_VIRTUAL_ORGS):
        if not org_2d[c]: continue
        yrs = sorted(org_2d[c].keys())
        pts = np.array([org_2d[c][yr] for yr in yrs])
        # split into historical vs predicted
        hist_mask = np.array([yr <= 2021 for yr in yrs])
        pred_mask = np.array([yr > 2021 for yr in yrs])
        # historical solid
        h_pts = pts[hist_mask]
        if len(h_pts) >= 2:
            ax.plot(h_pts[:, 0], h_pts[:, 1], "-", color=org_colors[c],
                    lw=3, alpha=0.95, zorder=6, solid_capstyle="round")
        # predicted dashed with arrow
        p_pts = pts[pred_mask]
        if len(p_pts) >= 1 and len(h_pts) >= 1:
            # connect from last historical to predicted
            seg = np.vstack([h_pts[-1:], p_pts])
            ax.plot(seg[:, 0], seg[:, 1], "--", color=org_colors[c],
                    lw=2.2, alpha=0.85, zorder=6)
            # arrow at end of prediction
            ax.annotate("", xy=p_pts[-1], xytext=p_pts[-2] if len(p_pts) >= 2 else h_pts[-1],
                        arrowprops=dict(arrowstyle="->", color=org_colors[c], lw=2.2))
        # marker: start (historical), present (=2021), future (predicted end)
        if len(h_pts) > 0:
            ax.scatter(h_pts[0, 0], h_pts[0, 1], s=110, color=org_colors[c],
                       edgecolors="white", linewidth=2.0, marker="o", zorder=7)  # start
            ax.scatter(h_pts[-1, 0], h_pts[-1, 1], s=180, color=org_colors[c],
                       edgecolors="black", linewidth=1.8, marker="s", zorder=7)  # present
        if len(p_pts) > 0:
            ax.scatter(p_pts[-1, 0], p_pts[-1, 1], s=250, color=org_colors[c],
                       edgecolors="black", linewidth=1.8, marker="*", zorder=7)  # future
            # label
            label = f"Org-{c}\n({org_info[c]['label']})\n[{org_info[c]['size']} 発明者]"
            ax.annotate(label, p_pts[-1],
                        xytext=(15, 10), textcoords="offset points", fontsize=8.5,
                        color="black", fontweight="bold",
                        bbox=dict(facecolor=org_colors[c], edgecolor="black", lw=1,
                                  alpha=0.7, boxstyle="round,pad=0.3"))
        legend_lines.append(plt.Line2D([], [], color=org_colors[c], lw=3,
                                       label=f"Org-{c}: {org_info[c]['label']}"))

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    # Marker legend
    from matplotlib.lines import Line2D
    marker_legend = [
        Line2D([], [], marker="o", color="w", markerfacecolor="gray", markersize=11,
               markeredgecolor="white", markeredgewidth=2, label="2013 (起点)"),
        Line2D([], [], marker="s", color="w", markerfacecolor="gray", markersize=12,
               markeredgecolor="black", markeredgewidth=1.5, label="2021 (現在)"),
        Line2D([], [], marker="*", color="w", markerfacecolor="gray", markersize=16,
               markeredgecolor="black", markeredgewidth=1.5,
               label=f"{YEAR_BASE+PREDICT_END_T} (PI-SDE 予測)"),
        Line2D([], [], color="black", lw=2.5, label="実履歴 (2013-2021)"),
        Line2D([], [], color="black", lw=2.5, ls="--", label="SDE 予測 (2022-2030)"),
    ]
    ax.legend(handles=marker_legend, loc="lower right", fontsize=9, framealpha=0.95,
              facecolor="white", edgecolor="#444", title="凡例")

    # Colorbars
    cbar = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.01)
    cbar.set_label("成長ポテンシャル (赤=山頂=成長, 青=谷=衰退)", fontsize=10)
    cbar2 = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.022)
    cbar2.set_label("topic 実成長率 g", fontsize=10)

    ax.set_title(f"企業 (仮想 N={N_VIRTUAL_ORGS}) × 技術 × 時系列予測 — Patent Energy Y02\n"
                 f"山頂 = 成長ゾーン / 矢印 = 技術空間の流れ / 履歴 (2013-2021) + 予測 (~{YEAR_BASE+PREDICT_END_T})",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
