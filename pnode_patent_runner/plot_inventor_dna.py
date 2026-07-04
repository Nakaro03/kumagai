"""
Phase 1 Quick Win: Inventor DNA on Patent Energy landscape.

1. bipartite_energy.csv から inventor-CPC edges を読み込み
2. 我々の 50 topics に絞る (Y02 family の top50)
3. Top-N の prolific inventor を抽出
4. 各 inventor × 各年 で CPC distribution → e_i(t) = Σ_j w_{ij}(t) c_j(t)
5. 学習済み Φ_θ で各 e_i(t) を評価
6. UMAP に landscape を描画 → inventor trajectory を上書き
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
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib import font_manager

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
YEAR_BASE = 2013          # Patent Energy timepoint t=0
LAST_T = 11               # 2024
TIME_RANGE = list(range(2013, 2022))  # bipartite data ends 2021
TOP_INVENTORS = 30        # how many inventors to track


def main():
    # Load model + data
    pat = f"{ROOT}/*{TAG_SUFFIX}/seed_{SEED}/alltime"
    cands = [Path(p) for p in glob.glob(pat) if "_lx" not in Path(p).parent.name]
    out_dir = cands[0]
    out_png = Path("RESULTS/inventor_dna_patent_energy.png")

    print(f"Loading {DATA_PT}...")
    data = torch.load(DATA_PT, weights_only=False)
    xp = data["xp"]
    y  = data["y"]
    topic_names = data["topic_names"]   # CPC codes
    centroids = data["centroids"]       # list of (n_topics, D)
    n_topics = data["n_topics"]
    n_T = len(y)

    topic_set = set(topic_names)
    topic_to_idx = {tn: j for j, tn in enumerate(topic_names)}
    print(f"  {n_topics} topics, T={n_T}")

    config = SimpleNamespace(**torch.load(out_dir / "config.pt", weights_only=False))
    config.x_dim = xp[0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardSDE(config).to(device)
    ckpts = sorted(glob.glob(str(out_dir / "train.epoch_*.pt")))
    ckpt = ckpts[-1] if ckpts else str(out_dir / "train.best.pt")
    state = torch.load(ckpt, weights_only=False, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Loaded model: {ckpt}")

    # Load bipartite
    print(f"Loading {BIPARTITE} (large file, ~1.7M rows)...")
    df = pd.read_csv(BIPARTITE)
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    df_filt = df[df["i"].isin(topic_set)]
    df_filt = df_filt[(df_filt["year"] >= TIME_RANGE[0]) & (df_filt["year"] <= TIME_RANGE[-1])]
    print(f"  filtered: {len(df_filt)} rows (Y02 top50 × 2013-2021)")
    print(f"  unique inventors in filter: {df_filt['u'].nunique()}")

    # Top-N most prolific inventors who are active across multiple years
    multi_yr = df_filt.groupby("u")["year"].nunique()
    multi_yr_inv = multi_yr[multi_yr >= 3].index  # active in 3+ years
    df_active = df_filt[df_filt["u"].isin(multi_yr_inv)]
    top = df_active["u"].value_counts().head(TOP_INVENTORS).index
    df_top = df_active[df_active["u"].isin(top)]
    print(f"  Top {TOP_INVENTORS} prolific inventors (active 3+ yrs)")

    # Build inventor × year × topic count matrix
    rows = []
    for inv in top:
        sub = df_top[df_top["u"] == inv]
        for yr in TIME_RANGE:
            sub_yr = sub[sub["year"] == yr]
            if len(sub_yr) == 0:
                continue
            counts = sub_yr["i"].value_counts()
            total = counts.sum()
            for cpc, n in counts.items():
                j = topic_to_idx.get(cpc)
                if j is not None:
                    rows.append({"u": inv, "year": yr, "j": j, "w": n / total, "count_yr": total})
    counts_df = pd.DataFrame(rows)
    print(f"  built {len(counts_df)} (inv, year, topic) records")

    # Compute e_i(t) for each inventor each year
    inventor_emb = {}     # {inv: {year: e_vec (D,)}}
    for inv in top:
        inventor_emb[inv] = {}
        sub = counts_df[counts_df["u"] == inv]
        for yr, g in sub.groupby("year"):
            # find time index in our model
            t_idx = yr - YEAR_BASE  # since YEAR_BASE = 2013 = t=0
            if t_idx < 0 or t_idx >= n_T:
                continue
            cents_t = centroids[t_idx].numpy()  # (n_topics, D)
            w_vec = np.zeros(n_topics)
            for _, r in g.iterrows():
                w_vec[int(r["j"])] = r["w"]
            e = (w_vec[:, None] * cents_t).sum(axis=0)  # (D,)
            if np.linalg.norm(e) > 1e-6:
                inventor_emb[inv][yr] = e

    # Drop inventors with < 3 years of data
    inventor_emb = {inv: v for inv, v in inventor_emb.items() if len(v) >= 3}
    print(f"  {len(inventor_emb)} inventors with ≥3 years of embedding")

    # Compute Φ at each inventor position each year
    print("Evaluating Φ at inventor positions...")
    inv_phi = {}
    for inv, ev in inventor_emb.items():
        inv_phi[inv] = {}
        for yr, e in ev.items():
            t_val = float(yr - YEAR_BASE)
            xt = torch.cat([
                torch.tensor(e, dtype=torch.float32).unsqueeze(0),
                torch.tensor([[t_val]], dtype=torch.float32)
            ], dim=1).to(device).requires_grad_()
            phi = model._func._pot(xt).squeeze().item()
            inv_phi[inv][yr] = phi

    # UMAP fit on union of all observations + centroids + inventor embeddings
    print("UMAP fitting...")
    import umap
    x_all_pts = []
    for xp_t in xp:
        # sub-sample for UMAP speed
        idx = np.random.RandomState(42).choice(len(xp_t), min(2000, len(xp_t)), replace=False)
        x_all_pts.append(xp_t.numpy()[idx])
    x_all = np.concatenate(x_all_pts)

    # Include all centroids
    cent_all = np.concatenate([centroids[t].numpy() for t in range(n_T)])
    # Include all inventor embeddings
    inv_all = np.array([e for ev in inventor_emb.values() for e in ev.values()])
    big = np.concatenate([x_all, cent_all, inv_all])
    print(f"  fitting UMAP on {big.shape}...")
    um = umap.UMAP(n_components=2, n_neighbors=30, random_state=42, transform_seed=42)
    big_2d = um.fit_transform(big)
    x_2d = big_2d[:len(x_all)]
    cent_2d = big_2d[len(x_all):len(x_all)+len(cent_all)]
    inv_2d_flat = big_2d[-len(inv_all):]
    print(f"  UMAP done")

    # Compute Φ on all observation points for heatmap (using last year t=LAST_T)
    print(f"Evaluating Φ heatmap at t={LAST_T} ({YEAR_BASE+LAST_T})...")
    x_all_dev = torch.tensor(x_all, dtype=torch.float32, device=device)
    t_col = torch.full((x_all_dev.shape[0], 1), float(y[LAST_T]), device=device)
    with torch.enable_grad():
        xt_all = torch.cat([x_all_dev, t_col], dim=1).requires_grad_()
        phi_all = model._func._pot(xt_all).squeeze(-1).detach().cpu().numpy()

    # Map inventor → 2D
    inv_2d = {}
    iflat = 0
    for inv, ev in inventor_emb.items():
        inv_2d[inv] = {}
        for yr in sorted(ev.keys()):
            inv_2d[inv][yr] = inv_2d_flat[iflat]
            iflat += 1

    # ────────────────────────── Figure ──────────────────────────
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(1, 2, width_ratios=[2, 1], wspace=0.18)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1])

    # Background: Φ heatmap at last year
    order = np.argsort(phi_all)
    sc = ax_map.scatter(x_2d[order, 0], x_2d[order, 1], c=phi_all[order],
                        s=2, cmap="RdYlBu_r", alpha=0.35,
                        vmin=np.percentile(phi_all, 2),
                        vmax=np.percentile(phi_all, 98))
    fig.colorbar(sc, ax=ax_map, label="Φ (low = valley = predicted growing)",
                 fraction=0.04, pad=0.02)

    # Centroids (at last year)
    cent_last_2d = cent_2d[(LAST_T) * n_topics: (LAST_T+1) * n_topics]
    ax_map.scatter(cent_last_2d[:, 0], cent_last_2d[:, 1], s=80,
                   facecolor="none", edgecolors="black", linewidths=1.2, zorder=4)

    # Inventor trajectories
    n_inv = len(inv_2d)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_inv, 1)))
    legend_handles = []
    for ci, (inv, traj) in enumerate(inv_2d.items()):
        yrs = sorted(traj.keys())
        pts = np.array([traj[yr] for yr in yrs])
        c = colors[ci]
        ax_map.plot(pts[:, 0], pts[:, 1], "-", color=c, lw=1.8, alpha=0.7, zorder=5)
        # arrow at end indicating direction
        if len(pts) >= 2:
            ax_map.annotate("", xy=(pts[-1, 0], pts[-1, 1]),
                            xytext=(pts[-2, 0], pts[-2, 1]),
                            arrowprops=dict(arrowstyle="->", color=c, lw=1.8))
        ax_map.scatter(pts[-1, 0], pts[-1, 1], color=c, s=80,
                       edgecolors="black", linewidths=0.8, zorder=6)
        # label short name
        short = inv.replace("fl:", "").replace("ln:", "").replace("_", " ")[:14]
        ax_map.annotate(short, (pts[-1, 0], pts[-1, 1]),
                        xytext=(7, 7), textcoords="offset points",
                        fontsize=7, color="black",
                        bbox=dict(facecolor="white", edgecolor=c, lw=0.5,
                                  alpha=0.8, boxstyle="round,pad=0.15"))
        legend_handles.append(Line2D([], [], color=c, lw=2, label=short))

    ax_map.set_title(f"発明者 DNA 軌跡 — Patent Energy CPC Y02 (Top {n_inv} prolific inventors, 2013-2021)",
                     fontsize=13, fontweight="bold")
    ax_map.set_xlabel("UMAP1"); ax_map.set_ylabel("UMAP2")
    ax_map.set_xticks([]); ax_map.set_yticks([])

    # Side panel: ranking by ΔΦ (motion toward growth)
    ax_side.axis("off")
    ax_side.text(0.5, 0.97, "発明者の DNA 変化 ランキング",
                 fontsize=13, fontweight="bold", ha="center",
                 transform=ax_side.transAxes)
    ax_side.text(0.5, 0.93, "(過去5年で 成長領域 [-Φ↑] へ動いた人物)",
                 fontsize=9, ha="center", color="#555",
                 transform=ax_side.transAxes)

    # ΔΦ = Φ(latest) − Φ(earliest)  (negative = moved toward valley = good)
    rank = []
    for inv, phi_dict in inv_phi.items():
        yrs = sorted(phi_dict.keys())
        if len(yrs) < 2: continue
        dphi = phi_dict[yrs[-1]] - phi_dict[yrs[0]]
        rank.append((inv, dphi, phi_dict[yrs[-1]]))
    rank.sort(key=lambda x: x[1])   # smallest (most negative) first

    ax_side.text(0.03, 0.88, "★ 成長領域へ移動した発明者 TOP 7",
                 fontsize=11, fontweight="bold", color="#0a5",
                 transform=ax_side.transAxes)
    for i, (inv, dp, ph) in enumerate(rank[:7]):
        short = inv.replace("fl:", "").replace("ln:", "").replace("_", " ")[:25]
        ax_side.text(0.05, 0.83 - i*0.038,
                     f"{i+1}. {short:<26}  ΔΦ={dp:+.2f}",
                     fontsize=9, transform=ax_side.transAxes,
                     family="monospace")

    ax_side.text(0.03, 0.50, "✗ 衰退領域へ移動した発明者 BOTTOM 7",
                 fontsize=11, fontweight="bold", color="#a00",
                 transform=ax_side.transAxes)
    for i, (inv, dp, ph) in enumerate(rank[-7:][::-1]):
        short = inv.replace("fl:", "").replace("ln:", "").replace("_", " ")[:25]
        ax_side.text(0.05, 0.45 - i*0.038,
                     f"{i+1}. {short:<26}  ΔΦ={dp:+.2f}",
                     fontsize=9, transform=ax_side.transAxes,
                     family="monospace")

    ax_side.text(0.5, 0.05,
                 "ΔΦ<0:  成長領域へ移動\nΔΦ>0:  衰退領域へ移動",
                 ha="center", fontsize=9, color="#444",
                 transform=ax_side.transAxes,
                 bbox=dict(facecolor="#f0f0f0", edgecolor="#888",
                           boxstyle="round,pad=0.4"))

    plt.suptitle("Phase 1: 発明者 DNA on Patent Energy Φ Landscape (Top 30 prolific inventors)",
                 fontsize=14, fontweight="bold", y=1.005)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"\nSaved -> {out_png}")


if __name__ == "__main__":
    main()
