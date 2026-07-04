"""
クリーンな 2D 技術空間 — Patent Energy & JP Construction.

ノイズ無し:
  - topic centroid のみ (背景点群なし)
  - 色 = 実成長率
  - サイズ = 件数
  - 家族別の薄い領域マーカー
"""
from __future__ import annotations

import os, sys, glob
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

for cand in ["Noto Sans CJK JP", "IPAGothic", "TakaoGothic"]:
    if cand in [f.name for f in font_manager.fontManager.ttflist]:
        plt.rcParams["font.family"] = cand
        break

CPC_FAMILIES = {
    "Y02A": ("気候適応",     "#5b9bd5"),
    "Y02B": ("建物省エネ",   "#a5a5a5"),
    "Y02C": ("GHG 削減",     "#ffc000"),
    "Y02D": ("ICT 省エネ",   "#70ad47"),
    "Y02E": ("クリーンエネ", "#ed7d31"),
    "Y02P": ("製造省エネ",   "#4472c4"),
    "Y02T": ("交通省エネ",   "#7030a0"),
    "Y02W": ("廃棄物",       "#c00000"),
}

DOMAINS = {
    "Patent Energy (CPC Y02, 2024)": {
        "data": "data/PNode_Patent_Energy_X1_top50/alltime/fate_train.pt",
        "last_t": 11,
        "family_fn": lambda n: n[:4] if n else "?",
        "family_map": CPC_FAMILIES,
    },
    "JP Construction (J-STAGE, 2025)": {
        "data": "data/PNode_JP_Construction_X1/alltime/fate_train.pt",
        "last_t": 10,
        "family_fn": None,        # k-means cluster names, no family
        "family_map": None,
    },
}


def project_2d(centroids, method="umap", random_state=42):
    """Project topic centroids to 2D. UMAP gives better cluster separation than PCA."""
    if method == "umap":
        try:
            import umap
            um = umap.UMAP(n_components=2, n_neighbors=min(15, len(centroids) - 1),
                           min_dist=0.5, metric="euclidean",
                           random_state=random_state, transform_seed=random_state)
            return um.fit_transform(centroids), um
        except Exception:
            pass
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(centroids), pca


def make_panel(ax, cfg, title):
    data = torch.load(cfg["data"], weights_only=False)
    centroids = data["centroids"]
    topic_names = data["topic_names"]
    n_topics = data["n_topics"]
    growth = data["growth"]
    last_t = cfg["last_t"]

    # Centroids at last year, only active topics
    cent = centroids[last_t].numpy()
    active = cent.sum(axis=-1) != 0
    cent_act = cent[active]
    names_act = [topic_names[i] for i in range(n_topics) if active[i]]
    g_act = growth[last_t].numpy()[active]

    # Topic volume (cumulative across years)
    vol = np.zeros(n_topics)
    for k in range(len(centroids)):
        t_k = data["topics"][k].numpy()
        for j in range(n_topics):
            vol[j] += (t_k == j).sum()
    vol_act = vol[active]

    # 2D PCA
    pts_2d, _ = project_2d(cent_act)

    # Family-based background hulls (only for Patent)
    if cfg["family_fn"] is not None and cfg["family_map"] is not None:
        fam_of = [cfg["family_fn"](n) for n in names_act]
        for fam, (label, color) in cfg["family_map"].items():
            mask = np.array([f == fam for f in fam_of])
            if mask.sum() < 3:
                continue
            pts = pts_2d[mask]
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                ax.fill(hull_pts[:, 0], hull_pts[:, 1], color=color,
                        alpha=0.12, edgecolor=color, lw=1.0, zorder=1)
                # family label at centroid
                cx, cy = pts.mean(axis=0)
                ax.annotate(f"{fam}\n{label}", (cx, cy),
                            fontsize=10, fontweight="bold", color=color,
                            ha="center", va="center", zorder=2, alpha=0.65)
            except Exception:
                pass

    # Topic markers — size = log(volume+1), color = growth
    g_max = max(abs(g_act.max()), abs(g_act.min())) + 1e-6
    sizes = 80 + 800 * (vol_act / vol_act.max())
    sc = ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c=g_act, cmap="RdYlGn",
                    s=sizes, edgecolors="black", linewidths=1.0,
                    vmin=-g_max, vmax=g_max, zorder=5, alpha=0.92)

    # Annotate top growth (3) + top decline (3) only — less clutter
    order = np.argsort(-g_act)
    for i in order[:3]:
        short = names_act[i].split(":")[-1][:18] if ":" in names_act[i] else names_act[i][:18]
        ax.annotate(f"↑ {short}\n  g={g_act[i]:+.2f}", pts_2d[i],
                    xytext=(10, 8), textcoords="offset points",
                    fontsize=9.5, fontweight="bold", color="#073",
                    bbox=dict(facecolor="white", edgecolor="#073", lw=1.0,
                              alpha=0.95, boxstyle="round,pad=0.25"), zorder=6,
                    arrowprops=dict(arrowstyle="->", color="#073", lw=1.2))
    for i in order[-3:]:
        short = names_act[i].split(":")[-1][:18] if ":" in names_act[i] else names_act[i][:18]
        ax.annotate(f"↓ {short}\n  g={g_act[i]:+.2f}", pts_2d[i],
                    xytext=(10, -24), textcoords="offset points",
                    fontsize=9.5, color="#a00",
                    bbox=dict(facecolor="white", edgecolor="#a00", lw=1.0,
                              alpha=0.95, boxstyle="round,pad=0.25"), zorder=6,
                    arrowprops=dict(arrowstyle="->", color="#a00", lw=1.2))

    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    return sc


fig = plt.figure(figsize=(20, 9))
gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.18)

ax1 = fig.add_subplot(gs[0, 0])
sc1 = make_panel(ax1, DOMAINS["Patent Energy (CPC Y02, 2024)"], "Patent Energy 2D 技術空間")
cb1 = plt.colorbar(sc1, ax=ax1, fraction=0.04, pad=0.01)
cb1.set_label("実成長率 g  (緑 = 成長, 赤 = 衰退)", fontsize=10)

ax2 = fig.add_subplot(gs[0, 1])
sc2 = make_panel(ax2, DOMAINS["JP Construction (J-STAGE, 2025)"], "JP Construction 2D 技術空間")
cb2 = plt.colorbar(sc2, ax=ax2, fraction=0.04, pad=0.01)
cb2.set_label("実成長率 g  (緑 = 成長, 赤 = 衰退)", fontsize=10)

plt.suptitle("クリーンな 2D 技術空間 — Topic Centroid のみ表示  "
             "(マーカー大 = 件数多 / 色 = 成長率 / 薄色領域 = 技術家族)",
             fontsize=14, fontweight="bold", y=1.01)

out = Path("RESULTS/fig12_clean_2d_space.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")

import shutil
shutil.copy(out, "figures/fig12_clean_2d_space.png")
print("Copied to figures/")
