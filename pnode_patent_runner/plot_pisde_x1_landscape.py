"""
PI-SDE + X1 学習済みモデルの UMAP 可視化 (paper Fig. style)。

4 パネル構成:
  [A] 観測点 (年色分け)
  [B] Φ heatmap (UMAP scatter colored by Φ)
  [C] トピック centroid 配置 + 実成長率 g_j 注釈
  [D] Φ ランキング vs 実 g ランキング (相関プロット)
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

from src.model import ForwardSDE

X1_DIR = Path("RESULTS/PNode_Paper_X1/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01/seed_42/alltime")
DATA_PT = "data/PNode_Paper_X1/alltime/fate_train.pt"
OUT_PNG = X1_DIR / "trajectories_x1_umap.png"
EVAL_T  = 3   # 可視化時点 (2025)

# データ
data = torch.load(DATA_PT, weights_only=False)
xp = data["xp"]
y  = data["y"]
topics = data["topics"]
topic_names = data["topic_names"]
centroids = data["centroids"]
growth_raw = data["growth"]
growth_norm = data["growth_norm"]
n_topics = data["n_topics"]
print(f"topics: {n_topics}, time points: {y}")

# モデルロード
from types import SimpleNamespace
config = SimpleNamespace(**torch.load(X1_DIR / "config.pt", weights_only=False))
config.x_dim = xp[0].shape[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ForwardSDE(config).to(device)
# best または最新 epoch checkpoint
import glob
ckpts = sorted(glob.glob(str(X1_DIR / "train.epoch_*.pt")))
ckpt_path = ckpts[-1] if ckpts else str(X1_DIR / "train.best.pt")
print(f"Loading: {ckpt_path}")
ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# UMAP fit on all data
print("UMAP fitting...")
import umap
x_all = torch.cat(xp).numpy()
um = umap.UMAP(n_components=2, n_neighbors=30, metric="euclidean",
               random_state=42, transform_seed=42)
x_all_2d = um.fit_transform(x_all)

# 全観測点に対応する年ラベル
y_all = np.concatenate([np.full(v.shape[0], y[k]) for k, v in enumerate(xp)])
topics_all = np.concatenate([t.numpy() for t in topics])

# Φ(全観測点, t)
print("Computing Φ values...")
xt_all = torch.cat([torch.tensor(x_all, dtype=torch.float32),
                     torch.tensor(y_all, dtype=torch.float32).unsqueeze(1)], dim=1)
xt_all_dev = xt_all.to(device).requires_grad_()
phi_all = model._func._pot(xt_all_dev).squeeze(-1).detach().cpu().numpy()

# トピック centroid を UMAP に投影
cent_t = centroids[EVAL_T].numpy()
active_mask = cent_t.sum(axis=-1) != 0
cent_active = cent_t[active_mask]
cent_2d = um.transform(cent_active)
g_t = growth_raw[EVAL_T].numpy()[active_mask]
topic_names_active = [topic_names[i] for i in range(n_topics) if active_mask[i]]

# Φ at centroids
xt_cent = torch.cat([torch.tensor(cent_active, dtype=torch.float32),
                     torch.full((len(cent_active), 1), float(y[EVAL_T]))], dim=1)
phi_cent = model._func._pot(xt_cent.to(device).requires_grad_()).squeeze(-1).detach().cpu().numpy()

# ── プロット ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 6))
gs = GridSpec(1, 4, width_ratios=[1, 1, 1.1, 1], wspace=0.25)

# [A] 観測点 (年色分け)
ax0 = plt.subplot(gs[0, 0])
years_int = y_all.astype(int)
for yt in [0, 1, 2, 3]:
    mask = years_int == yt
    ax0.scatter(x_all_2d[mask, 0], x_all_2d[mask, 1], s=2, alpha=0.4,
                color=plt.cm.viridis(yt / 3.0), label=f"t={yt} (year {2022+yt})")
ax0.set_title("[A] Observed papers by year", fontsize=11, fontweight="bold")
ax0.set_xlabel("UMAP1"); ax0.set_ylabel("UMAP2")
ax0.legend(fontsize=8); ax0.set_xticks([]); ax0.set_yticks([])

# [B] Φ heatmap
ax1 = plt.subplot(gs[0, 1])
ci = np.argsort(phi_all)
sc1 = ax1.scatter(x_all_2d[ci, 0], x_all_2d[ci, 1], c=phi_all[ci],
                  s=2, cmap="RdYlBu_r")
ax1.set_title(f"[B] X1-trained Φ(x, t)\n  (low = valley = predicted growing)", fontsize=11, fontweight="bold")
ax1.set_xlabel("UMAP1"); ax1.set_ylabel("UMAP2")
ax1.set_xticks([]); ax1.set_yticks([])
plt.colorbar(sc1, ax=ax1, label="Φ", fraction=0.04)

# [C] Centroids + g labels
ax2 = plt.subplot(gs[0, 2])
ax2.scatter(x_all_2d[:, 0], x_all_2d[:, 1], s=1, color="lightgray", alpha=0.3)
# centroid を g で色分け
sc2 = ax2.scatter(cent_2d[:, 0], cent_2d[:, 1], c=g_t, cmap="RdYlGn",
                  s=180, edgecolors="black", linewidths=1.0, vmin=-0.5, vmax=0.5, zorder=5)
# 注釈
for i, name in enumerate(topic_names_active):
    fc = "white" if abs(g_t[i]) > 0.2 else "black"
    fs = 8 if abs(g_t[i]) > 0.2 else 6.5
    ax2.annotate(name.replace("cs.", ""), (cent_2d[i, 0], cent_2d[i, 1]),
                 ha="center", va="center", fontsize=fs, color=fc,
                 fontweight="bold" if abs(g_t[i]) > 0.3 else "normal")
ax2.set_title(f"[C] Topic centroids @ t={EVAL_T} (year {2022+EVAL_T})\n"
              f"  Color = actual growth rate g_j", fontsize=11, fontweight="bold")
ax2.set_xlabel("UMAP1"); ax2.set_ylabel("UMAP2")
ax2.set_xticks([]); ax2.set_yticks([])
plt.colorbar(sc2, ax=ax2, label="g_j (growth)", fraction=0.04)

# [D] Φ-rank vs g-rank scatter
ax3 = plt.subplot(gs[0, 3])
phi_rank = np.argsort(np.argsort(phi_cent))     # 低 Φ = rank 1
g_rank   = np.argsort(np.argsort(-g_t))         # 高 g  = rank 1
r, p = stats.spearmanr(phi_cent, g_t)

# 散布図: x = Φ rank, y = g rank
sc3 = ax3.scatter(phi_rank, g_rank, c=g_t, cmap="RdYlGn", s=100,
                  edgecolors="black", linewidths=0.7, vmin=-0.5, vmax=0.5)
# 注釈 (主要なトピックのみ)
for i, name in enumerate(topic_names_active):
    if abs(g_t[i]) > 0.2 or phi_rank[i] < 5 or phi_rank[i] >= len(phi_rank) - 5:
        ax3.annotate(name.replace("cs.", ""), (phi_rank[i], g_rank[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

# 完璧な対応線 (rank が同じ = 完全な一致は y = x ではなく y = -(rank))
# Spearman 完璧なら scatter が y = x 上に並ぶ (低Φ→高g→低 g_rank)
# 低Φ(rank0) → 高g → g_rank 0
# つまり y = x の直線
n_t = len(phi_rank)
ax3.plot([0, n_t-1], [0, n_t-1], "k--", lw=0.8, alpha=0.5, label="Perfect (Φ ↑ ⇔ g ↑)")
sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
ax3.set_title(f"[D] Φ rank vs Growth rank\n  Spearman r = {r:+.3f}{sig} (p={p:.4f})",
              fontsize=11, fontweight="bold")
ax3.set_xlabel("Φ rank (1=valley=predicted top growing)")
ax3.set_ylabel("g rank (1=actual top growing)")
ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

fig.suptitle(
    f"PI-SDE + X1 (Topic-Anchored Potential) on ArXiv CS  |  seed=42, year 2025\n"
    f"X1 makes Φ landscape interpretable: low Φ region = high actual growth rate",
    fontsize=12, fontweight="bold", y=1.02,
)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"Saved -> {OUT_PNG}")
