"""
PI-SDE 論文 (Visual_Potential.ipynb + Plot_vector.ipynb) と同形式の UMAP 可視化。

3 パネル構成 (PI-SDE 論文の Fig 風):
  [A] UMAP 観測点 (年で色分け)
  [B] UMAP 観測点 を Φ(x, t) で色分け (Potential heatmap)
  [C] Vector field: sample points からの drift 矢印 (UMAP空間)

出力: trajectories_umap.png
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

warnings.filterwarnings("ignore")

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

from src.model import ForwardSDE

ALLTIME_DIR = Path("RESULTS/PNode_Paper/softplus-400_400-0.5-const-0.1-0.1-0.005/seed_42/alltime")
DATA_PT     = "data/PNode_Paper/alltime/fate_train.pt"
OUT_PNG     = ALLTIME_DIR / "trajectories_umap.png"

# データ
data = torch.load(DATA_PT, weights_only=False)
xp = data["xp"]
y  = data["y"]
print("Loaded data:", [v.shape for v in xp])

# config & model
from types import SimpleNamespace
config = SimpleNamespace(**torch.load(ALLTIME_DIR / "config.pt", weights_only=False))
config.x_dim = xp[0].shape[-1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ForwardSDE(config).to(device)
ckpt = torch.load(ALLTIME_DIR / "train.epoch_000500.pt", weights_only=False, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 全観測点を統合 + 年ラベル
x_all = torch.cat(xp, dim=0)                                       # (Ntot, 50)
y_all = np.concatenate([np.full(v.shape[0], y[k]) for k, v in enumerate(xp)])
print(f"x_all: {x_all.shape}, y_all unique: {np.unique(y_all)}")

# UMAP fit (PI-SDE 論文と同設定: n_components=2, metric='euclidean', n_neighbors=30)
import umap
print("UMAP fitting (n_neighbors=30)...")
um = umap.UMAP(n_components=2, metric="euclidean", n_neighbors=30,
               random_state=42, transform_seed=42)
x_all_2d = um.fit_transform(x_all.cpu().numpy())                   # (Ntot, 2)
print(f"UMAP done. shape: {x_all_2d.shape}")

# Φ(x_all, t_corresponding) を計算
print("Computing Φ for all observed points...")
xt_all = torch.cat([x_all, torch.tensor(y_all, dtype=torch.float32).unsqueeze(1)], dim=1)
xt_all_dev = xt_all.to(device).requires_grad_()
phi_all = model._func._pot(xt_all_dev).squeeze(-1).detach().cpu().numpy()
print(f"Φ range: [{phi_all.min():.3f}, {phi_all.max():.3f}]")

# ── プロット ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 6))
gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.04], wspace=0.25)

# [A] UMAP scatter colored by time
ax0 = plt.subplot(gs[0, 0])
years_int = y_all.astype(int)
cmap_t = plt.cm.viridis
for yt in [0, 1, 2, 3]:
    mask = years_int == yt
    ax0.scatter(x_all_2d[mask, 0], x_all_2d[mask, 1],
                s=3, alpha=0.5, color=cmap_t(yt / 3.0),
                label=f"t={yt} (year {2022+yt})")
ax0.set_title("[A] Observed papers by year", fontsize=12)
ax0.set_xlabel("UMAP1"); ax0.set_ylabel("UMAP2")
ax0.legend(fontsize=9, loc="upper right")
ax0.set_xticks([]); ax0.set_yticks([])

# [B] Φ heatmap
ax1 = plt.subplot(gs[0, 1])
ci = np.argsort(phi_all)
sc1 = ax1.scatter(x_all_2d[ci, 0], x_all_2d[ci, 1], c=phi_all[ci],
                  s=3, cmap="RdYlBu_r")
ax1.set_title("[B] Learned potential Φ(x, t)\n  (low=valley=attractor)", fontsize=12)
ax1.set_xlabel("UMAP1"); ax1.set_ylabel("UMAP2")
ax1.set_xticks([]); ax1.set_yticks([])

# [C] Vector field: drift on sampled starts → UMAP arrows (PI-SDE Plot_vector.ipynb 流)
ax2 = plt.subplot(gs[0, 2])
ax2.scatter(x_all_2d[:, 0], x_all_2d[:, 1], s=1, color="gray", alpha=0.35)

# 各時点 t について 30 個サンプルを取り drift を計算
print("Computing drift vectors for vector field...")
np.random.seed(42)
for k in range(len(xp)):
    n_samp = min(30, xp[k].shape[0])
    idx = np.random.choice(xp[k].shape[0], n_samp, replace=False)
    x_samp = xp[k][idx]                                                # (n, 50)
    t_col = torch.full((n_samp, 1), float(y[k]))
    Xt = torch.cat([x_samp, t_col], dim=1).to(device).requires_grad_()
    # drift = -∇_x Φ (PI-SDE の _drift)
    drift_x = model._func._drift(Xt).detach().cpu().numpy()            # (n, 50)
    # X_start, X_end を UMAP に
    X_start = x_samp.cpu().numpy()
    X_end   = X_start + drift_x
    x_start_umap = um.transform(X_start)
    x_end_umap   = um.transform(X_end)
    xv = x_end_umap - x_start_umap
    norm = np.linalg.norm(xv, axis=1, keepdims=True)
    xv = xv / (norm + 1e-10) * 0.7  # 正規化 (PI-SDE 論文と同様)
    color = plt.cm.viridis(k / 3.0)
    ax2.quiver(x_start_umap[:, 0], x_start_umap[:, 1], xv[:, 0], xv[:, 1],
               scale=1.5, scale_units="xy", width=0.005, color=color,
               alpha=0.85, label=f"t={k}")

ax2.set_title("[C] Drift vector field -∇Φ\n(UMAP-projected per-sample arrows)", fontsize=12)
ax2.set_xlabel("UMAP1"); ax2.set_ylabel("UMAP2")
ax2.legend(fontsize=9, loc="upper right")
ax2.set_xticks([]); ax2.set_yticks([])

# colorbar
cax = plt.subplot(gs[0, 3])
cbar = plt.colorbar(sc1, cax=cax)
cbar.set_label("Φ", fontsize=12)

fig.suptitle(
    "PI-SDE on ArXiv CS  |  UMAP visualization (paper Fig. style)\n"
    "[A] observed clouds by year   [B] learned potential Φ   [C] drift vector field −∇Φ",
    fontsize=12, fontweight="bold",
)
fig.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved -> {OUT_PNG}")
