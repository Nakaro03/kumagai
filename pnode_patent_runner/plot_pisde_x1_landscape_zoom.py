"""
PI-SDE + X1 (seed=42) の Φ 景観を詳細可視化。

仕様:
  - UMAP 2D 投影で全 15,241 論文を散布
  - 背景: Φ contour (補間で smooth に)
  - トピック centroid を大きなマーカーで表示:
      色 = 実 g_j (緑=成長, 赤=衰退)
      サイズ = |g_j|
  - ラベル: top 10 成長 + top 10 衰退トピックのみ表示
  - 矢印: -∇Φ ベクトル場 (research flow)
"""
from __future__ import annotations

import os, sys, warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata

warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

from src.model import ForwardSDE

X1_DIR = Path("RESULTS/PNode_Paper_X1/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01/seed_42/alltime")
DATA_PT = "data/PNode_Paper_X1/alltime/fate_train.pt"
OUT_PNG = X1_DIR / "landscape_zoom.png"
EVAL_T = 3

# データ
data = torch.load(DATA_PT, weights_only=False)
xp = data["xp"]
y  = data["y"]
topic_names = data["topic_names"]
centroids   = data["centroids"]
growth_raw  = data["growth"]
n_topics    = data["n_topics"]

# モデル
from types import SimpleNamespace
config = SimpleNamespace(**torch.load(X1_DIR / "config.pt", weights_only=False))
config.x_dim = xp[0].shape[-1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ForwardSDE(config).to(device)
ckpt = torch.load(X1_DIR / "train.epoch_000500.pt", weights_only=False, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# UMAP
import umap
x_all = torch.cat(xp).numpy()
um = umap.UMAP(n_components=2, n_neighbors=30, metric="euclidean",
               random_state=42, transform_seed=42)
x_all_2d = um.fit_transform(x_all)

# Φ on all observed points at year EVAL_T
y_all = np.concatenate([np.full(v.shape[0], y[k]) for k, v in enumerate(xp)])
xt_all = torch.cat([torch.tensor(x_all, dtype=torch.float32),
                    torch.tensor(y_all, dtype=torch.float32).unsqueeze(1)], dim=1)
phi_all = model._func._pot(xt_all.to(device).requires_grad_()).squeeze(-1).detach().cpu().numpy()

# トピック centroids @ EVAL_T
cent = centroids[EVAL_T].numpy()
active = cent.sum(axis=-1) != 0
cent_a = cent[active]
g_a = growth_raw[EVAL_T].numpy()[active]
names_a = [topic_names[i] for i in range(n_topics) if active[i]]
cent_2d = um.transform(cent_a)

# Φ at centroids
xt_cent = torch.cat([torch.tensor(cent_a, dtype=torch.float32),
                     torch.full((len(cent_a), 1), float(y[EVAL_T]))], dim=1)
phi_cent = model._func._pot(xt_cent.to(device).requires_grad_()).squeeze(-1).detach().cpu().numpy()

# Φ contour grid (interpolation from observed Φ values)
RES = 200
xmin, xmax = x_all_2d[:, 0].min() - 1, x_all_2d[:, 0].max() + 1
ymin, ymax = x_all_2d[:, 1].min() - 1, x_all_2d[:, 1].max() + 1
gx = np.linspace(xmin, xmax, RES)
gy = np.linspace(ymin, ymax, RES)
GX, GY = np.meshgrid(gx, gy)
# 同一年のものだけで補間 (year-specific landscape)
mask_yr = y_all == y[EVAL_T]
phi_grid = griddata(x_all_2d[mask_yr], phi_all[mask_yr], (GX, GY), method="linear")

# Vector field (UMAP 空間で簡易表示: drift を gradient of interpolated Φ で近似)
# 補間 Φ の数値勾配
phi_grid_filled = np.where(np.isnan(phi_grid), np.nanmean(phi_grid), phi_grid)
gPhi_y, gPhi_x = np.gradient(phi_grid_filled, gy, gx)
U = -gPhi_x
V = -gPhi_y
# normalize for visualization
mag = np.sqrt(U**2 + V**2)
U_n = U / (mag + 1e-10)
V_n = V / (mag + 1e-10)

# ── プロット ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10))

# 背景 contour
levels = 30
cf = ax.contourf(GX, GY, phi_grid_filled, levels=levels, cmap="RdYlBu_r", alpha=0.55)
ax.contour(GX, GY, phi_grid_filled, levels=15, colors="white", linewidths=0.4, alpha=0.6)
cb = plt.colorbar(cf, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("Φ(x, t=3)  (low = valley = growing)", fontsize=11)

# 全観測点 (淡く)
ax.scatter(x_all_2d[:, 0], x_all_2d[:, 1], s=1, color="black", alpha=0.06, zorder=1)

# Vector field (subsample)
step = 12
ax.quiver(GX[::step, ::step], GY[::step, ::step],
          U_n[::step, ::step], V_n[::step, ::step],
          color="white", alpha=0.75, scale=30, width=0.0035, zorder=2)

# トピック centroid (色: g, サイズ: |g|)
sizes = 100 + 250 * np.abs(g_a)   # 100〜600 程度
sc = ax.scatter(cent_2d[:, 0], cent_2d[:, 1], c=g_a, cmap="RdYlGn", s=sizes,
                vmin=-0.5, vmax=1.0, edgecolors="black", linewidths=1.4, zorder=5)

# ラベル: TOP 10 grow + TOP 10 decline + 主要 (cs.AI, cs.LG, cs.CV)
order_g = np.argsort(-g_a)
order_decline = np.argsort(g_a)
key_topics = set()
for i in order_g[:10]:    key_topics.add(i)
for i in order_decline[:10]: key_topics.add(i)
for n in ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE"]:
    if n in names_a:
        key_topics.add(names_a.index(n))

for i in key_topics:
    name = names_a[i].replace("cs.", "")
    g = g_a[i]
    fc = "white" if abs(g) > 0.3 else "black"
    fs = 11 if abs(g) > 0.3 else 9
    ax.annotate(name, (cent_2d[i, 0], cent_2d[i, 1]),
                ha="center", va="center", fontsize=fs, color=fc,
                fontweight="bold" if abs(g) > 0.3 else "normal",
                zorder=6)

# サイドカラーバー (g 用)
ax_cb2 = fig.add_axes([0.92, 0.12, 0.012, 0.30])
import matplotlib.cm as cm
sm = cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(-0.5, 1.0))
sm.set_array([])
cb2 = plt.colorbar(sm, cax=ax_cb2)
cb2.set_label("Actual growth rate g_j", fontsize=11)

# 凡例 (size)
from matplotlib.lines import Line2D
legend_sizes = [0.1, 0.5, 1.0]
legend_elems = []
for s in legend_sizes:
    legend_elems.append(Line2D([0],[0], marker="o", markersize=np.sqrt(100+250*s),
                                color="w", markerfacecolor="lightgreen",
                                markeredgecolor="black", label=f"|g|={s}"))
ax.legend(handles=legend_elems, loc="upper left", title="Marker size",
          fontsize=9, title_fontsize=10)

ax.set_xlabel("UMAP1", fontsize=12)
ax.set_ylabel("UMAP2", fontsize=12)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(
    f"PI-SDE + X1 Topic-Anchored Landscape  (seed=42, year 2025)\n"
    f"Background: learned Φ (blue=valley=predicted growing)   |   "
    f"Dots: topic centroids (color=actual growth g, size=|g|)   |   "
    f"White arrows: -∇Φ flow",
    fontsize=12, fontweight="bold",
)

fig.subplots_adjust(right=0.90)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved -> {OUT_PNG}")
