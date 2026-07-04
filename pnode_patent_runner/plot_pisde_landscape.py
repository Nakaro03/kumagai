"""
PI-SDE 学習済みモデルの Φ landscape + vector field 可視化 (PI-SDE 論文式)。

手法:
  1. 全観測点を PCA 2D に投影
  2. 各観測点で Φ(x, t) を計算 (50D x → 1D scalar)
  3. 2D 平面で観測点 (PCA座標, Φ) を Triangulation で interpolate して contour
  4. ∇_x Φ を autograd で計算し、PCA 2D に正射影して quiver
  5. 各時点 t ∈ {0, 1, 2, 3} ごとにパネル

出力: trajectories_with_landscape.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

from src.model import ForwardSDE

ALLTIME_DIR = Path("RESULTS/PNode_Paper/softplus-400_400-0.5-const-0.1-0.1-0.005/seed_42/alltime")
DATA_PT     = "data/PNode_Paper/alltime/fate_train.pt"
OUT_PNG     = ALLTIME_DIR / "trajectories_with_landscape.png"

# データロード
data = torch.load(DATA_PT, weights_only=False)
xp = data["xp"]
y  = data["y"]
print("Time points:", y, "Sample sizes:", [v.shape[0] for v in xp])

# config & model
from types import SimpleNamespace
config = SimpleNamespace(**torch.load(ALLTIME_DIR / "config.pt", weights_only=False))
config.x_dim = xp[0].shape[-1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ForwardSDE(config).to(device)
ckpt = torch.load(ALLTIME_DIR / "train.epoch_000500.pt", weights_only=False, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# PCA: 全観測データで fit
print("PCA fitting...")
from sklearn.decomposition import PCA
all_obs = np.vstack([v.numpy() for v in xp])
pca2 = PCA(n_components=2, random_state=42)
pca2.fit(all_obs)
# 主成分行列 V_2 ∈ ℝ^(50 × 2)  (50D → 2D の射影行列)
V2 = pca2.components_.T  # (50, 2)
mean50 = pca2.mean_       # (50,)


def compute_phi(x_50d, t_val):
    """Φ(x, t) を計算。入力 x_50d: (N, 50)"""
    x_t = torch.tensor(x_50d, dtype=torch.float32, device=device)
    t_col = torch.ones(x_t.shape[0], 1, device=device) * float(t_val)
    xt = torch.cat([x_t, t_col], dim=1).requires_grad_()
    phi = model._func._pot(xt).squeeze(-1)
    return phi


def compute_grad_phi(x_50d, t_val):
    """∇_x Φ(x, t) を計算 → 50D"""
    x_t = torch.tensor(x_50d, dtype=torch.float32, device=device)
    t_col = torch.ones(x_t.shape[0], 1, device=device) * float(t_val)
    xt = torch.cat([x_t, t_col], dim=1).requires_grad_()
    phi = model._func._pot(xt)
    grad = torch.autograd.grad(phi.sum(), xt)[0]
    grad_x = grad[:, :-1].detach().cpu().numpy()  # (N, 50)
    return grad_x


# SDE rollout で予測位置を取得
print("SDE rollout...")
n_traj = 100
x_0_sub = xp[0][np.random.choice(xp[0].shape[0], n_traj, replace=False)]
r_0 = torch.zeros(n_traj).unsqueeze(1)
x_r_0 = torch.cat([x_0_sub, r_0], dim=1).to(device).requires_grad_()
x_r_s = model([np.float64(yt) for yt in y], x_r_0)
preds_50 = [s[:, :-1].detach().cpu().numpy() for s in x_r_s]   # 各時点の予測 50D

# 観測 2D 座標
xp_2d = [pca2.transform(v.numpy()) for v in xp]
preds_2d = [pca2.transform(p) for p in preds_50]

# 描画範囲
all_2d = np.vstack(xp_2d + preds_2d)
xmin, xmax = all_2d[:, 0].min() - 0.5, all_2d[:, 0].max() + 0.5
ymin, ymax = all_2d[:, 1].min() - 0.5, all_2d[:, 1].max() + 0.5

# グリッド作成 (Φ contour & vector field 用)
RES = 30
gx = np.linspace(xmin, xmax, RES)
gy = np.linspace(ymin, ymax, RES)
GX, GY = np.meshgrid(gx, gy)
# 2D グリッド点を 50D に逆変換 (mean + V2 @ [gx, gy])
grid_2d = np.stack([GX.flatten(), GY.flatten()], axis=1)         # (RES², 2)
grid_50 = mean50[None, :] + grid_2d @ V2.T                        # (RES², 50)

# ── 描画 ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(y), figsize=(5.5 * len(y), 5), sharex=True, sharey=True)

for k, (t_val, ax) in enumerate(zip(y, axes)):
    # Φ on 50D grid (PCA-lifted)
    phi_grid = compute_phi(grid_50, t_val).detach().cpu().numpy().reshape(RES, RES)
    # contour
    cf = ax.contourf(GX, GY, phi_grid, levels=20, cmap="RdYlGn_r", alpha=0.55)
    ax.contour(GX, GY, phi_grid, levels=10, colors="white", linewidths=0.4, alpha=0.6)

    # vector field: -∇_x Φ → PCA 2D に射影
    grad_50 = compute_grad_phi(grid_50, t_val)                # (RES², 50)
    grad_2d = grad_50 @ V2                                     # (RES², 2)
    U = -grad_2d[:, 0].reshape(RES, RES)
    V = -grad_2d[:, 1].reshape(RES, RES)
    step = 2
    ax.quiver(GX[::step, ::step], GY[::step, ::step],
              U[::step, ::step], V[::step, ::step],
              color="white", alpha=0.7, scale=None, width=0.0035)

    # 観測点 (灰色, 小サイズ)
    obs = xp_2d[k]
    ax.scatter(obs[:, 0], obs[:, 1], s=2, alpha=0.18, color="#374151",
               zorder=2, label=f"Observed (n={obs.shape[0]})")

    # 予測点 (青)
    pred = preds_2d[k]
    ax.scatter(pred[:, 0], pred[:, 1], s=14, alpha=0.7, color="#1d4ed8",
               edgecolors="white", linewidths=0.4, zorder=3,
               label=f"PI-SDE pred (n={pred.shape[0]})")

    # 軌跡 (前年からの線)
    if k > 0:
        prev = preds_2d[k - 1]
        for j in range(min(40, n_traj)):
            ax.plot([prev[j, 0], pred[j, 0]], [prev[j, 1], pred[j, 1]],
                    color="#1d4ed8", alpha=0.35, lw=0.6, zorder=2.5)

    ax.set_title(f"t={int(t_val)}  (year {2022+int(t_val)})", fontsize=11)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2" if k == 0 else "")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.2)

# colorbar
cax = fig.add_axes([0.92, 0.18, 0.012, 0.65])
plt.colorbar(cf, cax=cax, label="Φ(x, t)  (low = valley = attractor)")

fig.suptitle(
    "PI-SDE Waddington-style landscape on ArXiv CS papers  |  seed=42\n"
    "Background = potential Φ(x,t)  |  White arrows = −∇Φ (research flow direction)  |  "
    "Blue dots = predicted positions  |  Gray = actual papers",
    fontsize=11, fontweight="bold", y=1.02,
)
fig.subplots_adjust(right=0.90, wspace=0.15)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved -> {OUT_PNG}")
