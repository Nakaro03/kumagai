"""
PI-SDE 学習済みモデルから 2D 軌跡を生成して可視化。

50-D PCA 空間で SDE を回し、結果を最初の 2 主成分に投影してプロット。
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

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

from src.model import ForwardSDE

# 設定
ALLTIME_DIR = Path("RESULTS/PNode_Paper/softplus-400_400-0.5-const-0.1-0.1-0.005/seed_42/alltime")
DATA_PT     = "data/PNode_Paper/alltime/fate_train.pt"
OUT_PNG     = ALLTIME_DIR / "trajectories.png"

# データロード
data = torch.load(DATA_PT, weights_only=False)
xp = data["xp"]        # list of (N_t, 50)
y  = data["y"]         # [0,1,2,3]
print("Time points:", y)
print("Sample sizes:", [v.shape[0] for v in xp])

# モデルロード
config_dict = torch.load(ALLTIME_DIR / "config.pt", weights_only=False)
from types import SimpleNamespace
config = SimpleNamespace(**config_dict)
config.x_dim = xp[0].shape[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ForwardSDE(config).to(device)
ckpt = torch.load(ALLTIME_DIR / "train.epoch_000500.pt", weights_only=False, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 観測データを最初の 2 主成分に投影
print("Projecting to 2D via PCA on all observed data...")
from sklearn.decomposition import PCA
all_obs = np.vstack([v.numpy() for v in xp])
pca2 = PCA(n_components=2, random_state=42)
pca2.fit(all_obs)
xp_2d = [pca2.transform(v.numpy()) for v in xp]

# 予測軌跡を生成
print("Running SDE forward from t=0...")
n_traj = 200
x_0_sub = xp[0][np.random.choice(xp[0].shape[0], n_traj, replace=False)]
r_0 = torch.zeros(n_traj).unsqueeze(1)
x_r_0 = torch.cat([x_0_sub, r_0], dim=1).to(device).requires_grad_()

# 各時点での予測を取得 (t=0 → t=1 → t=2 → t=3 の rollout)
x_r_s = model([np.float64(yt) for yt in y], x_r_0)
# x_r_s: list of (n_traj, 51) for each time point
preds_2d = [pca2.transform(s[:, :-1].detach().cpu().numpy()) for s in x_r_s]

# ── プロット ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(y), figsize=(5 * len(y), 4.5), sharex=True, sharey=True)
for k, (t_val, ax) in enumerate(zip(y, axes)):
    # 観測分布
    obs = xp_2d[k]
    ax.scatter(obs[:, 0], obs[:, 1], s=4, alpha=0.25, color="#9ca3af",
               label=f"Observed (n={obs.shape[0]})", zorder=1)
    # 予測分布
    pred = preds_2d[k]
    ax.scatter(pred[:, 0], pred[:, 1], s=10, alpha=0.6, color="#3b82f6",
               label=f"PI-SDE predicted (n={pred.shape[0]})", zorder=2,
               edgecolors="black", linewidths=0.2)
    # 軌跡 (一部だけ表示)
    if k > 0:
        prev = preds_2d[k - 1]
        n_show = 30
        for j in range(n_show):
            ax.plot([prev[j, 0], pred[j, 0]], [prev[j, 1], pred[j, 1]],
                    color="#3b82f6", alpha=0.3, lw=0.6, zorder=1.5)
    ax.set_title(f"t={int(t_val)} (year {2022+int(t_val)})", fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle(
    "PI-SDE rollout from t=0 (year 2022): observed vs predicted distributions in 2D PCA\n"
    "Gray cloud = actual ArXiv papers each year   |   Blue dots = SDE-predicted positions",
    fontsize=11, fontweight="bold",
)
fig.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"Saved -> {OUT_PNG}")
