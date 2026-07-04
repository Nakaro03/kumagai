"""
3 ドメイン (Paper / Patent Energy / Patent Construction) の X1 学習済み
ベクトル場 + Φ景観 を一枚に並べた可視化 (PI-SDE 論文 Plot_vector.ipynb 形式)。

3 行 × 3 列:
  各行: Paper / Patent Energy / Patent Construction
  各列:
    [A] 観測点 UMAP scatter (年色分け)
    [B] Φ heatmap (UMAP scatter colored by Φ)
    [C] Drift vector field -∇Φ (UMAP-projected per-sample arrows)
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

warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

from src.model import ForwardSDE
from types import SimpleNamespace

# ── 3 ドメイン設定 ───────────────────────────────────────────────
DOMAINS = [
    {
        "name": "Paper (ArXiv CS)",
        "ckpt_dir": "RESULTS/PNode_Paper_X1/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01/seed_42/alltime",
        "data_pt":  "data/PNode_Paper_X1/alltime/fate_train.pt",
        "n_topics_label": "n_topics=32",
    },
    {
        "name": "Patent Energy",
        "ckpt_dir": "RESULTS/PNode_Patent_Energy_X1_top50/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01/seed_42/alltime",
        "data_pt":  "data/PNode_Patent_Energy_X1_top50/alltime/fate_train.pt",
        "n_topics_label": "n_topics=50",
    },
    {
        "name": "Patent Construction",
        "ckpt_dir": "RESULTS/PNode_Patent_Construction_X1_top50/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01/seed_42/alltime",
        "data_pt":  "data/PNode_Patent_Construction_X1_top50/alltime/fate_train.pt",
        "n_topics_label": "n_topics=50",
    },
]

OUT_PNG = Path("RESULTS/x1_3domain_vectorfield.png")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_domain(cfg):
    """ドメインのデータとモデルをロード"""
    data = torch.load(cfg["data_pt"], weights_only=False)
    config = SimpleNamespace(**torch.load(Path(cfg["ckpt_dir"]) / "config.pt", weights_only=False))
    config.x_dim = data["xp"][0].shape[-1]

    model = ForwardSDE(config).to(device)
    # epoch_000200 or epoch_000500
    import glob
    ckpts = sorted(glob.glob(str(Path(cfg["ckpt_dir"]) / "train.epoch_*.pt")))
    ckpt_path = ckpts[-1] if ckpts else str(Path(cfg["ckpt_dir"]) / "train.best.pt")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return data, model, config, ckpt_path


def compute_umap_and_phi(data, model, config):
    """UMAP 投影 + Φ 計算"""
    import umap
    xp = data["xp"]; y = data["y"]
    x_all = torch.cat(xp).numpy()
    y_all = np.concatenate([np.full(v.shape[0], y[k]) for k, v in enumerate(xp)])

    # UMAP
    um = umap.UMAP(n_components=2, n_neighbors=30, metric="euclidean",
                   random_state=42, transform_seed=42)
    x_all_2d = um.fit_transform(x_all)

    # Φ at all points
    xt_all = torch.cat([torch.tensor(x_all, dtype=torch.float32),
                        torch.tensor(y_all, dtype=torch.float32).unsqueeze(1)], dim=1)
    phi_all = model._func._pot(xt_all.to(device).requires_grad_()).squeeze(-1).detach().cpu().numpy()

    return x_all_2d, y_all, phi_all, um


def compute_drift_arrows(data, model, um, n_per_year=25):
    """各時点でサンプル点の drift を計算し UMAP に投影"""
    np.random.seed(42)
    arrows = []
    xp = data["xp"]; y = data["y"]
    for k, t_val in enumerate(y):
        n_samp = min(n_per_year, xp[k].shape[0])
        idx = np.random.choice(xp[k].shape[0], n_samp, replace=False)
        x_samp = xp[k][idx]
        t_col = torch.full((n_samp, 1), float(t_val))
        Xt = torch.cat([x_samp, t_col], dim=1).to(device).requires_grad_()
        drift_x = model._func._drift(Xt).detach().cpu().numpy()
        X_start = x_samp.cpu().numpy()
        X_end = X_start + drift_x
        x_start_umap = um.transform(X_start)
        x_end_umap = um.transform(X_end)
        xv = x_end_umap - x_start_umap
        norm = np.linalg.norm(xv, axis=1, keepdims=True)
        xv = xv / (norm + 1e-10) * 0.7
        arrows.append({"start": x_start_umap, "v": xv, "t": k})
    return arrows


# ── プロット (3 row × 3 col) ─────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(len(DOMAINS), 3, figure=fig, hspace=0.32, wspace=0.18,
              left=0.04, right=0.92, top=0.95, bottom=0.04)

print("Generating domain visualizations...")
for row_idx, cfg in enumerate(DOMAINS):
    print(f"\n[{row_idx+1}/{len(DOMAINS)}] {cfg['name']}")
    data, model, config, ckpt_path = load_domain(cfg)
    print(f"  ckpt: {ckpt_path.split('/')[-1]}")
    x_all_2d, y_all, phi_all, um = compute_umap_and_phi(data, model, config)
    print(f"  Φ range: [{phi_all.min():.3f}, {phi_all.max():.3f}]")
    arrows = compute_drift_arrows(data, model, um, n_per_year=25)

    # [A] 年色分け
    ax0 = fig.add_subplot(gs[row_idx, 0])
    n_y = len(data["y"])
    for k in range(n_y):
        mask = y_all == data["y"][k]
        ax0.scatter(x_all_2d[mask, 0], x_all_2d[mask, 1], s=2, alpha=0.4,
                    color=plt.cm.viridis(k / max(n_y - 1, 1)),
                    label=f"t={int(data['y'][k])}" if k in (0, n_y-1) else None)
    ax0.set_title(f"{cfg['name']}  ({cfg['n_topics_label']})\n[A] Observed (year-coded)",
                   fontsize=10)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_ylabel("UMAP2", fontsize=9)
    ax0.legend(fontsize=7, loc="upper right")

    # [B] Φ heatmap
    ax1 = fig.add_subplot(gs[row_idx, 1])
    ci = np.argsort(phi_all)
    sc = ax1.scatter(x_all_2d[ci, 0], x_all_2d[ci, 1], c=phi_all[ci],
                      s=2, cmap="RdYlBu_r")
    ax1.set_title(f"[B] Learned Φ (low=valley=growing)\nΦ range: [{phi_all.min():.2f}, {phi_all.max():.2f}]",
                   fontsize=10)
    ax1.set_xticks([]); ax1.set_yticks([])
    plt.colorbar(sc, ax=ax1, fraction=0.04, label="Φ")

    # [C] Drift vector field
    ax2 = fig.add_subplot(gs[row_idx, 2])
    ax2.scatter(x_all_2d[:, 0], x_all_2d[:, 1], s=1, color="gray", alpha=0.25)
    for arr in arrows:
        color = plt.cm.viridis(arr["t"] / max(n_y - 1, 1))
        ax2.quiver(arr["start"][:, 0], arr["start"][:, 1],
                   arr["v"][:, 0], arr["v"][:, 1],
                   scale=1.5, scale_units="xy", width=0.005,
                   color=color, alpha=0.85)
    ax2.set_title(f"[C] Drift -∇Φ (arrows = research flow)\n5 seed mean Spearman: see below",
                   fontsize=10)
    ax2.set_xticks([]); ax2.set_yticks([])

    if row_idx == len(DOMAINS) - 1:
        ax0.set_xlabel("UMAP1", fontsize=9)
        ax1.set_xlabel("UMAP1", fontsize=9)
        ax2.set_xlabel("UMAP1", fontsize=9)

fig.suptitle(
    "PI-SDE + X1 Topic-Anchored Vector Fields  |  3 Domains × seed=42 (representative)\n"
    "[A] year-coded observations   [B] learned Φ (blue=valley=growing)   "
    "[C] drift −∇Φ (UMAP-projected, colored by time)",
    fontsize=12, fontweight="bold", y=0.99,
)

fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")
