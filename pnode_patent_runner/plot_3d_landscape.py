"""
3D 技術地形 — Φ を z 軸として真の山岳地形を表示.
"""
from __future__ import annotations

import os, sys, glob
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import font_manager
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

for cand in ["Noto Sans CJK JP", "IPAGothic", "TakaoGothic"]:
    if cand in [f.name for f in font_manager.fontManager.ttflist]:
        plt.rcParams["font.family"] = cand
        break

from src.model import ForwardSDE
from types import SimpleNamespace

CPC_LABELS = {
    "Y02A": "気候適応", "Y02B": "建物省エネ", "Y02C": "GHG 削減",
    "Y02D": "ICT 省エネ", "Y02E": "クリーンエネ", "Y02P": "製造省エネ",
    "Y02T": "交通省エネ", "Y02W": "廃棄物",
}

DATA_PT = "data/PNode_Patent_Energy_X1_top50/alltime/fate_train.pt"
ROOT = "RESULTS/PNode_Patent_Energy_X1_top50"
TAG_SUFFIX = "-x1_v1.0_g0.1_b0.01"
SEED = 42
YEAR_BASE = 2013
LAST_T = 11
GRID_RES = 70


def main():
    pat = f"{ROOT}/*{TAG_SUFFIX}/seed_{SEED}/alltime"
    cands = [Path(p) for p in glob.glob(pat) if "_lx" not in Path(p).parent.name]
    out_dir = cands[0]

    print("Loading...")
    data = torch.load(DATA_PT, weights_only=False)
    xp = data["xp"]
    y = data["y"]
    topic_names = data["topic_names"]
    centroids = data["centroids"]
    n_topics = data["n_topics"]
    growth = data["growth"]

    config = SimpleNamespace(**torch.load(out_dir / "config.pt", weights_only=False))
    config.x_dim = xp[0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardSDE(config).to(device)
    ckpts = sorted(glob.glob(str(out_dir / "train.epoch_*.pt")))
    ckpt = ckpts[-1] if ckpts else str(out_dir / "train.best.pt")
    state = torch.load(ckpt, weights_only=False, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # UMAP
    print("UMAP...")
    import umap
    rng = np.random.RandomState(42)
    x_samples = [v.numpy()[rng.choice(len(v), min(800, len(v)), replace=False)] for v in xp]
    x_all = np.concatenate(x_samples)
    cent_act_all = []
    for k in range(len(centroids)):
        mask = centroids[k].numpy().sum(axis=-1) != 0
        cent_act_all.append(centroids[k].numpy()[mask])
    cent_concat = np.concatenate(cent_act_all)
    big = np.concatenate([x_all, cent_concat])
    um = umap.UMAP(n_components=2, n_neighbors=30, random_state=42, transform_seed=42)
    big_2d = um.fit_transform(big)
    obs_2d = big_2d[:len(x_all)]

    # Φ at obs at LAST_T
    x_dev = torch.tensor(x_all, dtype=torch.float32, device=device)
    t_col = torch.full((x_dev.shape[0], 1), float(y[LAST_T]), device=device)
    with torch.enable_grad():
        xt = torch.cat([x_dev, t_col], dim=1).requires_grad_()
        phi_obs = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()
    height_obs = -phi_obs

    # Centroids at LAST_T projected
    cent_last = centroids[LAST_T].numpy()
    active = cent_last.sum(axis=-1) != 0
    cent_act = cent_last[active]
    names_act = [topic_names[i] for i in range(n_topics) if active[i]]
    g_act = growth[LAST_T].numpy()[active]
    # Project using same UMAP (use the centroids portion of big_2d via re-fit transform)
    cent_2d = um.transform(cent_act)
    # Φ at centroids
    cent_dev = torch.tensor(cent_act, dtype=torch.float32, device=device)
    t_col2 = torch.full((cent_dev.shape[0], 1), float(y[LAST_T]), device=device)
    with torch.enable_grad():
        xt2 = torch.cat([cent_dev, t_col2], dim=1).requires_grad_()
        phi_cent = model._func._pot(xt2).squeeze(-1).detach().cpu().numpy()
    height_cent = -phi_cent

    # Grid
    pad = 0.5
    x_min, x_max = obs_2d[:, 0].min() - pad, obs_2d[:, 0].max() + pad
    y_min, y_max = obs_2d[:, 1].min() - pad, obs_2d[:, 1].max() + pad
    gx = np.linspace(x_min, x_max, GRID_RES)
    gy = np.linspace(y_min, y_max, GRID_RES)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    lin = LinearNDInterpolator(obs_2d, height_obs)
    nn = NearestNDInterpolator(obs_2d, height_obs)
    H = lin(grid_pts)
    nan_mask = np.isnan(H); H[nan_mask] = nn(grid_pts[nan_mask])
    H = H.reshape(GX.shape)
    H_lo, H_hi = np.percentile(H, 5), np.percentile(H, 95)
    H = np.clip(H, H_lo, H_hi)

    # ── 3D figure with 2 viewing angles
    fig = plt.figure(figsize=(20, 9))

    for panel_idx, (elev, azim, title) in enumerate([
        (35, -60, "斜め視点 (北東より)"),
        (35, 35, "斜め視点 (南西より)")
    ]):
        ax = fig.add_subplot(1, 2, panel_idx + 1, projection="3d")
        # Surface plot
        cmap = matplotlib.colormaps["terrain"]
        surf = ax.plot_surface(GX, GY, H, cmap=cmap, alpha=0.85,
                               linewidth=0, antialiased=True, rstride=2, cstride=2)
        # Contour underneath
        ax.contour(GX, GY, H, zdir="z", offset=H_lo - 0.05, levels=10,
                   colors="gray", alpha=0.4, linewidths=0.5)

        # Topic centroids as 3D balls
        g_max = max(abs(g_act.max()), abs(g_act.min())) + 1e-6
        sc = ax.scatter(cent_2d[:, 0], cent_2d[:, 1], height_cent,
                        c=g_act, cmap="RdYlGn", vmin=-g_max, vmax=g_max,
                        s=70, edgecolors="black", linewidth=0.8, zorder=10)

        # Annotate top 3 growers (only on first panel for clarity)
        if panel_idx == 0:
            order = np.argsort(-g_act)
            for i in order[:3]:
                cat = CPC_LABELS.get(names_act[i][:4], "")
                ax.text(cent_2d[i, 0], cent_2d[i, 1], height_cent[i] + 0.05,
                        f"  ★ {names_act[i]}\n  ({cat})",
                        fontsize=8, color="#073", fontweight="bold")
            for i in order[-2:]:
                cat = CPC_LABELS.get(names_act[i][:4], "")
                ax.text(cent_2d[i, 0], cent_2d[i, 1], height_cent[i] + 0.05,
                        f"  ↓ {names_act[i]}\n  ({cat})",
                        fontsize=8, color="#a00")

        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.set_zlabel("成長ポテンシャル (-Φ)")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.view_init(elev=elev, azim=azim)

    fig.colorbar(surf, ax=fig.axes, fraction=0.025, pad=0.04,
                 label="標高 (= 成長ポテンシャル -Φ)")

    plt.suptitle(f"3D 技術地形 — Patent Energy CPC Y02, year {YEAR_BASE+LAST_T}\n"
                 "山頂 = 成長期待ゾーン / 谷底 = 衰退ゾーン / 球 = topic centroid (色 = 実成長率)",
                 fontsize=14, fontweight="bold", y=1.005)

    out = Path("RESULTS/fig15_3d_landscape.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")

    import shutil
    shutil.copy(out, "figures/fig15_3d_landscape.png")
    print("Copied to figures/")


if __name__ == "__main__":
    main()
