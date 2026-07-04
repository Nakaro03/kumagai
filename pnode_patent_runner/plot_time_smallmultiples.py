"""
Time Small Multiples — 各年の Φ 場を小パネルで並べる.
"""
from __future__ import annotations

import os, sys, glob
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

DATA_PT = "data/PNode_Patent_Energy_X1_top50/alltime/fate_train.pt"
ROOT = "RESULTS/PNode_Patent_Energy_X1_top50"
TAG_SUFFIX = "-x1_v1.0_g0.1_b0.01"
SEED = 42
YEAR_BASE = 2013
GRID_RES = 60


def main():
    pat = f"{ROOT}/*{TAG_SUFFIX}/seed_{SEED}/alltime"
    cands = [Path(p) for p in glob.glob(pat) if "_lx" not in Path(p).parent.name]
    out_dir = cands[0]

    print("Loading model + data...")
    data = torch.load(DATA_PT, weights_only=False)
    xp = data["xp"]
    y = data["y"]
    centroids = data["centroids"]
    n_topics = data["n_topics"]
    growth = data["growth"]
    n_T = len(y)

    config = SimpleNamespace(**torch.load(out_dir / "config.pt", weights_only=False))
    config.x_dim = xp[0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardSDE(config).to(device)
    ckpts = sorted(glob.glob(str(out_dir / "train.epoch_*.pt")))
    ckpt = ckpts[-1] if ckpts else str(out_dir / "train.best.pt")
    state = torch.load(ckpt, weights_only=False, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Common UMAP (fit once on union)
    print("UMAP fitting...")
    import umap
    rng = np.random.RandomState(42)
    x_samples = [v.numpy()[rng.choice(len(v), min(800, len(v)), replace=False)] for v in xp]
    x_all = np.concatenate(x_samples)
    cent_all = np.concatenate([centroids[t].numpy() for t in range(n_T)])
    big = np.concatenate([x_all, cent_all])
    um = umap.UMAP(n_components=2, n_neighbors=30, metric="euclidean",
                   random_state=42, transform_seed=42)
    big_2d = um.fit_transform(big)
    obs_2d = big_2d[:len(x_all)]
    cent_2d_all = big_2d[len(x_all):]

    # Compute Φ at all obs points for each year
    print("Computing Φ for all years...")
    x_dev = torch.tensor(x_all, dtype=torch.float32, device=device)
    phi_per_t = []
    for k in range(n_T):
        t_col = torch.full((x_dev.shape[0], 1), float(y[k]), device=device)
        with torch.enable_grad():
            xt = torch.cat([x_dev, t_col], dim=1).requires_grad_()
            phi = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()
        phi_per_t.append(-phi)   # invert sign: high = growth

    # Common color range across all years (for fair comparison)
    all_phi = np.concatenate(phi_per_t)
    H_lo = np.percentile(all_phi, 5)
    H_hi = np.percentile(all_phi, 95)

    # Grid in 2D
    pad = 0.5
    x_min, x_max = obs_2d[:, 0].min() - pad, obs_2d[:, 0].max() + pad
    y_min, y_max = obs_2d[:, 1].min() - pad, obs_2d[:, 1].max() + pad
    gx = np.linspace(x_min, x_max, GRID_RES)
    gy = np.linspace(y_min, y_max, GRID_RES)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

    # Compute H grid per year
    print("Interpolating Φ grids per year...")
    H_per_t = []
    for k in range(n_T):
        lin = LinearNDInterpolator(obs_2d, phi_per_t[k])
        nn = NearestNDInterpolator(obs_2d, phi_per_t[k])
        H = lin(grid_pts)
        nan_mask = np.isnan(H); H[nan_mask] = nn(grid_pts[nan_mask])
        H_per_t.append(np.clip(H.reshape(GX.shape), H_lo, H_hi))

    # ── Figure: 3 rows × 4 cols = 12 panels (year 2013-2024)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    cmap = matplotlib.colormaps["RdBu_r"]
    levels = np.linspace(H_lo, H_hi, 25)

    for k in range(n_T):
        ax = axes[k]
        cf = ax.contourf(GX, GY, H_per_t[k], levels=levels, cmap=cmap, alpha=0.92, extend="both")
        ax.contour(GX, GY, H_per_t[k], levels=8, colors="black", linewidths=0.3, alpha=0.45)

        # Topic centroids for this year
        cent_2d_t = cent_2d_all[k * n_topics:(k + 1) * n_topics]
        active = centroids[k].numpy().sum(axis=-1) != 0
        g_t = growth[k].numpy()[active]
        g_max = max(abs(g_t.max()), abs(g_t.min())) + 1e-6 if len(g_t) else 1.0
        ax.scatter(cent_2d_t[active, 0], cent_2d_t[active, 1], c=g_t, cmap="RdYlGn",
                   s=18, edgecolors="black", linewidth=0.4, vmin=-g_max, vmax=g_max, zorder=5)

        year = YEAR_BASE + int(y[k])
        ax.set_title(f"{year}", fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        # Year-progress bar
        from matplotlib.patches import Rectangle
        prog = (k + 1) / n_T
        ax.add_patch(Rectangle((x_min, y_min), (x_max - x_min) * prog, 0.3,
                                color="#1f4f7a", alpha=0.4, zorder=10))

    # Hide unused panels
    for k in range(n_T, len(axes)):
        axes[k].axis("off")

    plt.suptitle("Patent Energy CPC Y02 — 学習済み Φ 場 (-Φ) の年次変化  "
                 "(赤 = 成長期待ゾーン / 青 = 衰退ゾーン / 共通カラースケール)",
                 fontsize=14, fontweight="bold", y=0.995)

    # Single colorbar
    cbar = fig.colorbar(cf, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("成長ポテンシャル (-Φ_θ)", fontsize=10)

    out = Path("RESULTS/fig14_time_smallmultiples.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")

    import shutil
    shutil.copy(out, "figures/fig14_time_smallmultiples.png")
    print("Copied to figures/")


if __name__ == "__main__":
    main()
