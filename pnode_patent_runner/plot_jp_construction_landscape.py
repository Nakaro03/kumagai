"""
JP Construction X1 学習済みモデルの UMAP 可視化。
plot_pisde_x1_landscape.py の JP 対応版。
"""
from __future__ import annotations

import os, sys, glob, warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

# Japanese-capable font (IPA / Noto CJK) discovery
def _try_set_jp_font():
    for cand in ["Noto Sans CJK JP", "IPAGothic", "IPAPGothic",
                 "TakaoGothic", "VL PGothic", "IPAexGothic",
                 "Hiragino Sans", "Yu Gothic"]:
        try:
            if cand in [f.name for f in font_manager.fontManager.ttflist]:
                plt.rcParams["font.family"] = cand
                return cand
        except Exception:
            pass
    return None

JP_FONT = _try_set_jp_font()
print(f"JP font: {JP_FONT}")

from src.model import ForwardSDE
from types import SimpleNamespace

DATA_PT = "data/PNode_JP_Construction_X1/alltime/fate_train.pt"
ROOT = "RESULTS/PNode_JP_Construction_X1"
TAG_SUFFIX = "-x1_v1.0_g0.1_b0.01"
SEED = 42
LAST_T = 10
YEAR_BASE = 2015


def find_outdir():
    pat = f"{ROOT}/*{TAG_SUFFIX}/seed_{SEED}/alltime"
    cands = [Path(p) for p in glob.glob(pat) if "_lx" not in Path(p).parent.name]
    if not cands:
        raise FileNotFoundError(f"no checkpoint at {pat}")
    return cands[0]


def main():
    out_dir = find_outdir()
    out_png = out_dir / "trajectories_jp_landscape.png"

    print(f"Loading {DATA_PT}...")
    data = torch.load(DATA_PT, weights_only=False)
    xp = data["xp"]
    y  = data["y"]
    topics = data["topics"]
    topic_names = data["topic_names"]
    centroids = data["centroids"]
    growth = data["growth"]
    n_topics = data["n_topics"]

    config = SimpleNamespace(**torch.load(out_dir / "config.pt", weights_only=False))
    config.x_dim = xp[0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardSDE(config).to(device)
    ckpts = sorted(glob.glob(str(out_dir / "train.epoch_*.pt")))
    ckpt = ckpts[-1] if ckpts else str(out_dir / "train.best.pt")
    state = torch.load(ckpt, weights_only=False, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Loaded: {ckpt}")

    # UMAP
    print("UMAP fitting on full point cloud...")
    import umap
    x_all = torch.cat(xp).numpy()
    um = umap.UMAP(n_components=2, n_neighbors=30, metric="euclidean",
                   random_state=42, transform_seed=42)
    x_all_2d = um.fit_transform(x_all)
    y_all = np.concatenate([np.full(v.shape[0], y[k]) for k, v in enumerate(xp)])

    # Φ on all points
    xt_all = torch.cat([torch.tensor(x_all, dtype=torch.float32),
                        torch.tensor(y_all, dtype=torch.float32).unsqueeze(1)], dim=1)
    phi_all = model._func._pot(xt_all.to(device).requires_grad_()).squeeze(-1).detach().cpu().numpy()

    # Centroids
    cent_t = centroids[LAST_T].numpy()
    active = cent_t.sum(axis=-1) != 0
    cent_active = cent_t[active]
    cent_2d = um.transform(cent_active)
    g_t = growth[LAST_T].numpy()[active]
    names_active = [topic_names[i] for i in range(n_topics) if active[i]]

    xt_cent = torch.cat([torch.tensor(cent_active, dtype=torch.float32),
                         torch.full((len(cent_active), 1), float(y[LAST_T]))], dim=1)
    phi_cent = model._func._pot(xt_cent.to(device).requires_grad_()).squeeze(-1).detach().cpu().numpy()

    r, p_val = stats.spearmanr(phi_cent, g_t)

    fig = plt.figure(figsize=(22, 6))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1.2, 1], wspace=0.25)

    # [A] year color
    ax0 = plt.subplot(gs[0, 0])
    yrs = y_all.astype(int)
    n_yrs = len(set(yrs))
    for yt in sorted(set(yrs)):
        m = yrs == yt
        ax0.scatter(x_all_2d[m, 0], x_all_2d[m, 1], s=2, alpha=0.35,
                    color=plt.cm.viridis(yt / max(1, n_yrs - 1)),
                    label=f"t={yt} ({YEAR_BASE+yt})")
    ax0.set_title("[A] Observed papers by year", fontsize=11, fontweight="bold")
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.legend(fontsize=7, ncol=2)

    # [B] Φ heatmap
    ax1 = plt.subplot(gs[0, 1])
    order = np.argsort(phi_all)
    sc = ax1.scatter(x_all_2d[order, 0], x_all_2d[order, 1], c=phi_all[order],
                     s=2, cmap="RdYlBu_r")
    ax1.set_title("[B] X1-trained Φ(x, t)\n  (low = valley = predicted growing)",
                  fontsize=11, fontweight="bold")
    ax1.set_xticks([]); ax1.set_yticks([])
    plt.colorbar(sc, ax=ax1, label="Φ", fraction=0.04)

    # [C] Centroids + g labels
    ax2 = plt.subplot(gs[0, 2])
    ax2.scatter(x_all_2d[:, 0], x_all_2d[:, 1], s=1, color="lightgray", alpha=0.3)
    vmax_g = np.percentile(np.abs(g_t), 95)
    sc2 = ax2.scatter(cent_2d[:, 0], cent_2d[:, 1], c=g_t, cmap="RdYlGn",
                      s=160, edgecolors="black", linewidths=1.0,
                      vmin=-vmax_g, vmax=vmax_g, zorder=5)
    # label: shorten topic_name
    for i, nm in enumerate(names_active):
        short = nm.split(":")[-1][:14] if ":" in nm else nm[:14]
        fc = "white" if abs(g_t[i]) > 0.3 * vmax_g else "black"
        fs = 7.5 if abs(g_t[i]) > 0.5 * vmax_g else 6
        ax2.annotate(short, (cent_2d[i, 0], cent_2d[i, 1]),
                     ha="center", va="center", fontsize=fs, color=fc)
    ax2.set_title(f"[C] Topic centroids @ t={LAST_T} ({YEAR_BASE+LAST_T})\n  Color = actual growth rate g_j",
                  fontsize=11, fontweight="bold")
    ax2.set_xticks([]); ax2.set_yticks([])
    plt.colorbar(sc2, ax=ax2, label="g_j", fraction=0.04)

    # [D] rank scatter
    ax3 = plt.subplot(gs[0, 3])
    phi_rank = np.argsort(np.argsort(phi_cent))
    g_rank   = np.argsort(np.argsort(-g_t))
    ax3.scatter(phi_rank, g_rank, c=g_t, cmap="RdYlGn", s=80,
                edgecolors="black", linewidths=0.6, vmin=-vmax_g, vmax=vmax_g)
    n = len(phi_cent)
    ax3.plot([0, n-1], [0, n-1], "--", color="gray", lw=1, label="Perfect (R=-1)")
    ax3.set_title(f"[D] Φ rank vs Growth rank\n  Spearman r = {r:+.3f} (p={p_val:.4g})",
                  fontsize=11, fontweight="bold")
    ax3.set_xlabel("Φ rank (low=valley)"); ax3.set_ylabel("g rank (high=growing)")
    ax3.legend(fontsize=8)

    plt.suptitle(f"JP Construction (J-STAGE) — PI-SDE + X1  (seed={SEED}, year {YEAR_BASE+LAST_T})",
                 fontsize=13, fontweight="bold", y=1.02)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
