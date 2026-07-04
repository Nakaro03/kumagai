"""
2D 技術空間 + 学習済み Φ 等高線 (薄く背景表示).
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
from scipy.spatial import ConvexHull
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

for cand in ["Noto Sans CJK JP", "IPAGothic", "TakaoGothic"]:
    if cand in [f.name for f in font_manager.fontManager.ttflist]:
        plt.rcParams["font.family"] = cand
        break

from src.model import ForwardSDE
from types import SimpleNamespace

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
        "root": "RESULTS/PNode_Patent_Energy_X1_top50",
        "last_t": 11,
        "family_fn": lambda n: n[:4] if n else "?",
        "family_map": CPC_FAMILIES,
    },
    "JP Construction (J-STAGE, 2025)": {
        "data": "data/PNode_JP_Construction_X1/alltime/fate_train.pt",
        "root": "RESULTS/PNode_JP_Construction_X1",
        "last_t": 10,
        "family_fn": None,
        "family_map": None,
    },
}
TAG_SUFFIX = "-x1_v1.0_g0.1_b0.01"
SEED = 42
GRID_RES = 80


def find_model(root):
    pat = f"{root}/*{TAG_SUFFIX}/seed_{SEED}/alltime"
    cands = [Path(p) for p in glob.glob(pat) if "_lx" not in Path(p).parent.name]
    return cands[0]


def make_panel(ax, cfg, title):
    data = torch.load(cfg["data"], weights_only=False)
    xp = data["xp"]
    y = data["y"]
    centroids = data["centroids"]
    topic_names = data["topic_names"]
    n_topics = data["n_topics"]
    growth = data["growth"]
    last_t = cfg["last_t"]

    out_dir = find_model(cfg["root"])
    config = SimpleNamespace(**torch.load(out_dir / "config.pt", weights_only=False))
    config.x_dim = xp[0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardSDE(config).to(device)
    ckpts = sorted(glob.glob(str(out_dir / "train.epoch_*.pt")))
    ckpt = ckpts[-1] if ckpts else str(out_dir / "train.best.pt")
    state = torch.load(ckpt, weights_only=False, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    cent = centroids[last_t].numpy()
    active = cent.sum(axis=-1) != 0
    cent_act = cent[active]
    names_act = [topic_names[i] for i in range(n_topics) if active[i]]
    g_act = growth[last_t].numpy()[active]

    vol = np.zeros(n_topics)
    for k in range(len(centroids)):
        t_k = data["topics"][k].numpy()
        for j in range(n_topics):
            vol[j] += (t_k == j).sum()
    vol_act = vol[active]

    # Subsample observed points for UMAP
    rng = np.random.RandomState(42)
    x_samples = []
    for v in xp:
        n_take = min(800, len(v))
        idx = rng.choice(len(v), n_take, replace=False)
        x_samples.append(v.numpy()[idx])
    x_obs = np.concatenate(x_samples)

    # UMAP on union of observed points + centroids
    print(f"  UMAP for {title} ({len(x_obs)} obs + {len(cent_act)} centroids)...")
    import umap
    big = np.concatenate([x_obs, cent_act])
    um = umap.UMAP(n_components=2, n_neighbors=30, metric="euclidean",
                   random_state=42, transform_seed=42)
    big_2d = um.fit_transform(big)
    obs_2d = big_2d[:len(x_obs)]
    cent_2d = big_2d[len(x_obs):]

    # Compute Φ on observed points at last_t
    print(f"  computing Φ at t={last_t} ({y[last_t]:.0f})...")
    x_dev = torch.tensor(x_obs, dtype=torch.float32, device=device)
    t_col = torch.full((x_dev.shape[0], 1), float(y[last_t]), device=device)
    with torch.enable_grad():
        xt = torch.cat([x_dev, t_col], dim=1).requires_grad_()
        phi = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()
    height = -phi  # so peaks = growth

    # Interpolate to grid
    pad = 0.5
    x_min, x_max = obs_2d[:, 0].min() - pad, obs_2d[:, 0].max() + pad
    y_min, y_max = obs_2d[:, 1].min() - pad, obs_2d[:, 1].max() + pad
    gx = np.linspace(x_min, x_max, GRID_RES)
    gy = np.linspace(y_min, y_max, GRID_RES)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    lin = LinearNDInterpolator(obs_2d, height)
    nn = NearestNDInterpolator(obs_2d, height)
    H = lin(grid_pts)
    nan_mask = np.isnan(H); H[nan_mask] = nn(grid_pts[nan_mask])
    H = H.reshape(GX.shape)
    H_lo, H_hi = np.percentile(H, 3), np.percentile(H, 97)
    H = np.clip(H, H_lo, H_hi)

    # Background: very subtle Φ contour
    cmap = matplotlib.colormaps["RdBu_r"]
    levels = np.linspace(H_lo, H_hi, 25)
    cf = ax.contourf(GX, GY, H, levels=levels, cmap=cmap, alpha=0.32, extend="both")
    cs = ax.contour(GX, GY, H, levels=8, colors="gray", linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f", colors="#666")

    # Family hulls (Patent only)
    if cfg["family_fn"] is not None and cfg["family_map"] is not None:
        fam_of = [cfg["family_fn"](n) for n in names_act]
        for fam, (label, color) in cfg["family_map"].items():
            mask = np.array([f == fam for f in fam_of])
            if mask.sum() < 3: continue
            pts = cent_2d[mask]
            try:
                hull = ConvexHull(pts)
                hp = pts[hull.vertices]
                ax.fill(hp[:, 0], hp[:, 1], color=color, alpha=0.10,
                        edgecolor=color, lw=1.0, zorder=2)
                cx, cy = pts.mean(axis=0)
                ax.annotate(f"{fam}\n{label}", (cx, cy), fontsize=10,
                            fontweight="bold", color=color, ha="center",
                            va="center", zorder=3, alpha=0.85)
            except Exception:
                pass

    # Topic markers
    g_max = max(abs(g_act.max()), abs(g_act.min())) + 1e-6
    sizes = 90 + 700 * (vol_act / vol_act.max())
    sc = ax.scatter(cent_2d[:, 0], cent_2d[:, 1], c=g_act, cmap="RdYlGn",
                    s=sizes, edgecolors="black", linewidths=1.1,
                    vmin=-g_max, vmax=g_max, zorder=6, alpha=0.95)

    order = np.argsort(-g_act)
    for i in order[:3]:
        short = names_act[i].split(":")[-1][:18] if ":" in names_act[i] else names_act[i][:18]
        ax.annotate(f"↑ {short}\n  g={g_act[i]:+.2f}", cent_2d[i],
                    xytext=(12, 10), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#073",
                    bbox=dict(facecolor="white", edgecolor="#073", lw=1.0,
                              alpha=0.95, boxstyle="round,pad=0.25"),
                    zorder=7,
                    arrowprops=dict(arrowstyle="->", color="#073", lw=1.2))
    for i in order[-3:]:
        short = names_act[i].split(":")[-1][:18] if ":" in names_act[i] else names_act[i][:18]
        ax.annotate(f"↓ {short}\n  g={g_act[i]:+.2f}", cent_2d[i],
                    xytext=(12, -28), textcoords="offset points",
                    fontsize=9, color="#a00",
                    bbox=dict(facecolor="white", edgecolor="#a00", lw=1.0,
                              alpha=0.95, boxstyle="round,pad=0.25"),
                    zorder=7,
                    arrowprops=dict(arrowstyle="->", color="#a00", lw=1.2))

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title(title, fontsize=12, fontweight="bold")
    return sc


fig = plt.figure(figsize=(20, 9))
gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.18)

ax1 = fig.add_subplot(gs[0, 0])
sc1 = make_panel(ax1, DOMAINS["Patent Energy (CPC Y02, 2024)"], "Patent Energy 2D 技術空間  (背景: 学習済み Φ)")
cb1 = plt.colorbar(sc1, ax=ax1, fraction=0.04, pad=0.01)
cb1.set_label("実成長率 g", fontsize=10)

ax2 = fig.add_subplot(gs[0, 1])
sc2 = make_panel(ax2, DOMAINS["JP Construction (J-STAGE, 2025)"], "JP Construction 2D 技術空間  (背景: 学習済み Φ)")
cb2 = plt.colorbar(sc2, ax=ax2, fraction=0.04, pad=0.01)
cb2.set_label("実成長率 g", fontsize=10)

plt.suptitle("2D 技術空間 + Φ 等高線背景  "
             "(背景: -Φ_θ で赤=成長期待ゾーン, 青=衰退ゾーン / マーカー: 各 topic の実成長率)",
             fontsize=14, fontweight="bold", y=1.005)

out = Path("RESULTS/fig13_2d_phi_contour.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")

import shutil
shutil.copy(out, "figures/fig13_2d_phi_contour.png")
print("Copied to figures/")
