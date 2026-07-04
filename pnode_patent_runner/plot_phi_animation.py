"""
Φ(x, t) 地形のアニメーション — Patent Energy 学習済みモデルで
2013-2024 年の年次変動を可視化。
出力: GIF + 各フレーム PNG。
"""
from __future__ import annotations

import os, sys, glob
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager

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

CPC_LABELS = {
    "Y02A": "気候適応", "Y02B": "建物省エネ", "Y02C": "温室効果ガス削減",
    "Y02D": "ICT省エネ", "Y02E": "クリーンエネルギー", "Y02P": "製造省エネ",
    "Y02T": "交通省エネ・EV", "Y02W": "廃棄物・水",
}

def cat(n):
    return CPC_LABELS.get(n[:4], "その他") if n else "その他"


def main():
    pat = f"{ROOT}/*{TAG_SUFFIX}/seed_{SEED}/alltime"
    cands = [Path(p) for p in glob.glob(pat) if "_lx" not in Path(p).parent.name]
    out_dir = cands[0]
    out_gif = Path("RESULTS/phi_animation_patent_energy.gif")
    frames_dir = Path("RESULTS/phi_animation_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATA_PT}...")
    data = torch.load(DATA_PT, weights_only=False)
    xp = data["xp"]
    y = data["y"]
    n_T = len(y)
    centroids = data["centroids"]
    topic_names = data["topic_names"]
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

    # UMAP one-shot fit on full point cloud
    print("UMAP fitting (this may take a minute)...")
    import umap
    x_all = torch.cat(xp).numpy()
    um = umap.UMAP(n_components=2, n_neighbors=30, metric="euclidean",
                   random_state=42, transform_seed=42)
    x_all_2d = um.fit_transform(x_all)
    print(f"  shape: {x_all_2d.shape}")

    # 各時点 t での Φ at all observation points を pre-compute (12 × 48000 評価)
    print("Pre-computing Φ for all (point, time) combinations...")
    phi_per_t = []
    x_all_tensor = torch.tensor(x_all, dtype=torch.float32, device=device)
    for k in range(n_T):
        with torch.enable_grad():
            t_col = torch.full((x_all_tensor.shape[0], 1), float(y[k]), device=device)
            xt = torch.cat([x_all_tensor, t_col], dim=1).requires_grad_()
            phi_k = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()
        phi_per_t.append(phi_k)
        print(f"  t={k} ({YEAR_BASE+k}): Φ range [{phi_k.min():.2f}, {phi_k.max():.2f}]")

    # 共通カラースケール (全期間で固定)
    all_phi = np.concatenate(phi_per_t)
    vmin = np.percentile(all_phi, 2)
    vmax = np.percentile(all_phi, 98)
    print(f"Color range: vmin={vmin:.2f}, vmax={vmax:.2f}")

    # Centroids per year, projected
    cent_2d_per_t = []
    for k in range(n_T):
        c_k = centroids[k].numpy()
        active = c_k.sum(-1) != 0
        c_act = c_k[active]
        if len(c_act):
            c_2d = um.transform(c_act)
        else:
            c_2d = np.empty((0, 2))
        cent_2d_per_t.append((c_2d, active, c_act))

    # アニメ用 figure
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1])

    def render_frame(k):
        ax.clear(); ax_side.clear()

        # メイン: Φ heatmap as colored scatter
        order = np.argsort(phi_per_t[k])
        sc = ax.scatter(x_all_2d[order, 0], x_all_2d[order, 1],
                        c=phi_per_t[k][order], s=2.5, cmap="RdYlBu_r",
                        vmin=vmin, vmax=vmax, alpha=0.55)

        # Centroids at this year
        c_2d, active_mask, _ = cent_2d_per_t[k]
        g_k = growth[k].numpy()[active_mask]
        names_act = [topic_names[i] for i in range(n_topics) if active_mask[i]]
        if len(c_2d):
            g_vmax = np.percentile(np.abs(g_k), 95) if g_k.size else 1.0
            ax.scatter(c_2d[:, 0], c_2d[:, 1], c=g_k, cmap="RdYlGn",
                       s=140, edgecolors="black", linewidths=1.0,
                       vmin=-g_vmax, vmax=g_vmax, zorder=5)
            # annotate TOP-3 growers + TOP-3 decliners
            if len(g_k):
                order_g = np.argsort(-g_k)
                for ii, i in enumerate(order_g[:3]):
                    ax.annotate(f"↑ {names_act[i]}\n({cat(names_act[i])})",
                                (c_2d[i, 0], c_2d[i, 1]),
                                xytext=(10, 10), textcoords="offset points",
                                fontsize=8, color="#0a5",
                                bbox=dict(facecolor="white", edgecolor="#0a5",
                                          alpha=0.85, boxstyle="round,pad=0.2"))
                for i in order_g[-3:]:
                    ax.annotate(f"↓ {names_act[i]}\n({cat(names_act[i])})",
                                (c_2d[i, 0], c_2d[i, 1]),
                                xytext=(10, -25), textcoords="offset points",
                                fontsize=8, color="#a00",
                                bbox=dict(facecolor="white", edgecolor="#a00",
                                          alpha=0.85, boxstyle="round,pad=0.2"))
        year = YEAR_BASE + int(y[k])
        ax.set_title(f"Φ(x, t)  技術地形図   t = {k}  /  year = {year}",
                     fontsize=15, fontweight="bold")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(x_all_2d[:, 0].min() - 1, x_all_2d[:, 0].max() + 1)
        ax.set_ylim(x_all_2d[:, 1].min() - 1, x_all_2d[:, 1].max() + 1)

        # サイドパネル: TOP-5 / BOTTOM-5 成長 topic
        ax_side.axis("off")
        ax_side.text(0.5, 0.97, f"年: {year}", ha="center", fontsize=18,
                     fontweight="bold", transform=ax_side.transAxes)
        if len(g_k):
            order_g = np.argsort(-g_k)
            ax_side.text(0.03, 0.88, "成長 TOP-5", fontsize=12, fontweight="bold",
                         color="#0a5", transform=ax_side.transAxes)
            for ii, i in enumerate(order_g[:5]):
                ax_side.text(0.05, 0.84 - ii * 0.04,
                             f"{ii+1}. {names_act[i]:<14}  g={g_k[i]:+.2f}  [{cat(names_act[i])}]",
                             fontsize=9.5, transform=ax_side.transAxes,
                             color="#073" if g_k[i] > 0 else "#444")

            ax_side.text(0.03, 0.45, "衰退 BOTTOM-5", fontsize=12, fontweight="bold",
                         color="#a00", transform=ax_side.transAxes)
            for ii, i in enumerate(order_g[-5:][::-1]):
                ax_side.text(0.05, 0.41 - ii * 0.04,
                             f"{ii+1}. {names_act[i]:<14}  g={g_k[i]:+.2f}  [{cat(names_act[i])}]",
                             fontsize=9.5, transform=ax_side.transAxes,
                             color="#a00")

        # 進行バー
        ax_side.barh([0.04], [(k + 1) / n_T], left=0.0, height=0.04,
                     color="#1f4f7f", transform=ax_side.transAxes)
        ax_side.text(0.5, 0.005, f"  t = {k+1} / {n_T}",
                     fontsize=9, ha="center", transform=ax_side.transAxes, color="#444")

        return [sc]

    print(f"\nRendering {n_T} frames...")
    fig.suptitle("クリーンエネルギー特許 — Φ 技術地形 アニメーション (Patent Energy CPC Y02)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save individual PNGs (for debugging / paper supp)
    for k in range(n_T):
        render_frame(k)
        path = frames_dir / f"frame_{k:02d}_year_{YEAR_BASE+int(y[k])}.png"
        plt.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
        print(f"  frame {k} -> {path}")

    # GIF
    ani = animation.FuncAnimation(fig, render_frame, frames=n_T,
                                  interval=1200, blit=False, repeat=True)
    print(f"\nWriting GIF -> {out_gif}")
    ani.save(out_gif, writer="pillow", fps=1, dpi=100)
    print(f"Saved GIF -> {out_gif}")


if __name__ == "__main__":
    main()
