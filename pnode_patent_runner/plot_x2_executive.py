"""
X2 経営者向けダッシュボード — Multi-Task (Φ + growth + uncertainty) を活用.

4 パネル構成:
  [A] TOP-10 投資推奨 + 信頼区間 (g_pred ± φ_std)
  [B] BOTTOM-10 撤退検討 + 信頼区間
  [C] Φ landscape + g_pred 色分け (予測 vs 実測比較)
  [D] Uncertainty heatmap (どこが確信高い / 低い)
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
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

for cand in ["Noto Sans CJK JP", "IPAGothic", "TakaoGothic"]:
    if cand in [f.name for f in font_manager.fontManager.ttflist]:
        plt.rcParams["font.family"] = cand
        break

sys.path.insert(0, "pnode_patent_runner")
from run_pisde_x2 import TopicCrossAttention, GrowthHead, D_CTX, N_HEADS, N_MC
from src.model import ForwardSDE
from types import SimpleNamespace

CPC_LABELS = {
    "Y02A": "気候適応", "Y02B": "建物省エネ", "Y02C": "GHG削減",
    "Y02D": "ICT省エネ", "Y02E": "クリーンエネ", "Y02P": "製造省エネ",
    "Y02T": "交通省エネ", "Y02W": "廃棄物",
}

DOMAIN_CFG = {
    "Patent Energy CPC Y02": {
        "data": "data/PNode_Patent_Energy_X1_top50/alltime/fate_train.pt",
        "root": "RESULTS_X2/PNode_Patent_Energy_X1_top50",
        "last_t": 11, "year_label": "2024",
        "cpc_label_fn": lambda n: CPC_LABELS.get(n[:4], "?") if n else "?",
    },
}
DOMAIN = "Patent Energy CPC Y02"   # 主軸ドメイン
SEED = 42
TAG_GLOB = "*x2_v*"
GRID_RES = 70


def main():
    cfg = DOMAIN_CFG[DOMAIN]
    out_dir_pat = f"{cfg['root']}/{TAG_GLOB}/seed_{SEED}/alltime"
    cands = list(Path(".").glob(out_dir_pat))
    out_dir = cands[0]
    print(f"Loading X2 model from {out_dir}")

    data = torch.load(cfg["data"], weights_only=False)
    xp = data["xp"]
    y = data["y"]
    topic_names = data["topic_names"]
    centroids = data["centroids"]
    n_topics = data["n_topics"]
    growth = data["growth"]
    last_t = cfg["last_t"]

    config = SimpleNamespace(**torch.load(out_dir / "config.pt", weights_only=False))
    config.x_dim = xp[0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardSDE(config).to(device)
    attention = TopicCrossAttention(x_dim=config.x_dim, d_model=D_CTX, n_heads=N_HEADS).to(device)
    growth_head = GrowthHead(x_dim=config.x_dim, d_ctx=D_CTX, hidden=64, dropout=0.1).to(device)
    ckpts = sorted(glob.glob(str(out_dir / "train.epoch_*.pt")))
    ckpt_p = ckpts[-1] if ckpts else str(out_dir / "train.best.pt")
    state = torch.load(ckpt_p, weights_only=False, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    attention.load_state_dict(state["attention_state_dict"])
    growth_head.load_state_dict(state["growth_head_state_dict"])
    model.eval(); attention.eval(); growth_head.eval()

    # ── 予測 & uncertainty ──
    cent = centroids[last_t].numpy()
    active = cent.sum(axis=-1) != 0
    cent_act = cent[active]
    names_act = [topic_names[i] for i in range(n_topics) if active[i]]
    g_actual = growth[last_t].numpy()[active]
    growth_norm_act = data["growth_norm"][last_t][active]
    K = len(cent_act)

    c_dev = torch.tensor(cent_act, dtype=torch.float32, device=device).requires_grad_()
    g_n_dev = growth_norm_act.to(device)
    t_val = float(y[last_t])
    t_col = torch.ones(K, 1, device=device) * t_val
    xt = torch.cat([c_dev, t_col], dim=1)

    # Φ point estimate
    phi_mean = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()

    # Growth head prediction
    ctx = attention(c_dev, g_n_dev, t_val)
    g_pred_norm = growth_head(c_dev, t_val, ctx).detach().cpu().numpy()

    # Monte Carlo uncertainty (perturbation around centroid)
    sigma_mc = 0.05
    phi_samples = []
    g_samples = []
    for _ in range(N_MC):
        c_p = c_dev + torch.randn_like(c_dev) * sigma_mc
        xt_p = torch.cat([c_p, t_col], dim=1)
        phi_p = model._func._pot(xt_p).squeeze(-1).detach().cpu().numpy()
        ctx_p = attention(c_p, g_n_dev, t_val)
        g_p = growth_head(c_p, t_val, ctx_p).detach().cpu().numpy()
        phi_samples.append(phi_p)
        g_samples.append(g_p)
    phi_std = np.std(np.stack(phi_samples), axis=0)
    g_std = np.std(np.stack(g_samples), axis=0)

    # UMAP layout
    print("UMAP...")
    import umap
    rng = np.random.RandomState(42)
    x_samples = [v.numpy()[rng.choice(len(v), min(800, len(v)), replace=False)] for v in xp]
    x_obs = np.concatenate(x_samples)
    big = np.concatenate([x_obs, cent_act])
    um = umap.UMAP(n_components=2, n_neighbors=30, random_state=42, transform_seed=42)
    big_2d = um.fit_transform(big)
    obs_2d = big_2d[:len(x_obs)]
    cent_2d = big_2d[len(x_obs):]

    # Φ heatmap on grid
    x_dev = torch.tensor(x_obs, dtype=torch.float32, device=device)
    t_col2 = torch.full((x_dev.shape[0], 1), t_val, device=device)
    with torch.enable_grad():
        xt_all = torch.cat([x_dev, t_col2], dim=1).requires_grad_()
        phi_obs = model._func._pot(xt_all).squeeze(-1).detach().cpu().numpy()
    H_lo, H_hi = np.percentile(-phi_obs, 3), np.percentile(-phi_obs, 97)
    height_obs = np.clip(-phi_obs, H_lo, H_hi)
    pad = 0.5
    x_min, x_max = obs_2d[:, 0].min() - pad, obs_2d[:, 0].max() + pad
    y_min, y_max = obs_2d[:, 1].min() - pad, obs_2d[:, 1].max() + pad
    gx = np.linspace(x_min, x_max, GRID_RES); gy = np.linspace(y_min, y_max, GRID_RES)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    H = LinearNDInterpolator(obs_2d, height_obs)(pts)
    nan_m = np.isnan(H); H[nan_m] = NearestNDInterpolator(obs_2d, height_obs)(pts[nan_m])
    H = H.reshape(GX.shape)

    # ────────────────────────── Figure ──────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.4, 1, 1], height_ratios=[1, 1],
                  wspace=0.32, hspace=0.32)

    # [A] TOP-10 投資推奨 + 信頼区間
    ax_top = fig.add_subplot(gs[0, 1])
    order = np.argsort(-g_pred_norm)
    top10 = order[:10]
    y_pos = np.arange(10)[::-1]
    means = g_pred_norm[top10]; stds = g_std[top10]
    actual_ranks = np.argsort(np.argsort(-g_actual))  # 0 = top
    bars = ax_top.barh(y_pos, means, xerr=stds, color="#2c7a2c",
                        edgecolor="black", alpha=0.85, capsize=3,
                        error_kw={"lw": 1, "ecolor": "black"})
    for k, i in enumerate(top10):
        cat = cfg["cpc_label_fn"](names_act[i])
        actual_r = actual_ranks[i] + 1
        hit = "✓" if actual_r <= 10 else "✗"
        ax_top.text(means[k] + stds[k] + 0.03, y_pos[k],
                    f"{names_act[i]} ({cat})  実rank={actual_r}{hit}",
                    fontsize=8, va="center")
    ax_top.set_yticks(y_pos); ax_top.set_yticklabels([f"#{k+1}" for k in range(10)], fontsize=9)
    ax_top.set_xlabel("予測成長率 (g_norm) ± MC uncertainty", fontsize=10)
    ax_top.set_title("[A] 投資推奨 TOP-10 (X2 growth-head)", fontsize=11, fontweight="bold")
    ax_top.grid(axis="x", alpha=0.3)
    ax_top.set_xlim(0, means.max() * 1.6)

    # [B] BOTTOM-10 撤退検討 + 信頼区間
    ax_bot = fig.add_subplot(gs[1, 1])
    bot10 = order[-10:][::-1]
    means_b = g_pred_norm[bot10]; stds_b = g_std[bot10]
    ax_bot.barh(y_pos, means_b, xerr=stds_b, color="#a02020",
                 edgecolor="black", alpha=0.85, capsize=3,
                 error_kw={"lw": 1, "ecolor": "black"})
    for k, i in enumerate(bot10):
        cat = cfg["cpc_label_fn"](names_act[i])
        actual_r = actual_ranks[i] + 1
        hit = "✓" if actual_r >= K - 9 else "✗"
        x_text = means_b[k] - stds_b[k] - 0.04
        ax_bot.text(x_text, y_pos[k],
                    f"{names_act[i]} ({cat})  実rank={actual_r}{hit}",
                    fontsize=8, va="center", ha="right")
    ax_bot.set_yticks(y_pos); ax_bot.set_yticklabels([f"#{k+1}" for k in range(10)], fontsize=9)
    ax_bot.set_xlabel("予測成長率 (g_norm) ± MC uncertainty", fontsize=10)
    ax_bot.set_title("[B] 撤退検討 BOTTOM-10", fontsize=11, fontweight="bold")
    ax_bot.grid(axis="x", alpha=0.3)
    ax_bot.set_xlim(means_b.min() * 1.5, 0)

    # [C] Φ landscape + 予測 (緑) と 実測 (青) を 2 重ドットで表示
    ax_land = fig.add_subplot(gs[:, 0])
    cmap = matplotlib.colormaps["RdBu_r"]
    levels = np.linspace(H_lo, H_hi, 28)
    ax_land.contourf(GX, GY, H, levels=levels, cmap=cmap, alpha=0.85, extend="both")
    ax_land.contour(GX, GY, H, levels=10, colors="black", linewidths=0.4, alpha=0.45)

    # 実測の TOP-3 を青リング、予測の TOP-3 を緑リング (一致なら同じ位置に重なる)
    sc1 = ax_land.scatter(cent_2d[:, 0], cent_2d[:, 1], c=g_actual, cmap="RdYlGn",
                          s=180, edgecolors="black", linewidths=0.8,
                          vmin=-abs(g_actual).max(), vmax=abs(g_actual).max(),
                          zorder=5, alpha=0.92)
    # 緑円: predicted TOP-3
    actual_top3 = np.argsort(-g_actual)[:3]
    pred_top3 = order[:3]
    for i in pred_top3:
        ax_land.scatter(cent_2d[i, 0], cent_2d[i, 1], s=320, facecolor="none",
                        edgecolor="lime", linewidth=2.5, zorder=6)
    for i in actual_top3:
        ax_land.scatter(cent_2d[i, 0], cent_2d[i, 1], s=400, facecolor="none",
                        edgecolor="blue", linewidth=2.0, linestyle="--", zorder=7)
        # label
        ax_land.annotate(f"実#{int(np.where(actual_top3==i)[0][0])+1} {names_act[i]}",
                         cent_2d[i], xytext=(10, 8), textcoords="offset points",
                         fontsize=8, color="blue", fontweight="bold",
                         bbox=dict(facecolor="white", edgecolor="blue", lw=1,
                                   alpha=0.9, boxstyle="round,pad=0.2"))
    ax_land.set_title(f"[C] Φ landscape + 予測・実測 TOP-3 比較\n"
                      "  緑円実線 = X2予測 TOP-3, 青円破線 = 実 TOP-3",
                      fontsize=11, fontweight="bold")
    ax_land.set_xticks([]); ax_land.set_yticks([])
    ax_land.set_xlabel("UMAP1"); ax_land.set_ylabel("UMAP2")

    # [D] Uncertainty heatmap
    ax_unc = fig.add_subplot(gs[:, 2])
    sc2 = ax_unc.scatter(cent_2d[:, 0], cent_2d[:, 1], c=g_std, cmap="plasma",
                          s=140, edgecolors="black", linewidths=0.8, zorder=5)
    for k, i in enumerate(np.argsort(-g_std)[:5]):
        ax_unc.annotate(f"不確実: {names_act[i]}", cent_2d[i],
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=8, color="#660066",
                        bbox=dict(facecolor="white", edgecolor="#660066",
                                  lw=0.8, alpha=0.9, boxstyle="round,pad=0.15"))
    plt.colorbar(sc2, ax=ax_unc, label="予測不確実性 σ(g_pred)", fraction=0.04)
    ax_unc.set_title("[D] X2 予測の不確実性マップ\n  (MC SDE rollout: σ大 = 投資判断は慎重に)",
                     fontsize=11, fontweight="bold")
    ax_unc.set_xticks([]); ax_unc.set_yticks([])
    ax_unc.set_xlabel("UMAP1"); ax_unc.set_ylabel("UMAP2")

    plt.suptitle(f"X2 PI-SDE 経営者ダッシュボード — {DOMAIN}, {cfg['year_label']} 予測",
                 fontsize=14, fontweight="bold", y=1.005)

    out = Path("RESULTS/fig17_x2_executive_dashboard.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")
    import shutil
    shutil.copy(out, "figures/fig17_x2_executive_dashboard.png")


if __name__ == "__main__":
    main()
