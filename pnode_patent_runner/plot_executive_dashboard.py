"""
経営者向けダッシュボード — Patent Energy 学習済みモデルから:
  [1] 投資 4 象限図 (現在量 × 成長予測)
  [2] TOP10 成長技術 / TOP10 衰退技術
  [3] 各 topic の Φ(t) 時系列 → 成熟タイミング
  [4] "技術地形図" + 流線 (-∇Φ)
"""
from __future__ import annotations

import os, sys, glob
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import font_manager

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

# Japanese-capable font
for cand in ["Noto Sans CJK JP", "IPAGothic", "TakaoGothic", "VL PGothic"]:
    if cand in [f.name for f in font_manager.fontManager.ttflist]:
        plt.rcParams["font.family"] = cand
        break

from src.model import ForwardSDE
from types import SimpleNamespace

DOMAIN = "patent_energy_top50"
DATA_PT = "data/PNode_Patent_Energy_X1_top50/alltime/fate_train.pt"
ROOT = "RESULTS/PNode_Patent_Energy_X1_top50"
TAG_SUFFIX = "-x1_v1.0_g0.1_b0.01"
SEED = 42
LAST_T = 11
YEAR_BASE = 2013   # paper started in 2013, t=0..11

# CPC コードの日本語解釈 (経営者向け)
CPC_LABELS = {
    "Y02A": "気候変動適応",
    "Y02B": "建物・住宅 (省エネ)",
    "Y02C": "温室効果ガス削減",
    "Y02D": "ICT (省エネ)",
    "Y02E": "クリーンエネルギー",
    "Y02P": "製造業 (省エネ)",
    "Y02T": "交通 (省エネ・EV)",
    "Y02W": "廃棄物・水",
}


def cpc_category(name):
    """e.g., 'Y02E10/52' -> 'Y02E' -> 'クリーンエネルギー'"""
    if not name: return "その他"
    head = name[:4]
    return CPC_LABELS.get(head, "その他")


def main():
    pat = f"{ROOT}/*{TAG_SUFFIX}/seed_{SEED}/alltime"
    cands = [Path(p) for p in glob.glob(pat) if "_lx" not in Path(p).parent.name]
    out_dir = cands[0]
    out_png = Path("RESULTS/executive_dashboard_patent_energy.png")

    print(f"Loading {DATA_PT}...")
    data = torch.load(DATA_PT, weights_only=False)
    xp = data["xp"]
    y  = data["y"]
    topic_names = data["topic_names"]
    centroids = data["centroids"]
    growth = data["growth"]
    n_topics = data["n_topics"]
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

    # Active topic last year
    cent_last = centroids[LAST_T].numpy()
    active = cent_last.sum(axis=-1) != 0
    cent_act = cent_last[active]
    names_act = [topic_names[i] for i in range(n_topics) if active[i]]
    g_last = growth[LAST_T].numpy()[active]

    # 現在量 = 累計 topic 内サンプル数 (全期間)
    volume = np.zeros(n_topics, dtype=float)
    for k in range(n_T):
        t_k = data["topics"][k].numpy()
        for j in range(n_topics):
            volume[j] += (t_k == j).sum()
    vol_act = volume[active]

    # Φ at last year for each active centroid
    xt_cent = torch.cat([
        torch.tensor(cent_act, dtype=torch.float32),
        torch.full((len(cent_act), 1), float(y[LAST_T]))
    ], dim=1).to(device).requires_grad_()
    phi_cent = model._func._pot(xt_cent).squeeze(-1).detach().cpu().numpy()

    # Φ time series (t=0..LAST_T) for each topic centroid → 成熟タイミング
    phi_ts = np.zeros((n_T, len(cent_act)))
    for k in range(n_T):
        cent_k = centroids[k].numpy()[active]   # use last-year centroid for fixed identity
        # Use stable identity: project topic last-year centroid through time
        xt_k = torch.cat([
            torch.tensor(cent_act, dtype=torch.float32),
            torch.full((len(cent_act), 1), float(y[k]))
        ], dim=1).to(device).requires_grad_()
        phi_ts[k] = model._func._pot(xt_k).squeeze(-1).detach().cpu().numpy()

    # ─────────────────────────── Figure ───────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1], height_ratios=[1, 1],
                          hspace=0.32, wspace=0.30)

    # ===== [1] 投資 4 象限図 =====
    ax1 = fig.add_subplot(gs[0:2, 0])
    # x: 現在量 (log scale), y: 成長予測 = -Φ (高い方が良い)
    growth_pred = -phi_cent
    log_vol = np.log10(vol_act + 1)
    ax1.scatter(log_vol, growth_pred, s=80 + 4 * (vol_act / vol_act.max() * 200),
                c=growth_pred, cmap="RdYlGn", edgecolors="black", linewidth=0.6,
                alpha=0.85)
    # quadrant lines
    vol_mid = np.median(log_vol)
    g_mid = np.median(growth_pred)
    ax1.axvline(vol_mid, color="gray", lw=0.8, linestyle=":")
    ax1.axhline(g_mid, color="gray", lw=0.8, linestyle=":")
    # quadrant labels
    ax1.text(log_vol.max() * 0.98, growth_pred.max() * 0.96, "[★] 主力\n(大規模・成長)",
             ha="right", va="top", fontsize=11, fontweight="bold", color="#2c7a2c",
             bbox=dict(facecolor="#e6f5e6", edgecolor="#2c7a2c", boxstyle="round,pad=0.4"))
    ax1.text(log_vol.min() * 1.05, growth_pred.max() * 0.96, "[新] 新規参入候補\n(小規模・成長)",
             ha="left", va="top", fontsize=11, fontweight="bold", color="#1f6f9c",
             bbox=dict(facecolor="#e6f1f8", edgecolor="#1f6f9c", boxstyle="round,pad=0.4"))
    ax1.text(log_vol.max() * 0.98, growth_pred.min() * 1.02, "[維] 安定維持\n(大規模・成熟)",
             ha="right", va="bottom", fontsize=11, fontweight="bold", color="#888",
             bbox=dict(facecolor="#f0f0f0", edgecolor="#888", boxstyle="round,pad=0.4"))
    ax1.text(log_vol.min() * 1.05, growth_pred.min() * 1.02, "[退] 撤退検討\n(小規模・衰退)",
             ha="left", va="bottom", fontsize=11, fontweight="bold", color="#a00",
             bbox=dict(facecolor="#fbe6e6", edgecolor="#a00", boxstyle="round,pad=0.4"))
    # annotate top-growth and key topics
    order_g = np.argsort(-growth_pred)
    annotated = set()
    for i in order_g[:6]:
        cat = cpc_category(names_act[i])
        ax1.annotate(f"{names_act[i]}\n({cat})", (log_vol[i], growth_pred[i]),
                     fontsize=8, xytext=(8, 8), textcoords="offset points",
                     bbox=dict(facecolor="white", edgecolor="#2c7a2c", boxstyle="round,pad=0.2", alpha=0.9))
        annotated.add(i)
    order_g_asc = np.argsort(growth_pred)
    for i in order_g_asc[:4]:
        if i in annotated: continue
        cat = cpc_category(names_act[i])
        ax1.annotate(f"{names_act[i]}\n({cat})", (log_vol[i], growth_pred[i]),
                     fontsize=8, xytext=(8, -16), textcoords="offset points",
                     bbox=dict(facecolor="white", edgecolor="#a00", boxstyle="round,pad=0.2", alpha=0.9))
    ax1.set_xlabel("現在の累計特許数  (log10)", fontsize=12)
    ax1.set_ylabel("成長予測スコア  (-Φ, 高い = 投資価値大)", fontsize=12)
    ax1.set_title("【経営判断】 4 象限 投資マップ — クリーンエネルギー特許",
                  fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3)

    # ===== [2] TOP/BOTTOM 10 ランキング =====
    ax2 = fig.add_subplot(gs[0, 1])
    top10 = order_g[:10]
    y_pos = np.arange(10)[::-1]
    colors = plt.cm.RdYlGn(0.7 + 0.3 * np.arange(10) / 10)
    ax2.barh(y_pos, growth_pred[top10], color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{names_act[i]}  [{cpc_category(names_act[i])}]" for i in top10],
                       fontsize=9)
    ax2.set_xlabel("成長予測スコア", fontsize=10)
    ax2.set_title("【投資推奨】 TOP-10 成長技術 (2024)", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    # ===== [3] BOTTOM 10 ランキング =====
    ax3 = fig.add_subplot(gs[1, 1])
    bot10 = order_g[-10:][::-1]
    colors = plt.cm.RdYlGn(0.3 * np.arange(10) / 10)
    ax3.barh(y_pos, growth_pred[bot10], color=colors, edgecolor="black", linewidth=0.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([f"{names_act[i]}  [{cpc_category(names_act[i])}]" for i in bot10],
                       fontsize=9)
    ax3.set_xlabel("成長予測スコア", fontsize=10)
    ax3.set_title("【撤退検討】 BOTTOM-10 衰退技術", fontsize=12, fontweight="bold")
    ax3.grid(axis="x", alpha=0.3)

    # ===== [4] Φ(t) 時系列 — 成熟タイミング =====
    ax4 = fig.add_subplot(gs[0, 2])
    years_axis = np.array([YEAR_BASE + int(yi) for yi in y])
    # Show top 5 growing + top 5 declining
    for i in order_g[:5]:
        ax4.plot(years_axis, -phi_ts[:, i], "-o", lw=2, markersize=4,
                 label=f"{names_act[i]}", alpha=0.9)
    ax4.set_xlabel("年", fontsize=10)
    ax4.set_ylabel("成長予測スコア (-Φ)", fontsize=10)
    ax4.set_title("【成長軌跡】 TOP-5 技術の Φ 時系列", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=8, loc="best")
    ax4.grid(alpha=0.3)
    ax4.axvline(YEAR_BASE + LAST_T, color="red", linestyle="--", alpha=0.5, label="現在")

    ax5 = fig.add_subplot(gs[1, 2])
    for i in order_g[-5:]:
        ax5.plot(years_axis, -phi_ts[:, i], "-o", lw=2, markersize=4,
                 label=f"{names_act[i]}", alpha=0.9)
    ax5.set_xlabel("年", fontsize=10)
    ax5.set_ylabel("成長予測スコア (-Φ)", fontsize=10)
    ax5.set_title("【衰退軌跡】 BOTTOM-5 技術の Φ 時系列", fontsize=12, fontweight="bold")
    ax5.legend(fontsize=8, loc="best")
    ax5.grid(alpha=0.3)
    ax5.axvline(YEAR_BASE + LAST_T, color="red", linestyle="--", alpha=0.5)

    plt.suptitle("経営者向け  技術投資ダッシュボード  —  クリーンエネルギー特許 (CPC Y02 分類)",
                 fontsize=15, fontweight="bold", y=0.995)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
