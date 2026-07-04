"""
ThemeRiver-style 可視化 — Patent Energy CPC Y02 のトピック流動 (2013-2030).

文献: Havre, Hetzler, Nowell (TVCG 2002) "ThemeRiver: Visualizing Thematic Changes in
                Large Document Collections"

履歴 (2013-2024) + PI-SDE 予測 (2025-2030).
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
from matplotlib.patches import Patch

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
LAST_T = 11
PREDICT_END_T = 17

CPC_FAMILIES = {
    "Y02A": ("気候適応", "#5b9bd5"),
    "Y02B": ("建物省エネ", "#a5a5a5"),
    "Y02C": ("GHG 削減", "#ffc000"),
    "Y02D": ("ICT 省エネ", "#70ad47"),
    "Y02E": ("クリーンエネ", "#ed7d31"),
    "Y02P": ("製造省エネ", "#4472c4"),
    "Y02T": ("交通省エネ", "#7030a0"),
    "Y02W": ("廃棄物", "#c00000"),
}


def main():
    pat = f"{ROOT}/*{TAG_SUFFIX}/seed_{SEED}/alltime"
    cands = [Path(p) for p in glob.glob(pat) if "_lx" not in Path(p).parent.name]
    out_dir = cands[0]
    out_png = Path("RESULTS/themeriver_patent_energy.png")

    print("Loading model + data...")
    data = torch.load(DATA_PT, weights_only=False)
    xp = data["xp"]
    y = data["y"]
    topic_names = data["topic_names"]
    centroids = data["centroids"]
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

    # ── 履歴: 各 topic の年あたり件数 (= bipartite から)
    import pandas as pd
    BIPARTITE = "data/processed/bipartite_energy.csv"
    print("Loading bipartite to compute historical counts...")
    df = pd.read_csv(BIPARTITE)
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    topic_set = set(topic_names)
    df_filt = df[df["i"].isin(topic_set)]
    df_filt = df_filt[(df_filt["year"] >= YEAR_BASE) & (df_filt["year"] <= 2021)]

    # 件数 (topic, year)
    counts = pd.crosstab(df_filt["year"], df_filt["i"])
    counts = counts.reindex(columns=topic_names, fill_value=0)
    counts = counts.sort_index()  # year ascending
    print(f"  data years: {list(counts.index)}")

    # ── 予測: PI-SDE rollout で各 topic centroid の Φ 推移を計算
    # Φ が低い = 成長中 → 件数増加に変換
    # シンプルなマッピング: count_pred(t) = count(t_last) * exp(-(Φ(c, t) - Φ(c, t_last)) * scale)
    print("Predicting future counts via PI-SDE Φ trajectory...")
    last_obs_year = counts.index.max()
    base_counts = counts.loc[last_obs_year].values  # (n_topics,) at year 2021

    fut_years = list(range(last_obs_year + 1, YEAR_BASE + PREDICT_END_T + 1))
    fut_counts = np.zeros((len(fut_years), n_topics))

    # Φ at last observed year and future years
    cents_last = centroids[last_obs_year - YEAR_BASE].numpy() if (last_obs_year - YEAR_BASE) < n_T else centroids[-1].numpy()
    cents_dev = torch.tensor(cents_last, dtype=torch.float32, device=device)

    phi_baseline = None
    for k, yr in enumerate([last_obs_year] + fut_years):
        t_val = float(yr - YEAR_BASE)
        with torch.enable_grad():
            xt = torch.cat([cents_dev,
                            torch.full((cents_dev.shape[0], 1), t_val, device=device)], dim=1).requires_grad_()
            phi = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()
        if k == 0:
            phi_baseline = phi
        else:
            delta_phi = phi - phi_baseline
            growth_pred = -delta_phi   # 低 Φ = growth
            # Apply growth-per-year compound to base counts
            fut_counts[k - 1] = base_counts * np.exp(0.15 * growth_pred * (yr - last_obs_year) / 4.0)

    # Combine historical + predicted into one DataFrame
    all_years = list(counts.index) + fut_years
    all_counts = np.vstack([counts.values, fut_counts])

    # ── Aggregate by family (Y02A, Y02B, ...) for readability
    fam_of = [tn[:4] for tn in topic_names]
    fam_list = sorted(set(fam_of), key=lambda f: list(CPC_FAMILIES.keys()).index(f) if f in CPC_FAMILIES else 99)
    fam_counts = np.zeros((len(all_years), len(fam_list)))
    for i, fam in enumerate(fam_list):
        idx = [j for j in range(n_topics) if fam_of[j] == fam]
        fam_counts[:, i] = all_counts[:, idx].sum(axis=1)

    # ── ThemeRiver plot
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.4]})

    ax = axes[0]
    # Center-aligned stacking (true ThemeRiver style)
    total = fam_counts.sum(axis=1)
    baseline = -total / 2

    for i, fam in enumerate(fam_list):
        label, color = CPC_FAMILIES.get(fam, (fam, "#777"))
        y_low = baseline + (fam_counts[:, :i]).sum(axis=1)
        y_high = y_low + fam_counts[:, i]
        ax.fill_between(all_years, y_low, y_high, color=color, alpha=0.95,
                        edgecolor="white", linewidth=0.4, label=f"{fam} {label}")

    # Mark prediction boundary
    bnd_year = last_obs_year + 0.5
    ax.axvline(bnd_year, color="red", lw=2, ls="--", alpha=0.7)
    ax.text(bnd_year + 0.1, baseline.min() * 0.95, "PI-SDE 予測 →",
            color="red", fontsize=11, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="red", boxstyle="round,pad=0.25"))
    ax.text(bnd_year - 0.1, baseline.min() * 0.95, "← 実観測",
            color="black", fontsize=11, fontweight="bold", ha="right",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.25"))

    ax.set_title("ThemeRiver: クリーンエネルギー特許 CPC 大分類 の時系列流動 (Patent Energy Y02)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("特許件数 (積み上げ)", fontsize=11)
    ax.legend(loc="upper left", ncol=4, fontsize=9, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_yticks([])
    # Show year tick every 2 years
    yr_ticks = list(range(YEAR_BASE, YEAR_BASE + PREDICT_END_T + 1, 2))
    ax.set_xticks(yr_ticks)
    ax.set_xticklabels(yr_ticks)

    # ── Bottom panel: TOP-5 individual CPC growth/decline projection
    ax2 = axes[1]
    # find top growing and top declining (predicted at year 2030)
    phi_2030 = phi
    rank = np.argsort(phi_2030 - phi_baseline)  # most negative = strongest growth
    top5_grow = rank[:5]
    top5_decl = rank[-5:][::-1]

    for k, idx in enumerate(top5_grow):
        cnt_traj = all_counts[:, idx]
        cnt_norm = cnt_traj / cnt_traj.max() if cnt_traj.max() > 0 else cnt_traj
        ax2.plot(all_years, cnt_norm, "-", lw=2.2, alpha=0.85,
                 label=f"↑ {topic_names[idx]} ({CPC_FAMILIES.get(topic_names[idx][:4], ('','#777'))[0]})",
                 color=plt.cm.Greens(0.4 + 0.12 * k))
    for k, idx in enumerate(top5_decl):
        cnt_traj = all_counts[:, idx]
        cnt_norm = cnt_traj / cnt_traj.max() if cnt_traj.max() > 0 else cnt_traj
        ax2.plot(all_years, cnt_norm, "--", lw=2.2, alpha=0.65,
                 label=f"↓ {topic_names[idx]} ({CPC_FAMILIES.get(topic_names[idx][:4], ('','#777'))[0]})",
                 color=plt.cm.Reds(0.4 + 0.12 * k))

    ax2.axvline(bnd_year, color="red", lw=2, ls="--", alpha=0.7)
    ax2.set_xlabel("年", fontsize=11)
    ax2.set_ylabel("正規化件数 (max=1)", fontsize=10)
    ax2.set_title("TOP-5 成長予測 (実線, 緑) vs TOP-5 衰退予測 (点線, 赤)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(yr_ticks)
    ax2.set_xticklabels(yr_ticks)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
