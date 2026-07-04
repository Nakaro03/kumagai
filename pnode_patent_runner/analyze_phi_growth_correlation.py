"""
Φ(topic) ランキング vs 実際の成長率の相関分析。

PC-PNODE の L_trend が正しく機能していれば：
  - 成長トピック（g > 0）→ 低Φ（谷）
  - 衰退トピック（g < 0）→ 高Φ（山）
  → Spearman 相関(Φ, growth) < 0 かつ有意

比較対象として P-NODE（L_trend なし）も同時評価する。

出力:
  - コンソール: 年ごとの相関係数・p値・上位/下位トピック
  - pnode_patent_runner/outputs/pnode_pc_landscape/phi_growth_correlation.png
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

from pnode_patent_runner.benchmark_vgae import BenchmarkTemporalVGAE
from pnode_patent_runner.checkpoint_utils import load_state_dict_skip_shape_mismatch
from pnode_patent_runner.cope_experiment import load_author_topic_graph_bundle

# ── 設定 ──────────────────────────────────────────────────────────────────────
CKPT_PC    = "pnode_patent_runner/outputs/pnode_pc_detach_fix/ckpt/pnode_pc_seed42.pt"
CKPT_PNODE = "pnode_patent_runner/outputs/ablation_pnode/ckpt/pnode_seed42.pt"
DATA_CSV   = "data/processed/arxiv_cs_embedded_2020-2026_full.csv"
OUT_PNG    = "pnode_patent_runner/outputs/pnode_pc_detach_fix/phi_growth_correlation.png"
YEAR_RANGE = (2022, 2025)
MIN_PAPERS = 5
TOP_N = 10  # 上位/下位表示数

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def load_model(ckpt_path: str, bundle, variant: str, device) -> BenchmarkTemporalVGAE:
    raw = torch.load(ckpt_path, map_location=device)
    sd = raw["state_dict"] if "state_dict" in raw else raw
    hidden = int(raw.get("hidden_dim", 128))
    latent = int(raw.get("latent_dim", 2))

    model = BenchmarkTemporalVGAE(
        num_nodes=bundle.total_n,
        num_corps=bundle.num_corps,
        input_dim=bundle.in_dim,
        hidden_dim=hidden,
        latent_dim=latent,
        initial_corp_vectors=bundle.init_vectors,
        link_score_mode="distance",
        variant=variant,
        pnode_potential_feature="mlp",
        year_min=YEAR_RANGE[0],
        year_max=YEAR_RANGE[1],
    ).to(device)

    skip, _ = load_state_dict_skip_shape_mismatch(model, sd)
    if skip:
        print(f"  [{variant}] スキップ {len(skip)} キー")
    model.eval()
    return model


@torch.no_grad()
def get_topic_phi(model, bundle, years_list, device):
    """年ごとにトピックノードの z と Φ(z) を返す。"""
    pot = model.temporal_predictor.potential_net
    num_left = bundle.num_corps  # 著者数
    results = {}

    for y in years_list:
        data_y = bundle.graphs[y].to(device)
        z, _, _ = model.encode(data_y.x, data_y.edge_index)

        # PC-PNODE: population を set
        if hasattr(pot, "set_population"):
            pot.set_population(z.detach())

        z_topics = z[num_left:]          # トピックノードだけ
        phi_t = pot(z_topics).squeeze(-1)  # (n_topics,)
        results[y] = {
            "z": z_topics.cpu().numpy(),
            "phi": phi_t.cpu().numpy(),
        }
    return results


def compute_correlation(phi_arr, growth_arr, topic_names=None, year=None, label=""):
    """Spearman 相関と上位/下位トピックを表示。"""
    mask = np.isfinite(phi_arr) & np.isfinite(growth_arr)
    phi_m = phi_arr[mask]
    g_m   = growth_arr[mask]
    names_m = [topic_names[i] for i in range(len(mask)) if mask[i]] if topic_names else None

    r, p = stats.spearmanr(phi_m, g_m)
    tau, p_tau = stats.kendalltau(phi_m, g_m)

    print(f"\n  [{label}] {year}年  n={mask.sum()}")
    print(f"    Spearman r={r:+.4f}  p={p:.4f}  {'**有意**' if p < 0.05 else '(n.s.)'}")
    print(f"    Kendall  τ={tau:+.4f}  p={p_tau:.4f}")

    if names_m is not None:
        order = np.argsort(phi_m)   # Φ 昇順 = 谷
        print(f"    Φ 最小（谷=注目予測）TOP{TOP_N}:  成長率")
        for i in order[:TOP_N]:
            print(f"      {names_m[i]:<30} Φ={phi_m[i]:+.3f}  g={g_m[i]:+.3f}")
        print(f"    Φ 最大（山=衰退予測）TOP{TOP_N}:  成長率")
        for i in order[-TOP_N:][::-1]:
            print(f"      {names_m[i]:<30} Φ={phi_m[i]:+.3f}  g={g_m[i]:+.3f}")
    return r, p, tau


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    bundle = load_author_topic_graph_bundle(
        DATA_CSV,
        min_papers=MIN_PAPERS,
        year_range=YEAR_RANGE,
    )
    years_list = sorted(bundle.graphs.keys())
    topics = bundle.right_nodes      # トピック名リスト（右ノード）
    growth_by_year = bundle.topic_growth_by_year or {}

    print(f"著者={bundle.num_corps}  トピック={len(topics)}  年={years_list}")

    # ── モデル読み込み ─────────────────────────────────────────────────────────
    models = {}
    print("\n[PC-PNODE] チェックポイント読み込み...")
    models["PC-PNODE"] = load_model(CKPT_PC, bundle, "pnode_pc", device)

    if Path(CKPT_PNODE).is_file():
        print("[P-NODE]   チェックポイント読み込み...")
        models["P-NODE"] = load_model(CKPT_PNODE, bundle, "pnode", device)

    # ── Φ と成長率の取得 ─────────────────────────────────────────────────────
    phi_data = {name: get_topic_phi(m, bundle, years_list, device)
                for name, m in models.items()}

    # ── 成長率を3年移動平均に平滑化 ─────────────────────────────────────────
    # 各年の前後±1年の平均を取ることでノイズを低減
    all_years_sorted = sorted(growth_by_year.keys())
    smoothed_growth: dict = {}
    for i, y in enumerate(all_years_sorted):
        neighbor_years = [yy for yy in all_years_sorted
                          if abs(yy - y) <= 1 and yy in growth_by_year]
        stacked = torch.stack([growth_by_year[yy] for yy in neighbor_years])
        smoothed_growth[y] = stacked.mean(dim=0)
    print(f"\n成長率を {len(smoothed_growth)} 年分 ±1年平均で平滑化")

    # ── 相関分析 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Φ(topic) vs 実際の成長率  Spearman 相関")
    print("=" * 60)
    print("  期待値: PC-PNODE r < 0（低Φ=谷 → 高成長）")
    print("          P-NODE   r ≈ 0（L_trend なし → 無相関）")

    corr_results = {name: {} for name in models}

    for name in models:
        print(f"\n【{name}】")
        for y in years_list:
            g = smoothed_growth.get(y)
            if g is None:
                continue
            phi_arr = phi_data[name][y]["phi"]
            g_arr   = g.numpy()
            n_topics = min(len(phi_arr), len(g_arr))
            r, p, tau = compute_correlation(
                phi_arr[:n_topics], g_arr[:n_topics],
                topic_names=[str(t) for t in topics[:n_topics]],
                year=y, label=name,
            )
            corr_results[name][y] = {"r": r, "p": p, "tau": tau}

    # ── 可視化 ────────────────────────────────────────────────────────────────
    avail_years = [y for y in years_list if smoothed_growth.get(y) is not None]
    n_years = len(avail_years)
    n_models = len(models)

    fig = plt.figure(figsize=(6 * n_years, 4 * (n_models + 1)))
    gs = gridspec.GridSpec(n_models + 1, n_years, figure=fig, hspace=0.5, wspace=0.35)

    # 散布図パネル
    for mi, (name, m_phi) in enumerate(phi_data.items()):
        for yi, y in enumerate(avail_years):
            g = smoothed_growth.get(y)
            if g is None:
                continue
            phi_arr = m_phi[y]["phi"]
            g_arr   = g.numpy()
            n = min(len(phi_arr), len(g_arr))
            phi_arr, g_arr = phi_arr[:n], g_arr[:n]

            ax = fig.add_subplot(gs[mi, yi])
            scatter = ax.scatter(phi_arr, g_arr, s=6, alpha=0.5,
                                 c=g_arr, cmap="RdYlGn", vmin=-1, vmax=1)
            # 回帰直線（Φ が定数のとき polyfit は失敗するので try-except）
            try:
                mask_fit = np.isfinite(phi_arr) & np.isfinite(g_arr)
                if mask_fit.sum() >= 2 and phi_arr[mask_fit].std() > 1e-8:
                    z_fit = np.polyfit(phi_arr[mask_fit], g_arr[mask_fit], 1)
                    xp = np.linspace(phi_arr.min(), phi_arr.max(), 100)
                    ax.plot(xp, np.polyval(z_fit, xp), "k--", lw=1, alpha=0.7)
            except Exception:
                pass

            rc = corr_results[name].get(y, {})
            r_val = rc.get("r", float("nan"))
            p_val = rc.get("p", float("nan"))
            sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
            ax.set_title(f"{name} {y}年\nSpearman r={r_val:+.3f}{sig}", fontsize=8)
            ax.set_xlabel("Φ(topic)", fontsize=7)
            ax.set_ylabel("成長率 g", fontsize=7)
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.axvline(phi_arr.mean(), color="gray", lw=0.5, ls="--")
            plt.colorbar(scatter, ax=ax, fraction=0.04, label="g")

    # 相関係数推移パネル
    ax_trend = fig.add_subplot(gs[n_models, :])
    x = np.arange(len(avail_years))
    for name, color, marker in zip(models, ["#3b82f6", "#ef4444"], ["o", "s"]):
        rs = [corr_results[name].get(y, {}).get("r", float("nan")) for y in avail_years]
        ps = [corr_results[name].get(y, {}).get("p", float("nan")) for y in avail_years]
        ax_trend.plot(x, rs, marker=marker, color=color, label=name, lw=1.5, ms=7)
        for xi, (r, p) in enumerate(zip(rs, ps)):
            if p < 0.05 and r == r:
                ax_trend.annotate(f"p={p:.3f}", (xi, r),
                                  textcoords="offset points", xytext=(4, 4),
                                  fontsize=6.5, color=color)

    ax_trend.axhline(0, color="gray", lw=0.8, ls="--")
    ax_trend.axhline(-0.3, color="green", lw=0.6, ls=":", alpha=0.6, label="r=-0.3 目安")
    ax_trend.set_xticks(x)
    ax_trend.set_xticklabels([str(y) for y in avail_years])
    ax_trend.set_ylabel("Spearman r (Φ vs 成長率)")
    ax_trend.set_title("Φ ランキング vs 実際の成長率 相関係数の推移\n"
                        "r < 0 = 低Φ(谷)が成長トピックに対応（PC-PNODE の期待動作）", fontsize=9)
    ax_trend.legend(fontsize=8)
    ax_trend.set_ylim(-1.05, 1.05)
    ax_trend.grid(alpha=0.3)

    fig.suptitle("PC-PNODE: Φ(topic) vs 実際の技術トレンド成長率  相関分析",
                 fontsize=11, fontweight="bold")

    out = Path(OUT_PNG)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nSaved -> {out}")

    # ── サマリー ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  サマリー: 平均 Spearman r (全年)")
    print("=" * 60)
    for name in models:
        rs = [v["r"] for v in corr_results[name].values() if "r" in v]
        mean_r = np.mean(rs) if rs else float("nan")
        print(f"  {name:<12}  mean r = {mean_r:+.4f}  {'✅ 期待通り' if mean_r < -0.1 else '❌ L_trend 効果薄'}")


if __name__ == "__main__":
    main()
