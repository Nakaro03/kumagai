#!/usr/bin/env python3
"""
TAP-NODE の解釈可能量をトピック単位で実データと突き合わせる研究用図。
2D 潜在の Φ 地形（井戸が融合して不可読）に代わる可視化。

- A: 井戸深さ log w_j(t) = κ·log1p(M_j) + b·D̃_j の年推移（スロープチャート、学習年のみ）
- B: モデルの集約引力 A_j(2024) = Σ_i s_ij vs 2025 年に実現した新規著者–トピックリンク数
     （holdout 学習モデル → リークなし）。人気度ベースライン ρ(M_j) と併記。
- C: 変化対変化のハードテスト: トレンド D̃_j(2024) vs 新規流入の変化 (2025−2024)

相関統計は同名 _stats JSON にも保存する。

例:
  python -m pnode_patent_runner.plot_tap_node_topic_dynamics \\
    --load-checkpoint pnode_patent_runner/outputs/tap_node/ckpt/tap_node_holdout_seed42.pt \\
    --output pnode_patent_runner/outputs/tap_node/tap_node_topic_dynamics_seed42.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from pnode_patent_runner.checkpoint_utils import load_state_dict_skip_shape_mismatch
from pnode_patent_runner.dual_force_data import load_dual_force_bundle
from pnode_patent_runner.tap_node_models import TAPVGAE

TREND_CMAP = "RdBu_r"  # 値=−D̃: 赤=衰退, 青=成長（landscape 図と同一セマンティクス）


def _bipartite_pairs(data, n_a: int) -> set:
    ei = data.edge_index
    m = (ei[0] < n_a) & (ei[1] >= n_a)
    return {(int(a), int(t)) for a, t in ei[:, m].t().tolist()}


def _spread_labels(ys: np.ndarray, min_gap: float) -> np.ndarray:
    """ラベルの縦位置を最小間隔 min_gap で上下に散らす（順序保存）。"""
    order = np.argsort(ys)
    adj = ys.astype(float).copy()
    for prev, cur in zip(order[:-1], order[1:]):
        if adj[cur] - adj[prev] < min_gap:
            adj[cur] = adj[prev] + min_gap
    return adj


def main() -> int:
    p = argparse.ArgumentParser(description="TAP-NODE topic dynamics plot")
    p.add_argument("--data", type=str, default="data/processed/arxiv_cs_embedded_2020-2026_full.csv")
    p.add_argument("--topic-column", type=str, default="topic")
    p.add_argument("--min-patents", type=int, default=5)
    p.add_argument("--load-checkpoint", type=Path, required=True)
    p.add_argument("--train-years", type=int, nargs="+", default=[2022, 2023, 2024])
    p.add_argument("--test-year", type=int, default=2025)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    ck = torch.load(str(args.load_checkpoint), map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TAPVGAE(
        ck["num_nodes"], ck["num_authors"], ck["input_dim"],
        hidden_dim=ck["hidden_dim"], latent_dim=ck["latent_dim"],
        initial_author_vectors=ck.get("initial_author_vectors"),
        link_score_mode=ck.get("link_score_mode", "distance"),
    ).to(device)
    load_state_dict_skip_shape_mismatch(model, ck["state_dict"])
    model.eval()
    ode = model.temporal_predictor.ode_func
    kappa, b = float(ode.kappa.item()), float(ode.b.item())

    bundle = load_dual_force_bundle(args.data, topic_column=args.topic_column, min_papers=args.min_patents)
    n_a = bundle.num_corps
    topics = list(bundle.right_nodes)
    J = len(topics)
    train_years = sorted(int(y) for y in args.train_years)
    ty = int(args.test_year)
    for y in train_years + [ty]:
        if y not in bundle.graphs:
            raise SystemExit(f"year {y} がグラフにありません")

    # --- A: 井戸深さの年推移（学習済み κ, b で評価） ---
    logw = {}
    dt_by_year = {}
    for y in train_years:
        g = bundle.graphs[y]
        m = g.topic_mass.numpy()
        dt = np.log1p(np.clip(g.topic_trend_plus.numpy(), 0, None)) - np.log1p(
            np.clip(g.topic_trend_minus.numpy(), 0, None)
        )
        logw[y] = kappa * np.log1p(m) + b * dt
        dt_by_year[y] = dt
    y_last = train_years[-1]

    # --- B: 集約引力 A_j(y_last)。学習期間にエッジを持つ著者のみで集計 ---
    g_last = bundle.graphs[y_last].to(device)
    with torch.no_grad():
        z, _, _ = model.encode(g_last.x, g_last.edge_index)
        ode.set_topic_info(
            z[n_a:], g_last.topic_mass.to(device),
            g_last.topic_trend_plus.to(device), g_last.topic_trend_minus.to(device),
        )
        _ei0 = [bundle.graphs[y].edge_index[0].cpu() for y in train_years]
        active = torch.unique(torch.cat([e[e < n_a] for e in _ei0])).to(device)
        s = F.softmax(ode._logits(z[active]), dim=1)  # (n_active, J)
        pull = s.sum(dim=0).cpu().numpy()  # A_j

    # --- 実現した新規リンク（学習窓に対して新規のペアのみ） ---
    hist_pairs: set = set()
    new_counts = {}
    for y in train_years + [ty]:
        pairs = _bipartite_pairs(bundle.graphs[y], n_a)
        if y > train_years[0]:
            fresh = pairs - hist_pairs
            cnt = np.zeros(J)
            for _, t in fresh:
                cnt[t - n_a] += 1
            new_counts[y] = cnt
        hist_pairs |= pairs
    inflow_test = new_counts[ty]
    inflow_prev = new_counts[y_last]
    d_inflow = inflow_test - inflow_prev

    mass_last = bundle.graphs[y_last].topic_mass.cpu().numpy()
    dt_last = dt_by_year[y_last]

    # --- 相関統計 ---
    def sp(x, yv):
        r = stats.spearmanr(x, yv)
        return float(r.statistic), float(r.pvalue)

    rho_pull, p_pull = sp(pull, inflow_test)
    rho_mass, p_mass = sp(mass_last, inflow_test)
    rho_logw, p_logw = sp(logw[y_last], inflow_test)
    rho_trend_d, p_trend_d = sp(dt_last, d_inflow)
    # 人気度統制の偏 Spearman（rank 化して偏相関）
    rk = lambda v: stats.rankdata(v)
    rp, rm, ri = rk(pull), rk(mass_last), rk(inflow_test)
    c_pm = np.corrcoef(rp, rm)[0, 1]
    c_pi = np.corrcoef(rp, ri)[0, 1]
    c_mi = np.corrcoef(rm, ri)[0, 1]
    partial = (c_pi - c_pm * c_mi) / np.sqrt((1 - c_pm**2) * (1 - c_mi**2))

    stats_out = {
        "checkpoint": str(args.load_checkpoint),
        "train_years": train_years,
        "test_year": ty,
        "n_topics": J,
        "kappa": kappa,
        "b": b,
        "spearman_pull_vs_new_inflow": {"rho": rho_pull, "p": p_pull},
        "spearman_mass_vs_new_inflow": {"rho": rho_mass, "p": p_mass},
        "spearman_logw_vs_new_inflow": {"rho": rho_logw, "p": p_logw},
        "partial_spearman_pull_given_mass": float(partial),
        "spearman_trend_vs_delta_inflow": {"rho": rho_trend_d, "p": p_trend_d},
    }

    # --- 描画 ---
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)
    dmax = max(np.abs(dt_last).max(), 1e-9)
    tcolors = plt.get_cmap(TREND_CMAP)(plt.Normalize(-dmax, dmax)(-dt_last))

    ax = axes[0]
    W = np.stack([logw[y] for y in train_years])  # (Y, J)
    delta = W[-1] - W[0]
    n_hl = 6
    hl = set(np.argsort(delta)[:n_hl]) | set(np.argsort(delta)[-n_hl:])
    for j in range(J):
        if j in hl:
            ax.plot(train_years, W[:, j], color=tcolors[j], lw=1.8, zorder=3)
        else:
            ax.plot(train_years, W[:, j], color="#c8c8c8", lw=0.7, alpha=0.8, zorder=1)
    hj = sorted(hl, key=lambda j: W[-1, j])
    ylab = _spread_labels(W[-1, np.array(hj)], min_gap=(W.max() - W.min()) * 0.035)
    for yy, j in zip(ylab, hj):
        ax.annotate(
            topics[j], (train_years[-1], W[-1, j]), xytext=(train_years[-1] + 0.06, yy),
            fontsize=7.5, color="#333333", va="center",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
        )
    ax.set_xlim(train_years[0] - 0.1, train_years[-1] + 0.55)
    ax.set_xticks(train_years)
    ax.set_xlabel("year")
    ax.set_ylabel("well depth  log w_j(t)")
    ax.set_title(f"A. Topic well depth (learned κ={kappa:.2f}, b={b:.2f})\ntop-{n_hl} risers/fallers labeled; color: red=declining, blue=growing")

    ax = axes[1]
    # ソフトマックス引力は勝者総取りで生値の散布図は退化する。報告統計（Spearman）に
    # 合わせて順位–順位で描く。
    xs, ys = rk(pull), rk(inflow_test)
    ax.scatter(xs, ys, s=42, c=tcolors, edgecolors="white", linewidths=0.7)
    for j in np.argsort(inflow_test)[-6:]:
        ax.annotate(topics[j], (xs[j], ys[j]), xytext=(3, 3), textcoords="offset points", fontsize=7.5, color="#333333")
    ax.set_xlabel(f"rank of model pull A_j({y_last})")
    ax.set_ylabel(f"rank of new links into topic, {ty}")
    ax.set_title(f"B. Model pull vs realized new inflow (holdout {ty})")
    ax.text(
        0.97, 0.03,
        f"Spearman ρ(pull) = {rho_pull:.2f} (p={p_pull:.1e})\n"
        f"baseline ρ(mass) = {rho_mass:.2f} (p={p_mass:.1e})\n"
        f"partial ρ(pull | mass) = {partial:.2f}",
        transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9),
    )

    ax = axes[2]
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.axvline(0, color="#bbbbbb", lw=0.8)
    ax.scatter(dt_last, d_inflow, s=42, c=tcolors, edgecolors="white", linewidths=0.7)
    for j in np.argsort(np.abs(d_inflow))[-6:]:
        ax.annotate(topics[j], (dt_last[j], d_inflow[j]), xytext=(3, 3), textcoords="offset points", fontsize=7.5, color="#333333")
    ax.set_xlabel(f"topic trend D̃_j({y_last})")
    ax.set_ylabel(f"Δ new inflow ({ty} − {y_last})")
    ax.set_title("C. Change-on-change (hard test)")
    ax.text(
        0.03, 0.97, f"Spearman ρ = {rho_trend_d:.2f} (p={p_trend_d:.1e})",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    sp_path = out.with_name(out.stem + "_stats.json")
    with open(sp_path, "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {out}")
    print(f"Wrote: {sp_path}")
    print(json.dumps(stats_out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    main()
