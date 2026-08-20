#!/usr/bin/env python3
"""
学習済み Dual-Force v2 checkpoint から、企業×技術の潜在空間上の速度場を静止画で出力する。

- 背景ヒートマップ: |v(z)|（ベクトル場の大きさ）
- 白線: ストリームライン（種点から力に従って積分した軌跡。ポテンシャルΦは定義しない
  ——Dual-Forceは非保存力場なので、TAP-NODEのようなΦ地形図の代わりにこちらを使う）
- 橙三角: 技術（CPC）アンカー、青点: 企業（遷移元年時点）
- 緑細線: 遷移先年に実際に起きた新規参入（離散イベントをそのまま重ねる、
  連続軌道としては描かない——ASPH-Flow Layer 3 と同じ原則）

注意: グリッド範囲は企業・トピック分布の1〜99パーセンタイルに基づく。範囲外（外挿域）の
色・矢印は訓練データの分布から外れた場所での外挿であり、意味を読み込みすぎないこと。

例:
  python -m pnode_patent_runner.plot_dual_force_flow \\
    --load-checkpoint pnode_patent_runner/outputs/dual_force_patent/ckpt/zscore_renorm_construction_seed42.pt \\
    --year-prev 2020 --year-next 2021 \\
    --output pnode_patent_runner/outputs/dual_force_patent/figures/flow_construction_seed42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
import matplotlib.pyplot as plt
import numpy as np
import torch

from pnode_patent_runner.checkpoint_utils import load_state_dict_skip_shape_mismatch
from pnode_patent_runner.dual_force_data_patent import load_dual_force_bundle_patent_domain
from pnode_patent_runner.dual_force_vgae import DualForceVGAE


def main() -> int:
    p = argparse.ArgumentParser(description="Dual-Force v2 速度場プロット（patentドメイン）")
    p.add_argument("--load-checkpoint", type=Path, required=True)
    p.add_argument("--year-prev", type=int, required=True, help="速度場を評価する年（この年のzを使う）")
    p.add_argument("--year-next", type=int, required=True, help="実際の新規参入を重ねる遷移先年")
    p.add_argument("--grid-n", type=int, default=45)
    p.add_argument("--n-streamlines", type=int, default=250)
    p.add_argument("--n-entries-drawn", type=int, default=400)
    p.add_argument("--n-predicted-drawn", type=int, default=80,
                    help="予測ベクトル矢印を描く企業数（多すぎると重なって見づらいため絞る）")
    p.add_argument("--output", type=Path, required=True, help="拡張子なし。.png と .pdf を両方出力")
    args = p.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(str(args.load_checkpoint), map_location=device, weights_only=False)

    model = DualForceVGAE(
        ckpt["num_nodes"], ckpt["num_authors"], ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"], latent_dim=ckpt["latent_dim"],
        initial_author_vectors=ckpt["initial_author_vectors"],
        link_score_mode=ckpt["link_score_mode"],
        d_scale_mode=ckpt["d_scale_mode"],
        renorm_masked_attention=ckpt["renorm_masked_attention"],
        gamma=ckpt["gamma_init"],
    ).to(device)
    skipped, _ = load_state_dict_skip_shape_mismatch(model, ckpt["state_dict"])
    model.eval()

    bundle = load_dual_force_bundle_patent_domain(
        ckpt["csv_path"], year_range=tuple(ckpt["year_range"]), min_events=2,
    )
    num_corps = ckpt["num_authors"]

    data0 = bundle.graphs[args.year_prev].to(device)
    data1 = bundle.graphs[args.year_next].to(device)

    with torch.no_grad():
        _, mu0, _ = model.encode(data0.x, data0.edge_index)
        z0 = mu0

    h = z0[:num_corps].numpy()
    P = z0[num_corps:].numpy()

    with torch.no_grad():
        # 評価（future_link_auc_scores_dual_force）と全く同じ呼び出し
        # ＝実在企業についてモデルが実際に出す予測後位置 h_i(year_next)
        z_pred = model.predict_future([z0], data0)
    h_pred = z_pred[:num_corps].numpy()

    d_j = (data0.topic_trend_plus - data0.topic_trend_minus).unsqueeze(-1)
    with torch.no_grad():
        ode_f = model.temporal_predictor.ode_func
        ode_f.set_topic_info(z0[num_corps:], d_j)

        all_pts = np.concatenate([h, P], axis=0)
        lo, hi_ = np.percentile(all_pts, [1, 99], axis=0)
        pad = 0.15 * (hi_ - lo)
        xmin, xmax = lo[0] - pad[0], hi_[0] + pad[0]
        ymin, ymax = lo[1] - pad[1], hi_[1] + pad[1]

        res = int(args.grid_n)
        xs = np.linspace(xmin, xmax, res)
        ys = np.linspace(ymin, ymax, res)
        XX, YY = np.meshgrid(xs, ys)
        grid = torch.tensor(np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=torch.float32)
        V = ode_f(torch.tensor(0.0), grid).numpy()
        speed = np.linalg.norm(V, axis=1).reshape(res, res)
        U = V[:, 0].reshape(res, res)
        Vv = V[:, 1].reshape(res, res)

        rng = np.random.default_rng(0)
        n_seed = min(int(args.n_streamlines), h.shape[0])
        seed_idx = rng.choice(h.shape[0], size=n_seed, replace=False)
        cur = torch.tensor(h[seed_idx], dtype=torch.float32)
        n_steps, dt = 20, 0.05
        for _ in range(n_steps):
            v = ode_f(torch.tensor(0.0), cur)
            cur = cur + dt * v

    mask0 = (data0.edge_index[0] < num_corps) & (data0.edge_index[1] >= num_corps)
    seen_prev = {
        (int(a), int(t_)) for a, t_ in data0.edge_index[:, mask0].numpy().T
    }
    mask1 = (data1.edge_index[0] < num_corps) & (data1.edge_index[1] >= num_corps)
    pos_edges_next = data1.edge_index[:, mask1].numpy()
    new_entries_all = [
        (a, t_) for a, t_ in pos_edges_next.T if (int(a), int(t_)) not in seen_prev
    ]

    # 「正解の参入先」と「モデルの予測ベクトル」を同じ企業サンプルに揃えて対で比較できるようにする
    entry_authors_all = sorted({a for a, _ in new_entries_all})
    n_pred = min(int(args.n_predicted_drawn), len(entry_authors_all))
    sampled_authors = set(
        np.random.default_rng(1).choice(entry_authors_all, size=n_pred, replace=False)
    ) if entry_authors_all else set()
    new_entries_sampled = [(a, t_) for a, t_ in new_entries_all if a in sampled_authors]
    # 背景の緑本数はサンプル外も少し混ぜて雰囲気を見せる（--n-entries-drawn 件まで）
    new_entries_bg = new_entries_all[: int(args.n_entries_drawn)]

    fig, ax = plt.subplots(figsize=(10, 8.5), facecolor="#11151b")
    ax.set_facecolor("#11151b")

    heat = ax.pcolormesh(XX, YY, speed, cmap="magma", shading="gouraud", alpha=0.85)
    cb = fig.colorbar(heat, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("|v(z)|  ベクトル場の大きさ", color="#93a2b3")
    cb.ax.yaxis.set_tick_params(color="#93a2b3")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#93a2b3")

    ax.streamplot(XX, YY, U, Vv, color="#e7ecf2", linewidth=0.5, density=1.1, arrowsize=0.7, arrowstyle="->")
    ax.scatter(P[:, 0], P[:, 1], s=10, c="#d9a15c", marker="^", label="技術(CPC)アンカー", zorder=5, alpha=0.8)
    ax.scatter(h[:, 0], h[:, 1], s=4, c="#5c93b8", alpha=0.3, label=f"企業（{args.year_prev}年時点）", zorder=4)

    for a, t_ in new_entries_bg:
        ax.annotate(
            "", xy=(P[t_ - num_corps, 0], P[t_ - num_corps, 1]), xytext=(h[a, 0], h[a, 1]),
            arrowprops=dict(arrowstyle="-", color="#6faf8f", lw=0.4, alpha=0.3), zorder=6,
        )

    # サンプルした企業だけ、正解（緑・太め）とモデル予測（シアン）を対で強調表示
    for a, t_ in new_entries_sampled:
        ax.annotate(
            "", xy=(P[t_ - num_corps, 0], P[t_ - num_corps, 1]), xytext=(h[a, 0], h[a, 1]),
            arrowprops=dict(arrowstyle="->", color="#6faf8f", lw=1.1, alpha=0.9,
                             shrinkA=0, shrinkB=0, mutation_scale=8), zorder=8,
        )
    for a in sampled_authors:
        ax.annotate(
            "", xy=(h_pred[a, 0], h_pred[a, 1]), xytext=(h[a, 0], h[a, 1]),
            arrowprops=dict(arrowstyle="->", color="#5ce8d8", lw=1.1, alpha=0.9,
                             shrinkA=0, shrinkB=0, mutation_scale=8), zorder=7,
        )
    ax.plot([], [], color="#6faf8f", lw=1.2, marker=">", markersize=4, label="実際の参入先（正解、抽出企業）")
    ax.plot([], [], color="#5ce8d8", lw=1.2, marker=">", markersize=4, label="モデルの予測ベクトル（同じ企業）")

    dom = ckpt.get("domain", "?")
    ax.set_title(
        f"Dual-Force v2 ({ckpt['d_scale_mode']}"
        f"{'+renorm' if ckpt['renorm_masked_attention'] else ''}) 速度場 "
        f"— {dom}, {args.year_prev}→{args.year_next}",
        color="#e7ecf2", fontsize=13,
    )
    ax.set_xlabel("latent dim 1", color="#93a2b3")
    ax.set_ylabel("latent dim 2", color="#93a2b3")
    ax.tick_params(colors="#5f6d7d")
    for spine in ax.spines.values():
        spine.set_color("#2e3946")
    ax.legend(loc="upper right", facecolor="#1a212b", edgecolor="#2e3946", labelcolor="#e7ecf2", fontsize=9)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out) + ".png", dpi=160, facecolor=fig.get_facecolor())
    plt.savefig(str(out) + ".pdf", facecolor=fig.get_facecolor())
    print(f"Wrote: {out}.png")
    print(f"Wrote: {out}.pdf")
    print(f"n_new_entries_drawn(背景): {len(new_entries_bg)}, n_predicted_drawn(サンプル対比): {len(sampled_authors)}")
    if skipped:
        print(f"skipped checkpoint keys (再設定済み): {skipped}")
    return 0


if __name__ == "__main__":
    main()
