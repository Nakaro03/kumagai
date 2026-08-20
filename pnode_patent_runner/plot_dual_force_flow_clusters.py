#!/usr/bin/env python3
"""
Dual-Force v2 の速度場を、潜在空間をクラスタ分割してズームしたPNGとして複数出力する。

`plot_dual_force_flow.py`（全体図、個別企業ごとの生の矢印）は密集領域で矢印が
重なって読みにくくなる。本スクリプトは:

1. 企業・トピックの潜在位置を KMeans でいくつかのクラスタ（＝地図上の領域）に分割
2. クラスタごとに、そのバウンディングボックスへズームした細かい格子で |v(z)| を再計算
   （同じ格子点数でも領域が狭い分、実質的な解像度が上がる＝粒度を細かくする）
3. 個別企業の生の矢印ではなく、粗いビン格子ごとに予測ベクトルを平均した
   quiver（矢印場）を描く——重なりをなくして見やすくする

例:
  python -m pnode_patent_runner.plot_dual_force_flow_clusters \\
    --load-checkpoint pnode_patent_runner/outputs/dual_force_patent/ckpt/zscore_renorm_construction_seed42.pt \\
    --year-prev 2020 --year-next 2021 --n-clusters 4 \\
    --output-dir pnode_patent_runner/outputs/dual_force_patent/figures/clusters_seed42
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
from sklearn.cluster import KMeans

from pnode_patent_runner.checkpoint_utils import load_state_dict_skip_shape_mismatch
from pnode_patent_runner.dual_force_data_patent import load_dual_force_bundle_patent_domain
from pnode_patent_runner.dual_force_vgae import DualForceVGAE


def binned_average(pos: np.ndarray, vec: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, min_count: int = 2):
    """pos の各点が落ちるビンごとに vec を平均し、方向の一致度（円形統計の合成ベクトル長 R）も返す。

    R = |mean_i(v_i / |v_i|)| ∈ [0, 1]。
    R=1: ビン内の全企業がほぼ同じ方向に予測されている（局所的に一貫した力）。
    R=0: 方向がバラバラ（その場所では予測方向が定まらない・企業ごとに事情が違う）。
    疎なビン（count<min_count）は捨てる。
    """
    xi = np.clip(np.digitize(pos[:, 0], xedges) - 1, 0, len(xedges) - 2)
    yi = np.clip(np.digitize(pos[:, 1], yedges) - 1, 0, len(yedges) - 2)
    nx, ny = len(xedges) - 1, len(yedges) - 1
    sum_u = np.zeros((ny, nx))
    sum_v = np.zeros((ny, nx))
    cnt = np.zeros((ny, nx))
    np.add.at(sum_u, (yi, xi), vec[:, 0])
    np.add.at(sum_v, (yi, xi), vec[:, 1])
    np.add.at(cnt, (yi, xi), 1)

    mag = np.linalg.norm(vec, axis=1)
    mag_safe = np.where(mag > 1e-12, mag, 1.0)
    unit = vec / mag_safe[:, None]
    sum_uu = np.zeros((ny, nx))
    sum_uv = np.zeros((ny, nx))
    np.add.at(sum_uu, (yi, xi), unit[:, 0])
    np.add.at(sum_uv, (yi, xi), unit[:, 1])

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_u = np.where(cnt >= min_count, sum_u / np.maximum(cnt, 1), np.nan)
        mean_v = np.where(cnt >= min_count, sum_v / np.maximum(cnt, 1), np.nan)
        R = np.where(
            cnt >= min_count,
            np.hypot(sum_uu, sum_uv) / np.maximum(cnt, 1),
            np.nan,
        )
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    XC, YC = np.meshgrid(xc, yc)
    return XC, YC, mean_u, mean_v, cnt, R


def main() -> int:
    p = argparse.ArgumentParser(description="Dual-Force v2 速度場 — クラスタ別ズーム＋ビン平均quiver")
    p.add_argument("--load-checkpoint", type=Path, required=True)
    p.add_argument("--year-prev", type=int, required=True)
    p.add_argument("--year-next", type=int, required=True)
    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--grid-n", type=int, default=60, help="クラスタごとの |v| 格子の分割数")
    p.add_argument("--n-bins", type=int, default=14, help="quiver 用ビン格子の分割数")
    p.add_argument("--min-bin-count", type=int, default=2)
    p.add_argument("--output-dir", type=Path, required=True)
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
    load_state_dict_skip_shape_mismatch(model, ckpt["state_dict"])
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
        z_pred = model.predict_future([z0], data0)
    h = z0[:num_corps].numpy()
    P = z0[num_corps:].numpy()
    h_pred = z_pred[:num_corps].numpy()
    pred_vec = h_pred - h  # 実企業ごとの「モデルが実際に出す予測変位」

    d_j = (data0.topic_trend_plus - data0.topic_trend_minus).unsqueeze(-1)
    ode_f = model.temporal_predictor.ode_func
    with torch.no_grad():
        ode_f.set_topic_info(z0[num_corps:], d_j)

    mask0 = (data0.edge_index[0] < num_corps) & (data0.edge_index[1] >= num_corps)
    seen_prev = {(int(a), int(t_)) for a, t_ in data0.edge_index[:, mask0].numpy().T}
    mask1 = (data1.edge_index[0] < num_corps) & (data1.edge_index[1] >= num_corps)
    pos_edges_next = data1.edge_index[:, mask1].numpy()
    new_entries = [(a, t_) for a, t_ in pos_edges_next.T if (int(a), int(t_)) not in seen_prev]
    entries_by_author: dict[int, list[int]] = {}
    for a, t_ in new_entries:
        entries_by_author.setdefault(int(a), []).append(int(t_))

    # --- クラスタ分割（企業＋トピック位置の全体で学習し、企業をそのラベルで分ける）---
    all_pts = np.concatenate([h, P], axis=0)
    km = KMeans(n_clusters=int(args.n_clusters), n_init=10, random_state=0).fit(all_pts)
    h_labels = km.labels_[: h.shape[0]]
    P_labels = km.labels_[h.shape[0]:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for c in range(int(args.n_clusters)):
        h_idx = np.where(h_labels == c)[0]
        P_idx = np.where(P_labels == c)[0]
        if len(h_idx) < 5:
            print(f"cluster {c}: 企業数が少なすぎるためスキップ ({len(h_idx)})")
            continue
        h_c = h[h_idx]
        P_c = P[P_idx] if len(P_idx) else P
        pred_c = pred_vec[h_idx]

        pts_c = np.concatenate([h_c, P_c], axis=0)
        lo, hi_ = np.percentile(pts_c, [2, 98], axis=0)
        pad = 0.2 * np.maximum(hi_ - lo, 1e-3)
        xmin, xmax = lo[0] - pad[0], hi_[0] + pad[0]
        ymin, ymax = lo[1] - pad[1], hi_[1] + pad[1]

        res = int(args.grid_n)
        xs = np.linspace(xmin, xmax, res)
        ys = np.linspace(ymin, ymax, res)
        XX, YY = np.meshgrid(xs, ys)
        grid = torch.tensor(np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=torch.float32)
        with torch.no_grad():
            V = ode_f(torch.tensor(0.0), grid).numpy()
        speed = np.linalg.norm(V, axis=1).reshape(res, res)
        U = V[:, 0].reshape(res, res)
        Vv = V[:, 1].reshape(res, res)

        nb = int(args.n_bins)
        xedges = np.linspace(xmin, xmax, nb + 1)
        yedges = np.linspace(ymin, ymax, nb + 1)
        XC, YC, mean_u, mean_v, cnt, R = binned_average(h_c, pred_c, xedges, yedges, args.min_bin_count)
        binw_x = (xmax - xmin) / nb
        binw_y = (ymax - ymin) / nb

        valid = ~np.isnan(mean_u)
        if valid.any():
            mags = np.hypot(mean_u[valid], mean_v[valid])
            typical_mag = np.median(mags[mags > 0]) if (mags > 0).any() else 1.0
            target_len = 0.07 * min(xmax - xmin, ymax - ymin)
            scale_boost = target_len / max(typical_mag, 1e-8)
        else:
            scale_boost = 1.0

        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.5, 7.2), facecolor="#11151b")
        for a_ in (ax, ax2):
            a_.set_facecolor("#11151b")

        # 左: |v(z)| と平均予測ベクトル（従来通り）
        heat = ax.pcolormesh(XX, YY, speed, cmap="magma", shading="gouraud", alpha=0.85)
        cb = fig.colorbar(heat, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("|v(z)|", color="#93a2b3")
        cb.ax.yaxis.set_tick_params(color="#93a2b3")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#93a2b3")
        ax.streamplot(XX, YY, U, Vv, color=(0.906, 0.925, 0.949, 0.4), linewidth=0.5, density=1.0, arrowsize=0.6)
        ax.scatter(h_c[:, 0], h_c[:, 1], s=5, c="#5c93b8", alpha=0.3, label=f"企業 (n={len(h_c)})", zorder=4)
        ax.scatter(P_c[:, 0], P_c[:, 1], s=12, c="#d9a15c", marker="^", alpha=0.85, label="技術(CPC)アンカー", zorder=5)
        if valid.any():
            ax.quiver(
                XC[valid], YC[valid], mean_u[valid] * scale_boost, mean_v[valid] * scale_boost,
                color="#5ce8d8", angles="xy", scale_units="xy", scale=1, width=0.0035,
                alpha=0.95, zorder=7, label=f"ビン平均予測ベクトル(×{scale_boost:.1f}表示)",
            )
        ax.set_title("|v(z)| と平均予測ベクトル", color="#e7ecf2", fontsize=11)

        # 右: 同一ビン内の企業間で予測方向がどれだけ揃っているか（R）
        Rplot = np.ma.masked_invalid(R)
        heat2 = ax2.pcolormesh(
            xedges, yedges, Rplot, cmap="viridis", vmin=0, vmax=1, shading="flat",
        )
        cb2 = fig.colorbar(heat2, ax=ax2, fraction=0.045, pad=0.02)
        cb2.set_label("方向の一致度 R（1=揃っている、0=バラバラ）", color="#93a2b3")
        cb2.ax.yaxis.set_tick_params(color="#93a2b3")
        plt.setp(plt.getp(cb2.ax.axes, "yticklabels"), color="#93a2b3")
        ax2.scatter(P_c[:, 0], P_c[:, 1], s=10, c="#e7ecf2", marker="^", alpha=0.6, zorder=5)
        if valid.any():
            ax2.quiver(
                XC[valid], YC[valid], mean_u[valid] * scale_boost, mean_v[valid] * scale_boost,
                color="#ffffff", angles="xy", scale_units="xy", scale=1, width=0.003,
                alpha=0.55, zorder=6,
            )
        ax2.set_title(f"予測方向の一致度 R（{nb}×{nb}ビン、最小{args.min_bin_count}社/ビン）", color="#e7ecf2", fontsize=11)

        for a_ in (ax, ax2):
            a_.set_xlabel("latent dim 1", color="#93a2b3")
            a_.set_ylabel("latent dim 2", color="#93a2b3")
            a_.tick_params(colors="#5f6d7d")
            for spine in a_.spines.values():
                spine.set_color("#2e3946")
            a_.set_xlim(xmin, xmax)
            a_.set_ylim(ymin, ymax)
        ax.legend(loc="upper right", facecolor="#1a212b", edgecolor="#2e3946", labelcolor="#e7ecf2", fontsize=7.5)

        fig.suptitle(
            f"cluster {c}  (企業{len(h_c)}社・技術{len(P_c)}件)  {args.year_prev}→{args.year_next}",
            color="#e7ecf2", fontsize=13,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        out_png = out_dir / f"cluster{c}.png"
        plt.savefig(str(out_png), dpi=155, facecolor=fig.get_facecolor())
        plt.savefig(str(out_dir / f"cluster{c}.pdf"), facecolor=fig.get_facecolor())
        plt.close(fig)
        valid_R = ~np.isnan(R)
        r_mean = float(np.nanmean(R)) if valid_R.any() else float("nan")
        print(
            f"cluster {c}: 企業{len(h_c)}社, 技術{len(P_c)}件, "
            f"quiverビン有効数={int(valid.sum())}, 平均R={r_mean:.3f} -> {out_png}"
        )

    return 0


if __name__ == "__main__":
    main()
