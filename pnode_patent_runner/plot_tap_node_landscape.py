#!/usr/bin/env python3
"""
学習済み TAP-NODE checkpoint から Φ(z, t) 地形の静的図を出力する。

- 左: Φ ヒートマップ（Greys, 谷=明・峰=暗）+ 等高線 + −∇Φ 矢印 +
      トピックアンカー（大きさ= log1p(質量), 色=トレンド D̃; 赤=衰退, 青=成長）+ 著者点
- 右: トレンド項の寄与 ΔΦ = Φ(b=学習値) − Φ(b=0)（RdBu_r; 赤=隆起=衰退帯, 青=沈降=成長帯）

例:
  python -m pnode_patent_runner.plot_tap_node_landscape \\
    --load-checkpoint pnode_patent_runner/outputs/tap_node/ckpt/tap_node_slr_seed42.pt \\
    --anchor-year 2024 \\
    --output pnode_patent_runner/outputs/tap_node/tap_node_landscape_seed42.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pnode_patent_runner.checkpoint_utils import load_state_dict_skip_shape_mismatch
from pnode_patent_runner.dual_force_data import load_dual_force_bundle
from pnode_patent_runner.tap_node_models import TAPVGAE


def main() -> int:
    p = argparse.ArgumentParser(description="TAP-NODE Φ landscape plot")
    p.add_argument("--data", type=str, default="data/processed/arxiv_cs_embedded_2020-2026_full.csv")
    p.add_argument("--topic-column", type=str, default="topic")
    p.add_argument("--min-patents", type=int, default=5)
    p.add_argument("--load-checkpoint", type=Path, required=True)
    p.add_argument("--anchor-year", type=int, default=2024, help="Φ を描く年（この年の質量・トレンドで井戸を変調）")
    p.add_argument("--grid-n", type=int, default=220)
    p.add_argument("--quiver-n", type=int, default=23)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    ck = torch.load(str(args.load_checkpoint), map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TAPVGAE(
        ck["num_nodes"],
        ck["num_authors"],
        ck["input_dim"],
        hidden_dim=ck["hidden_dim"],
        latent_dim=ck["latent_dim"],
        initial_author_vectors=ck.get("initial_author_vectors"),
        link_score_mode=ck.get("link_score_mode", "distance"),
    ).to(device)
    # トピック情報バッファ (P_j 等) は年ごとに set_topic_info で再設定するため形状不一致はスキップ
    load_state_dict_skip_shape_mismatch(model, ck["state_dict"])
    model.eval()
    if int(ck["latent_dim"]) != 2:
        raise SystemExit("latent_dim=2 の checkpoint のみ対応（直接 2D 描画のため）")

    bundle = load_dual_force_bundle(args.data, topic_column=args.topic_column, min_papers=args.min_patents)
    year = int(args.anchor_year)
    if year not in bundle.graphs:
        raise SystemExit(f"anchor-year={year} がグラフにありません: {sorted(bundle.graphs)}")
    data = bundle.graphs[year].to(device)

    with torch.no_grad():
        z, _, _ = model.encode(data.x, data.edge_index)
    n_a = model.num_corps
    ode = model.temporal_predictor.ode_func
    ode.set_topic_info(
        z[n_a:],
        data.topic_mass.to(device),
        data.topic_trend_plus.to(device),
        data.topic_trend_minus.to(device),
    )

    z_np = z.detach().cpu().numpy()
    za, zt = z_np[:n_a], z_np[n_a:]
    # 外れ著者で描画範囲が伸びないよう、著者は分位点・アンカーは全点で範囲を決める
    lo = np.minimum(np.percentile(za, 0.5, axis=0), zt.min(axis=0)) - 0.15
    hi = np.maximum(np.percentile(za, 99.5, axis=0), zt.max(axis=0)) + 0.15

    def phi_on_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.linspace(lo[0], hi[0], n)
        ys = np.linspace(lo[1], hi[1], n)
        X, Y = np.meshgrid(xs, ys)
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
        with torch.no_grad():
            P = ode.phi(pts).cpu().numpy().reshape(n, n)
        return X, Y, P

    X, Y, PHI = phi_on_grid(int(args.grid_n))

    # トレンド項の寄与: 同一アンカー・同一 κ,h で b のみ 0 に
    b_learned = float(ode.b.item())
    with torch.no_grad():
        ode.b.fill_(0.0)
        _, _, PHI_B0 = phi_on_grid(int(args.grid_n))
        ode.b.fill_(b_learned)
    # ポテンシャルは定数差に不変（勾配流は変わらない）ため、空間変動成分のみ描く
    DPHI = PHI - PHI_B0
    DPHI = DPHI - np.median(DPHI)

    # −∇Φ 矢印（粗い格子で）
    nq = int(args.quiver_n)
    xq = np.linspace(lo[0], hi[0], nq)
    yq = np.linspace(lo[1], hi[1], nq)
    XQ, YQ = np.meshgrid(xq, yq)
    ptsq = torch.tensor(np.stack([XQ.ravel(), YQ.ravel()], axis=1), dtype=torch.float32, device=device)
    with torch.no_grad():
        V = ode.forward(0.0, ptsq).cpu().numpy()
    U = V[:, 0].reshape(nq, nq)
    W = V[:, 1].reshape(nq, nq)

    mass = data.topic_mass.cpu().numpy()
    d_tilde = (
        np.log1p(np.clip(data.topic_trend_plus.cpu().numpy(), 0, None))
        - np.log1p(np.clip(data.topic_trend_minus.cpu().numpy(), 0, None))
    )
    sizes = 12 + 48 * (np.log1p(mass) / max(np.log1p(mass).max(), 1e-9))
    dmax = max(np.abs(d_tilde).max(), 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), constrained_layout=True)

    ax = axes[0]
    im0 = ax.pcolormesh(X, Y, PHI, cmap="Greys", shading="auto", rasterized=True)
    ax.contour(X, Y, PHI, levels=14, colors="#999999", linewidths=0.5, alpha=0.6)
    ax.quiver(XQ, YQ, U, W, color="#555555", alpha=0.55, width=0.0028, scale_units="xy")
    # 著者は密集して谷を塗り潰すため 2 割に間引き、薄く描く
    rng = np.random.default_rng(0)
    ia = rng.choice(za.shape[0], size=max(1, za.shape[0] // 5), replace=False)
    ax.scatter(za[ia, 0], za[ia, 1], s=2.5, c="#222222", alpha=0.07, linewidths=0, label="authors (20%)")
    sc = ax.scatter(
        zt[:, 0], zt[:, 1], s=sizes, c=-d_tilde, cmap="RdBu_r", vmin=-dmax, vmax=dmax,
        edgecolors="#ffffff", linewidths=0.7, label="topic anchors",
    )
    ax.set_title(f"TAP-NODE learned potential Φ(z, {year})   (valleys = light)")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    cb0 = fig.colorbar(im0, ax=ax, shrink=0.85, pad=0.01)
    cb0.set_label("Φ")
    cb1 = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.03)
    cb1.set_label("topic trend  (red = declining, blue = growing)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8, markerscale=1.5)

    ax = axes[1]
    vmax = np.percentile(np.abs(DPHI), 99)
    im2 = ax.pcolormesh(X, Y, DPHI, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    ax.scatter(
        zt[:, 0], zt[:, 1], s=sizes, c="none",
        edgecolors="#333333", linewidths=0.8,
    )
    ax.set_title(f"Trend-term contribution  ΔΦ = Φ(b={b_learned:.2f}) − Φ(b=0)")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    cb2 = fig.colorbar(im2, ax=ax, shrink=0.85, pad=0.01)
    cb2.set_label("ΔΦ  (red = raised near declining, blue = lowered near growing)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    main()
