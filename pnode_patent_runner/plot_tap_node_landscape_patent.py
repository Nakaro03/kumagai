#!/usr/bin/env python3
"""
学習済み TAP-NODE(+burst) checkpoint（patentドメイン）から Φ(z, t) 地形の静止画を出力する。

`plot_tap_node_landscape.py`（arXiv著者–トピック用）と同じ考え方だが、データローダーを
`dual_force_data_patent.load_dual_force_bundle_patent_domain` に差し替え、burst 項も
（学習済みなら）Φ に反映する。TAP-NODE は真の勾配流（保存力場）なので、Dual-Force
（非保存力場、`plot_dual_force_flow.py`）と違い、井戸（谷）を持つポテンシャル地形を直接描ける。

- 左: Φ ヒートマップ（谷=明・峰=暗）+ 等高線 + −∇Φ 矢印 +
      技術(CPC)アンカー（大きさ=log1p(件数), 色=トレンド D̃; 赤=衰退, 青=成長）+ 企業点
- 右: トレンド項の寄与 ΔΦ = Φ(b=学習値) − Φ(b=0)（RdBu_r; 赤=隆起=衰退帯, 青=沈降=成長帯）

例:
  python -m pnode_patent_runner.plot_tap_node_landscape_patent \\
    --load-checkpoint pnode_patent_runner/outputs/tap_node_patent/ckpt/tap_node_burst_construction_seed42.pt \\
    --anchor-year 2020 \\
    --output pnode_patent_runner/outputs/tap_node_patent/figures/landscape_construction_seed42
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
from pnode_patent_runner.tap_node_models import TAPVGAE


def main() -> int:
    p = argparse.ArgumentParser(description="TAP-NODE(+burst) Φ landscape プロット（patentドメイン）")
    p.add_argument("--load-checkpoint", type=Path, required=True)
    p.add_argument("--anchor-year", type=int, required=True, help="Φ を描く年（この年の質量・トレンド・burstで井戸を変調）")
    p.add_argument("--grid-n", type=int, default=220)
    p.add_argument("--quiver-n", type=int, default=23)
    p.add_argument("--output", type=Path, required=True, help="拡張子なし。.png と .pdf を両方出力")
    args = p.parse_args()

    device = torch.device("cpu")
    ck = torch.load(str(args.load_checkpoint), map_location=device, weights_only=False)

    model = TAPVGAE(
        ck["num_nodes"], ck["num_authors"], ck["input_dim"],
        hidden_dim=ck["hidden_dim"], latent_dim=ck["latent_dim"],
        initial_author_vectors=ck.get("initial_author_vectors"),
        link_score_mode=ck.get("link_score_mode", "distance"),
    ).to(device)
    if int(ck["latent_dim"]) != 2:
        raise SystemExit("latent_dim=2 の checkpoint のみ対応（直接 2D 描画のため）")
    skipped, _ = load_state_dict_skip_shape_mismatch(model, ck["state_dict"])
    model.eval()

    bundle = load_dual_force_bundle_patent_domain(
        ck["csv_path"], year_range=tuple(ck["year_range"]), min_events=2,
        burst_percentile=float(ck.get("burst_percentile", 80.0)),
    )
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
        data.topic_mass,
        data.topic_trend_plus,
        data.topic_trend_minus,
        getattr(data, "topic_burst", None),
    )

    z_np = z.detach().cpu().numpy()
    za, zt = z_np[:n_a], z_np[n_a:]
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

    b_learned = float(ode.b.item())
    with torch.no_grad():
        ode.b.fill_(0.0)
        _, _, PHI_B0 = phi_on_grid(int(args.grid_n))
        ode.b.fill_(b_learned)
    DPHI = PHI - PHI_B0
    DPHI = DPHI - np.median(DPHI)

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

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), constrained_layout=True, facecolor="#11151b")

    ax = axes[0]
    ax.set_facecolor("#11151b")
    im0 = ax.pcolormesh(X, Y, PHI, cmap="Greys_r", shading="auto", rasterized=True)
    ax.contour(X, Y, PHI, levels=14, colors="#5f6d7d", linewidths=0.5, alpha=0.6)
    ax.quiver(XQ, YQ, U, W, color="#93a2b3", alpha=0.6, width=0.0028, scale_units="xy")
    rng = np.random.default_rng(0)
    ia = rng.choice(za.shape[0], size=max(1, za.shape[0] // 5), replace=False)
    ax.scatter(za[ia, 0], za[ia, 1], s=2.5, c="#5c93b8", alpha=0.12, linewidths=0, label="企業（20%サンプル）")
    sc = ax.scatter(
        zt[:, 0], zt[:, 1], s=sizes, c=-d_tilde, cmap="RdBu_r", vmin=-dmax, vmax=dmax,
        edgecolors="#e7ecf2", linewidths=0.6, label="技術(CPC)アンカー",
    )
    dom = ck.get("domain", "?")
    ax.set_title(f"TAP-NODE(+burst) 学習済みポテンシャル Φ(z, {year}) — {dom}   (谷=明)", color="#e7ecf2", fontsize=11)
    ax.set_xlabel("latent dim 1", color="#93a2b3")
    ax.set_ylabel("latent dim 2", color="#93a2b3")
    ax.tick_params(colors="#5f6d7d")
    for spine in ax.spines.values():
        spine.set_color("#2e3946")
    cb0 = fig.colorbar(im0, ax=ax, shrink=0.85, pad=0.01)
    cb0.set_label("Φ", color="#93a2b3")
    cb0.ax.yaxis.set_tick_params(color="#93a2b3")
    plt.setp(plt.getp(cb0.ax.axes, "yticklabels"), color="#93a2b3")
    cb1 = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.03)
    cb1.set_label("技術トレンド（赤=衰退, 青=成長）", color="#93a2b3")
    cb1.ax.yaxis.set_tick_params(color="#93a2b3")
    plt.setp(plt.getp(cb1.ax.axes, "yticklabels"), color="#93a2b3")
    ax.legend(loc="upper right", framealpha=0.85, fontsize=8, markerscale=1.5,
              facecolor="#1a212b", edgecolor="#2e3946", labelcolor="#e7ecf2")

    ax = axes[1]
    ax.set_facecolor("#11151b")
    vmax = np.percentile(np.abs(DPHI), 99)
    im2 = ax.pcolormesh(X, Y, DPHI, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    ax.scatter(zt[:, 0], zt[:, 1], s=sizes, c="none", edgecolors="#e7ecf2", linewidths=0.7)
    ax.set_title(f"トレンド項の寄与  ΔΦ = Φ(b={b_learned:.2f}) − Φ(b=0)", color="#e7ecf2", fontsize=11)
    ax.set_xlabel("latent dim 1", color="#93a2b3")
    ax.set_ylabel("latent dim 2", color="#93a2b3")
    ax.tick_params(colors="#5f6d7d")
    for spine in ax.spines.values():
        spine.set_color("#2e3946")
    cb2 = fig.colorbar(im2, ax=ax, shrink=0.85, pad=0.01)
    cb2.set_label("ΔΦ（赤=衰退で隆起, 青=成長で沈降）", color="#93a2b3")
    cb2.ax.yaxis.set_tick_params(color="#93a2b3")
    plt.setp(plt.getp(cb2.ax.axes, "yticklabels"), color="#93a2b3")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=150, facecolor=fig.get_facecolor())
    fig.savefig(str(out) + ".pdf", facecolor=fig.get_facecolor())
    print(f"Wrote: {out}.png")
    print(f"Wrote: {out}.pdf")
    if skipped:
        print(f"skipped checkpoint keys (再設定済み): {skipped}")
    return 0


if __name__ == "__main__":
    main()
