#!/usr/bin/env python3
"""
学習済み UnifiedVGAE の **μ だけ**（特許ノード、eval では encode の第1出力＝μ）で
オフラインに KDE 密度 log p̂_t(z) と時間差 D_t(z)=log p̂_t - log p̂_{ref} を PNG 出力する。

目的: 「時間つき密度」の可視化が意味を持つか、ホットスポット（高 p̂）・衰退（D_t<0）が読めるかの確認。

例（kumagai 直下で）:
  python -m pnode_patent_runner.run_offline_mu_density_maps \\
    --data notebooks/work/dataset/topic_info3.csv \\
    --load-checkpoint pnode_patent_runner/outputs/cope_landscape/unified_vgae.pt \\
    --year-range 2010 2020 \\
    --delta-years 3 \\
    --output-dir pnode_patent_runner/outputs/offline_density_mu

注意: latent_dim=2 を想定（ヒートマップは 2D 平面）。
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pnode_patent_runner.data import (
    build_global_graphs,
    calculate_initial_corp_vectors,
    filter_active_corporations,
    preprocess_data,
)
from pnode_patent_runner.offline_density_maps import (
    log_density_grid,
    resolve_ref_year,
    union_bounds_with_margin,
)
from pnode_patent_runner.unified_vgae import UnifiedVGAE
from pnode_patent_runner.checkpoint_utils import load_state_dict_skip_shape_mismatch


def extract_patent_mu(
    mu_all: np.ndarray,
    num_corps: int,
    active_mask: torch.Tensor,
) -> np.ndarray:
    """その年にエッジを持つノードのみ True の active_mask で、特許ノードの μ を抽出。"""
    n = mu_all.shape[0]
    if n <= num_corps:
        return np.zeros((0, mu_all.shape[1]), dtype=np.float64)
    pat_slice = active_mask[num_corps:].cpu().numpy().astype(bool)
    pts = mu_all[num_corps:][pat_slice]
    return np.asarray(pts, dtype=np.float64)


def main():
    warnings.filterwarnings("ignore")

    repo = _REPO_ROOT
    default_csv = repo / "notebooks/work/dataset/topic_info3.csv"
    default_out = repo / "pnode_patent_runner/outputs/offline_density_mu"

    parser = argparse.ArgumentParser(
        description="μ の KDE による log p̂_t と D_t のオフライン可視化（PNG）",
    )
    parser.add_argument("--data", type=str, default=str(default_csv))
    parser.add_argument("--load-checkpoint", type=str, required=True)
    parser.add_argument(
        "--year-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="例: 2010 2020",
    )
    parser.add_argument("--min-patents", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(default_out))
    parser.add_argument("--resolution", type=int, default=80, help="グリッド解像度（各軸）")
    parser.add_argument("--margin", type=float, default=0.5, help="μ の外接矩形に足す余白")
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help="KDE 帯域幅（省略時は Scott 風の自動）",
    )
    parser.add_argument(
        "--delta-years",
        type=int,
        default=3,
        help="D_t = log p̂_t - log p̂_ref の ref を「t より少なくともこの年数前」に近い年から選ぶ",
    )
    parser.add_argument(
        "--cope-link-score",
        type=str,
        choices=("distance", "cosine"),
        default="cosine",
    )
    parser.add_argument("--cosine-logit-scale", type=float, default=5.0)
    parser.add_argument("--w-pot-init", type=float, default=0.05)
    parser.add_argument("--cope-density-calibrated", action="store_true")
    parser.add_argument("--cope-density-log-weight", type=float, default=1.0)
    parser.add_argument("--cope-density-ema-momentum", type=float, default=0.05)
    args = parser.parse_args()

    if args.latent_dim != 2:
        raise SystemExit("ヒートマップは latent_dim=2 のみ対応です。")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"データが見つかりません: {data_path}")
    ckpt = Path(args.load_checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"チェックポイントが見つかりません: {ckpt}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = preprocess_data(str(data_path))
    if len(df) == 0:
        raise SystemExit("前処理後データが空です")
    df = filter_active_corporations(df, min_patents=args.min_patents)

    graphs, corps, patents, total_n, hist_edges, in_dim = build_global_graphs(df)
    num_corps = len(corps)
    years_available = sorted(graphs.keys())

    if args.year_range is not None:
        y0, y1 = int(args.year_range[0]), int(args.year_range[1])
        if y0 > y1:
            y0, y1 = y1, y0
        years_list = [y for y in years_available if y0 <= y <= y1]
    else:
        years_list = list(years_available)

    if not years_list:
        raise SystemExit(f"対象年がありません。利用可能: {years_available}")

    init_vectors = calculate_initial_corp_vectors(df, num_corps, in_dim, corps)
    model = UnifiedVGAE(
        num_nodes=total_n,
        num_corps=num_corps,
        input_dim=in_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        initial_corp_vectors=init_vectors,
        w_pot_init=args.w_pot_init,
        link_score_mode=args.cope_link_score,
        cosine_logit_scale=args.cosine_logit_scale,
        density_calibrated_potential=bool(args.cope_density_calibrated),
        density_log_weight=float(args.cope_density_log_weight),
        density_ema_momentum=float(args.cope_density_ema_momentum),
    ).to(device)
    raw_sd = torch.load(str(ckpt), map_location=device)
    skip_log, _ = load_state_dict_skip_shape_mismatch(model, raw_sd)
    if skip_log:
        print("チェックポイント読込（スキップあり）:", len(skip_log), "キー")
    model.eval()

    # 各年の特許 μ 点列
    mu_by_year: dict[int, np.ndarray] = {}
    for y in years_list:
        data_y = graphs[y].to(device)
        with torch.no_grad():
            _, mu, _ = model.encode(data_y.x, data_y.edge_index)
            mu_np = mu.cpu().numpy()
        mu_by_year[y] = extract_patent_mu(mu_np, num_corps, data_y.active_mask)

    # 全期間の μ で外接矩形
    x_min, x_max, y_min, y_max = union_bounds_with_margin(
        [mu_by_year[y] for y in years_list], args.margin
    )

    # 各年の log p̂ グリッド（同じ境界・解像度で揃える）＋メッシュ（軸は z1=横, z2=縦）
    logp_by_year: dict[int, np.ndarray] = {}
    Xg = Yg = None
    for y in years_list:
        pts = mu_by_year[y]
        Xg, Yg, logp = log_density_grid(
            pts,
            x_min,
            x_max,
            y_min,
            y_max,
            args.resolution,
            bandwidth=args.bandwidth,
        )
        logp_by_year[y] = logp

    assert Xg is not None and Yg is not None

    # PNG: 各年 1 枚（左 log p̂_t、右 D_t）
    for y in years_list:
        ref = resolve_ref_year(years_list, y, args.delta_years)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

        logp_t = logp_by_year[y]
        im0 = axes[0].pcolormesh(
            Xg,
            Yg,
            logp_t,
            shading="auto",
            cmap="viridis",
        )
        axes[0].set_title(f"log p̂_{{{y}}}(z)  (KDE on μ, patents active in year)")
        axes[0].set_xlabel("z1")
        axes[0].set_ylabel("z2")
        axes[0].set_aspect("equal", adjustable="box")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        if ref is None or ref not in logp_by_year:
            axes[1].text(
                0.5,
                0.5,
                "参照年なし\n（より古い年が year-range にない）",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_axis_off()
        else:
            D = logp_t - logp_by_year[ref]
            vmax = float(np.nanmax(np.abs(D))) + 1e-9
            im1 = axes[1].pcolormesh(
                Xg,
                Yg,
                D,
                shading="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            axes[1].set_title(
                f"D_{{{y}}}(z) = log p̂_{{{y}}} − log p̂_{{{ref}}}  (Δ≥{args.delta_years}y → ref={ref})"
            )
            axes[1].set_xlabel("z1")
            axes[1].set_ylabel("z2")
            axes[1].set_aspect("equal", adjustable="box")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # 散布: μ 点（上に重ねる）
        pts = mu_by_year[y]
        if len(pts) > 0:
            axes[0].scatter(pts[:, 0], pts[:, 1], s=4, c="white", alpha=0.35, linewidths=0)
            if ref is not None and ref in mu_by_year and len(mu_by_year[ref]) > 0:
                pr = mu_by_year[ref]
                axes[1].scatter(pr[:, 0], pr[:, 1], s=3, c="k", alpha=0.2, linewidths=0)
            axes[1].scatter(pts[:, 0], pts[:, 1], s=4, c="yellow", alpha=0.4, linewidths=0)

        fig.suptitle(f"Offline μ density (checkpoint: {ckpt.name})", fontsize=10)
        fig.tight_layout()
        png_path = out_dir / f"mu_density_Dt_{y}_ref{ref if ref else 'none'}.png"
        fig.savefig(png_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote: {png_path}")

    # サマリー CSV（各年の特許点数）
    summary_path = out_dir / "mu_counts_by_year.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "n_patent_mu", "ref_year_for_D", "delta_years_target"])
        for y in years_list:
            ref = resolve_ref_year(years_list, y, args.delta_years)
            w.writerow([y, len(mu_by_year[y]), ref if ref else "", args.delta_years])
    print(f"Wrote: {summary_path}")
    print("解釈の目安: 左パネルで明るい＝高 log p̂（その年の μ が集まる谷／ホットスポット候補）。")
    print("右パネルで青系（負）＝ t より ref で密度が高かった領域＝衰退スポット候補。")


if __name__ == "__main__":
    main()
