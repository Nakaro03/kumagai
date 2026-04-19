#!/usr/bin/env python3
"""
CoPE-VGAE（UnifiedVGAE）学習済み重みから、
interactive_landscape_vector_field と同型のインタラクティブ HTML を出力する（ページ表記は CoPE 向け）。

例（リポジトリルートで。既定はダーク左パネル版テンプレ＝ `map_cope_alt_dark.html` 系）:
  python -m pnode_patent_runner.run_interactive_landscape_cope_vector_field \\
    --data notebooks/work/dataset/topic_info3.csv \\
    --year-range 2010 2020 \\
    --load-checkpoint pnode_patent_runner/outputs/cope_landscape/unified_vgae.pt \\
    --cope-link-score distance

Φ を学習した PotentialNet ではなく **密度由来（Φ=−log p̂, 特許 μ の KDE）**にしたい場合:
  ... 同上 ... --phi-source density_kde

注意: latent_dim=2 のチェックポイントのみ（ベクトル場・等高線は 2D 平面）。

別テンプレート（標準レイアウト）にしたい場合:
  ... --html-template pnode_patent_runner/interactive_vector_field_template.html \\
      --output pnode_patent_runner/outputs/cope_landscape/interactive_map_cope_vector_field.html
"""
import argparse
import sys
import warnings
from pathlib import Path

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
from pnode_patent_runner.interactive_landscape import build_interactive_payload
from pnode_patent_runner.interactive_landscape_vector_field import (
    alt_dark_ui_labels,
    compute_vector_field_density_potential_for_plotly,
    compute_vector_field_for_plotly,
    merge_payload_with_vector_field,
    write_interactive_vector_field_html,
)
from pnode_patent_runner.unified_vgae import METHOD_SHORT_NAME, UnifiedVGAE
from pnode_patent_runner.checkpoint_utils import load_state_dict_skip_shape_mismatch


def _patent_mu_xy(mu_all: np.ndarray, num_corps: int, active_mask: torch.Tensor) -> np.ndarray:
    """当該年グラフでアクティブな特許ノードの μ（形状 (k,2)）。"""
    if mu_all.shape[0] <= num_corps:
        return np.zeros((0, mu_all.shape[1]), dtype=np.float64)
    pat_mask = active_mask[num_corps:].detach().cpu().numpy().astype(bool)
    pts = mu_all[num_corps:][pat_mask]
    return np.asarray(pts, dtype=np.float64)


def main():
    warnings.filterwarnings("ignore")

    repo = _REPO_ROOT
    default_csv = repo / "notebooks/work/dataset/topic_info3.csv"
    # 論文・スクリーンショット用の既定: alt_dark テンプレ → map_cope_alt_dark.html（Downloads の同名と同系）
    default_out = repo / "pnode_patent_runner/outputs/cope_landscape/map_cope_alt_dark.html"
    default_html_template = repo / "pnode_patent_runner/interactive_vector_field_alt_dark.html"

    parser = argparse.ArgumentParser(
        description=f"{METHOD_SHORT_NAME}（UnifiedVGAE）— 潜在マップ + Φ / −∇Φ の HTML",
    )
    parser.add_argument("--data", type=str, default=str(default_csv))
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_out),
        help="出力 .html",
    )
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--years", type=str, default="")
    parser.add_argument("--all-years", action="store_true")
    parser.add_argument(
        "--year-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="例: --year-range 2010 2020",
    )
    parser.add_argument("--min-patents", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--load-checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-max-chars", type=int, default=450)
    parser.add_argument("--field-resolution", type=int, default=42)
    parser.add_argument("--field-margin", type=float, default=0.5)
    parser.add_argument("--quiver-stride", type=int, default=3, help="短矢印シードの間隔")
    parser.add_argument("--quiver-length", type=float, default=1.75, help="短矢印の長さ倍率")
    parser.add_argument("--n-mag-bins", type=int, default=5, help="|∇Φ| の色分けビン数（2〜8）")
    parser.add_argument(
        "--phi-heatmap-style",
        type=str,
        choices=("default", "valley_red_peak_blue"),
        default="valley_red_peak_blue",
        help="Φ ヒート: default=生の Φ（Viridis）/ valley_red_peak_blue=[−1,1] 正規化・谷赤・山青",
    )
    parser.add_argument(
        "--arrow-style",
        type=str,
        choices=("magnitude", "gray_grid"),
        default="gray_grid",
        help="矢印: magnitude=|∇Φ| で色分け / gray_grid=グレー・グリッド（等高線に直交する −∇Φ）",
    )
    parser.add_argument(
        "--phi-source",
        type=str,
        choices=("neural", "density_kde"),
        default="neural",
        help=(
            "Φ の定義: neural=PotentialNet（学習済み）／"
            "density_kde=Φ=−log p̂（特許 μ のガウス KDE）＋−∇Φ は数値勾配"
        ),
    )
    parser.add_argument(
        "--kde-bandwidth",
        type=float,
        default=None,
        help="phi-source=density_kde のときの KDE 帯域幅（省略時は Scott 風）",
    )
    parser.add_argument(
        "--density-phi-contour",
        type=str,
        choices=("global", "multi_peak"),
        default="multi_peak",
        help="density_kde 時の Φ 等高線: global=全域1系統 / multi_peak=log p̂ の複数局所最大ごとに別等高線",
    )
    parser.add_argument(
        "--density-peak-min-percentile",
        type=float,
        default=82.0,
        help="multi_peak: 頂点候補の log p 下位しきい（%）",
    )
    parser.add_argument(
        "--density-peak-min-sep",
        type=float,
        default=2.5,
        help="multi_peak: 頂点間の最小距離（グリッドセル換算）",
    )
    parser.add_argument(
        "--density-peak-level-drop",
        type=float,
        default=1.8,
        help="multi_peak: 各頂点から log p がこの幅だけ下がるまでをその峰の領域に含める",
    )
    parser.add_argument(
        "--density-peak-max",
        type=int,
        default=14,
        help="multi_peak: 描く峰の数の上限",
    )
    parser.add_argument(
        "--density-multi-peak-n-contours",
        type=int,
        default=5,
        help="multi_peak: 各峰あたりの等高線の本数（3〜8 程度）",
    )
    parser.add_argument(
        "--cope-link-score",
        type=str,
        choices=("distance", "cosine"),
        default="cosine",
        help="学習時と同じ link_score_mode（principled 既定は cosine）",
    )
    parser.add_argument("--cosine-logit-scale", type=float, default=5.0)
    parser.add_argument("--w-pot-init", type=float, default=0.05)
    parser.add_argument(
        "--cope-density-calibrated",
        action="store_true",
        help="学習済みチェックポイントと同じ案1（密度校準）アーキでモデルを構築",
    )
    parser.add_argument("--cope-density-log-weight", type=float, default=1.0)
    parser.add_argument("--cope-density-ema-momentum", type=float, default=0.05)
    parser.add_argument(
        "--html-template",
        type=str,
        default=str(default_html_template),
        help=(
            "Plotly HTML テンプレート（__PAYLOAD_B64__ 等）。"
            "既定は interactive_vector_field_alt_dark.html（map_cope_alt_dark 系のダーク UI）。"
            "従来レイアウトは interactive_vector_field_template.html を指定。"
        ),
    )
    args = parser.parse_args()

    if args.latent_dim != 2:
        raise SystemExit("ベクトル場 HTML は latent_dim=2 のみ対応です。")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"データが見つかりません: {data_path}")

    ckpt = Path(args.load_checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"チェックポイントが見つかりません: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = preprocess_data(str(data_path))
    if len(df) == 0:
        raise SystemExit("前処理後データが空です")
    df = filter_active_corporations(df, min_patents=args.min_patents)

    graphs, corps, patents, total_n, hist_edges, in_dim = build_global_graphs(df)
    num_corps = len(corps)
    years_available = sorted(graphs.keys())

    yr = getattr(args, "year_range", None)
    if yr is not None:
        if args.all_years or (args.years or "").strip():
            raise SystemExit("--year-range は --years / --all-years と同時に使えません。")
        y0, y1 = int(yr[0]), int(yr[1])
        if y0 > y1:
            y0, y1 = y1, y0
        years_list = [y for y in years_available if y0 <= y <= y1]
        if not years_list:
            raise SystemExit(
                f"年範囲 {y0}〜{y1} に当てはまるグラフ年がありません。利用可能: {years_available}"
            )
    elif args.all_years:
        years_list = list(years_available)
    elif (args.years or "").strip():
        years_list = []
        for part in args.years.split(","):
            part = part.strip()
            if not part:
                continue
            y = int(part)
            if y not in graphs:
                raise SystemExit(f"year={y} のグラフがありません。利用可能: {years_available}")
            years_list.append(y)
        years_list = sorted(set(years_list))
    else:
        y = args.year if args.year is not None else years_available[-1]
        if y not in graphs:
            raise SystemExit(f"year={y} がありません: {years_available}")
        years_list = [y]

    if args.year is not None:
        if args.year not in years_list:
            raise SystemExit(f"--year {args.year} は選択年リストに含めてください: {years_list}")
        default_year = str(args.year)
    else:
        default_year = str(max(years_list))

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
        print("チェックポイント読込: 次をスキップ（データと学習時でノード数等が異なる場合に発生）:")
        for line in skip_log[:20]:
            print("  ", line)
        if len(skip_log) > 20:
            print(f"  ... 他 {len(skip_log) - 20} 行")
    model.eval()

    if args.epochs > 0:
        from pnode_patent_runner.unified_training import train_model_improved

        model, _, _, _ = train_model_improved(
            model, graphs, num_corps, hist_edges, num_epochs=args.epochs
        )

    by_year = {}
    for y in years_list:
        data_y = graphs[y].to(device)
        with torch.no_grad():
            z, _, _ = model.encode(data_y.x, data_y.edge_index)
            z_np = z.cpu().numpy()
        base = build_interactive_payload(
            df,
            graphs[y],
            num_corps,
            corps,
            patents,
            z_np,
            y,
            summary_max=args.summary_max_chars,
        )
        if args.phi_source == "neural":
            with torch.enable_grad():
                vf = compute_vector_field_for_plotly(
                    model,
                    device,
                    z_np,
                    margin=args.field_margin,
                    resolution=args.field_resolution,
                    quiver_stride=args.quiver_stride,
                    quiver_length_mult=args.quiver_length,
                    n_mag_bins=args.n_mag_bins,
                    phi_heatmap_style=args.phi_heatmap_style,
                    arrow_style=args.arrow_style,
                )
        else:
            mu_pat = _patent_mu_xy(z_np, num_corps, data_y.active_mask)
            vf = compute_vector_field_density_potential_for_plotly(
                z_np,
                mu_pat,
                margin=args.field_margin,
                resolution=args.field_resolution,
                quiver_stride=args.quiver_stride,
                quiver_length_mult=args.quiver_length,
                n_mag_bins=args.n_mag_bins,
                phi_heatmap_style=args.phi_heatmap_style,
                arrow_style=args.arrow_style,
                bandwidth=args.kde_bandwidth,
                phi_contour_mode=str(args.density_phi_contour),
                peak_min_percentile=float(args.density_peak_min_percentile),
                peak_min_sep_cells=float(args.density_peak_min_sep),
                peak_level_drop=float(args.density_peak_level_drop),
                peak_max=int(args.density_peak_max),
                multi_peak_n_contours=int(args.density_multi_peak_n_contours),
            )
        by_year[str(y)] = merge_payload_with_vector_field(base, vf)

    out = Path(args.output)
    page_title = f"{METHOD_SHORT_NAME}: latent map + Φ / −∇Φ"
    heading = f"{METHOD_SHORT_NAME} — 潜在マップ ＋ Φ（等高線・ヒートマップ）・−∇Φ 矢印"
    tpl_opt = Path(args.html_template.strip()) if (args.html_template or "").strip() else None
    write_interactive_vector_field_html(
        by_year,
        out,
        default_year,
        page_title=page_title,
        heading=heading,
        template_path=tpl_opt,
        ui=alt_dark_ui_labels("patent"),
    )
    print(f"Wrote: {out}")
    print(f"  埋め込み年: {years_list}（初期表示: {default_year}）")
    print(f"  phi_source={args.phi_source}, link_score_mode={args.cope_link_score}, cosine_logit_scale={args.cosine_logit_scale}")
    for y in years_list:
        p = by_year[str(y)]
        print(f"    {y}: 特許 {len(p['patents'])} 点, 企業 {len(p['corporations'])} 点（active）")
    print("  HTML 内で等高線／ヒートマップ／矢印の表示・透明度を調整できます。")


if __name__ == "__main__":
    main()
