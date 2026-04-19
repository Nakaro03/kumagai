#!/usr/bin/env python3
"""
PNodeEnergyTD（Φ(z, year) 共有・純勾配流）学習済み checkpoint から、
`run_interactive_landscape_td_vector_field` と同型のインタラクティブ HTML を出力する。

チェックポイントは ``run_benchmark_comparison --save-checkpoint-dir`` が保存した辞書、
または raw state_dict（その場合は --year-min / --year-max 必須）を想定。

例（ベンチマーク後）:
  python -m pnode_patent_runner.run_benchmark_comparison \\
    ... --methods pnode_energy --save-checkpoint-dir pnode_patent_runner/outputs/cope_benchmark/ckpt

  python -m pnode_patent_runner.run_interactive_landscape_pnode_energy_td \\
    --data data/processed/topic_info3.csv \\
    --year-range 2015 2019 \\
    --load-checkpoint pnode_patent_runner/outputs/cope_benchmark/ckpt/pnode_energy_seed42.pt \\
    --output pnode_patent_runner/outputs/cope_landscape/map_pnode_energy_alt_dark_td.html
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pnode_patent_runner.benchmark_pnode_energy_td import METHOD_SHORT_NAME, PNodeEnergyTD
from pnode_patent_runner.data import (
    build_global_graphs,
    calculate_initial_corp_vectors,
    filter_active_corporations,
    preprocess_data,
)
from pnode_patent_runner.interactive_landscape import build_interactive_payload
from pnode_patent_runner.interactive_landscape_vector_field import (
    alt_dark_ui_labels,
    merge_payload_with_vector_field,
    write_interactive_vector_field_html,
)
from pnode_patent_runner.interactive_landscape_vector_field_td import (
    compute_vector_field_for_plotly_td,
)
from pnode_patent_runner.checkpoint_utils import load_state_dict_skip_shape_mismatch


def main():
    warnings.filterwarnings("ignore")

    repo = _REPO_ROOT
    default_csv = repo / "notebooks/work/dataset/topic_info3.csv"
    default_out = repo / "pnode_patent_runner/outputs/cope_landscape/map_pnode_energy_alt_dark_td.html"
    default_html_template = repo / "pnode_patent_runner/interactive_vector_field_alt_dark.html"

    parser = argparse.ArgumentParser(
        description=f"{METHOD_SHORT_NAME} — Φ(z, year) 潜在マップ + −∇Φ（HTML）",
    )
    parser.add_argument("--data", type=str, default=str(default_csv))
    parser.add_argument("--output", type=str, default=str(default_out))
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--years", type=str, default="")
    parser.add_argument("--all-years", action="store_true")
    parser.add_argument(
        "--year-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
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
    parser.add_argument("--quiver-stride", type=int, default=3)
    parser.add_argument("--quiver-length", type=float, default=1.75)
    parser.add_argument("--n-mag-bins", type=int, default=5)
    parser.add_argument(
        "--phi-heatmap-style",
        type=str,
        choices=("default", "valley_red_peak_blue"),
        default="valley_red_peak_blue",
    )
    parser.add_argument(
        "--arrow-style",
        type=str,
        choices=("magnitude", "gray_grid"),
        default="gray_grid",
    )
    parser.add_argument(
        "--cope-link-score",
        type=str,
        choices=("distance", "cosine"),
        default="distance",
    )
    parser.add_argument("--cosine-logit-scale", type=float, default=5.0)
    parser.add_argument(
        "--year-min",
        type=int,
        default=None,
        help="checkpoint に year 情報が無いとき必須",
    )
    parser.add_argument(
        "--year-max",
        type=int,
        default=None,
        help="checkpoint に year 情報が無いとき必須",
    )
    parser.add_argument(
        "--html-template",
        type=str,
        default=str(default_html_template),
    )
    args = parser.parse_args()

    if args.latent_dim != 2:
        raise SystemExit("ベクトル場 HTML は latent_dim=2 のみ対応です。")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"データが見つかりません: {data_path}")

    ckpt_path = Path(args.load_checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"チェックポイントが見つかりません: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    raw_ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        state_dict = raw_ckpt["state_dict"]
        meta_ym = int(raw_ckpt["year_min"])
        meta_yM = int(raw_ckpt["year_max"])
        hidden_dim = int(raw_ckpt.get("hidden_dim", args.hidden_dim))
        latent_dim = int(raw_ckpt.get("latent_dim", args.latent_dim))
        link_score = str(raw_ckpt.get("cope_link_score", args.cope_link_score))
        cos_scale = float(raw_ckpt.get("cosine_logit_scale", args.cosine_logit_scale))
    else:
        state_dict = raw_ckpt
        if args.year_min is None or args.year_max is None:
            raise SystemExit(
                "checkpoint に year_min/year_max が含まれません。"
                " --year-min と --year-max を指定してください。"
            )
        meta_ym, meta_yM = int(args.year_min), int(args.year_max)
        hidden_dim, latent_dim = args.hidden_dim, args.latent_dim
        link_score, cos_scale = args.cope_link_score, args.cosine_logit_scale

    df = preprocess_data(str(data_path))
    if len(df) == 0:
        raise SystemExit("前処理後データが空です")
    df = filter_active_corporations(df, min_patents=args.min_patents)

    graphs, corps, patents, total_n, hist_edges, in_dim = build_global_graphs(df)
    num_corps = len(corps)
    years_available = sorted(graphs.keys())

    yr = args.year_range
    if yr is not None:
        if args.all_years or (args.years or "").strip():
            raise SystemExit("--year-range は --years / --all-years と同時に使えません。")
        y0, y1 = int(yr[0]), int(yr[1])
        if y0 > y1:
            y0, y1 = y1, y0
        years_list = [y for y in years_available if y0 <= y <= y1]
        if not years_list:
            raise SystemExit(
                f"年範囲 {y0}〜{y1} にグラフがありません。利用可能: {years_available}"
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
                raise SystemExit(f"year={y} がありません: {years_available}")
            years_list.append(y)
        years_list = sorted(set(years_list))
    else:
        y = args.year if args.year is not None else years_available[-1]
        if y not in graphs:
            raise SystemExit(f"year={y} がありません: {years_available}")
        years_list = [y]

    for y in years_list:
        if y < meta_ym or y > meta_yM:
            raise SystemExit(
                f"表示年 {y} が学習範囲 [{meta_ym}, {meta_yM}] 外です。"
            )

    if args.year is not None:
        if args.year not in years_list:
            raise SystemExit(f"--year {args.year} は選択年リストに含めてください: {years_list}")
        default_year = str(args.year)
    else:
        default_year = str(max(years_list))

    init_vectors = calculate_initial_corp_vectors(df, num_corps, in_dim, corps)
    model = PNodeEnergyTD(
        num_nodes=total_n,
        num_corps=num_corps,
        input_dim=in_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        initial_corp_vectors=init_vectors,
        link_score_mode=link_score,
        cosine_logit_scale=cos_scale,
        year_min=meta_ym,
        year_max=meta_yM,
    ).to(device)
    skip_log, _ = load_state_dict_skip_shape_mismatch(model, state_dict)
    if skip_log:
        print("チェックポイント読込（一部スキップ）:")
        for line in skip_log[:15]:
            print("  ", line)
    model.eval()

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
        with torch.enable_grad():
            vf = compute_vector_field_for_plotly_td(
                model,
                device,
                z_np,
                calendar_year=int(y),
                margin=args.field_margin,
                resolution=args.field_resolution,
                quiver_stride=args.quiver_stride,
                quiver_length_mult=args.quiver_length,
                n_mag_bins=args.n_mag_bins,
                phi_heatmap_style=args.phi_heatmap_style,
                arrow_style=args.arrow_style,
            )
        by_year[str(y)] = merge_payload_with_vector_field(base, vf)

    out = Path(args.output)
    page_title = f"{METHOD_SHORT_NAME}: Φ(z,y) map + field"
    heading = f"{METHOD_SHORT_NAME} — Φ(z, 年) ＋ −∇Φ（時間依存・純勾配流）"
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
    print(f"  Φ 範囲: [{meta_ym}, {meta_yM}]（スライダー年と一致）")


if __name__ == "__main__":
    main()
