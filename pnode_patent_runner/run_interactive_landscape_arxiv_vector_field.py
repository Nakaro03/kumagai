#!/usr/bin/env python3
"""
著者–論文（ArXiv 風 CSV）で学習した UnifiedVGAE から、
潜在マップ + Φ / −∇Φ のインタラクティブ HTML を出力する。

**年の整合**: ``preprocess_arxiv_data`` の ``year_min`` / ``year_max``（CLI: ``--arxiv-year-*``）と
学習時を揃える。スライダーで切り替える年は ``--year-range`` / ``--years`` で束ねた
``bundle.graphs`` のキー（論文の ``year``）に対応する。

例:
  python -m pnode_patent_runner.run_train_unified_vgae_checkpoint \\
    --data-domain arxiv --data data/processed/arxiv_cs_embedded_2020-2026.csv \\
    --year-range 2020 2026 --arxiv-year-min 2020 --arxiv-year-max 2026 \\
    --min-patents 5 --epochs 20 --cope-link-score cosine \\
    --save pnode_patent_runner/outputs/arxiv_landscape/unified_vgae_arxiv.pt

  python -m pnode_patent_runner.run_interactive_landscape_arxiv_vector_field \\
    --data data/processed/arxiv_cs_embedded_2020-2026.csv \\
    --year-range 2020 2026 --arxiv-year-min 2020 --arxiv-year-max 2026 \\
    --min-patents 5 --load-checkpoint pnode_patent_runner/outputs/arxiv_landscape/unified_vgae_arxiv.pt \\
    --cope-link-score cosine \\
    --output pnode_patent_runner/outputs/arxiv_landscape/interactive_map_arxiv.html
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pnode_patent_runner.cope_experiment import (
    load_author_paper_graph_bundle,
    load_author_topic_graph_bundle,
)
from pnode_patent_runner.interactive_landscape import (
    build_interactive_payload_author_paper,
    build_interactive_payload_author_topic,
)
from pnode_patent_runner.interactive_landscape_vector_field import (
    alt_dark_ui_labels,
    compute_vector_field_for_plotly,
    merge_payload_with_vector_field,
    write_interactive_vector_field_html,
)
from pnode_patent_runner.unified_vgae import METHOD_SHORT_NAME, UnifiedVGAE
from pnode_patent_runner.checkpoint_utils import load_state_dict_skip_shape_mismatch


def _default_arxiv_csv(repo: Path) -> Path:
    for candidate in (
        repo / "data/processed/arxiv_cs_embedded_2020-2026_full.csv",
        repo / "notebooks/work/dataset/arxiv_cs_Data/arxiv_cs_embedded_2020-2026.csv",
        repo / "data/processed/arxiv_cs_embedded_2020-2026.csv",
    ):
        if candidate.is_file():
            return candidate
    return repo / "notebooks/work/dataset/arxiv_cs_Data/arxiv_cs_embedded_2020-2026.csv"


def main() -> None:
    warnings.filterwarnings("ignore")

    repo = _REPO_ROOT
    out_paper = repo / "pnode_patent_runner/outputs/arxiv_landscape/interactive_map_arxiv.html"
    out_topic = repo / "pnode_patent_runner/outputs/arxiv_landscape/interactive_map_arxiv_topic.html"
    tpl_paper = repo / "pnode_patent_runner/interactive_vector_field_author_paper.html"
    tpl_topic = repo / "pnode_patent_runner/interactive_vector_field_author_topic.html"

    parser = argparse.ArgumentParser(
        description=f"{METHOD_SHORT_NAME} — 著者–論文 / 著者–トピック 潜在マップ + Φ / −∇Φ（HTML）",
    )
    parser.add_argument(
        "--graph-mode",
        type=str,
        choices=("paper", "topic"),
        default="paper",
        help="paper=右ノードは論文 / topic=右ノードは topic 列（別グラフ・別学習の .pt が必要）",
    )
    parser.add_argument(
        "--topic-column",
        type=str,
        default="topic",
        help="graph-mode=topic のときのトピック列名",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="ArXiv 埋め込み CSV。省略時は dataset または data/processed を探索",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="省略時: paper→interactive_map_arxiv.html / topic→interactive_map_arxiv_topic.html",
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
        help="束ねる論文年（学習時の --year-range と一致推奨）",
    )
    parser.add_argument(
        "--min-patents",
        type=int,
        default=5,
        help="著者あたり最小論文（行）数（学習時と同一に）",
    )
    parser.add_argument(
        "--arxiv-year-min",
        type=int,
        default=2020,
        help="前処理の年下限（学習時と同一に）",
    )
    parser.add_argument(
        "--arxiv-year-max",
        type=int,
        default=2026,
        help="前処理の年上限（学習時と同一に）",
    )
    parser.add_argument(
        "--arxiv-no-year-filter",
        action="store_true",
        help="前処理の年フィルタを無効化（学習時と揃えること）",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--load-checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-max-chars", type=int, default=450)
    parser.add_argument("--field-resolution", type=int, default=42)
    parser.add_argument("--field-margin", type=float, default=0.5)
    parser.add_argument(
        "--quiver-stride",
        type=int,
        default=4,
        help="矢印グリッド間隔（大きいほど疎＝見やすい。gray_grid 推奨 3〜5）",
    )
    parser.add_argument("--quiver-length", type=float, default=1.75)
    parser.add_argument("--n-mag-bins", type=int, default=5)
    parser.add_argument(
        "--cope-link-score",
        type=str,
        choices=("distance", "cosine"),
        default="cosine",
    )
    parser.add_argument("--cosine-logit-scale", type=float, default=5.0)
    parser.add_argument("--w-pot-init", type=float, default=0.05)
    parser.add_argument(
        "--cope-density-calibrated",
        action="store_true",
        help="チェックポイントと同じ案1（密度校準）アーキでモデルを構築",
    )
    parser.add_argument("--cope-density-log-weight", type=float, default=1.0)
    parser.add_argument("--cope-density-ema-momentum", type=float, default=0.05)
    parser.add_argument(
        "--html-template",
        type=str,
        default="",
        help="省略時は graph-mode に応じた著者–論文／著者–トピック用テンプレ",
    )
    args = parser.parse_args()

    out_path = Path(args.output.strip()) if args.output.strip() else (
        out_topic if args.graph_mode == "topic" else out_paper
    )
    tpl_path = Path(args.html_template.strip()) if args.html_template.strip() else (
        tpl_topic if args.graph_mode == "topic" else tpl_paper
    )

    if args.latent_dim != 2:
        raise SystemExit("ベクトル場 HTML は latent_dim=2 のみ対応です。")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path((args.data or "").strip() or _default_arxiv_csv(repo))
    if not data_path.is_file():
        raise SystemExit(f"データが見つかりません: {data_path}")

    ckpt = Path(args.load_checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"チェックポイントが見つかりません: {ckpt}")

    ymin: Optional[int] = None if args.arxiv_no_year_filter else args.arxiv_year_min
    ymax: Optional[int] = None if args.arxiv_no_year_filter else args.arxiv_year_max
    yr = tuple(args.year_range) if args.year_range is not None else None

    try:
        if args.graph_mode == "topic":
            bundle = load_author_topic_graph_bundle(
                data_path,
                min_papers=args.min_patents,
                topic_column=args.topic_column,
                arxiv_year_min=ymin,
                arxiv_year_max=ymax,
                year_range=yr,
                years_csv=args.years,
                all_years=args.all_years,
            )
        else:
            bundle = load_author_paper_graph_bundle(
                data_path,
                min_papers=args.min_patents,
                arxiv_year_min=ymin,
                arxiv_year_max=ymax,
                year_range=yr,
                years_csv=args.years,
                all_years=args.all_years,
            )
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e))

    df = bundle.dataframe
    rights = bundle.right_nodes
    authors = bundle.corps
    if df is None or rights is None:
        raise SystemExit("内部エラー: bundle に dataframe / right_nodes がありません。")

    graphs = bundle.graphs
    num_authors = bundle.num_corps
    years_list: List[int] = sorted(graphs.keys())

    if args.year is not None:
        if args.year not in graphs:
            raise SystemExit(f"--year {args.year} が束にありません。利用可能: {years_list}")
        default_year = str(args.year)
    else:
        default_year = str(max(years_list))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"ArXiv 前処理の年: min={ymin}, max={ymax}（学習時と同一にしてください）"
    )
    print(
        f"graph_mode={args.graph_mode}, 束の年: {years_list} ／ "
        f"num_authors={num_authors}, total_n={bundle.total_n}, in_dim={bundle.in_dim}"
    )
    if args.graph_mode == "topic":
        print("  注意: チェックポイントは著者–トピックグラフで学習したものを使用してください。")

    model = UnifiedVGAE(
        num_nodes=bundle.total_n,
        num_corps=num_authors,
        input_dim=bundle.in_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        initial_corp_vectors=bundle.init_vectors,
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
            model,
            graphs,
            num_authors,
            bundle.hist_edges,
            num_epochs=args.epochs,
        )

    by_year = {}
    for y in years_list:
        data_y = graphs[y].to(device)
        with torch.no_grad():
            z, _, _ = model.encode(data_y.x, data_y.edge_index)
            z_np = z.cpu().numpy()
        if args.graph_mode == "topic":
            base = build_interactive_payload_author_topic(
                df,
                graphs[y],
                num_authors,
                authors,
                rights,
                z_np,
                y,
                topic_column=args.topic_column,
                summary_max=args.summary_max_chars,
            )
        else:
            base = build_interactive_payload_author_paper(
                df,
                graphs[y],
                num_authors,
                authors,
                rights,
                z_np,
                y,
                summary_max=args.summary_max_chars,
            )
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
                phi_heatmap_style="valley_red_peak_blue",
                arrow_style="gray_grid",
            )
        by_year[str(y)] = merge_payload_with_vector_field(base, vf)

    page_title = f"{METHOD_SHORT_NAME} (arxiv {args.graph_mode}): latent + Φ / −∇Φ"
    heading = (
        f"{METHOD_SHORT_NAME} — 著者–トピック 潜在マップ ＋ Φ・−∇Φ"
        if args.graph_mode == "topic"
        else f"{METHOD_SHORT_NAME} — 著者–論文 潜在マップ ＋ Φ・−∇Φ"
    )
    write_interactive_vector_field_html(
        by_year,
        out_path,
        default_year,
        page_title=page_title,
        heading=heading,
        template_path=tpl_path,
        ui=alt_dark_ui_labels(
            "author_topic" if args.graph_mode == "topic" else "author_paper"
        ),
    )
    print(f"Wrote: {out_path}")
    print(f"  埋め込み年: {years_list}（初期表示: {default_year}）")
    print(f"  link_score_mode={args.cope_link_score}")
    rn = "トピック" if args.graph_mode == "topic" else "論文"
    for y in years_list:
        p = by_year[str(y)]
        print(f"    {y}: {rn} {len(p['patents'])} 点, 著者 {len(p['corporations'])} 点（active）")


if __name__ == "__main__":
    main()
