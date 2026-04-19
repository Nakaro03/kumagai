#!/usr/bin/env python3
"""
現在の CSV・左ノード最小件数・年範囲でグラフ束を構築し、
UnifiedVGAE を **初期化から** ``train_model_improved`` で学習して ``state_dict`` を保存する。

- ``--data-domain patent``: ``load_cope_graph_bundle``（企業–特許）
- ``--data-domain arxiv``: ``load_author_paper_graph_bundle``（著者–論文）。
  **論文年**は ``--arxiv-year-min`` / ``--arxiv-year-max`` と ``--year-range`` を
  可視化スクリプトと揃えること。

例（特許）:
  python -m pnode_patent_runner.run_train_unified_vgae_checkpoint \\
    --data-domain patent --data notebooks/work/dataset/topic_info3.csv \\
    --year-range 2010 2020 --min-patents 2 --epochs 30 \\
    --cope-link-score distance --save pnode_patent_runner/outputs/cope_landscape/unified_vgae.pt

例（著者–論文）:
  埋め込み列に ``...`` が入った省略 CSV は使えません。``*_full.csv`` など全次元版を指定してください。
  python -m pnode_patent_runner.run_train_unified_vgae_checkpoint \\
    --data-domain arxiv \\
    --data data/processed/arxiv_cs_embedded_2020-2026_full.csv \\
    --year-range 2020 2026 --arxiv-year-min 2020 --arxiv-year-max 2026 \\
    --min-patents 5 --cope-link-score cosine --epochs 20 \\
    --save pnode_patent_runner/outputs/arxiv_landscape/unified_vgae_arxiv.pt
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pnode_patent_runner.cope_experiment import (
    load_author_paper_graph_bundle,
    load_author_topic_graph_bundle,
    load_cope_graph_bundle,
)
from pnode_patent_runner.unified_training import (
    README_DEFAULT_BETA,
    README_DEFAULT_FUTURE_LINK_WEIGHT,
    README_DEFAULT_LATENT_PRED_WEIGHT,
    README_DEFAULT_NUM_NEG_FUTURE,
    README_DEFAULT_NUM_NEG_RECON,
    README_DEFAULT_POS_WEIGHT,
    README_DEFAULT_POTENTIAL_WEIGHT,
    README_DEFAULT_TRAJECTORY_WEIGHT,
    evaluate_val_auc,
    train_model_improved,
)
from pnode_patent_runner.unified_vgae import METHOD_SHORT_NAME, UnifiedVGAE


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
    default_csv_patent = repo / "notebooks/work/dataset/topic_info3.csv"
    default_save_patent = repo / "pnode_patent_runner/outputs/cope_landscape/unified_vgae.pt"
    default_save_arxiv = repo / "pnode_patent_runner/outputs/arxiv_landscape/unified_vgae_arxiv.pt"
    default_save_topic = repo / "pnode_patent_runner/outputs/arxiv_landscape/unified_vgae_arxiv_topic.pt"

    p = argparse.ArgumentParser(
        description=f"{METHOD_SHORT_NAME}（UnifiedVGAE）— 現在のデータで学習し .pt を保存",
    )
    p.add_argument(
        "--data-domain",
        type=str,
        choices=("patent", "arxiv", "author_topic"),
        default="patent",
        help="patent=企業–特許 / arxiv=著者–論文 / author_topic=著者–トピック",
    )
    p.add_argument(
        "--topic-column",
        type=str,
        default="topic",
        help="author_topic のトピック列名",
    )
    p.add_argument(
        "--data",
        type=str,
        default="",
        help="CSV パス。省略時はドメイン別の既定（arxiv は dataset / data/processed を探索）",
    )
    p.add_argument(
        "--year-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
    )
    p.add_argument("--years", type=str, default="")
    p.add_argument("--all-years", action="store_true")
    p.add_argument(
        "--min-patents",
        type=int,
        default=None,
        help="patent: 企業あたり最小特許数（既定2）/ arxiv・author_topic: 著者あたり最小行数（既定5）",
    )
    p.add_argument(
        "--arxiv-year-min",
        type=int,
        default=2020,
        help="arxiv の前処理年下限（--arxiv-no-year-filter で無効）",
    )
    p.add_argument(
        "--arxiv-year-max",
        type=int,
        default=2026,
        help="arxiv の前処理年上限",
    )
    p.add_argument(
        "--arxiv-no-year-filter",
        action="store_true",
        help="arxiv の年範囲フィルタを無効化",
    )
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument(
        "--save",
        type=str,
        default="",
        help="出力 .pt（ドメイン別の既定パスあり）",
    )
    p.add_argument(
        "--cope-link-score",
        type=str,
        choices=("distance", "cosine"),
        default="distance",
    )
    p.add_argument("--cosine-logit-scale", type=float, default=5.0)
    p.add_argument("--w-pot-init", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=README_DEFAULT_BETA)
    p.add_argument("--pos-weight", type=float, default=README_DEFAULT_POS_WEIGHT)
    p.add_argument(
        "--potential-weight", type=float, default=README_DEFAULT_POTENTIAL_WEIGHT
    )
    p.add_argument(
        "--trajectory-weight", type=float, default=README_DEFAULT_TRAJECTORY_WEIGHT
    )
    p.add_argument(
        "--latent-pred-weight", type=float, default=README_DEFAULT_LATENT_PRED_WEIGHT
    )
    p.add_argument(
        "--future-link-weight", type=float, default=README_DEFAULT_FUTURE_LINK_WEIGHT
    )
    p.add_argument("--num-neg-recon", type=int, default=README_DEFAULT_NUM_NEG_RECON)
    p.add_argument("--num-neg-future", type=int, default=README_DEFAULT_NUM_NEG_FUTURE)
    p.add_argument(
        "--cope-density-calibrated",
        action="store_true",
        help="案1: μ の EMA 対角ガウス log p で Φ を校準",
    )
    p.add_argument("--cope-density-log-weight", type=float, default=1.0)
    p.add_argument("--cope-density-ema-momentum", type=float, default=0.05)
    args = p.parse_args()

    if args.latent_dim != 2:
        raise SystemExit("可視化用に latent_dim=2 を推奨します（--latent-dim 2）。")

    data_str = (args.data or "").strip()
    if data_str:
        data_path = Path(data_str)
    elif args.data_domain == "patent":
        data_path = default_csv_patent
    else:
        data_path = _default_arxiv_csv(repo)
    if not data_path.is_file():
        raise SystemExit(f"データが見つかりません: {data_path}")

    min_left = (
        args.min_patents
        if args.min_patents is not None
        else (5 if args.data_domain in ("arxiv", "author_topic") else 2)
    )
    yr = tuple(args.year_range) if args.year_range is not None else None
    ymin: Optional[int] = None if args.arxiv_no_year_filter else args.arxiv_year_min
    ymax: Optional[int] = None if args.arxiv_no_year_filter else args.arxiv_year_max
    try:
        if args.data_domain == "patent":
            bundle = load_cope_graph_bundle(
                data_path,
                min_patents=min_left,
                year_range=yr,
                years_csv=args.years,
                all_years=args.all_years,
            )
        elif args.data_domain == "arxiv":
            bundle = load_author_paper_graph_bundle(
                data_path,
                min_papers=min_left,
                arxiv_year_min=ymin,
                arxiv_year_max=ymax,
                year_range=yr,
                years_csv=args.years,
                all_years=args.all_years,
            )
        else:
            bundle = load_author_topic_graph_bundle(
                data_path,
                min_papers=min_left,
                topic_column=args.topic_column,
                arxiv_year_min=ymin,
                arxiv_year_max=ymax,
                year_range=yr,
                years_csv=args.years,
                all_years=args.all_years,
            )
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"data_domain={args.data_domain}, years={sorted(bundle.graphs.keys())}, "
        f"num_corps={bundle.num_corps}, total_n={bundle.total_n}, in_dim={bundle.in_dim}"
    )
    if args.data_domain in ("arxiv", "author_topic"):
        print(f"arxiv 前処理年: min={ymin}, max={ymax}（可視化時も同じに）")

    model = UnifiedVGAE(
        num_nodes=bundle.total_n,
        num_corps=bundle.num_corps,
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

    _, _, best_auc, hist = train_model_improved(
        model,
        bundle.graphs,
        bundle.num_corps,
        bundle.hist_edges,
        num_epochs=args.epochs,
        potential_weight=args.potential_weight,
        trajectory_weight=args.trajectory_weight,
        lr=args.lr,
        latent_pred_weight=args.latent_pred_weight,
        future_link_weight=args.future_link_weight,
        num_neg_recon=args.num_neg_recon,
        num_neg_future=args.num_neg_future,
        beta=args.beta,
        pos_weight=args.pos_weight,
    )
    final_auc = evaluate_val_auc(model, bundle.graphs, bundle.num_corps, device)
    print(f"best_val_auc (途中最大): {best_auc:.4f}")
    print(f"final_val_auc: {final_auc:.4f}")
    if hist["val_auc"]:
        print(f"val_auc per epoch: {[round(x, 4) for x in hist['val_auc']]}")

    save_str = (args.save or "").strip()
    if save_str:
        out = Path(save_str)
    else:
        if args.data_domain == "patent":
            out = Path(default_save_patent)
        elif args.data_domain == "arxiv":
            out = Path(default_save_arxiv)
        else:
            out = Path(default_save_topic)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out))
    print(f"Wrote checkpoint: {out}")
    ys = sorted(bundle.graphs.keys())
    yr_hint = f"{ys[0]} {ys[-1]}" if ys else ""
    if args.data_domain == "patent":
        print(
            f"  可視化例: python -m pnode_patent_runner.run_interactive_landscape_cope_vector_field \\\n"
            f"    --data {data_path} --year-range {yr_hint} --load-checkpoint {out} \\\n"
            f"    --cope-link-score {args.cope_link_score} --output ..."
        )
    else:
        arxiv_year_flags = (
            "    --arxiv-no-year-filter \\\n"
            if ymin is None
            else f"    --arxiv-year-min {ymin} --arxiv-year-max {ymax} \\\n"
        )
        gm = "topic" if args.data_domain == "author_topic" else "paper"
        tc = (
            f"    --topic-column {args.topic_column} \\\n"
            if args.data_domain == "author_topic"
            else ""
        )
        print(
            f"  可視化例: python -m pnode_patent_runner.run_interactive_landscape_arxiv_vector_field \\\n"
            f"    --graph-mode {gm} \\\n"
            f"{tc}"
            f"    --data {data_path} --year-range {yr_hint} \\\n"
            f"{arxiv_year_flags}"
            f"    --min-patents {min_left} --load-checkpoint {out} \\\n"
            f"    --cope-link-score {args.cope_link_score} --output ..."
        )


if __name__ == "__main__":
    main()
