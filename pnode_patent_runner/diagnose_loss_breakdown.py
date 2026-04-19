#!/usr/bin/env python3
"""
ベンチマーク JSON の last_epoch_train_breakdown を集計する、または短い学習で内訳を取得する。

リポジトリ kumagai ルートで:

  python -m pnode_patent_runner.diagnose_loss_breakdown \\
    --json-glob 'pnode_patent_runner/outputs/cope_benchmark/cope_pnode_td_cross/arxiv/p*.json'

  # 内訳が無い旧 JSON の場合は、再学習スモーク（数 epoch）
  python -m pnode_patent_runner.diagnose_loss_breakdown --smoke-train \\
    --data-domain arxiv --holdout-test-year 2026 --epochs 3 --methods cope,cope \\
    --smoke-td-compare
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parents[1]
if str(_ROOT := _REPO) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _fractions(b: Dict[str, float]) -> Dict[str, float]:
    t = float(b.get("total", 0.0)) or 1e-12
    keys = ("recon", "kl", "latent_pred", "future_link", "potential", "trajectory")
    return {k: float(b.get(k, 0.0)) / t for k in keys}


def _print_breakdown_table(title: str, b: Dict[str, float]) -> None:
    fr = _fractions(b)
    print(f"\n=== {title} ===")
    print(f"  total (last epoch avg): {b.get('total', 0):.6f}")
    for k in ("recon", "kl", "latent_pred", "future_link", "potential", "trajectory"):
        print(f"  {k:14s}  raw={b.get(k, 0):10.6f}  frac_of_total={fr.get(k, 0):7.4%}")


def summarize_json_files(paths: List[Path]) -> None:
    for p in paths:
        if not p.is_file():
            print(f"skip (missing): {p}", file=sys.stderr)
            continue
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        tag = raw.get("time_dependent_potential", "?")
        dom = raw.get("data_domain", "?")
        print(f"\n## file={p.name}  domain={dom}  TD={tag}")
        for row in raw.get("results", []):
            key = row.get("key", "?")
            leb = row.get("last_epoch_train_breakdown")
            if not leb:
                print(f"  [{key}] last_epoch_train_breakdown なし（旧形式の JSON の可能性。run_benchmark_comparison を再実行）")
                continue
            _print_breakdown_table(f"{p.name} :: {key}", leb)


def _load_bundle_and_holdout(args: argparse.Namespace, repo: Path):
    from pnode_patent_runner.cope_experiment import (
        load_author_paper_graph_bundle,
        load_author_topic_graph_bundle,
        load_cope_graph_bundle,
        split_bundle_holdout_test_year,
    )

    data_str = (args.data or "").strip()
    data_path = Path(data_str) if data_str else None
    if args.data_domain == "patent":
        from pnode_patent_runner.run_benchmark_comparison import _default_csv_for_domain

        dp = data_path or _default_csv_for_domain(repo, "patent")
        yr = (args.year_range[0], args.year_range[1]) if args.year_range else None
        bundle = load_cope_graph_bundle(
            str(dp),
            min_patents=args.min_patents,
            year_range=yr,
        )
    else:
        from pnode_patent_runner.run_benchmark_comparison import _default_csv_for_domain

        dp = data_path or _default_csv_for_domain(repo, args.data_domain)
        ymin = None if args.arxiv_no_year_filter else args.arxiv_year_min
        ymax = None if args.arxiv_no_year_filter else args.arxiv_year_max
        yr = tuple(args.year_range) if args.year_range else None
        if args.data_domain == "arxiv":
            bundle = load_author_paper_graph_bundle(
                str(dp),
                min_papers=args.min_patents,
                arxiv_year_min=ymin,
                arxiv_year_max=ymax,
                year_range=yr,
            )
        else:
            bundle = load_author_topic_graph_bundle(
                str(dp),
                min_papers=args.min_patents,
                topic_column=args.topic_column,
                arxiv_year_min=ymin,
                arxiv_year_max=ymax,
                year_range=yr,
            )

    holdout = None
    if args.holdout_test_year is not None:
        holdout = split_bundle_holdout_test_year(bundle, int(args.holdout_test_year))
    g_tr = holdout.graphs_train if holdout is not None else bundle.graphs
    h_tr = holdout.hist_edges_train if holdout is not None else bundle.hist_edges
    return bundle, g_tr, h_tr, holdout


def smoke_train_compare(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    from pnode_patent_runner.cope_experiment import ModelBuildKw, build_baseline_model
    from pnode_patent_runner.unified_training import train_model_improved
    from pnode_patent_runner.unified_training_td import train_model_td

    repo = _REPO
    bundle, g_tr, h_tr, holdout = _load_bundle_and_holdout(args, repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _train_years = sorted(g_tr.keys())
    _y_min, _y_max = int(_train_years[0]), int(_train_years[-1])
    if args.time_dependent_potential and holdout is not None:
        _fy = sorted(holdout.graphs_full.keys())
        _y_min, _y_max = int(_fy[0]), int(_fy[-1])

    def run_cope(td: bool) -> Tuple[str, Dict[str, Any]]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        mb = ModelBuildKw(
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            link_score_mode=args.cope_link_score,
            time_dependent_potential=td,
            year_min=_y_min,
            year_max=_y_max,
        )
        model = build_baseline_model("cope", device, bundle, mb)
        if td:
            _, _, _, hist = train_model_td(
                model,
                g_tr,
                bundle.num_corps,
                h_tr,
                num_epochs=args.epochs,
            )
        else:
            _, _, _, hist = train_model_improved(
                model,
                g_tr,
                bundle.num_corps,
                h_tr,
                num_epochs=args.epochs,
            )
        leb = hist.get("last_epoch_breakdown") or {}
        return ("CoPE TD" if td else "CoPE non-TD", leb)

    if args.smoke_td_compare:
        for title, leb in (run_cope(False), run_cope(True)):
            if leb:
                _print_breakdown_table(f"smoke {title}", leb)
            else:
                print(f"{title}: breakdown 未取得")
    else:
        td = bool(args.time_dependent_potential)
        title = "CoPE TD" if td else "CoPE non-TD"
        _, leb = run_cope(td)
        if leb:
            _print_breakdown_table(f"smoke {title}", leb)


def main() -> None:
    p = argparse.ArgumentParser(description="損失内訳の集計・スモーク学習")
    p.add_argument(
        "--json-glob",
        type=str,
        default="",
        help="ベンチマーク JSON への glob（相対は kumagai ルート基準）",
    )
    p.add_argument("--smoke-train", action="store_true", help="短い学習で内訳を表示")
    p.add_argument(
        "--smoke-td-compare",
        action="store_true",
        help="smoke-train 時に CoPE の TD 有無を両方回す",
    )
    p.add_argument("--data-domain", choices=("patent", "arxiv", "author_topic"), default="arxiv")
    p.add_argument("--data", type=str, default="")
    p.add_argument("--min-patents", type=int, default=5)
    p.add_argument("--year-range", nargs=2, type=int, default=None)
    p.add_argument("--arxiv-year-min", type=int, default=2020)
    p.add_argument("--arxiv-year-max", type=int, default=2026)
    p.add_argument("--arxiv-no-year-filter", action="store_true")
    p.add_argument("--topic-column", type=str, default="topic")
    p.add_argument("--holdout-test-year", type=int, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--cope-link-score", choices=("distance", "cosine"), default="distance")
    p.add_argument("--time-dependent-potential", action="store_true")
    p.add_argument("--methods", type=str, default="cope", help="未使用（互換用）")
    args = p.parse_args()

    if args.smoke_train:
        smoke_train_compare(args)
        return

    if not args.json_glob.strip():
        raise SystemExit("--json-glob か --smoke-train のいずれかを指定してください。")

    repo = _REPO
    pattern = args.json_glob.strip()
    paths = sorted({Path(p) for p in glob.glob(str(repo / pattern))})
    if not paths:
        paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        raise SystemExit(f"一致する JSON がありません: {pattern}")
    summarize_json_files(paths)


if __name__ == "__main__":
    main()
