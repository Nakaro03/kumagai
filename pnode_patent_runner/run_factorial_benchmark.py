#!/usr/bin/env python3
"""
3 ドメイン × 3 HPO モード × 2（時間依存 Φ あり/なし）の因子実験を連続実行する（シェル版の Python 移植）。

リポジトリ kumagai のルートで実行すること:
  python -m pnode_patent_runner.run_factorial_benchmark --help

HPO モード:
  fixed      … Optuna JSON なし（ハイパラは CLI 既定）
  cope_only  … --optuna-best-json（CoPE のみ最適化結果を適用）
  all        … --optuna-best-json-map（各手法に対称 HPO）

条件を一つずつ確かめる例（--dry-run でコマンドだけ確認）:

  # 1) グラフ: patent / 固定 HPO / 非TD / ホールドアウトなし
  python -m pnode_patent_runner.run_factorial_benchmark --domains patent --hpo-mode fixed --dry-run

  # 2) 同上 + ホールドアウト最終年 2020（--year-range に 2020 を含めること）
  python -m pnode_patent_runner.run_factorial_benchmark --domains patent --hpo-mode fixed \\
    --holdout-test-year 2020 --patent-year-end 2020 --dry-run

  # 3) 対称 HPO + 非TD
  python -m pnode_patent_runner.run_factorial_benchmark --domains patent --hpo-mode all \\
    --optuna-map-json pnode_patent_runner/outputs/optuna/symmetric_patent_20260411_234808/optuna_paths_by_method.json --dry-run

  # 4) 時間依存 Φ（TD=1）
  python -m pnode_patent_runner.run_factorial_benchmark --domains patent --hpo-mode fixed \\
    --time-dependent-potential --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]


def build_hpo_args(
    hpo_mode: str,
    *,
    optuna_cope_json: Path,
    optuna_map_json: Optional[Path],
) -> Tuple[str, List[str]]:
    """戻り値: (スラグ, run_benchmark_comparison へ渡す追加 argv)。"""
    if hpo_mode == "fixed":
        return "fixed", []
    if hpo_mode == "cope_only":
        return "cope_only", ["--optuna-best-json", str(optuna_cope_json)]
    if hpo_mode == "all":
        if optuna_map_json is None or not optuna_map_json.is_file():
            raise SystemExit(
                "HPO_MODE=all には存在する --optuna-map-json を指定してください。"
            )
        return "all", ["--optuna-best-json-map", str(optuna_map_json)]
    raise SystemExit(f"不明な HPO_MODE: {hpo_mode!r}（fixed|cope_only|all）")


def domain_argv(
    domain: str,
    *,
    patent_csv: Path,
    patent_year_start: int,
    patent_year_end: int,
    arxiv_csv: Optional[Path],
    min_papers: int,
    arxiv_year_min: int,
    arxiv_year_max: int,
    arxiv_year_start: Optional[int],
    arxiv_year_end: Optional[int],
) -> List[str]:
    """--data-domain に続くデータ関連 argv。"""
    if domain == "patent":
        return [
            "--data",
            str(patent_csv),
            "--year-range",
            str(patent_year_start),
            str(patent_year_end),
        ]
    if domain in ("arxiv", "author_topic"):
        out: List[str] = []
        if arxiv_csv is not None and arxiv_csv.is_file():
            out.extend(["--data", str(arxiv_csv)])
        out.extend(
            [
                "--min-patents",
                str(min_papers),
                "--arxiv-year-min",
                str(arxiv_year_min),
                "--arxiv-year-max",
                str(arxiv_year_max),
            ]
        )
        if arxiv_year_start is not None and arxiv_year_end is not None:
            out.extend(
                [
                    "--year-range",
                    str(arxiv_year_start),
                    str(arxiv_year_end),
                ]
            )
        return out
    raise SystemExit(f"不明なドメイン: {domain!r}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="run_benchmark_comparison を因子設計で複数回呼び出す。",
    )
    p.add_argument(
        "--domains",
        nargs="+",
        default=["patent"],
        choices=("patent", "arxiv", "author_topic"),
        help="評価する data-domain（複数可）",
    )
    p.add_argument(
        "--hpo-mode",
        type=str,
        choices=("fixed", "cope_only", "all"),
        default="fixed",
        help="fixed=固定ハイパラ / cope_only=CoPE のみ Optuna / all=各手法対称 HPO",
    )
    p.add_argument(
        "--time-dependent-potential",
        action="store_true",
        help="各実行に --time-dependent-potential を付ける（TD=1 相当）",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--cope-link-score",
        choices=("distance", "cosine"),
        default="distance",
    )
    p.add_argument("--min-patents", type=int, default=5)
    p.add_argument(
        "--out-root",
        type=Path,
        default=_REPO_ROOT / "pnode_patent_runner/outputs/cope_benchmark/factorial",
        help="出力 JSON のディレクトリ",
    )
    p.add_argument(
        "--optuna-cope-json",
        type=Path,
        default=_REPO_ROOT
        / "pnode_patent_runner/outputs/optuna/best_params_unified_vgae_cope.json",
        help="HPO_MODE=cope_only 用",
    )
    p.add_argument(
        "--optuna-map-json",
        type=Path,
        default=None,
        help="HPO_MODE=all 用（optuna_paths_by_method.json 等）",
    )
    p.add_argument(
        "--patent-csv",
        type=Path,
        default=_REPO_ROOT / "notebooks/work/dataset/topic_info3.csv",
    )
    p.add_argument("--patent-year-start", type=int, default=2010)
    p.add_argument("--patent-year-end", type=int, default=2020)
    p.add_argument(
        "--arxiv-csv",
        type=Path,
        default=_REPO_ROOT / "data/processed/arxiv_cs_embedded_2020-2026_full.csv",
    )
    p.add_argument("--arxiv-year-min", type=int, default=2020)
    p.add_argument("--arxiv-year-max", type=int, default=2026)
    p.add_argument(
        "--arxiv-year-start",
        type=int,
        default=None,
        help="arxiv/author_topic の --year-range 下限（省略時は付けない）",
    )
    p.add_argument("--arxiv-year-end", type=int, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="コマンドを表示するだけ（実行しない）",
    )
    p.add_argument(
        "--holdout-test-year",
        type=int,
        default=None,
        metavar="Y",
        help=(
            "run_benchmark_comparison に渡す。Y 年のエッジは学習に含めず、"
            "final_val_* は (Y の直前年→Y) の指標。--year-range に Y を含めること。"
        ),
    )
    args = p.parse_args()

    repo = _REPO_ROOT
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    hpo_slug, hpo_extra = build_hpo_args(
        args.hpo_mode,
        optuna_cope_json=args.optuna_cope_json,
        optuna_map_json=args.optuna_map_json,
    )

    td_flag = 1 if args.time_dependent_potential else 0
    arxiv_csv = args.arxiv_csv if args.arxiv_csv.is_file() else None

    for domain in args.domains:
        data_args = domain_argv(
            domain,
            patent_csv=args.patent_csv,
            patent_year_start=args.patent_year_start,
            patent_year_end=args.patent_year_end,
            arxiv_csv=arxiv_csv,
            min_papers=args.min_patents,
            arxiv_year_min=args.arxiv_year_min,
            arxiv_year_max=args.arxiv_year_max,
            arxiv_year_start=args.arxiv_year_start,
            arxiv_year_end=args.arxiv_year_end,
        )

        ho_tag = f"_holdout{args.holdout_test_year}" if args.holdout_test_year else ""
        out_json = (
            out_root
            / f"{domain}_hpo{hpo_slug}_td{td_flag}{ho_tag}_seed{args.seed}.json"
        )

        cmd: List[str] = [
            sys.executable,
            "-m",
            "pnode_patent_runner.run_benchmark_comparison",
            "--data-domain",
            domain,
            *data_args,
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--methods",
            "all",
            "--cope-link-score",
            args.cope_link_score,
            "--output-json",
            str(out_json),
        ]
        if args.time_dependent_potential:
            cmd.append("--time-dependent-potential")
        if args.holdout_test_year is not None:
            cmd.extend(["--holdout-test-year", str(int(args.holdout_test_year))])
        cmd.extend(hpo_extra)

        ho_msg = (
            f" holdout={args.holdout_test_year}"
            if args.holdout_test_year is not None
            else ""
        )
        print(
            f"=== domain={domain} HPO={hpo_slug} TD={td_flag}{ho_msg} -> {out_json} ==="
        )
        if args.dry_run:
            print(" ".join(cmd))
            continue

        r = subprocess.run(cmd, cwd=str(repo))
        if r.returncode != 0:
            raise SystemExit(r.returncode)

    if not args.dry_run:
        print(f"Done. Outputs under {out_root}")


if __name__ == "__main__":
    main()
