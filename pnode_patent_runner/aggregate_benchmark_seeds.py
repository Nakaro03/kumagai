#!/usr/bin/env python3
"""
複数シードの `run_benchmark_comparison` JSON を読み、手法ごとの mean / std / SE を出力する。

査読用の事前登録・主表ドラフトは [docs/STATS_PREREGISTRATION.md](docs/STATS_PREREGISTRATION.md)。

例:
  python -m pnode_patent_runner.aggregate_benchmark_seeds \\
    pnode_patent_runner/outputs/cope_benchmark/benchmark_patent_seed{42,43,44}.json \\
    --markdown

  python -m pnode_patent_runner.aggregate_benchmark_seeds --glob \\
    "pnode_patent_runner/outputs/cope_benchmark/benchmark_patent_seed*.json"

  # 長期 future-link（インデックス差 K）の AUC: [docs/LONG_HORIZON_PREREGISTRATION.md]
  python -m pnode_patent_runner.aggregate_benchmark_seeds --glob "*.json" \\
    --horizon-gap 2 --horizon-field auc --horizon-split final --markdown \\
    --paired-pnode-vs neural_ode
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_rows(path: Path) -> Tuple[int, str, List[Dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    seed = int(raw.get("seed", -1))
    domain = str(raw.get("data_domain", ""))
    rows = raw.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"results が無い: {path}")
    return seed, domain, rows


def _horizon_metric(
    row: Dict[str, Any],
    gap: int,
    field: str,
    split: str,
) -> float:
    """`final_metrics_by_horizon_gap` / `train_split_metrics_by_horizon_gap` から 1 値。"""
    block_key = (
        "final_metrics_by_horizon_gap"
        if split == "final"
        else "train_split_metrics_by_horizon_gap"
    )
    block = row.get(block_key)
    if not isinstance(block, dict):
        return float("nan")
    hk = str(int(gap))
    cell = block.get(hk)
    if not isinstance(cell, dict):
        return float("nan")
    v = cell.get(field)
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return float(v)
    return float("nan")


def _collect(
    paths: List[Path],
    metric: str,
    horizon_gap: Optional[int],
    horizon_field: str,
    horizon_split: str,
) -> Tuple[str, Dict[str, List[float]], str]:
    """
    domain（先頭ファイル由来）、各手法 key について paths と同じ順のメトリクス列。
    1 ファイル 1 値（同一 JSON 内の results は key ごとに 1 行想定）。

    horizon_gap が非 None のときは metric トップレベルキーではなく
    `{final|train}_metrics_by_horizon_gap[str(gap)][horizon_field]` を読む。
    """
    domain = ""
    per_file: List[Dict[str, float]] = []
    metric_label = metric
    if horizon_gap is not None:
        split_key = (
            "final_metrics_by_horizon_gap"
            if horizon_split == "final"
            else "train_split_metrics_by_horizon_gap"
        )
        metric_label = (
            f'{split_key}[{int(horizon_gap)}]["{horizon_field}"]'
        )
    for p in paths:
        _seed, dom, rows = _load_rows(p)
        if not domain and dom:
            domain = dom
        m: Dict[str, float] = {}
        for row in rows:
            key = str(row.get("key", ""))
            if not key:
                continue
            if horizon_gap is not None:
                v = _horizon_metric(row, horizon_gap, horizon_field, horizon_split)
            else:
                raw = row.get(metric)
                if isinstance(raw, (int, float)) and not (
                    isinstance(raw, float) and math.isnan(raw)
                ):
                    v = float(raw)
                else:
                    v = float("nan")
            m[key] = v
        per_file.append(m)
    all_keys = sorted({k for mp in per_file for k in mp})
    by_key: Dict[str, List[float]] = {
        k: [mp.get(k, float("nan")) for mp in per_file] for k in all_keys
    }
    return domain, by_key, metric_label


def _se(std: float, n: int) -> float:
    if n <= 1:
        return float("nan")
    return float(std / math.sqrt(n))


def _wilcoxon_paired(
    a: List[float], b: List[float],
) -> Tuple[Optional[float], str]:
    """
    Returns (p_value, err_msg). err_msg is non-empty iff p_value is None.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None, "Wilcoxon をスキップ: `scipy` が未インストール（`pip install scipy`）。"
    aa = [x for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    bb = [y for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    n = len(aa)
    if n < 1:
        return None, "Wilcoxon をスキップ: 有効ペアが 0（NaN または長さ不一致）。"
    if n < 3:
        return (
            None,
            "Wilcoxon をスキップ: 有効ペア n="
            f"{n}（JSON 本数と同じ）。査読用の符号検定には **少なくとも 3 本**の"
            "シード別 `benchmark_*_seed*.json` を `--glob` または引数で渡してください。",
        )
    res = wilcoxon(aa, bb, alternative="greater", zero_method="wilcox")
    return float(res.pvalue), ""


def main() -> None:
    p = argparse.ArgumentParser(description="複数シード benchmark JSON の集約")
    p.add_argument(
        "json_paths",
        nargs="*",
        type=Path,
        help="benchmark_*_seed*.json（複数）",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="",
        help="例: 'pnode_patent_runner/outputs/cope_benchmark/benchmark_patent_seed*.json'",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="final_val_auc",
        help="集約する JSON キー（--horizon-gap 未指定時。例: final_val_ap）",
    )
    p.add_argument(
        "--horizon-gap",
        type=int,
        default=None,
        metavar="K",
        help="指定時は results[].final_metrics_by_horizon_gap 等から "
        "インデックス差 K の値を読む（--metric は無視）",
    )
    p.add_argument(
        "--horizon-field",
        type=str,
        default="auc",
        choices=("auc", "ap", "ece"),
        help="--horizon-gap 指定時の辞書キー（auc / ap / ece）",
    )
    p.add_argument(
        "--horizon-split",
        type=str,
        default="final",
        choices=("final", "train"),
        help="final_metrics_by_horizon_gap か train_split_metrics_by_horizon_gap か",
    )
    p.add_argument(
        "--paired-pnode-vs",
        type=str,
        default="",
        metavar="BASELINE_KEY",
        help="指定時、各 JSON 内で pnode と baseline の metric 差に対し片側 Wilcoxon（pnode が大きい）",
    )
    p.add_argument("--markdown", action="store_true", help="Markdown 表を stdout に出す")
    args = p.parse_args()

    paths: List[Path] = []
    if args.glob.strip():
        paths = sorted(Path(p) for p in glob(args.glob.strip()))
    paths.extend(args.json_paths)
    paths = sorted(set(paths), key=lambda x: str(x))
    if not paths:
        raise SystemExit("JSON パスが空です。ファイルを渡すか --glob を指定してください。")
    for q in paths:
        if not q.is_file():
            raise SystemExit(f"ファイルが見つかりません: {q}")

    n_paths = len(paths)
    paired_flag = bool((args.paired_pnode_vs or "").strip())
    if paired_flag and n_paths < 3 and not args.markdown:
        print(
            f"注意: JSON が {n_paths} 本のみです。`--paired-pnode-vs` の Wilcoxon には通常 3 本以上の"
            "シード別 JSON が必要です（[docs/STATS_PREREGISTRATION.md]）。",
            file=sys.stderr,
        )

    domain, by_key, metric_label = _collect(
        paths,
        args.metric,
        args.horizon_gap,
        args.horizon_field,
        args.horizon_split,
    )
    n_files = len(paths)

    rows_out: List[Tuple[str, int, float, float, float]] = []
    for key in sorted(by_key.keys()):
        vals = by_key[key]
        if len(vals) != n_files:
            print(
                f"警告: key={key} の件数 {len(vals)} != ファイル数 {n_files}",
                file=sys.stderr,
            )
        clean = [v for v in vals if not math.isnan(v)]
        m = len(clean)
        if m == 0:
            mean_v = float("nan")
            std_v = float("nan")
            se_v = float("nan")
        else:
            mean_v = sum(clean) / m
            if m == 1:
                std_v = 0.0
            else:
                var = sum((x - mean_v) ** 2 for x in clean) / (m - 1)
                std_v = math.sqrt(var)
            se_v = _se(std_v, m)
        rows_out.append((key, m, mean_v, std_v, se_v))

    if args.markdown:
        print(f"## Aggregated `{metric_label}` (n_json={n_files}, data_domain={domain or '?'})")
        print()
        print("| method_key | n_valid | mean | std | SE |")
        print("|------------|---------|------|-----|-----|")
        for key, m, mean_v, std_v, se_v in rows_out:
            print(
                f"| {key} | {m} | {mean_v:.6f} | {std_v:.6f} | {se_v:.6f} |"
            )
        print()

    paired = (args.paired_pnode_vs or "").strip()
    if paired:
        if "pnode" not in by_key or paired not in by_key:
            print(
                f"Wilcoxon スキップ: pnode または {paired} の系列が無い",
                file=sys.stderr,
            )
        else:
            pv, werr = _wilcoxon_paired(by_key["pnode"], by_key[paired])
            if pv is None:
                if args.markdown:
                    print()
                    print(f"> **Wilcoxon:** {werr} 手計算は [docs/STATS_PREREGISTRATION.md]。")
                else:
                    print(werr + " 手計算は [docs/STATS_PREREGISTRATION.md]。", file=sys.stderr)
            else:
                msg = f"Wilcoxon one-sided (pnode > {paired}) p-value ≈ {pv:.6g}"
                if args.markdown:
                    print(f"**{msg}**")
                else:
                    print(msg)

    if not args.markdown:
        print(f"domain={domain} n_files={n_files} metric={metric_label}")
        for key, m, mean_v, std_v, se_v in rows_out:
            print(f"  {key:14s} n_valid={m} mean={mean_v:.6f} std={std_v:.6f} SE={se_v:.6f}")


if __name__ == "__main__":
    main()
