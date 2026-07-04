#!/usr/bin/env python3
"""
数値的 ODE 妥当性 + benchmark JSON 検査 + **各ファイルの主指標** + **複数シードの集約と有意差**
（片側 Wilcoxon: pnode > baseline）を1本の .md にまとめる。

例:
  python -m pnode_patent_runner.run_validity_report \\
    -o pnode_patent_runner/outputs/validity/validity_report.md

  python -m pnode_patent_runner.run_validity_report \\
    --benchmark-glob "pnode_patent_runner/outputs/pnode_BD_vgae_compare/benchmark_*.json" \\
    -o pnode_patent_runner/outputs/validity/validity_report.md

  # 有意差は **同一条件・異 seed の JSON が3本以上** あるときに算出（`aggregate_benchmark_seeds` 準拠）
  python -m pnode_patent_runner.run_validity_report \\
    --aggregate-seeds-glob "pnode_patent_runner/outputs/cope_benchmark/benchmark_author_topic_seed*.json" \\
    --paired-pnode-vs neural_ode \\
    -o pnode_patent_runner/outputs/validity/validity_report.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pnode_patent_runner.aggregate_benchmark_seeds import (
    _collect,
    _se,
    _wilcoxon_paired,
)
from pnode_patent_runner.benchmark_json_validity import (
    _markdown_table,
    validate_paths,
)
from pnode_patent_runner.ode_numerical_check import run_checks


def _metrics_block_one_file(path: Path) -> str:
    """1 JSON の results を Markdown 表にする。"""
    with open(path, encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)
    seed = raw.get("seed", "?")
    dom = raw.get("data_domain", "?")
    rows = raw.get("results")
    if not isinstance(rows, list) or not rows:
        return f"### `{path}`\n\n_（results 空）_\n\n"
    lines: List[str] = [
        f"### `{path.name}`\n\n",
        f"- `seed`={seed} | `data_domain`={dom}\n\n",
        "| key | final_val_auc | final_val_ap | final_val_ece |\n",
        "|-----|---------------|--------------|---------------|\n",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        k = str(row.get("key", ""))
        a = row.get("final_val_auc", float("nan"))
        apv = row.get("final_val_ap", float("nan"))
        e = row.get("final_val_ece", float("nan"))
        def _fmt(v: object) -> str:
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                return f"{float(v):.6f}"
            return "nan"

        lines.append(f"| {k} | {_fmt(a)} | {_fmt(apv)} | {_fmt(e)} |\n")
    lines.append("\n")
    return "".join(lines)


def _aggregate_significance_md(
    paths: List[Path],
    metric: str,
    horizon_gap: Optional[int],
    horizon_field: str,
    horizon_split: str,
    paired_baseline: str,
) -> str:
    if len(paths) < 1:
        return "_（JSON パス無し）_\n\n"
    dom, by_key, metric_label = _collect(
        paths, metric, horizon_gap, horizon_field, horizon_split
    )
    n_files = len(paths)
    out: List[str] = [
        f"- 使用メトリクス: `{metric_label}`\n",
        f"- 同一条件 JSON 本数: **{n_files}**（domain≈`{dom or '?'}`）\n",
    ]
    if n_files == 1:
        out.append(
            "- **注**: シード1本のため **平均=その1本の値**；**有意差**には"
            " `aggregate-seeds-glob` で**同条件**の `*_seed*.json` を**3本以上**渡す。\n\n"
        )
    else:
        out.append("\n")
    out.append(
        "| method_key | n_valid | mean | std | SE |\n"
        "|------------|---------|------|-----|----|\n"
    )
    for key in sorted(by_key.keys()):
        vals = by_key[key]
        clean = [v for v in vals if not math.isnan(v)]
        m = len(clean)
        if m == 0:
            mean_v, std_v, se_v = float("nan"), float("nan"), float("nan")
        else:
            mean_v = sum(clean) / m
            if m == 1:
                std_v = 0.0
            else:
                var = sum((x - mean_v) ** 2 for x in clean) / (m - 1)
                std_v = math.sqrt(var)
            se_v = _se(std_v, m) if m > 1 else 0.0
        out.append(
            f"| {key} | {m} | {mean_v:.6f} | {std_v:.6f} | {se_v:.6f} |\n"
        )
    out.append("\n")
    if not paired_baseline:
        return "".join(out)

    if "pnode" not in by_key or paired_baseline not in by_key:
        out.append(
            f"> 有意差スキップ: `pnode` または `{paired_baseline}` の列がありません。\n\n"
        )
        return "".join(out)

    if len(paths) < 3:
        out.append(
            f"> **Wilcoxon 片側** (pnode > {paired_baseline}): "
            f"有効 n={len(paths)} — **3本以上**のシード別 JSON があるときに p 値を報告"
            f"（[docs/STATS_PREREGISTRATION.md](docs/STATS_PREREGISTRATION.md)）。\n\n"
        )
        return "".join(out)

    pv, werr = _wilcoxon_paired(
        by_key["pnode"], by_key[paired_baseline]
    )
    if pv is not None and not werr:
        a_mean = _mean_list(by_key["pnode"])
        b_mean = _mean_list(by_key[paired_baseline])
        diff = a_mean - b_mean if a_mean is not None and b_mean is not None else float("nan")
        out.append("### 有意差（探索的。事前登録の主解析と同じ定義に揃えること）\n\n")
        out.append(
            f"- **検定**: ペア片側 Wilcoxon（H1: 各シードで `pnode` > `{paired_baseline}` ）\n"
        )
        out.append(
            f"- **p 値 (one-sided)**: **{pv:.6g}**\n"
        )
        if not math.isnan(diff):
            out.append(
                f"- シード平均の差 (`pnode` - `{paired_baseline}`): **{diff:+.6f}**（`{metric_label}`）\n"
            )
        if pv < 0.05:
            out.append(
                f"- 解釈: `p` < 0.05 のため、`{metric_label}` 上で `pnode` 優位の仮説に**一定の**支持"
                f"（**多重比較**・探索なら 1 主指標 + FDR を明記）。\n\n"
            )
        else:
            out.append(
                "- 解釈: `p` >= 0.05 — 有意差**なし**とは限らないが、**従来の**有意**主張**は弱い；"
                "効果量・`n` を併記。\n\n"
            )
    else:
        out.append(f"> **Wilcoxon:** {werr}\n\n")
    return "".join(out)


def _mean_list(xs: List[float]) -> Optional[float]:
    cl = [x for x in xs if not math.isnan(x)]
    if not cl:
        return None
    return sum(cl) / len(cl)


def main() -> int:
    p = argparse.ArgumentParser(
        description="ODE 数値整合 + benchmark JSON 検査の統合レポート"
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="保存先 .md（既定: pnode_patent_runner/outputs/validity/validity_report_<UTC>.md）",
    )
    p.add_argument(
        "--benchmark-glob",
        type=str,
        default="",
        help="run_benchmark の JSON へ glob（省略時はスキップ）",
    )
    p.add_argument(
        "benchmark_jsons",
        nargs="*",
        type=Path,
        help="追加の benchmark JSON パス",
    )
    p.add_argument(
        "--no-fail",
        action="store_true",
        help="JSON 検査で issue があっても exit 0",
    )
    p.add_argument(
        "--ode-only",
        action="store_true",
        help="ODE 数値検査のみ（benchmark 省略）",
    )
    p.add_argument(
        "--no-bench-metrics",
        action="store_true",
        help="節3–4（各JSONの主指標・集約/有意差）を省略",
    )
    p.add_argument(
        "--aggregate-seeds-glob",
        type=str,
        default="",
        help="シード集約＆Wilcoxon 用（省略時は --benchmark で渡した各 JSON）。"
        "例: '.../cope_benchmark/benchmark_author_topic_seed*.json'",
    )
    p.add_argument(
        "--aggregate-metric",
        type=str,
        default="final_val_auc",
        help="集約の主キー（例: final_val_ap）",
    )
    p.add_argument(
        "--horizon-gap",
        type=int,
        default=None,
        metavar="K",
        help="指定時 --aggregate-metric を無視し、長期 K の指標に切替",
    )
    p.add_argument(
        "--horizon-field",
        type=str,
        default="auc",
        choices=("auc", "ap", "ece"),
    )
    p.add_argument(
        "--horizon-split",
        type=str,
        default="final",
        choices=("final", "train"),
    )
    p.add_argument(
        "--paired-pnode-vs",
        type=str,
        default="neural_ode",
        help="片側 Wilcoxon の帰納: pnode の指標 > この key（空でスキップ）",
    )
    args = p.parse_args()

    out = args.output
    if out is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = (
            Path(__file__).resolve().parent
            / "outputs"
            / "validity"
            / f"validity_report_{ts}.md"
        )

    lines: List[str] = []
    lines.append("# P-NODE 妥当性レポート\n\n")
    lines.append(f"生成時刻 (UTC): {datetime.now(timezone.utc).isoformat()}\n\n")

    lines.append("## 1. ODE / Potential 数値整合\n\n")
    try:
        ode = run_checks()
    except Exception as e:  # noqa: BLE001
        lines.append(f"**FAIL**: 例外 — `{e!r}`\n\n")
        code_ode = 1
    else:
        for s in ode.lines:
            lines.append(f"- {s}\n")
        code_ode = 0
        if not (ode.ok_grad_mlp and ode.ok_ode_func):
            lines.append("\n**結果: ODE 数値検査 FAIL**\n\n")
            code_ode = 1
        elif ode.max_rel_grad_mlp > 0.05:
            lines.append(
                f"\n**注意**: max_rel {ode.max_rel_grad_mlp} > 既定 tol 0.05\n\n"
            )
            code_ode = 1
        else:
            lines.append("\n**結果: ODE 数値検査 OK**\n\n")

    code_json = 0
    bench_paths: List[Path] = []
    if not args.ode_only:
        if (args.benchmark_glob or "").strip():
            bench_paths.extend(Path(x) for x in sorted(glob(args.benchmark_glob.strip())))
        bench_paths.extend(args.benchmark_jsons)
        bench_paths = sorted(set(bench_paths), key=str)
        missing = [p for p in bench_paths if not p.is_file()]
        for q in missing:
            lines.append(f"\n**警告**: ファイルが見つかりません: `{q}`\n\n")
        exist = [p for p in bench_paths if p.is_file()]

        lines.append("## 2. benchmark JSON 構造\n\n")
        if not exist:
            lines.append("_（`--benchmark-glob` または .json パス未指定のためスキップ）_\n\n")
        else:
            all_ok, rows_out = validate_paths(exist)
            lines.append(_markdown_table(rows_out))
            for path, ok, n_res, issues, notes, plateau in rows_out:
                if issues:
                    lines.append(f"\n### Issues: `{path}`\n\n")
                    for msg in issues:
                        lines.append(f"- {msg}\n")
                if notes:
                    lines.append(f"\n### Notes: `{path}`\n\n")
                    for msg in notes:
                        lines.append(f"- {msg}\n")
            if not all_ok:
                lines.append("\n**結果: いずれかの JSON に issue あり**\n\n")
                code_json = 1
            else:
                lines.append("\n**結果: 全 JSON OK**\n\n")

        if not args.no_bench_metrics and not args.ode_only and exist:
            lines.append("## 3. ベンチ主指標（各 JSON の最終行）\n\n")
            lines.append(
                "同一ファイル内 `results[]` の `final_val_auc` / `ap` / `ece` です（"
                "サブサンプリング手続に依存）。\n\n"
            )
            for bp in exist:
                lines.append(_metrics_block_one_file(Path(bp).resolve()))

            ag_paths: List[Path] = []
            if (args.aggregate_seeds_glob or "").strip():
                ag_paths = sorted(
                    {Path(p) for p in glob(args.aggregate_seeds_glob.strip())}
                )
            else:
                ag_paths = list(exist)
            ag_paths = [p for p in ag_paths if p.is_file()]

            lines.append("## 4. 複数シードの集約と有意差\n\n")
            lines.append(
                "事前登録の**主**解析に合わせること（[docs/STATS_PREREGISTRATION.md](docs/STATS_PREREGISTRATION.md)）。\n\n"
            )
            paired = (args.paired_pnode_vs or "").strip()
            lines.append(
                _aggregate_significance_md(
                    ag_paths,
                    args.aggregate_metric,
                    args.horizon_gap,
                    args.horizon_field,
                    args.horizon_split,
                    paired,
                )
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    report = "".join(lines)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote: {out}", file=sys.stderr)

    if args.no_fail:
        return code_ode
    return max(code_ode, code_json)


if __name__ == "__main__":
    sys.exit(main())
