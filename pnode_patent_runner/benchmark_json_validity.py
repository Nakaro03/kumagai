#!/usr/bin/env python3
"""
run_benchmark_comparison の JSON を機械的に検査する。

- 必須キー: seed, data_domain, results
- 各 results[]: key, final_val_auc, final_val_ap, final_val_ece
- results[*].final_metrics_by_horizon_gap[K] = {auc, ap, ece}（任意ブロック）

例:
  python -m pnode_patent_runner.benchmark_json_validity \\
    pnode_patent_runner/outputs/pnode_BD_vgae_compare/benchmark_pnode_BD_vs_baselines_seed42.json \\
    --output pnode_patent_runner/outputs/validity/json_check.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REQUIRED_ROOT = ("seed", "data_domain", "results")
# 厳格必須は key + 主1指標（旧 JSON は ap/ece 欠損のことがある）
RESULT_REQUIRED = ("key", "final_val_auc")
RESULT_OPTIONAL_WARN = ("final_val_ap", "final_val_ece")


def _in_unit_interval(x: float) -> bool:
    if math.isnan(x) or math.isinf(x):
        return False
    return 0.0 <= x <= 1.0


def _valid_ece(x: float) -> bool:
    if math.isnan(x) or math.isinf(x):
        return False
    return 0.0 <= x


def _validate_horizon_block(
    block: Any, path_label: str, issues: List[str]
) -> None:
    if not isinstance(block, dict):
        issues.append(f"{path_label}: horizon ブロックが dict ではない")
        return
    for gk, cell in sorted(
        block.items(),
        key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0,
    ):
        if not isinstance(cell, dict):
            issues.append(f"{path_label}[{gk}] が dict ではない")
            continue
        for fld in ("auc", "ap"):
            v = cell.get(fld)
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                if not _in_unit_interval(float(v)):
                    issues.append(
                        f"{path_label}[{gk}].{fld}={v} は [0,1] 外"
                    )
        ve = cell.get("ece")
        if isinstance(ve, (int, float)) and not math.isnan(float(ve)):
            if not _valid_ece(float(ve)):
                issues.append(f"{path_label}[{gk}].ece={ve} 異常")


def validate_one(
    path: Path,
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    issues: List[str] = []
    notes: List[str] = []  # 旧形式など（all_ok には影響しない）
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    for k in REQUIRED_ROOT:
        if k not in raw:
            issues.append(f"欠損: トップレベル '{k}'")
    if "data" not in raw:
        # 警告は issues から外し stderr のみ（strict で落とさない）
        pass
    results = raw.get("results")
    if not isinstance(results, list) or not results:
        issues.append("results が空、または list でない")
        return False, issues, notes, raw
    for i, row in enumerate(results):
        if not isinstance(row, dict):
            issues.append(f"results[{i}] が dict ではない")
            continue
        pl = f"results[{i}]"
        for k in RESULT_REQUIRED:
            if k not in row:
                issues.append(f"{pl}.{k} 欠損 (key={row.get('key')!r})")
        for k in RESULT_OPTIONAL_WARN:
            if k not in row:
                notes.append(
                    f"{pl}.{k} 欠損 (旧形式の可 key={row.get('key')!r})"
                )
        key = str(row.get("key", ""))
        for m in ("final_val_auc", "final_val_ap"):
            v = row.get(m)
            if isinstance(v, (int, float)) and not math.isnan(v):
                if not _in_unit_interval(float(v)):
                    issues.append(f"{pl} key={key} {m}={v} は [0,1] 外")
        ve = row.get("final_val_ece")
        if isinstance(ve, (int, float)) and not math.isnan(float(ve)):
            if not _valid_ece(float(ve)):
                issues.append(f"{pl} key={key} final_val_ece={ve} 異常")
        for split in ("final_metrics_by_horizon_gap", "train_split_metrics_by_horizon_gap"):
            bm = row.get(split)
            if bm is None:
                continue
            _validate_horizon_block(bm, f"{pl}.{split}", issues)
    return len(issues) == 0, issues, notes, raw


def _baseline_plateau(
    rows: List[Dict[str, Any]],
    keys: Tuple[str, str, str] = ("static", "rnn", "neural_ode"),
) -> Optional[float]:
    by_k = {str(r.get("key")): r for r in rows}
    aucs: List[float] = []
    for k in keys:
        r = by_k.get(k)
        if not r:
            return None
        v = r.get("final_val_auc")
        if not isinstance(v, (int, float)) or math.isnan(float(v)):
            return None
        aucs.append(float(v))
    if max(aucs) - min(aucs) < 1e-6:
        return sum(aucs) / 3.0
    return None


def validate_paths(
    paths: List[Path],
) -> Tuple[
    bool, List[Tuple[Path, bool, int, List[str], List[str], Optional[float]]]
]:
    """(path, ok, n_results, issues, notes, baseline_plateau_auc)."""
    rows_out: List[
        Tuple[Path, bool, int, List[str], List[str], Optional[float]]
    ] = []
    all_ok = True
    for path in paths:
        ok, issues, notes, raw = validate_one(path)
        if not ok:
            all_ok = False
        n_res = len(raw.get("results") or [])
        plateau = _baseline_plateau(list(raw.get("results") or []))
        rows_out.append((path, ok, n_res, issues, notes, plateau))
    return all_ok, rows_out


def _markdown_table(
    rows_out: List[
        Tuple[Path, bool, int, List[str], List[str], Optional[float]]
    ],
) -> str:
    lines = [
        "| path | n_results | valid | static=rnn=neural_ode? |",
        "|------|-----------|-------|------------------------|",
    ]
    for path, ok, n_res, issues, _notes, plateau in rows_out:
        st = "OK" if ok else "NG"
        pl = f"≈ {plateau:.4f}" if plateau is not None else "—"
        lines.append(f"| `{path}` | {n_res} | {st} | {pl} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="run_benchmark の JSON を妥当性検査"
    )
    ap.add_argument(
        "json_paths",
        nargs="*",
        type=Path,
        help="benchmark_*.json（空なら --glob のみ）",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="",
        help="例: 'pnode_patent_runner/outputs/**/*.json'",
    )
    ap.add_argument(
        "--no-fail",
        action="store_true",
        help="検出 issue があっても exit 0（手元確認用）",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Markdown レポートを保存",
    )
    args = ap.parse_args()

    from glob import glob

    paths: List[Path] = []
    if (args.glob or "").strip():
        paths.extend(Path(p) for p in sorted(glob(args.glob.strip())))
    paths.extend(args.json_paths)
    paths = sorted(set(paths), key=lambda x: str(x))
    if not paths:
        raise SystemExit("JSON を渡すか --glob を指定してください。")
    for q in paths:
        if not q.is_file():
            raise SystemExit(f"ファイルが見つかりません: {q}")

    all_ok, rows_out = validate_paths(paths)

    text_blocks: List[str] = []
    text_blocks.append("# benchmark_json_validity\n")
    text_blocks.append(f"generated: {datetime.now(timezone.utc).isoformat()}\n")
    text_blocks.append("\n")
    text_blocks.append(_markdown_table(rows_out))

    for path, ok, n_res, issues, notes, plateau in rows_out:
        if issues:
            text_blocks.append(f"\n## Issues: `{path}`\n\n")
            for msg in issues:
                text_blocks.append(f"- {msg}\n")
        if notes:
            text_blocks.append(f"\n### Notes: `{path}`\n\n")
            for msg in notes:
                text_blocks.append(f"- {msg}\n")

    report = "".join(text_blocks)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote: {args.output}", file=sys.stderr)

    print(report)
    if not all_ok and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
