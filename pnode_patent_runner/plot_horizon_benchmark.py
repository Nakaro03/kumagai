#!/usr/bin/env python3
"""
`run_benchmark_comparison` の JSON（`final_metrics_by_horizon_gap` 付き）から
ホライズン別 AUC（および任意で AP）の折れ線図を PNG で保存する。

例:
  python -m pnode_patent_runner.plot_horizon_benchmark \\
    pnode_patent_runner/outputs/cope_benchmark/benchmark_patent_seed42.json \\
    --output pnode_patent_runner/outputs/cope_benchmark/horizon_auc.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _collect_series(
    payload: Dict[str, Any], metric: str
) -> Tuple[List[int], Dict[str, List[float]]]:
    gaps_set: set[int] = set()
    for row in payload.get("results", []):
        hmap = row.get("final_metrics_by_horizon_gap") or row.get(
            "train_split_metrics_by_horizon_gap"
        )
        if not isinstance(hmap, dict):
            continue
        for gk in hmap.keys():
            try:
                gaps_set.add(int(gk))
            except ValueError:
                continue
    gaps = sorted(gaps_set)
    method_to_vals: Dict[str, List[float]] = {}
    for row in payload.get("results", []):
        key = str(row.get("key", ""))
        hmap = row.get("final_metrics_by_horizon_gap") or row.get(
            "train_split_metrics_by_horizon_gap"
        )
        if not isinstance(hmap, dict) or not hmap:
            continue
        vals: List[float] = []
        for g in gaps:
            block = hmap.get(str(g))
            if not isinstance(block, dict):
                vals.append(float("nan"))
            else:
                vals.append(float(block.get(metric, float("nan"))))
        method_to_vals[key] = vals
    return gaps, method_to_vals


def main() -> None:
    p = argparse.ArgumentParser(description="ベンチマーク JSON のホライズン曲線を PNG 化")
    p.add_argument("json_path", type=Path, help="benchmark_*.json")
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="出力 PNG（省略時は入力と同じディレクトリに horizon_<metric>.png）",
    )
    p.add_argument(
        "--metric",
        choices=("auc", "ap", "ece"),
        default="auc",
        help="縦軸指標",
    )
    args = p.parse_args()
    path = args.json_path
    if not path.is_file():
        raise SystemExit(f"JSON が見つかりません: {path}")

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    gaps, series = _collect_series(payload, args.metric)
    if not gaps or not series:
        raise SystemExit(
            "ホライズン指標がありません。`run_benchmark_comparison --eval-horizon-gaps 1,2,3` "
            "で再実行してください。"
        )

    out = args.output
    if out is None:
        out = path.parent / f"horizon_{args.metric}.png"

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.2))
    for key, vals in sorted(series.items()):
        plt.plot(gaps, vals, marker="o", linewidth=1.6, label=key)
    plt.xlabel("Horizon gap k (steps on sorted year keys)")
    plt.ylabel(args.metric.upper())
    plt.title(f"Future-link {args.metric.upper()} vs horizon ({path.name})")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote: {out.resolve()}")


if __name__ == "__main__":
    main()
