#!/usr/bin/env python3
"""
複数の run_benchmark_comparison 出力 JSON を横並びで比較する。

同一 seed・同一データであれば、static / rnn / neural_ode は P-NODE 用 CLI に依存しないため
各列で一致する（不一致なら設定または実装の確認用）。

例:
  python -m pnode_patent_runner.compare_benchmark_ablations \\
    legacy.json B_mlp_K1.json D_gru_K4.json \\
    --labels legacy B_only D_only

  python -m pnode_patent_runner.compare_benchmark_ablations \\
    a.json b.json --metric final_val_ap --markdown
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_results(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    rows = raw.get("results")
    if not isinstance(rows, list):
        raise SystemExit(f"results が無い: {path}")
    return raw, rows


def _row_by_key(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = r.get("key")
        if isinstance(k, str):
            out[k] = r
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="ベンチマーク JSON 横比較（ablation 用）")
    p.add_argument(
        "json_paths",
        nargs="+",
        type=Path,
        help="run_benchmark_comparison の出力 JSON",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="列見出し（json と同じ個数。省略時はファイル名幹）",
    )
    p.add_argument(
        "--metric",
        default="final_val_auc",
        help="比較する results[*] のキー（既定: final_val_auc）",
    )
    p.add_argument("--markdown", action="store_true", help="Markdown 表を出力")
    args = p.parse_args()

    paths = [Path(x).resolve() for x in args.json_paths]
    if args.labels is not None:
        if len(args.labels) != len(paths):
            raise SystemExit("--labels の個数は json_paths と同じにしてください")
        labels = list(args.labels)
    else:
        labels = [x.stem for x in paths]

    payloads: List[Dict[str, Any]] = []
    key_maps: List[Dict[str, Dict[str, Any]]] = []
    for path in paths:
        raw, rows = _load_results(path)
        payloads.append(raw)
        key_maps.append(_row_by_key(rows))

    all_keys: List[str] = sorted(set().union(*(km.keys() for km in key_maps)))

    def _getf(d: Optional[Dict[str, Any]]) -> float:
        if not d:
            return float("nan")
        v = d.get(args.metric)
        if isinstance(v, bool):
            return float("nan")
        if isinstance(v, (int, float)):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return float("nan")
            return float(v)
        return float("nan")

    if args.markdown:
        header = "| method | " + " | ".join(labels) + " |"
        sep = "|---|" + "|".join(["---"] * len(labels)) + "|"
        print(header)
        print(sep)
        for key in all_keys:
            cells = [f"{_getf(km.get(key)):.4f}" for km in key_maps]
            print(f"| {key} | " + " | ".join(cells) + " |")
    else:
        w = max(len(args.metric), 8)
        colw = 12
        print(f"{'method':<22}", end="")
        for lb in labels:
            print(f"{lb:>{colw}}", end="")
        print()
        print("-" * (22 + colw * len(labels)))
        for key in all_keys:
            print(f"{key:<22}", end="")
            for km in key_maps:
                v = _getf(km.get(key))
                s = "nan" if math.isnan(v) else f"{v:.4f}"
                print(f"{s:>{colw}}", end="")
            print()

    # 参照用: 先頭 JSON の seed / pnode 設定
    meta0 = payloads[0]
    print("\n# meta (先頭 JSON)", file=sys.stderr)
    print(f"  seed={meta0.get('seed')}  data_domain={meta0.get('data_domain')}", file=sys.stderr)


if __name__ == "__main__":
    main()
