#!/usr/bin/env python3
"""
ArXiv 系 CSV 用: ``description_embedding`` を **省略なし**の数値ベクトル文字列に再エクスポートする。

``data_arxiv.preprocess_arxiv_data`` / ``safe_parse_embedding`` は、セル内が
**数値の空白区切り（またはカンマ区切り）**であることを想定している。

入力パターン（いずれか一方）:

1. **``--from-json-column``** … 各セルが **JSON 配列**（例: ``[0.1,-0.2,...]``）で、かつ省略なし。
2. **``--from-columns-prefix``** … 列 ``emb_0``, ``emb_1``, ... のように **1次元1列**でベクトルが入っている。

埋め込みを **まだ持っていない**場合は、別途モデルでベクトルを計算し、上記いずれかの形で CSV を用意してからこのスクリプトを使う。

例:

  python -m pnode_patent_runner.reexport_arxiv_embeddings \\
    --input data/processed/wide_embeddings.csv \\
    --output data/processed/arxiv_cs_embedded_fixed.csv \\
    --from-columns-prefix emb_

  python -m pnode_patent_runner.reexport_arxiv_embeddings \\
    --input data/raw.json_array_in_cell.csv \\
    --output data/processed/fixed.csv \\
    --from-json-column description_embedding
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pnode_patent_runner.data import safe_parse_embedding


def _embedding_to_cell(vec: np.ndarray) -> str:
    v = np.asarray(vec, dtype=np.float64).ravel()
    if v.size == 0:
        return ""
    # 科学表記で短くしつつ、省略記号は出さない
    return " ".join(f"{float(x):.10g}" for x in v)


def _columns_for_prefix(df: pd.DataFrame, prefix: str) -> List[str]:
    pat = re.compile(r"^" + re.escape(prefix) + r"(\d+)$")
    numbered: List[tuple] = []
    for c in df.columns:
        if not isinstance(c, str):
            continue
        m = pat.match(c)
        if m:
            numbered.append((int(m.group(1)), c))
    if not numbered:
        raise SystemExit(
            f"列名が '{prefix}<整数>' 形式のものがありません。例: {prefix}0, {prefix}1, ..."
        )
    numbered.sort(key=lambda x: x[0])
    return [c for _, c in numbered]


def main() -> None:
    p = argparse.ArgumentParser(description="description_embedding を全要素・空白区切りで再エクスポート")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--from-json-column",
        type=str,
        metavar="COL",
        help="JSON 配列文字列の列名（省略 '...' 不可）",
    )
    g.add_argument(
        "--from-columns-prefix",
        type=str,
        metavar="PREFIX",
        help="emb_0,emb_1,... のプレフィックス（例: emb_）",
    )
    p.add_argument(
        "--target-column",
        type=str,
        default="description_embedding",
        help="出力 CSV に書く列名（既定: description_embedding）",
    )
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        raise SystemExit(f"入力が見つかりません: {inp}")

    df = pd.read_csv(inp)
    out_col = args.target_column

    if args.from_json_column:
        col = args.from_json_column
        if col not in df.columns:
            raise SystemExit(f"列 '{col}' がありません。利用可能: {list(df.columns)}")

        cells: List[str] = []
        for i, raw in enumerate(df[col].tolist()):
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                cells.append("")
                continue
            s = str(raw).strip()
            if "..." in s or "\u2026" in s:
                raise SystemExit(
                    f"行 {i}: セルに '...' が含まれます。JSON 全要素が必要です（この行は復元不可）。"
                )
            try:
                arr = np.asarray(json.loads(s), dtype=np.float64).ravel()
            except json.JSONDecodeError as e:
                raise SystemExit(f"行 {i}: JSON として解釈できません: {e}") from e
            if arr.size == 0:
                cells.append("")
            else:
                cells.append(_embedding_to_cell(arr))
        df[out_col] = cells

    else:
        prefix = args.from_columns_prefix
        emb_cols = _columns_for_prefix(df, prefix)
        vecs: List[str] = []
        for _, row in df.iterrows():
            vals = row[emb_cols].to_numpy(dtype=np.float64, copy=True)
            if np.isnan(vals).all():
                vecs.append("")
            else:
                vecs.append(_embedding_to_cell(vals))
        df[out_col] = vecs

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_ok = 0
    for i in range(min(len(df), 50)):
        v = safe_parse_embedding(df[out_col].iloc[i])
        if v is not None and v.size > 0:
            n_ok += 1
    print(f"Wrote: {out}  （先頭最大50行のうちパース成功: {n_ok}）")
    if n_ok == 0 and len(df) > 0:
        print("警告: 先頭行で safe_parse_embedding がすべて失敗しました。列を確認してください。")


if __name__ == "__main__":
    main()
