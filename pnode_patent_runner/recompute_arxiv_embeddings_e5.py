#!/usr/bin/env python3
"""
ArXiv 系 CSV の ``description_embedding`` を
``intfloat/multilingual-e5-large`` で **全次元** 再計算し、
``data_arxiv.safe_parse_embedding`` が読める **空白区切り1セル** で保存する。

``notebooks/work/python/compute_paper_embeds.ipynb`` の設定に揃えている:

- 入力テキスト: ``description`` 列（欠損は空文字）
- ``model.encode(..., normalize_embeddings=True)``
- 既定 batch_size=32

※ 以前の ``to_csv`` 後に表示用の省略（``...``）が入った CSV は、このスクリプトで上書きすればパイプライン用に使える。

例（リポジトリルート）:

  python -m pnode_patent_runner.recompute_arxiv_embeddings_e5 \\
    --input data/processed/arxiv_cs_embedded_2020-2026.csv \\
    --output data/processed/arxiv_cs_embedded_2020-2026_full.csv

初回は Hugging Face からモデル取得（ネットワーク必須）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pnode_patent_runner.data import safe_parse_embedding


def embedding_to_csv_cell(vec: np.ndarray) -> str:
    v = np.asarray(vec, dtype=np.float64).ravel()
    if v.size == 0:
        return ""
    return " ".join(f"{float(x):.10g}" for x in v)


def main() -> None:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "要: sentence-transformers, torch。例: pip install sentence-transformers torch"
        ) from e

    repo = _REPO_ROOT
    default_in = repo / "data/processed/arxiv_cs_embedded_2020-2026.csv"

    p = argparse.ArgumentParser(
        description="multilingual-e5-large で description_embedding を再計算（省略なしCSV）",
    )
    p.add_argument("--input", type=str, default=str(default_in))
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="省略時は --input を上書き（上書き前にバックアップ推奨）",
    )
    p.add_argument(
        "--model",
        type=str,
        default="intfloat/multilingual-e5-large",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda / cpu / 空なら自動",
    )
    p.set_defaults(normalize_embeddings=True)
    p.add_argument(
        "--no-normalize-embeddings",
        action="store_false",
        dest="normalize_embeddings",
        help="model.encode の normalize_embeddings を False（既定は True＝ノートブックと同じ）",
    )
    p.add_argument(
        "--e5-passage-prefix",
        action="store_true",
        help="各テキスト先頭に \"passage: \"（E5 の passage 用。ノートブックは付けていない）",
    )
    p.add_argument(
        "--text-mode",
        type=str,
        choices=("description", "title_and_description"),
        default="description",
        help="description=ノートブック準拠 / title_and_description=title + 改行 + description",
    )
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        raise SystemExit(f"入力が見つかりません: {inp}")

    out = Path(args.output) if args.output.strip() else inp

    device = (args.device or "").strip() or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device={device}, model={args.model}")

    df = pd.read_csv(inp, low_memory=False)
    if "description" not in df.columns:
        raise SystemExit(f"列 'description' がありません: {list(df.columns)}")

    texts = []
    for _, row in df.iterrows():
        desc = row.get("description", "")
        if pd.isna(desc):
            desc = ""
        desc = str(desc).strip()
        if args.text_mode == "title_and_description":
            title = row.get("title", "")
            if pd.isna(title):
                title = ""
            title = str(title).strip()
            body = f"{title}\n{desc}" if title else desc
        else:
            body = desc
        if args.e5_passage_prefix and body:
            body = "passage: " + body
        texts.append(body)

    print(f"encode 対象: {len(texts)} 行, batch_size={args.batch_size}")
    model = SentenceTransformer(args.model, device=device)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=args.normalize_embeddings,
    )
    emb = np.asarray(embeddings, dtype=np.float32)
    print(f"embedding shape: {emb.shape}")

    df["description_embedding"] = [embedding_to_csv_cell(emb[i]) for i in range(len(emb))]

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")

    n_check = min(20, len(df))
    ok = sum(
        1
        for i in range(n_check)
        if safe_parse_embedding(df["description_embedding"].iloc[i]) is not None
    )
    print(f"Wrote: {out}")
    print(f"先頭 {n_check} 行のパース成功: {ok}/{n_check}")
    if ok < n_check:
        print("警告: 一部パース失敗。列を確認してください。")


if __name__ == "__main__":
    main()
