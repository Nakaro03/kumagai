"""
特許 CSV（topic_info3 系）から年次の企業–特許二部グラフを構築する。

処理の流れ（概要）
-----------------
1. **preprocess_data**
   - CSV を読み込み、各行の `description_embedding` と `metadata_embedding` をパースして連結し
     `combined_vector`（入力特徴）を作る。どちらか欠ける行は落とす。
   - `corporation` をリスト化（文字列なら `ast.literal_eval`）。
   - `year_month` を日付にし、欠損除去のうえ年範囲でフィルタする。

2. **filter_active_corporations**
   - 全期間で「紐づく特許行数」が `min_patents` 未満の企業を除外し、各行の出願人リストからも削る。
   - 出願人が空になった行を落とす。

3. **build_global_graphs**
   - ノード: 企業 0..C-1、特許 C..C+P-1（全期間で固定のインデックス）。
   - 年ごとに、その年に観測された企業–特許エッジだけで `torch_geometric.data.Data` を構築（無向のため両方向エッジ）。
   - 特許ノードのみ `x` に `combined_vector` を入れ、企業ノードは 0（学習時は `corp_embeddings` で上書き）。
   - **hist_edges**: これまでに出現した (企業idx, 特許idx) の集合（負例サンプリング等で使用）。

4. **calculate_initial_corp_vectors**
   - 各企業について、その企業が行に含まれるすべての特許の `combined_vector` の平均を初期埋め込みとする。
   - 一度も現れない企業は小さな正規乱数。

想定 CSV 列（既存ノートブックと同様）
------------------------------------
- 必須に近い: ``description_embedding``, ``metadata_embedding``, ``corporation``,
  ``year_month``, ``patent_number``
- 埋め込み列は文字列化された数値ベクトルでも可（正規表現でパース）。
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def safe_parse_embedding(x):
    """文字列 / リスト / NaN から float32 ベクトルを取り出す。失敗時は None。"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (list, np.ndarray)):
        arr = np.asarray(x, dtype=np.float32)
        return arr if arr.size > 0 else None
    if isinstance(x, str):
        if not x.strip():
            return None
        try:
            s = re.sub(r"[\[\]\n]", "", x)
            s = s.replace(",", " ")
            vals = np.array([float(v) for v in s.split() if v], dtype=np.float32)
            return vals if len(vals) else None
        except Exception:
            return None
    return None


def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    特許レコード CSV を読み、結合埋め込み・企業リスト・年を整える。

    Returns
    -------
    空の場合は空の DataFrame。列 ``combined_vector``（np.ndarray）, ``corporation``（list）,
    ``year_month``（datetime）, ``patent_number`` を含む。
    """
    path = Path(file_path)
    if not path.is_file():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    combined: List[np.ndarray] = []
    valid_idx: List[int] = []
    for i, row in df.iterrows():
        desc = safe_parse_embedding(row.get("description_embedding", ""))
        meta = safe_parse_embedding(row.get("metadata_embedding", ""))
        if desc is not None and meta is not None and len(desc) > 0 and len(meta) > 0:
            combined.append(np.concatenate([desc, meta]))
            valid_idx.append(i)

    if not combined:
        return pd.DataFrame()

    df = df.iloc[valid_idx].copy()
    df["combined_vector"] = combined

    def parse_corp(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else x
        except Exception:
            return [x] if isinstance(x, str) else x

    if "corporation" in df.columns:
        df["corporation"] = df["corporation"].apply(parse_corp)
        df["corporation"] = df["corporation"].apply(lambda x: x if isinstance(x, list) else [])

    if "year_month" in df.columns:
        df["year_month"] = pd.to_datetime(df["year_month"], errors="coerce")
        df = df.dropna(subset=["year_month"])
        df = df[(df["year_month"] >= "2010-01-01") & (df["year_month"] <= "2030-12-31")]

    return df


def filter_active_corporations(df: pd.DataFrame, min_patents: int = 2) -> pd.DataFrame:
    """
    全期間で ``min_patents`` 件未満しか特許行に出てこない企業を除外する。
    """
    if len(df) == 0 or "corporation" not in df.columns:
        return df.copy()

    flat = [c for sub in df["corporation"] for c in sub]
    counts = Counter(flat)
    active = {c for c, n in counts.items() if n >= min_patents}
    out = df.copy()
    out["corporation"] = out["corporation"].apply(lambda xs: [c for c in xs if c in active])
    out = out[out["corporation"].map(len) > 0]
    return out


def calculate_initial_corp_vectors(
    df: pd.DataFrame,
    num_corps: int,
    input_dim: int,
    all_corporations: List,
) -> torch.Tensor:
    """
    企業ごとに、その企業が関与する特許行の ``combined_vector`` の平均を初期ベクトルにする。
    """
    corp_vectors = np.zeros((num_corps, input_dim), dtype=np.float32)
    corp_counts = np.zeros(num_corps, dtype=np.float32)
    corp_to_idx = {c: i for i, c in enumerate(all_corporations)}
    for _, row in df.iterrows():
        vec = row["combined_vector"]
        for c in row["corporation"]:
            if c in corp_to_idx:
                idx = corp_to_idx[c]
                corp_vectors[idx] += vec
                corp_counts[idx] += 1
    for i in range(num_corps):
        if corp_counts[i] > 0:
            corp_vectors[i] /= corp_counts[i]
        else:
            corp_vectors[i] = np.random.normal(0, 0.01, input_dim).astype(np.float32)
    return torch.tensor(corp_vectors, dtype=torch.float32)


def build_global_graphs(
    df: pd.DataFrame,
) -> Tuple[Dict[int, Data], List, List, int, Set[Tuple[int, int]], int]:
    """
    年次二部グラフの辞書と補助情報を返す。

    Returns
    -------
    graphs : Dict[year, Data]
        ``x``, ``edge_index``, ``active_mask``, ``num_nodes`` を持つ。
    corps, patents : list
        ノード順序（企業は 0..C-1、特許はその後）。
    total_n : int
        全ノード数 C + P。
    hist_edges : set of (corp_idx, patent_idx)
        過去に現れた無向対応の企業–特許ペア（有向はグラフ内で両方向）。
    in_dim : int
        特許特徴ベクトル次元（``combined_vector`` の長さ）。
    """
    if len(df) == 0:
        return {}, [], [], 0, set(), 0

    all_corporations = sorted({c for corps in df["corporation"] for c in corps})
    all_patents = sorted(df["patent_number"].unique().tolist())
    corp_to_idx = {c: i for i, c in enumerate(all_corporations)}
    patent_to_idx = {p: i + len(all_corporations) for i, p in enumerate(all_patents)}
    total_nodes = len(all_corporations) + len(all_patents)

    patent_features: Dict = {}
    for _, row in df.iterrows():
        patent_features[row["patent_number"]] = row["combined_vector"]
    input_dim = len(next(iter(patent_features.values())))

    global_graph_dict: Dict[int, Data] = {}
    hist_edges: Set[Tuple[int, int]] = set()
    year_groups = df.groupby(df["year_month"].dt.year)

    for year, group in year_groups:
        edges: List[List[int]] = []
        for _, row in group.iterrows():
            if row["patent_number"] not in patent_to_idx:
                continue
            p_idx = patent_to_idx[row["patent_number"]]
            for corp in row["corporation"]:
                if corp in corp_to_idx:
                    c_idx = corp_to_idx[corp]
                    edges.append([c_idx, p_idx])
                    edges.append([p_idx, c_idx])
                    hist_edges.add((c_idx, p_idx))
        if not edges:
            continue

        x = torch.zeros(total_nodes, input_dim)
        for p_num, p_idx in patent_to_idx.items():
            if p_num in patent_features:
                x[p_idx] = torch.tensor(patent_features[p_num], dtype=torch.float32)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        active_mask = torch.zeros(total_nodes, dtype=torch.bool)
        active_mask[torch.unique(edge_index)] = True
        global_graph_dict[int(year)] = Data(
            x=x,
            edge_index=edge_index,
            active_mask=active_mask,
            num_nodes=total_nodes,
        )

    return global_graph_dict, all_corporations, all_patents, total_nodes, hist_edges, input_dim


# ノートブック名との互換
preprocess_csv = preprocess_data
