"""
著者–論文（ArXiv 風）CSV から年次二部グラフを構築する。

`P-NODE_paper.ipynb` / `P-NODE_paper copy 3.ipynb` の
`preprocess_arxiv_data`, `filter_active_authors`, `build_author_paper_graphs`
に相当する処理をモジュール化したもの。

想定 CSV 列（ノートブック例: `arxiv_cs_embedded_2020-2026.csv`）
----------------------------------------------------------------
- ``description_embedding`` … 論文埋め込み（文字列 or 配列）。**numpy 表示の省略（``...``）付き文字列は不可**（パース不能のため 0 行になる）。
- ``authors`` … カンマ区切り著者名
- ``year`` … 整数年
- ``url``（または ``arxiv_id`` / ``paper_id``）… 論文 ID。内部で ``paper_id`` 列に正規化。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from pnode_patent_runner.data import safe_parse_embedding


def preprocess_arxiv_data(
    file_path: str,
    *,
    year_min: Optional[int] = 2020,
    year_max: Optional[int] = 2026,
) -> pd.DataFrame:
    """埋め込み・著者リスト・年でフィルタした DataFrame。列: ``paper_vector``, ``authors_list``, ``year``, ``paper_id``。"""
    path = Path(file_path)
    if not path.is_file():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if len(df) == 0:
        return pd.DataFrame()

    embeddings: List[np.ndarray] = []
    valid_indices: List[int] = []
    for i, row in df.iterrows():
        desc = safe_parse_embedding(row.get("description_embedding", ""))
        if desc is not None and len(desc) > 0:
            embeddings.append(desc.astype(np.float32, copy=False))
            valid_indices.append(i)
    if not embeddings:
        msg = (
            "ArXiv 前処理: 有効な description_embedding が1行もありません。"
            " 文字列は数値のみの空白区切り／カンマ区切りである必要があります。"
        )
        if "description_embedding" not in df.columns:
            raise ValueError(
                msg + f" 列 'description_embedding' がありません。利用可能: {list(df.columns)}"
            )
        ellipsis = False
        for v in df["description_embedding"].dropna().head(8):
            if isinstance(v, str) and ("..." in v or "\u2026" in v):
                ellipsis = True
                break
        if ellipsis:
            msg += (
                " セルに '...' が含まれます（numpy や pandas の省略表示で CSV に書き出された可能性）。"
                " 全次元を記録したファイル（例: 1行1次元のワイド CSV、np.save、parquet）を使うか、"
                " 埋め込みを再計算してエクスポートし直してください。"
            )
            stem = path.stem
            if not stem.endswith("_full"):
                alt = path.parent / f"{stem}_full{path.suffix}"
                if alt.is_file():
                    msg += f" ヒント: 同じフォルダに {alt.name} があります（全次元版のことが多いです）。--data にそのパスを指定してください。"
        raise ValueError(msg)

    df = df.iloc[valid_indices].copy()
    df["paper_vector"] = embeddings

    def parse_authors(x):
        if isinstance(x, str):
            return [a.strip() for a in x.split(",") if a.strip()]
        return []

    df["authors_list"] = df["authors"].fillna("").apply(parse_authors)

    id_col = "url" if "url" in df.columns else None
    if id_col is None:
        for cand in ("arxiv_id", "paper_id", "id"):
            if cand in df.columns:
                id_col = cand
                break
    if id_col is None:
        df["paper_id"] = [str(i) for i in range(len(df))]
    else:
        df["paper_id"] = df[id_col].astype(str)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year"])
        df["year"] = df["year"].astype(int)
        if year_min is not None:
            df = df[df["year"] >= year_min]
        if year_max is not None:
            df = df[df["year"] <= year_max]

    return df


def filter_active_authors(df: pd.DataFrame, min_papers: int = 5) -> pd.DataFrame:
    """全期間で ``min_papers`` 本未満の著者を除外（ノートブックの ``filter_active_authors`` と同型）。"""
    if len(df) == 0 or "authors_list" not in df.columns:
        return df.copy()
    flat = [a for sub in df["authors_list"] for a in sub]
    counts = Counter(flat)
    active = {a for a, c in counts.items() if c >= min_papers}
    out = df.copy()
    out["authors_list"] = out["authors_list"].apply(lambda al: [a for a in al if a in active])
    out = out[out["authors_list"].map(len) > 0]
    return out


def calculate_initial_author_vectors(
    df: pd.DataFrame,
    num_authors: int,
    input_dim: int,
    all_authors: List,
) -> torch.Tensor:
    author_vectors = np.zeros((num_authors, input_dim), dtype=np.float32)
    author_counts = np.zeros(num_authors, dtype=np.float32)
    author_to_idx = {a: i for i, a in enumerate(all_authors)}
    for _, row in df.iterrows():
        vec = row["paper_vector"]
        if vec is None or len(vec) != input_dim:
            continue
        for a in row["authors_list"]:
            if a in author_to_idx:
                idx = author_to_idx[a]
                author_vectors[idx] += vec
                author_counts[idx] += 1
    for i in range(num_authors):
        if author_counts[i] > 0:
            author_vectors[i] /= author_counts[i]
        else:
            author_vectors[i] = np.random.normal(0, 0.01, input_dim).astype(np.float32)
    return torch.tensor(author_vectors, dtype=torch.float32)


def build_author_paper_graphs(df: pd.DataFrame) -> Tuple[Dict[int, Data], List, List, int, Set[Tuple[int, int]], int]:
    """
    著者 0..A-1、論文 A..A+P-1 の固定インデックスで年次 ``Data`` と ``hist_edges`` を返す。
    ``df`` には ``preprocess_arxiv_data`` 後の ``paper_id`` 列が必要。
    """
    if len(df) == 0:
        return {}, [], [], 0, set(), 0

    paper_id_column = "paper_id"
    if paper_id_column not in df.columns:
        raise ValueError(f"列 '{paper_id_column}' がありません。利用可能: {list(df.columns)}")

    all_authors = sorted({a for auths in df["authors_list"] for a in auths})
    all_papers = sorted(df[paper_id_column].astype(str).unique().tolist())
    author_to_idx = {auth: i for i, auth in enumerate(all_authors)}
    paper_to_idx = {pid: i + len(all_authors) for i, pid in enumerate(all_papers)}
    total_nodes = len(all_authors) + len(all_papers)

    paper_features: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        pid = str(row[paper_id_column])
        paper_features[pid] = row["paper_vector"]

    input_dim = len(next(iter(paper_features.values())))
    global_graph_dict: Dict[int, Data] = {}
    hist_edges: Set[Tuple[int, int]] = set()

    for year, group in df.groupby("year"):
        edges: List[List[int]] = []
        for _, row in group.iterrows():
            pid = str(row[paper_id_column])
            if pid not in paper_to_idx:
                continue
            p_idx = paper_to_idx[pid]
            for auth in row["authors_list"]:
                if auth in author_to_idx:
                    a_idx = author_to_idx[auth]
                    edges.extend([[a_idx, p_idx], [p_idx, a_idx]])
                    hist_edges.add((a_idx, p_idx))
        if not edges:
            continue
        x = torch.zeros(total_nodes, input_dim)
        for purl, p_idx in paper_to_idx.items():
            if purl in paper_features:
                x[p_idx] = torch.tensor(paper_features[purl], dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        active_mask = torch.zeros(total_nodes, dtype=torch.bool)
        active_mask[torch.unique(edge_index)] = True
        global_graph_dict[int(year)] = Data(
            x=x,
            edge_index=edge_index,
            active_mask=active_mask,
            num_nodes=total_nodes,
        )

    return global_graph_dict, all_authors, all_papers, total_nodes, hist_edges, input_dim


def preprocess_author_topic_data(
    file_path: str,
    *,
    year_min: Optional[int] = 2020,
    year_max: Optional[int] = 2026,
    topic_column: str = "topic",
) -> pd.DataFrame:
    """
    著者–トピック用。`preprocess_arxiv_data` と同じく ``paper_vector`` / ``authors_list`` / ``year`` を作り、
    ``topic_column`` 列の存在を確認する（`P-NODE_paper_topic.ipynb` 想定）。
    """
    df = preprocess_arxiv_data(file_path, year_min=year_min, year_max=year_max)
    if len(df) == 0:
        return df
    if topic_column not in df.columns:
        raise ValueError(
            f"著者–トピックには列 '{topic_column}' が必要です。利用可能: {list(df.columns)}"
        )
    return df


def build_author_topic_graphs(
    df: pd.DataFrame,
    *,
    topic_column: str = "topic",
) -> Tuple[Dict[int, Data], List, List, int, Set[Tuple[int, int]], int]:
    """
    著者 0..A-1、トピック A..A+T-1 の二部グラフ（年次）。
    トピックノード特徴は **全期間** の該当行 ``paper_vector`` の平均（セントロイド）。
    各行は (著者, その行のトピック) のエッジを張る（同年・同著者・同トピックは重複除去）。
    """
    if len(df) == 0:
        return {}, [], [], 0, set(), 0
    if "paper_vector" not in df.columns or "authors_list" not in df.columns:
        raise ValueError("paper_vector と authors_list が必要です（preprocess_arxiv_data 後）")
    if "year" not in df.columns:
        raise ValueError("year 列が必要です")
    if topic_column not in df.columns:
        raise ValueError(f"列 '{topic_column}' がありません")

    all_topics = sorted(df[topic_column].dropna().unique().tolist(), key=lambda x: str(x))
    topic_to_idx = {t: i for i, t in enumerate(all_topics)}
    num_topics = len(all_topics)

    sample_vec = df["paper_vector"].iloc[0]
    input_dim = len(sample_vec)

    topic_centroids = np.zeros((num_topics, input_dim), dtype=np.float32)
    for t_val, t_idx in topic_to_idx.items():
        sub = df[df[topic_column] == t_val]["paper_vector"]
        if len(sub) == 0:
            continue
        vectors = np.stack(sub.values)
        topic_centroids[t_idx] = np.mean(vectors, axis=0).astype(np.float32)

    all_authors = sorted({a for auths in df["authors_list"] for a in auths})
    author_to_idx = {auth: i for i, auth in enumerate(all_authors)}
    num_authors = len(all_authors)
    total_nodes = num_authors + num_topics
    topic_offset = num_authors

    global_graph_dict: Dict[int, Data] = {}
    hist_edges: Set[Tuple[int, int]] = set()

    for year, group in df.groupby("year"):
        edge_set: Set[Tuple[int, int]] = set()
        for _, row in group.iterrows():
            t_val = row[topic_column]
            if pd.isna(t_val) or t_val not in topic_to_idx:
                continue
            t_node_id = topic_to_idx[t_val] + topic_offset
            for auth in row["authors_list"]:
                if auth in author_to_idx:
                    a_idx = author_to_idx[auth]
                    edge_set.add((a_idx, t_node_id))
                    edge_set.add((t_node_id, a_idx))
                    hist_edges.add((a_idx, t_node_id))
        if not edge_set:
            continue
        edges = [list(e) for e in edge_set]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.zeros(total_nodes, input_dim)
        for i in range(num_topics):
            x[topic_offset + i] = torch.tensor(topic_centroids[i], dtype=torch.float32)
        active_mask = torch.zeros(total_nodes, dtype=torch.bool)
        active_mask[torch.unique(edge_index)] = True
        global_graph_dict[int(year)] = Data(
            x=x,
            edge_index=edge_index,
            active_mask=active_mask,
            num_nodes=total_nodes,
        )

    return global_graph_dict, all_authors, all_topics, total_nodes, hist_edges, input_dim
