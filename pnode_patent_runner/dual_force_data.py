import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Set
from torch_geometric.data import Data
from pnode_patent_runner.data_arxiv import preprocess_author_topic_data, build_author_topic_graphs

def compute_topic_mass_and_trend(df: pd.DataFrame, topic_column: str = "topic") -> Dict[int, Dict[str, float]]:
    """
    年ごとの各トピックの質量 (M_j) と勢い (D_j) を計算する。
    M_j(t) = 時刻 t におけるトピック j の論文数
    D_j(t) = M_j(t) - M_j(t-1)
    """
    years = sorted(df["year"].unique())
    topics = sorted(df[topic_column].unique())
    
    # 年ごとのトピック出現数を集計
    counts = df.groupby(["year", topic_column]).size().unstack(fill_value=0)
    
    topic_stats = {}
    for i, year in enumerate(years):
        year_stats = {}
        for topic in topics:
            m_t = float(counts.loc[year, topic])
            if i == 0:
                d_t = 0.0
            else:
                prev_year = years[i-1]
                m_prev = float(counts.loc[prev_year, topic])
                d_t = m_t - m_prev
            
            year_stats[topic] = {
                "mass": m_t,
                "trend": d_t,
                "trend_plus": max(0.0, d_t),
                "trend_minus": max(0.0, -d_t)
            }
        topic_stats[year] = year_stats
        
    return topic_stats

def load_dual_force_bundle(csv_path: str, topic_column: str = "topic", min_papers: int = 5):
    from pnode_patent_runner.cope_experiment import CopeGraphBundle, select_year_list, subset_graphs
    from pnode_patent_runner.data_arxiv import filter_active_authors, calculate_initial_author_vectors
    
    df = preprocess_author_topic_data(csv_path, topic_column=topic_column)
    df = filter_active_authors(df, min_papers=min_papers)
    
    # トピックの統計情報を計算
    topic_stats = compute_topic_mass_and_trend(df, topic_column=topic_column)
    
    graphs_full, authors, topics, total_nodes, hist_edges, input_dim = build_author_topic_graphs(df, topic_column=topic_column)
    
    num_authors = len(authors)
    init_vectors = calculate_initial_author_vectors(df, num_authors, input_dim, authors)
    
    # 各年のDataオブジェクトにトピック統計情報を追加
    for year, data in graphs_full.items():
        year_stats = topic_stats.get(year, {})
        # トピック順に並べる
        masses = []
        trends_plus = []
        trends_minus = []
        for t_val in topics:
            stats = year_stats.get(t_val, {"mass": 0.0, "trend_plus": 0.0, "trend_minus": 0.0})
            masses.append(stats["mass"])
            trends_plus.append(stats["trend_plus"])
            trends_minus.append(stats["trend_minus"])
            
        data.topic_mass = torch.tensor(masses, dtype=torch.float32)
        data.topic_trend_plus = torch.tensor(trends_plus, dtype=torch.float32)
        data.topic_trend_minus = torch.tensor(trends_minus, dtype=torch.float32)
    
    return CopeGraphBundle(
        graphs=graphs_full,
        num_corps=num_authors,
        hist_edges=hist_edges,
        in_dim=input_dim,
        total_n=total_nodes,
        corps=authors,
        init_vectors=init_vectors,
        dataframe=df,
        right_nodes=topics
    )
