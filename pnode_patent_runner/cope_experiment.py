"""
README_COPE.md に沿った実験用ユーティリティ。

- 特許 CSV → `load_cope_graph_bundle`（企業–特許）
- 著者–論文 CSV → `load_author_paper_graph_bundle`（`P-NODE_paper.ipynb` 相当）
- 年次サブグラフの切り出し
- ベースライン手法名と `UnifiedVGAE` / `BenchmarkTemporalVGAE` の生成
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data

from pnode_patent_runner.benchmark_vgae import BenchmarkTemporalVGAE
from pnode_patent_runner.data import (
    build_global_graphs,
    calculate_initial_corp_vectors,
    filter_active_corporations,
    preprocess_data,
)
from pnode_patent_runner.data_arxiv import (
    build_author_paper_graphs,
    build_author_topic_graphs,
    calculate_initial_author_vectors,
    filter_active_authors,
    preprocess_arxiv_data,
    preprocess_author_topic_data,
)
from pnode_patent_runner.unified_vgae import METHOD_SHORT_NAME, UnifiedVGAE
from pnode_patent_runner.unified_vgae_td import UnifiedVGAETD


@dataclass
class CopeGraphBundle:
    """学習・評価に必要なグラフと補助情報（README のデータパイプライン完了状態）。

    ``dataframe`` … 前処理・フィルタ後の DataFrame（可視化ホバー用。特許・著者系で設定）。
    ``right_nodes`` … 右パーティション id 列（特許番号 / ``paper_id`` / トピック名など）。
    """

    graphs: Dict[int, Data]
    num_corps: int
    hist_edges: Set[Tuple[int, int]]
    in_dim: int
    total_n: int
    corps: List
    init_vectors: torch.Tensor
    dataframe: Optional[pd.DataFrame] = None
    right_nodes: Optional[List] = None


BASELINE_METHOD_SPECS: Tuple[Tuple[str, str], ...] = (
    ("Static", "static"),
    ("RNN+VGAE", "rnn"),
    ("NeuralODE", "neural_ode"),
    ("P-NODE", "pnode"),
    ("P-NODE-ExplicitA", "pnode_explicit"),
    ("P-NODE+Res", "pnode_residual"),
    ("PC-PNODE", "pnode_pc"),
    ("P-NODE-Energy-TD", "pnode_energy"),
    (METHOD_SHORT_NAME, "cope"),
)

BASELINE_KEYS: Tuple[str, ...] = tuple(k for _, k in BASELINE_METHOD_SPECS)


def parse_methods_csv(s: str) -> List[str]:
    s = (s or "").strip().lower()
    if s in ("", "all"):
        return list(BASELINE_KEYS)
    allowed = set(BASELINE_KEYS)
    out: List[str] = []
    for part in s.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p not in allowed:
            raise ValueError(
                f"未知の手法キー: {p}。利用可能: {sorted(allowed)} または all"
            )
        out.append(p)
    return out or list(BASELINE_KEYS)


def select_year_list(
    graphs_full: Dict[int, Any],
    *,
    year_range: Optional[Tuple[int, int]] = None,
    years_csv: str = "",
    all_years: bool = False,
) -> List[int]:
    """
    `graphs_full` のキー（年）から、README と同様のルールで部分列を返す。

    - `year_range=(a,b)` … 閉区間 [a,b] に含まれる年（`all_years` / `years_csv` と併用不可）
    - `years_csv` … カンマ区切り年リスト
    - `all_years` … 全て
    - いずれも無ければ全て
    """
    years_available = sorted(graphs_full.keys())
    if year_range is not None:
        if all_years or (years_csv or "").strip():
            raise ValueError("year_range は all_years / years_csv と同時に使えません。")
        y0, y1 = int(year_range[0]), int(year_range[1])
        if y0 > y1:
            y0, y1 = y1, y0
        years_list = [y for y in years_available if y0 <= y <= y1]
        if not years_list:
            raise ValueError(
                f"年範囲 {y0}〜{y1} にグラフがありません。利用可能: {years_available}"
            )
        return years_list
    if all_years:
        return list(years_available)
    if (years_csv or "").strip():
        years_list: List[int] = []
        for part in years_csv.split(","):
            part = part.strip()
            if not part:
                continue
            y = int(part)
            if y not in graphs_full:
                raise ValueError(f"year={y} のグラフがありません。利用可能: {years_available}")
            years_list.append(y)
        return sorted(set(years_list))
    return list(years_available)


def subset_graphs(
    graphs_full: Dict[int, Data], years_list: Sequence[int]
) -> Dict[int, Data]:
    return {y: graphs_full[y] for y in years_list if y in graphs_full}


def hist_edges_union_from_graphs(
    graphs: Dict[int, Data], num_corps: int
) -> Set[Tuple[int, int]]:
    """
    与えられた年次グラフに現れる (企業idx, 右ノードidx) の和集合。
    学習に使う年だけに絞った ``hist_edges``（負例サンプリング用）に使う。
    """
    hist: Set[Tuple[int, int]] = set()
    for _y in sorted(graphs.keys()):
        ei = graphs[_y].edge_index
        for k in range(ei.size(1)):
            a, b = int(ei[0, k]), int(ei[1, k])
            if a < num_corps <= b:
                hist.add((a, b))
            elif b < num_corps <= a:
                hist.add((b, a))
    return hist


@dataclass(frozen=True)
class HoldoutSplit:
    """``holdout_test_year`` のエッジを学習損失に含めない分割。"""

    graphs_train: Dict[int, Data]
    hist_edges_train: Set[Tuple[int, int]]
    graphs_full: Dict[int, Data]
    holdout_test_year: int
    year_prev: int
    train_years: Tuple[int, ...]


def split_bundle_holdout_test_year(
    bundle: CopeGraphBundle, holdout_test_year: int
) -> HoldoutSplit:
    """
    - **学習**: ``y < holdout_test_year`` のグラフのみ。``hist_edges`` はその期間のエッジのみで再構成。
    - **最終テスト**: 時系列上 ``holdout_test_year`` の直前の年 ``year_prev`` から ``holdout_test_year`` への future-link
      （``year_prev`` は利用可能な年キーの並びで決める）。
    """
    years = sorted(bundle.graphs.keys())
    if holdout_test_year not in bundle.graphs:
        raise ValueError(
            f"holdout_test_year={holdout_test_year} がグラフにありません: {years}"
        )
    idx = years.index(holdout_test_year)
    if idx == 0:
        raise ValueError("holdout_test_year は最初の年にはできません。")
    train_years = [y for y in years if y < holdout_test_year]
    if len(train_years) < 2:
        raise ValueError(
            f"学習に連続ペアが必要です。{holdout_test_year} より前に2年以上必要。現状: {train_years}"
        )
    year_prev = years[idx - 1]
    graphs_train = subset_graphs(bundle.graphs, train_years)
    hist_tr = hist_edges_union_from_graphs(graphs_train, bundle.num_corps)
    return HoldoutSplit(
        graphs_train=graphs_train,
        hist_edges_train=hist_tr,
        graphs_full=bundle.graphs,
        holdout_test_year=holdout_test_year,
        year_prev=year_prev,
        train_years=tuple(train_years),
    )


def load_cope_graph_bundle(
    csv_path: Union[str, Path],
    *,
    min_patents: int = 2,
    year_range: Optional[Tuple[int, int]] = None,
    years_csv: str = "",
    all_years: bool = False,
) -> CopeGraphBundle:
    """
    README「データパイプライン」どおりに CSV からグラフ束を構築する。
    `year_range` / `years_csv` / `all_years` で年を絞った **部分グラフ辞書** を返す。
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"データが見つかりません: {path}")

    df = preprocess_data(str(path))
    if len(df) == 0:
        raise ValueError("前処理後データが空です")
    df = filter_active_corporations(df, min_patents=min_patents)

    graphs_full, corps, patents, total_n, hist_edges, in_dim = build_global_graphs(df)
    num_corps = len(corps)
    years_list = select_year_list(
        graphs_full,
        year_range=year_range,
        years_csv=years_csv,
        all_years=all_years,
    )
    graphs = subset_graphs(graphs_full, years_list)
    if len(graphs) < 2:
        raise ValueError(
            f"年次グラフが2年未満です（学習に連続ペアが必要）: {sorted(graphs.keys())}"
        )

    init_vectors = calculate_initial_corp_vectors(df, num_corps, in_dim, corps)
    return CopeGraphBundle(
        graphs=graphs,
        num_corps=num_corps,
        hist_edges=hist_edges,
        in_dim=in_dim,
        total_n=total_n,
        corps=corps,
        init_vectors=init_vectors,
        dataframe=df,
        right_nodes=patents,
    )


def load_author_paper_graph_bundle(
    csv_path: Union[str, Path],
    *,
    min_papers: int = 5,
    arxiv_year_min: Optional[int] = 2020,
    arxiv_year_max: Optional[int] = 2026,
    year_range: Optional[Tuple[int, int]] = None,
    years_csv: str = "",
    all_years: bool = False,
) -> CopeGraphBundle:
    """
    著者–論文二部グラフ（`data_arxiv` / `P-NODE_paper.ipynb` と同型）。
    ``CopeGraphBundle.num_corps`` は **著者数**（`UnifiedVGAE.num_corps` として左側パーティション）。
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"データが見つかりません: {path}")

    df = preprocess_arxiv_data(
        str(path), year_min=arxiv_year_min, year_max=arxiv_year_max
    )
    if len(df) == 0:
        raise ValueError("ArXiv 前処理後データが空です（列: description_embedding, authors, year, url 等）")
    df = filter_active_authors(df, min_papers=min_papers)

    graphs_full, authors, papers, total_n, hist_edges, in_dim = build_author_paper_graphs(df)
    num_authors = len(authors)
    if num_authors == 0:
        raise ValueError("著者が 0 です。min_papers を下げるか CSV を確認してください。")

    years_list = select_year_list(
        graphs_full,
        year_range=year_range,
        years_csv=years_csv,
        all_years=all_years,
    )
    graphs = subset_graphs(graphs_full, years_list)
    if len(graphs) < 2:
        raise ValueError(
            f"年次グラフが2年未満です: {sorted(graphs.keys())}。"
            " --arxiv-year-min / --arxiv-year-max や --year-range を調整してください。"
        )

    init_vectors = calculate_initial_author_vectors(df, num_authors, in_dim, authors)
    return CopeGraphBundle(
        graphs=graphs,
        num_corps=num_authors,
        hist_edges=hist_edges,
        in_dim=in_dim,
        total_n=total_n,
        corps=authors,
        init_vectors=init_vectors,
        dataframe=df,
        right_nodes=papers,
    )


def load_author_topic_graph_bundle(
    csv_path: Union[str, Path],
    *,
    min_papers: int = 5,
    topic_column: str = "topic",
    arxiv_year_min: Optional[int] = 2020,
    arxiv_year_max: Optional[int] = 2026,
    year_range: Optional[Tuple[int, int]] = None,
    years_csv: str = "",
    all_years: bool = False,
) -> CopeGraphBundle:
    """
    著者–トピック二部グラフ（`P-NODE_paper_topic.ipynb` の ``build_author_topic_graphs_from_col`` 相当）。
    右パーティションは論文ではなく **トピック**；`CopeGraphBundle.num_corps` は著者数。
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"データが見つかりません: {path}")

    df = preprocess_author_topic_data(
        str(path),
        year_min=arxiv_year_min,
        year_max=arxiv_year_max,
        topic_column=topic_column,
    )
    if len(df) == 0:
        raise ValueError(
            "著者–トピック前処理後データが空です（description_embedding, authors, year, "
            f"{topic_column} を確認）"
        )
    df = filter_active_authors(df, min_papers=min_papers)

    graphs_full, authors, topics, total_n, hist_edges, in_dim = build_author_topic_graphs(
        df, topic_column=topic_column
    )
    num_authors = len(authors)
    if num_authors == 0:
        raise ValueError("著者が 0 です。min_papers を下げるか CSV を確認してください。")

    years_list = select_year_list(
        graphs_full,
        year_range=year_range,
        years_csv=years_csv,
        all_years=all_years,
    )
    graphs = subset_graphs(graphs_full, years_list)
    if len(graphs) < 2:
        raise ValueError(
            f"年次グラフが2年未満です: {sorted(graphs.keys())}。"
            " --arxiv-year-min / --arxiv-year-max や --year-range を調整してください。"
        )

    init_vectors = calculate_initial_author_vectors(df, num_authors, in_dim, authors)
    return CopeGraphBundle(
        graphs=graphs,
        num_corps=num_authors,
        hist_edges=hist_edges,
        in_dim=in_dim,
        total_n=total_n,
        corps=authors,
        init_vectors=init_vectors,
        dataframe=df,
        right_nodes=topics,
    )


@dataclass
class ModelBuildKw:
    hidden_dim: int = 128
    latent_dim: int = 2
    link_score_mode: str = "distance"
    cosine_logit_scale: float = 5.0
    w_pot_init: float = 0.05
    rnn_history_len: int = 4
    """P-NODE 用: 直近何年分の z を結合して ODE 初値に射影するか（1 で従来どおり）。"""
    pnode_history_len: int = 1
    pnode_ode_method: str = "dopri5"
    pnode_ode_n_steps: int = 4
    density_calibrated_potential: bool = False
    density_log_weight: float = 1.0
    density_ema_momentum: float = 0.05
    time_dependent_potential: bool = False
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    #: ベンチマーク P-NODE / P-NODE+Res: ``rff``（sin/cos）または ``mlp``（z から直接）
    pnode_potential_feature: str = "mlp"
    #: ``rff`` 時のみ: True なら従来どおり B を固定（再現性用）
    pnode_rff_frozen_basis: bool = False
    #: P-NODE 系で CalibratedPotentialNet（Φ = φ_nn - w log p_hist）を使う
    pnode_density_calibrated: bool = False
    pnode_density_log_weight: float = 1.0
    pnode_density_ema_momentum: float = 0.05
    #: 履歴 K>1 のとき ``linear``（従来）または ``gru``
    pnode_hist_fuse_mode: str = "gru"
    #: PC-PNODE: w_ρ 初期値（密度引力強度）
    pc_w_rho_init: float = 0.1
    #: PC-PNODE: w_Δ 初期値（トレンド引力強度）
    pc_w_delta_init: float = 0.1
    #: PC-PNODE: KDE 帯域幅の log 初期値
    pc_log_bandwidth_init: float = 0.0
    #: PC-PNODE: density-align 損失の重み（λ_da）
    pc_density_align_weight: float = 0.005


def build_baseline_model(
    key: str,
    device: torch.device,
    bundle: CopeGraphBundle,
    kw: ModelBuildKw,
) -> nn.Module:
    """`key` は `static` / `rnn` / `neural_ode` / `pnode` / `pnode_explicit` / `pnode_residual` / `pnode_energy` / `cope`。"""
    common = dict(
        num_nodes=bundle.total_n,
        num_corps=bundle.num_corps,
        input_dim=bundle.in_dim,
        hidden_dim=kw.hidden_dim,
        latent_dim=kw.latent_dim,
        initial_corp_vectors=bundle.init_vectors,
        link_score_mode=kw.link_score_mode,
        cosine_logit_scale=kw.cosine_logit_scale,
    )
    if key == "cope":
        if kw.time_dependent_potential:
            y0 = int(kw.year_min) if kw.year_min is not None else 2010
            y1 = int(kw.year_max) if kw.year_max is not None else 2020
            return UnifiedVGAETD(
                **common,
                year_min=y0,
                year_max=y1,
                w_pot_init=kw.w_pot_init,
            ).to(device)
        return UnifiedVGAE(
            **common,
            w_pot_init=kw.w_pot_init,
            density_calibrated_potential=kw.density_calibrated_potential,
            density_log_weight=kw.density_log_weight,
            density_ema_momentum=kw.density_ema_momentum,
        ).to(device)
    if key == "pnode_energy":
        from pnode_patent_runner.benchmark_pnode_energy_td import PNodeEnergyTD

        y_min = int(kw.year_min) if kw.year_min is not None else 2010
        y_max = int(kw.year_max) if kw.year_max is not None else 2020
        return PNodeEnergyTD(
            **common,
            year_min=y_min,
            year_max=y_max,
        ).to(device)
    y_min = int(kw.year_min) if kw.year_min is not None else 2010
    y_max = int(kw.year_max) if kw.year_max is not None else 2020
    return BenchmarkTemporalVGAE(
        **common,
        w_pot_init=0.0,
        variant=key,
        rnn_history_len=kw.rnn_history_len,
        pnode_history_len=int(kw.pnode_history_len),
        time_dependent_potential=bool(kw.time_dependent_potential)
        and key in ("pnode", "neural_ode"),
        year_min=y_min,
        year_max=y_max,
        pnode_ode_method=str(kw.pnode_ode_method),
        pnode_ode_n_steps=int(kw.pnode_ode_n_steps),
        pnode_potential_feature=str(kw.pnode_potential_feature),
        pnode_rff_frozen_basis=bool(kw.pnode_rff_frozen_basis),
        pnode_density_calibrated=bool(kw.pnode_density_calibrated),
        pnode_density_log_weight=float(kw.pnode_density_log_weight),
        pnode_density_ema_momentum=float(kw.pnode_density_ema_momentum),
        pnode_hist_fuse_mode=str(kw.pnode_hist_fuse_mode),
        pc_w_rho_init=float(kw.pc_w_rho_init),
        pc_w_delta_init=float(kw.pc_w_delta_init),
        pc_log_bandwidth_init=float(kw.pc_log_bandwidth_init),
    ).to(device)
