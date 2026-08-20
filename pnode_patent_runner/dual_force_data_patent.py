"""
Patent (CPC) ドメイン用の Dual-Force/TAP-NODE データローダー。

`cope_experiment.load_bipartite_domain_graph_bundle` が返す CopeGraphBundle に、
`dual_force_data.load_dual_force_bundle`（arXiv 著者–トピック用）と同じインターフェース
(`.topic_mass` / `.topic_trend_plus` / `.topic_trend_minus`) を各年の Data に付与する。
burst 特徴は `gem_skill.py` の定義（対数モメンタムの年内80パーセンタイル）と完全に同一にし、
`docs/DUAL_FORCE_REDESIGN.md` / `docs/FILING_COUNT_FORECAST_DESIGN.md` の burst_j(t) を共有する。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from pnode_patent_runner.cope_experiment import CopeGraphBundle, load_bipartite_domain_graph_bundle

BURST_PERCENTILE = 80.0


def _topic_degree(data: Data, num_corps: int, n_items: int) -> np.ndarray:
    ei = data.edge_index.cpu()
    mask = (ei[0] < num_corps) & (ei[1] >= num_corps)
    items = (ei[1, mask] - num_corps).numpy()
    return np.bincount(items, minlength=n_items).astype(float)


def attach_topic_dynamics(
    bundle: CopeGraphBundle, burst_percentile: float = BURST_PERCENTILE
) -> CopeGraphBundle:
    """年ごとに mass / trend_plus / trend_minus / burst を計算し、各年の Data に付与する（in-place）。

    burst_j(t) は `gem_skill.YearContext` と同一定義:
    mom_j(t) = log1p(mass_j(t)) - log1p(mass_j(t-1))、
    burst_j(t) = 1[mom_j(t) >= 80th percentile of positive mom_j(t) within year]。
    t 以前（<=t）の情報のみを使うためリークなし。
    """
    years = sorted(bundle.graphs.keys())
    n_items = (
        len(bundle.right_nodes)
        if bundle.right_nodes is not None
        else int(max(int(bundle.graphs[y].edge_index[1].max()) for y in years) - bundle.num_corps + 1)
    )
    deg_by_year: Dict[int, np.ndarray] = {
        y: _topic_degree(bundle.graphs[y], bundle.num_corps, n_items) for y in years
    }
    for k, y in enumerate(years):
        mass = deg_by_year[y]
        if k == 0:
            d = np.zeros(n_items)
            mom = np.zeros(n_items)
        else:
            mass_prev = deg_by_year[years[k - 1]]
            d = mass - mass_prev
            mom = np.log1p(mass) - np.log1p(mass_prev)
        pos_mom = mom[mom > 0]
        thr = np.percentile(pos_mom, burst_percentile) if pos_mom.size else np.inf
        burst = (mom >= thr).astype(float)

        data = bundle.graphs[y]
        data.topic_mass = torch.tensor(mass, dtype=torch.float32)
        data.topic_trend_plus = torch.tensor(np.clip(d, 0.0, None), dtype=torch.float32)
        data.topic_trend_minus = torch.tensor(np.clip(-d, 0.0, None), dtype=torch.float32)
        data.topic_burst = torch.tensor(burst, dtype=torch.float32)
    return bundle


def load_dual_force_bundle_patent_domain(
    csv_path: str,
    *,
    year_range: Tuple[int, int],
    min_events: int = 2,
    year_min: int = 2000,
    year_max: int = 2021,
    burst_percentile: float = BURST_PERCENTILE,
) -> CopeGraphBundle:
    bundle = load_bipartite_domain_graph_bundle(
        csv_path,
        min_events=min_events,
        year_min=year_min,
        year_max=year_max,
        year_range=year_range,
    )
    return attach_topic_dynamics(bundle, burst_percentile=burst_percentile)
