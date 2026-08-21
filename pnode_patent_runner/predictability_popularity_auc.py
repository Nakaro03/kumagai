#!/usr/bin/env python3
"""
学習モデルと同一の future-link 評価ペア上で、訓練不要スコアの AUC を測る（検証 C）。

- popularity: 遷移元年のアイテム次数 log1p(deg_i(t))
- seen_before: 学習期間に (u, i) が観測済みなら 1（記憶ベースライン）
- seen+pop: 上記の辞書式結合

評価ペアの生成は `unified_training.future_link_auc_scores` /
`dual_force_eval.future_link_auc_scores_dual_force` と同一（正例 perm は年キー決定的、
負例は年シード付き乱数）なので、学習モデルの holdout AUC と直接比較できる。

例:
  python -m pnode_patent_runner.predictability_popularity_auc \\
    --output-json pnode_patent_runner/outputs/predictability_map/popularity_auc.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

from pnode_patent_runner.unified_training import (
    _future_link_pos_perm,
    future_link_metrics_from_scores,
)


def eval_pairs(
    graphs: Dict[int, "object"],
    num_corps: int,
    year_prev: int,
    year_next: int,
    max_pos: int = 1500,
    neg_ratio: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """future_link_auc_scores と同一手続きで (pos_ei, neg_ei) を返す。"""
    device = torch.device("cpu")
    data_next = graphs[year_next]
    ei = data_next.edge_index.cpu()
    mask = (ei[0] < num_corps) & (ei[1] >= num_corps)
    pos_ei_full = ei[:, mask]
    # 負例棄却は年内の全正例（サブサンプル前）に対して行う。理由は
    # unified_training.future_link_auc_scores と同一（2026-08-21 修正）。
    full_pos_set = {tuple(pos_ei_full[:, i].tolist()) for i in range(pos_ei_full.size(1))}

    n_pos = min(max_pos, pos_ei_full.size(1))
    perm = _future_link_pos_perm(year_prev, year_next, max_pos, neg_ratio, pos_ei_full.size(1), device)[:n_pos]
    pos_ei = pos_ei_full[:, perm]

    active_c = torch.unique(pos_ei[0])
    active_p = torch.unique(pos_ei[1])
    neg_seed = (
        (int(year_prev) % 500_000) * 800_009 + (int(year_next) % 500_000) * 400_009 + n_pos * 11
    ) % (2**32 - 1)
    rng = np.random.default_rng(int(neg_seed))
    ac = active_c.numpy()
    ap = active_p.numpy()
    neg_list = []
    neg_set = set()
    tries = 0
    while len(neg_list) < n_pos * neg_ratio and tries < n_pos * neg_ratio * 50:
        tries += 1
        c = int(rng.choice(ac))
        p = int(rng.choice(ap))
        if (c, p) in full_pos_set or (c, p) in neg_set:
            continue
        neg_set.add((c, p))
        neg_list.append([c, p])
    neg_ei = torch.tensor(neg_list, dtype=torch.long).t()
    return pos_ei, neg_ei


def score_pairs(
    pos_ei: torch.Tensor,
    neg_ei: torch.Tensor,
    item_deg: np.ndarray,
    hist_pairs: Set[Tuple[int, int]],
) -> Dict[str, Dict[str, float]]:
    y = np.concatenate([np.ones(pos_ei.size(1)), np.zeros(neg_ei.size(1))])
    all_ei = torch.cat([pos_ei, neg_ei], dim=1)
    pop = np.log1p(item_deg[all_ei[1].numpy()])
    seen = np.array([1.0 if (int(a), int(t)) in hist_pairs else 0.0 for a, t in all_ei.t().tolist()])
    out = {}
    for name, s in [("popularity", pop), ("seen_before", seen), ("seen_plus_pop", seen * 1e6 + pop)]:
        out[name] = future_link_metrics_from_scores(y, s)
    return out


def run_domain(graphs, num_corps, train_years: List[int], year_prev: int, year_next: int):
    deg = None
    g_prev = graphs[year_prev]
    ei = g_prev.edge_index.cpu()
    mask = (ei[0] < num_corps) & (ei[1] >= num_corps)
    n_nodes = int(max(int(ei.max()) + 1, num_corps + 1))
    deg = np.zeros(n_nodes)
    np.add.at(deg, ei[1, mask].numpy(), 1.0)

    hist: Set[Tuple[int, int]] = set()
    for y in train_years:
        eiy = graphs[y].edge_index.cpu()
        m = (eiy[0] < num_corps) & (eiy[1] >= num_corps)
        hist |= {tuple(p) for p in eiy[:, m].t().tolist()}

    pos_ei, neg_ei = eval_pairs(graphs, num_corps, year_prev, year_next)
    return score_pairs(pos_ei, neg_ei, deg, hist)


def main() -> int:
    p = argparse.ArgumentParser(description="training-free popularity/memorization AUC on identical eval pairs")
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument(
        "--patent-domains", type=str, nargs="+", default=["agrifood", "construction", "energy"]
    )
    args = p.parse_args()

    results = {}

    # author_topic: 2022–2025, holdout 遷移 2024→2025
    from pnode_patent_runner.dual_force_data import load_dual_force_bundle

    b = load_dual_force_bundle(
        "data/processed/arxiv_cs_embedded_2020-2026_full.csv", topic_column="topic", min_papers=5
    )
    graphs = {y: g for y, g in b.graphs.items() if 2022 <= y <= 2025}
    results["author_topic"] = run_domain(graphs, b.num_corps, [2022, 2023, 2024], 2024, 2025)
    print("author_topic done")

    # 特許 bipartite: 2017–2021, holdout 遷移 2020→2021
    from pnode_patent_runner.cope_experiment import load_bipartite_domain_graph_bundle

    for dom in args.patent_domains:
        bb = load_bipartite_domain_graph_bundle(
            f"data/processed/bipartite_{dom}.csv", year_range=(2017, 2021)
        )
        graphs = {y: g for y, g in bb.graphs.items() if 2017 <= y <= 2021}
        results[dom] = run_domain(graphs, bb.num_corps, [2017, 2018, 2019, 2020], 2020, 2021)
        print(f"{dom} done")

    out = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "same eval pairs as learned-model holdout eval (deterministic pos perm + year-seeded negatives)",
        "results": results,
    }
    oj = Path(args.output_json)
    oj.parent.mkdir(parents=True, exist_ok=True)
    with open(oj, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {oj}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    main()
