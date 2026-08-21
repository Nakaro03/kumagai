"""
Dual-Force VGAE 用: 年ごとに `Data` を ODE に渡す future-link 評価（`unified_training` と同じ手続き）。

`future_link_auc_scores` の実装に合わせ、唯一ロールアウト部だけ `predict_future(z_hist, data_t[year])` 連鎖に差し替える。
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from pnode_patent_runner.unified_training import (
    _future_link_pos_perm,
    decode_bipartite,
    future_link_metrics_from_scores,
)


def rollout_z_pred_dual_force(
    model: torch.nn.Module,
    z_hist: List[torch.Tensor],
    years_sorted: List[int],
    idx_start: int,
    n_steps: int,
    graphs: Dict[int, Data],
    device: torch.device,
) -> torch.Tensor:
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if idx_start + n_steps >= len(years_sorted):
        raise ValueError("rollout exceeds year range")
    z: torch.Tensor = z_hist[-1]
    for k in range(n_steps):
        yk = int(years_sorted[idx_start + k])
        d_k = graphs[yk].to(device, non_blocking=True)
        if k == 0:
            z = model.predict_future(z_hist, d_k)
        else:
            z = model.predict_future([z], d_k)
    return z


def future_link_auc_scores_dual_force(
    model: torch.nn.Module,
    graphs: Dict[int, Data],
    num_corps: int,
    device: torch.device,
    year_prev: int,
    year_next: int,
    max_pos: int = 1500,
    neg_ratio: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    years_sorted = sorted(graphs.keys())
    if year_prev not in graphs or year_next not in graphs:
        return np.array([0, 1]), np.array([0.5, 0.5])
    idx0 = years_sorted.index(year_prev)
    idx1 = years_sorted.index(year_next)
    n_steps = idx1 - idx0
    if n_steps < 1:
        return np.array([0, 1]), np.array([0.5, 0.5])
    history_len = int(getattr(model, "temporal_history_len", 1))
    start_idx = max(0, idx0 - history_len + 1)
    data_next = graphs[year_next]
    with torch.no_grad():
        z_hist: List[torch.Tensor] = []
        for j in range(start_idx, idx0):
            d = graphs[years_sorted[j]].to(device)
            zj, _, _ = model.encode(d.x, d.edge_index)
            z_hist.append(zj)
        data_prev = graphs[year_prev].to(device)
        z, _, _ = model.encode(data_prev.x, data_prev.edge_index)
        z_hist.append(z)
        z_pred = rollout_z_pred_dual_force(
            model, z_hist, years_sorted, idx0, n_steps, graphs, device
        )

    ei = data_next.edge_index.to(device)
    mask = (ei[0] < num_corps) & (ei[1] >= num_corps)
    pos_ei_full = ei[:, mask]
    if pos_ei_full.size(1) == 0:
        return np.array([0, 1]), np.array([0.5, 0.5])

    # 負例棄却は年内の全正例（サブサンプル前）に対して行う。理由は
    # unified_training.future_link_auc_scores と同一（2026-08-21 修正）。
    full_pos_set = {tuple(pos_ei_full[:, i].tolist()) for i in range(pos_ei_full.size(1))}

    n_pos = min(max_pos, pos_ei_full.size(1))
    perm = _future_link_pos_perm(
        year_prev, year_next, max_pos, neg_ratio, pos_ei_full.size(1), device
    )[:n_pos]
    pos_ei = pos_ei_full[:, perm]

    active_c = torch.unique(pos_ei[0])
    active_p = torch.unique(pos_ei[1])

    neg_list = []
    neg_set = set()
    neg_seed = (
        (int(year_prev) % 500_000) * 800_009 + (int(year_next) % 500_000) * 400_009 + n_pos * 11
    ) % (2**32 - 1)
    rng = np.random.default_rng(int(neg_seed))
    ac = active_c.cpu().numpy()
    ap = active_p.cpu().numpy()
    tries = 0
    while len(neg_list) < n_pos * neg_ratio and tries < n_pos * neg_ratio * 50:
        tries += 1
        c = int(rng.choice(ac))
        p = int(rng.choice(ap))
        if (c, p) in full_pos_set or (c, p) in neg_set:
            continue
        neg_set.add((c, p))
        neg_list.append([c, p])
    if len(neg_list) < 4:
        return np.array([0, 1]), np.array([0.5, 0.5])

    neg_ei = torch.tensor(neg_list[: n_pos * neg_ratio], dtype=torch.long, device=device).t()

    with torch.no_grad():
        s_pos = decode_bipartite(model, z_pred, pos_ei, year_next)
        s_neg = decode_bipartite(model, z_pred, neg_ei, year_next)

    y_true = np.concatenate([np.ones(s_pos.numel()), np.zeros(s_neg.numel())])
    y_score = np.concatenate([s_pos.cpu().numpy(), s_neg.cpu().numpy()])
    return y_true, y_score


def evaluate_dual_force_future_link_metrics(
    model: torch.nn.Module,
    graphs: Dict[int, Data],
    num_corps: int,
    device: torch.device,
) -> Dict[str, float]:
    years = sorted(graphs.keys())
    if len(years) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "ece": float("nan")}
    y0, y1 = years[-2], years[-1]
    yt, yscore = future_link_auc_scores_dual_force(
        model, graphs, num_corps, device, y0, y1
    )
    return future_link_metrics_from_scores(yt, yscore)
