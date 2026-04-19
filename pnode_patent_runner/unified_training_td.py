"""
UnifiedVGAETD 用の損失・学習ループ（年 y0, y1 を Φ(z,·) に明示）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from pnode_patent_runner.unified_training import (
    README_DEFAULT_BETA,
    README_DEFAULT_FUTURE_LINK_WEIGHT,
    README_DEFAULT_LATENT_PRED_WEIGHT,
    README_DEFAULT_NUM_NEG_FUTURE,
    README_DEFAULT_NUM_NEG_RECON,
    README_DEFAULT_POS_WEIGHT,
    README_DEFAULT_POTENTIAL_WEIGHT,
    README_DEFAULT_TRAJECTORY_WEIGHT,
)
from pnode_patent_runner.unified_vgae_td import UnifiedVGAETD


def sample_hard_negatives_td(
    model: UnifiedVGAETD,
    z: torch.Tensor,
    active_corps: torch.Tensor,
    active_patents: torch.Tensor,
    pos_set: Set[Tuple[int, int]],
    historical_edges: Set[Tuple[int, int]],
    calendar_year: int,
    num_samples: int = 400,
):
    """decode(z, ei, calendar_year) を使う負例サンプリング。"""
    if active_corps.numel() == 0 or active_patents.numel() == 0:
        return None
    device = z.device
    ac = active_corps.detach().cpu().tolist()
    ap = active_patents.detach().cpu().tolist()
    rng = np.random.default_rng()
    edges: List[List[int]] = []
    seen: Set[Tuple[int, int]] = set()
    max_tries = max(5000, num_samples * 80)
    tries = 0
    while len(edges) < num_samples and tries < max_tries:
        tries += 1
        c = int(rng.choice(ac))
        p = int(rng.choice(ap))
        key = (c, p)
        if key in pos_set or key in historical_edges or key in seen:
            continue
        seen.add(key)
        edges.append([c, p])
    if len(edges) < max(10, num_samples // 20):
        return None
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


def compute_loss_standardized_td(
    model: UnifiedVGAETD,
    data_t: Data,
    data_t1: Data,
    num_corps: int,
    z_history_for_prediction: List[torch.Tensor],
    historical_edges: Set[Tuple[int, int]],
    y0: int,
    y1: int,
    beta: float = README_DEFAULT_BETA,
    pos_weight: float = README_DEFAULT_POS_WEIGHT,
    latent_pred_weight: float = README_DEFAULT_LATENT_PRED_WEIGHT,
    future_link_weight: float = README_DEFAULT_FUTURE_LINK_WEIGHT,
    potential_weight: float = README_DEFAULT_POTENTIAL_WEIGHT,
    trajectory_weight: float = README_DEFAULT_TRAJECTORY_WEIGHT,
    num_neg_recon: int = README_DEFAULT_NUM_NEG_RECON,
    num_neg_future: int = README_DEFAULT_NUM_NEG_FUTURE,
    precomputed_z_mu_logvar: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = data_t.x.device
    pn = model.temporal_predictor.potential_net

    if precomputed_z_mu_logvar is not None:
        z_t, mu_t, logvar_t = precomputed_z_mu_logvar
    else:
        z_t, mu_t, logvar_t = model.encode(data_t.x, data_t.edge_index)
    with torch.no_grad():
        _, mu_t1_actual, _ = model.encode(data_t1.x, data_t1.edge_index)

    pos_edge_index = data_t.edge_index
    active_corps = torch.unique(pos_edge_index[0][pos_edge_index[0] < num_corps])
    active_patents = torch.unique(pos_edge_index[1][pos_edge_index[1] >= num_corps])
    pos_set = {tuple(p.tolist()) for p in pos_edge_index.t()}

    neg_edge_index = sample_hard_negatives_td(
        model,
        z_t,
        active_corps,
        active_patents,
        pos_set,
        historical_edges,
        y0,
        num_samples=num_neg_recon,
    )

    recon_loss = torch.tensor(0.0, device=device)
    if neg_edge_index is not None:
        pos_pred = model.decode(z_t, pos_edge_index, y0)
        neg_pred = model.decode(z_t, neg_edge_index, y0)
        node_degrees = torch.zeros(model.num_nodes, device=device)
        node_degrees.index_add_(
            0,
            data_t.edge_index.view(-1),
            torch.ones(data_t.edge_index.numel(), device=device),
        )
        edge_rarity = 1.0 / torch.log(
            node_degrees[pos_edge_index[0]] * node_degrees[pos_edge_index[1]] + 1e-15
        )
        recon_loss = -(torch.log(pos_pred + 1e-15) * edge_rarity).mean() * pos_weight - torch.log(
            1 - neg_pred + 1e-15
        ).mean()

    kl_loss = -0.5 * torch.mean(
        1
        + logvar_t[data_t.active_mask]
        - mu_t[data_t.active_mask].pow(2)
        - logvar_t[data_t.active_mask].exp()
    )

    potential_loss = torch.tensor(0.0, device=device)
    trajectory_loss = torch.tensor(0.0, device=device)

    yi0 = pn.year_tensor(y0, z_t.size(0), device)
    phi_z = pn(z_t, yi0)
    if phi_z.dim() > 1:
        phi_z = phi_z.squeeze(-1)

    if potential_weight > 0:
        potential_loss = 0.01 * (phi_z ** 2).mean()

    if trajectory_weight > 0 and data_t.active_mask.any():
        grad_z = torch.autograd.grad(
            phi_z.sum(), z_t, retain_graph=True, create_graph=True,
        )[0]
        v_theory = -grad_z
        am = data_t.active_mask
        delta = mu_t1_actual.detach() - z_t
        d = delta[am]
        v = v_theory[am]
        cos = F.cosine_similarity(
            F.normalize(d, dim=1, eps=1e-8),
            F.normalize(v, dim=1, eps=1e-8),
            dim=1,
        )
        trajectory_loss = (1.0 - cos).mean()

    z_t1_pred = model.predict_future(z_history_for_prediction, y0)
    latent_pred_loss = F.mse_loss(
        z_t1_pred[data_t1.active_mask], mu_t1_actual[data_t1.active_mask]
    )

    future_link_loss = torch.tensor(0.0, device=device)
    pos_edge_index_t1 = data_t1.edge_index
    if pos_edge_index_t1.size(1) > 0:
        active_c_t1 = torch.unique(pos_edge_index_t1[0][pos_edge_index_t1[0] < num_corps])
        active_p_t1 = torch.unique(pos_edge_index_t1[1][pos_edge_index_t1[1] >= num_corps])
        pos_set_t1 = {tuple(p.tolist()) for p in pos_edge_index_t1.t()}

        neg_t1 = sample_hard_negatives_td(
            model,
            z_t1_pred,
            active_c_t1,
            active_p_t1,
            pos_set_t1,
            historical_edges,
            y1,
            num_samples=num_neg_future,
        )
        if neg_t1 is not None:
            future_link_loss = -torch.log(
                model.decode(z_t1_pred, pos_edge_index_t1, y1) + 1e-15
            ).mean() * pos_weight - torch.log(1 - model.decode(z_t1_pred, neg_t1, y1) + 1e-15).mean()

    weighted_kl = beta * kl_loss
    weighted_latent = latent_pred_weight * latent_pred_loss
    weighted_future = future_link_weight * future_link_loss
    weighted_potential = potential_weight * potential_loss
    weighted_trajectory = trajectory_weight * trajectory_loss

    total_loss = (
        recon_loss
        + weighted_kl
        + weighted_latent
        + weighted_future
        + weighted_potential
        + weighted_trajectory
    )

    breakdown = {
        "total": float(total_loss.item()),
        "recon": float(recon_loss.item()),
        "kl": float(weighted_kl.item()),
        "latent_pred": float(weighted_latent.item()),
        "future_link": float(weighted_future.item()),
        "potential": float(weighted_potential.item()),
        "trajectory": float(weighted_trajectory.item()),
    }
    return total_loss, breakdown


def future_link_auc_scores_td(
    model: UnifiedVGAETD,
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
        z_pred = model.predict_future(z_hist, year_prev)

    ei = data_next.edge_index.to(device)
    mask = (ei[0] < num_corps) & (ei[1] >= num_corps)
    pos_ei = ei[:, mask]
    if pos_ei.size(1) == 0:
        return np.array([0, 1]), np.array([0.5, 0.5])

    n_pos = min(max_pos, pos_ei.size(1))
    perm = torch.randperm(pos_ei.size(1), device=device)[:n_pos]
    pos_ei = pos_ei[:, perm]

    active_c = torch.unique(pos_ei[0])
    active_p = torch.unique(pos_ei[1])
    pos_set = {tuple(pos_ei[:, i].tolist()) for i in range(pos_ei.size(1))}

    neg_list = []
    rng = np.random.default_rng(0)
    ac = active_c.cpu().numpy()
    ap = active_p.cpu().numpy()
    tries = 0
    while len(neg_list) < n_pos * neg_ratio and tries < n_pos * neg_ratio * 50:
        tries += 1
        c = int(rng.choice(ac))
        p = int(rng.choice(ap))
        if (c, p) in pos_set:
            continue
        neg_list.append([c, p])
    if len(neg_list) < 4:
        return np.array([0, 1]), np.array([0.5, 0.5])

    neg_ei = torch.tensor(neg_list[: n_pos * neg_ratio], dtype=torch.long, device=device).t()

    with torch.no_grad():
        s_pos = model.decode(z_pred, pos_ei, year_next)
        s_neg = model.decode(z_pred, neg_ei, year_next)

    y_true = np.concatenate([np.ones(s_pos.numel()), np.zeros(s_neg.numel())])
    y_score = np.concatenate([s_pos.cpu().numpy(), s_neg.cpu().numpy()])
    return y_true, y_score


def evaluate_val_auc_td(
    model: UnifiedVGAETD,
    graphs: Dict[int, Data],
    num_corps: int,
    device: torch.device,
) -> float:
    from sklearn.metrics import roc_auc_score

    years = sorted(graphs.keys())
    if len(years) < 2:
        return float("nan")
    y0, y1 = years[-2], years[-1]
    yt, yscore = future_link_auc_scores_td(
        model, graphs, num_corps, device, year_prev=y0, year_next=y1
    )
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, yscore))


def evaluate_val_future_link_metrics_td(
    model: UnifiedVGAETD,
    graphs: Dict[int, Data],
    num_corps: int,
    device: torch.device,
) -> Dict[str, float]:
    """最終2年の future-link について ROC-AUC と AP（`evaluate_val_future_link_metrics` と同型）。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    years = sorted(graphs.keys())
    if len(years) < 2:
        return {"auc": float("nan"), "ap": float("nan")}
    y0, y1 = years[-2], years[-1]
    yt, yscore = future_link_auc_scores_td(
        model, graphs, num_corps, device, year_prev=y0, year_next=y1
    )
    if len(np.unique(yt)) < 2:
        return {"auc": float("nan"), "ap": float("nan")}
    return {
        "auc": float(roc_auc_score(yt, yscore)),
        "ap": float(average_precision_score(yt, yscore)),
    }


def evaluate_val_future_link_metrics_for_years_td(
    model: UnifiedVGAETD,
    graphs: Dict[int, Data],
    num_corps: int,
    device: torch.device,
    year_prev: int,
    year_next: int,
) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if year_prev not in graphs or year_next not in graphs:
        return {"auc": float("nan"), "ap": float("nan")}
    yt, yscore = future_link_auc_scores_td(
        model,
        graphs,
        num_corps,
        device,
        year_prev=year_prev,
        year_next=year_next,
    )
    if len(np.unique(yt)) < 2:
        return {"auc": float("nan"), "ap": float("nan")}
    return {
        "auc": float(roc_auc_score(yt, yscore)),
        "ap": float(average_precision_score(yt, yscore)),
    }


def train_one_epoch_td(
    model: UnifiedVGAETD,
    graphs: Dict[int, Data],
    num_corps: int,
    hist_edges: Set[Tuple[int, int]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_kw: Dict[str, Any],
) -> Dict[str, float]:
    model.train()
    years = sorted(graphs.keys())
    pairs = list(zip(years[:-1], years[1:]))
    agg = {k: 0.0 for k in ("total", "recon", "kl", "latent_pred", "future_link", "potential", "trajectory")}
    n = 0
    history_len = int(getattr(model, "temporal_history_len", 1))
    for y0, y1 in pairs:
        data_t = graphs[y0].to(device)
        data_t1 = graphs[y1].to(device)
        idx0 = years.index(y0)
        start_idx = max(0, idx0 - history_len + 1)
        z_hist: List[torch.Tensor] = []
        for j in range(start_idx, idx0):
            d = graphs[years[j]].to(device)
            with torch.no_grad():
                zj, _, _ = model.encode(d.x, d.edge_index)
            z_hist.append(zj)
        z_t, mu_t, logvar_t = model.encode(data_t.x, data_t.edge_index)
        z_hist.append(z_t)
        optimizer.zero_grad()
        loss, br = compute_loss_standardized_td(
            model,
            data_t,
            data_t1,
            num_corps,
            z_hist,
            hist_edges,
            y0,
            y1,
            precomputed_z_mu_logvar=(z_t, mu_t, logvar_t),
            **loss_kw,
        )
        loss.backward()
        optimizer.step()
        for k in agg:
            agg[k] += br.get(k, 0.0)
        n += 1
    if n == 0:
        return agg
    return {k: v / n for k, v in agg.items()}


def train_model_td(
    model: UnifiedVGAETD,
    graphs: Dict[int, Data],
    num_corps: int,
    hist_edges: Set[Tuple[int, int]],
    num_epochs: int = 20,
    potential_weight: float = README_DEFAULT_POTENTIAL_WEIGHT,
    trajectory_weight: float = README_DEFAULT_TRAJECTORY_WEIGHT,
    lr: float = 0.001,
    num_neg_recon: int = README_DEFAULT_NUM_NEG_RECON,
    num_neg_future: int = README_DEFAULT_NUM_NEG_FUTURE,
    latent_pred_weight: float = README_DEFAULT_LATENT_PRED_WEIGHT,
    future_link_weight: float = README_DEFAULT_FUTURE_LINK_WEIGHT,
    beta: float = README_DEFAULT_BETA,
    pos_weight: float = README_DEFAULT_POS_WEIGHT,
):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_kw: Dict[str, Any] = dict(
        beta=beta,
        pos_weight=pos_weight,
        latent_pred_weight=latent_pred_weight,
        future_link_weight=future_link_weight,
        potential_weight=potential_weight,
        trajectory_weight=trajectory_weight,
        num_neg_recon=num_neg_recon,
        num_neg_future=num_neg_future,
    )
    history: Dict[str, Any] = {"loss": [], "val_auc": []}
    _comp_keys = ("recon", "kl", "latent_pred", "future_link", "potential", "trajectory")
    history["train_components"] = {k: [] for k in _comp_keys}
    last_breakdown: Optional[Dict[str, float]] = None
    best_auc = 0.0
    for _ in range(num_epochs):
        tr = train_one_epoch_td(
            model, graphs, num_corps, hist_edges, optimizer, device, loss_kw
        )
        last_breakdown = dict(tr)
        va = evaluate_val_auc_td(model, graphs, num_corps, device)
        history["loss"].append(tr["total"])
        history["val_auc"].append(va)
        for k in _comp_keys:
            history["train_components"][k].append(float(tr.get(k, 0.0)))
        if not np.isnan(va):
            best_auc = max(best_auc, va)
    model.eval()
    if last_breakdown is not None:
        history["last_epoch_breakdown"] = last_breakdown
    return model, 0.0, best_auc, history
