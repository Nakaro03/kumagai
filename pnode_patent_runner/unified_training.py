"""
UnifiedVGAE（CoPE-VGAE）用の損失・負例サンプリング・短い学習ループ。

損失の既定係数は README_COPE.md「損失関数」と一致させる。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# README_COPE.md に記載の既定（`compute_loss_standardized` 引数既定と同一）
README_DEFAULT_BETA = 0.01
README_DEFAULT_POS_WEIGHT = 5.0
README_DEFAULT_LATENT_PRED_WEIGHT = 1.0
README_DEFAULT_FUTURE_LINK_WEIGHT = 10.0
README_DEFAULT_POTENTIAL_WEIGHT = 0.01
README_DEFAULT_TRAJECTORY_WEIGHT = 0.05
README_DEFAULT_NUM_NEG_RECON = 800
README_DEFAULT_NUM_NEG_FUTURE = 400
README_DEFAULT_DENSITY_ALIGN_WEIGHT = 0.0
# pnode_explicit: 左パーティションの A(z)=exp(-Phi) を次数ベース正規化目標に近づける
README_DEFAULT_ATTENTION_GROUND_WEIGHT = 0.5
# Φ 正則の形（l2=従来 0.01·mean(Φ²) 相当の中身）
README_DEFAULT_POTENTIAL_REG_MODE = "l2"
# 軌道損失: 差分の基準 z 再パラム vs μ、損失形、||∇Φ|| 下限ペナ
README_DEFAULT_TRAJECTORY_DELTA_SOURCE = "z"
README_DEFAULT_TRAJECTORY_LOSS_TYPE = "cosine"
README_DEFAULT_TRAJECTORY_GRAD_FLOOR = 0.0
README_DEFAULT_TRAJECTORY_GRAD_FLOOR_WEIGHT = 0.0
# L_pot 内部スケール（従来互換・TD 損失からも参照可）
POTENTIAL_REG_INNER_SCALE = 0.01


def loss_aux_ramp_factor(epoch_idx: int, warmup_epochs: int) -> float:
    """
    λ_pot・λ_traj の線形ランプ係数（0〜1）。
    warmup_epochs<=0 は常に 1。N>1 のとき epoch 0 で 0、epoch N-1 で 1。
    N==1 はランプ無し（常に 1）として従来挙動に近づける。
    """
    if warmup_epochs <= 0:
        return 1.0
    if warmup_epochs == 1:
        return 1.0
    return min(1.0, float(epoch_idx) / float(warmup_epochs - 1))


def decode_bipartite(
    model: torch.nn.Module,
    z: torch.Tensor,
    edge_index: torch.Tensor,
    calendar_year: Optional[int] = None,
) -> torch.Tensor:
    """PNodeEnergyTD 等は decode に暦年が必要。"""
    if calendar_year is not None and getattr(model, "decode_requires_calendar_year", False):
        return model.decode(z, edge_index, int(calendar_year))
    return model.decode(z, edge_index)


def sample_hard_negatives_v2(
    model: torch.nn.Module,
    z: torch.Tensor,
    active_corps: torch.Tensor,
    active_patents: torch.Tensor,
    pos_set: Set[Tuple[int, int]],
    historical_edges: Set[Tuple[int, int]],
    num_samples: int = 400,
) -> Optional[torch.Tensor]:
    """二部の (企業idx, 特許idx) をランダムに選び、正例・履歴に無い負例エッジを返す。"""
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


def compute_loss_standardized(
    model: torch.nn.Module,
    data_t: Data,
    data_t1: Data,
    num_corps: int,
    z_history_for_prediction: List[torch.Tensor],
    historical_edges: Set[Tuple[int, int]],
    beta: float = README_DEFAULT_BETA,
    pos_weight: float = README_DEFAULT_POS_WEIGHT,
    latent_pred_weight: float = README_DEFAULT_LATENT_PRED_WEIGHT,
    future_link_weight: float = README_DEFAULT_FUTURE_LINK_WEIGHT,
    potential_weight: float = README_DEFAULT_POTENTIAL_WEIGHT,
    trajectory_weight: float = README_DEFAULT_TRAJECTORY_WEIGHT,
    num_neg_recon: int = README_DEFAULT_NUM_NEG_RECON,
    num_neg_future: int = README_DEFAULT_NUM_NEG_FUTURE,
    precomputed_z_mu_logvar: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    y0: Optional[int] = None,
    y1: Optional[int] = None,
    density_align_weight: float = README_DEFAULT_DENSITY_ALIGN_WEIGHT,
    attention_ground_weight: float = 0.0,
    potential_reg_mode: str = README_DEFAULT_POTENTIAL_REG_MODE,
    trajectory_delta_source: str = README_DEFAULT_TRAJECTORY_DELTA_SOURCE,
    trajectory_loss_type: str = README_DEFAULT_TRAJECTORY_LOSS_TYPE,
    trajectory_grad_floor: float = README_DEFAULT_TRAJECTORY_GRAD_FLOOR,
    trajectory_grad_floor_weight: float = README_DEFAULT_TRAJECTORY_GRAD_FLOOR_WEIGHT,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = data_t.x.device

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

    neg_edge_index = sample_hard_negatives_v2(
        model,
        z_t,
        active_corps,
        active_patents,
        pos_set,
        historical_edges,
        num_samples=num_neg_recon,
    )

    recon_loss = torch.tensor(0.0, device=device)
    if neg_edge_index is not None:
        pos_pred = decode_bipartite(model, z_t, pos_edge_index, y0)
        neg_pred = decode_bipartite(model, z_t, neg_edge_index, y0)
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
    grad_phi_l2_for_breakdown = 0.0
    has_potential = hasattr(model.temporal_predictor, "potential_net")
    td_phi = bool(getattr(model, "pnode_energy_td", False)) or (
        getattr(model, "time_dependent_potential", False)
        and getattr(model, "variant", None) == "pnode"
    )

    if has_potential:
        pn = model.temporal_predictor.potential_net
        if td_phi:
            if y0 is None:
                raise ValueError("y0 is required for time-dependent P-NODE / PNodeEnergyTD loss")
            yi0 = pn.year_tensor(int(y0), z_t.size(0), device)
            phi_z = pn(z_t, yi0)
        else:
            phi_z = pn(z_t)
        if phi_z.dim() > 1:
            phi_z = phi_z.squeeze(-1)

        if potential_weight > 0:
            prm = (potential_reg_mode or "l2").strip().lower()
            if prm == "l2":
                potential_loss = POTENTIAL_REG_INNER_SCALE * (phi_z ** 2).mean()
            elif prm in ("log1p_sq", "log1p", "log"):
                potential_loss = POTENTIAL_REG_INNER_SCALE * torch.log(
                    1.0 + phi_z**2
                ).mean()
            elif prm in ("centered_l2", "center_l2", "variance_l2"):
                pc = phi_z - phi_z.mean()
                potential_loss = POTENTIAL_REG_INNER_SCALE * (pc**2).mean()
            else:
                raise ValueError(
                    f"unknown potential_reg_mode: {potential_reg_mode!r} "
                    f"(use l2, log1p_sq, centered_l2)"
                )

        if trajectory_weight > 0 and data_t.active_mask.any():
            grad_z = torch.autograd.grad(
                phi_z.sum(), z_t, retain_graph=True, create_graph=True,
            )[0]
            v_theory = -grad_z
            am = data_t.active_mask
            tds = (trajectory_delta_source or "z").strip().lower()
            if tds in ("z", "reparam", "sample"):
                anchor = z_t
            elif tds in ("mu", "mean", "encoder"):
                anchor = mu_t.detach()
            else:
                raise ValueError(
                    f"unknown trajectory_delta_source: {trajectory_delta_source!r} "
                    f"(use z, mu)"
                )
            delta = mu_t1_actual.detach() - anchor
            d = delta[am]
            v = v_theory[am]
            grad_phi_l2_for_breakdown = float(
                v.norm(p=2, dim=1).mean().detach().item()
            )
            du = F.normalize(d, dim=1, eps=1e-8)
            vu = F.normalize(v, dim=1, eps=1e-8)
            cos = F.cosine_similarity(du, vu, dim=1)
            tls = (trajectory_loss_type or "cosine").strip().lower()
            if tls in ("cosine", "cos"):
                trajectory_loss = (1.0 - cos).mean()
            elif tls in ("smooth1mcos", "smooth_l1_mcos", "smooth_cos"):
                trajectory_loss = F.smooth_l1_loss(
                    1.0 - cos, torch.zeros_like(cos), beta=0.1, reduction="mean"
                )
            elif tls in ("huber_vec", "vec_smooth_l1", "smoothed_l1_dir"):
                trajectory_loss = F.smooth_l1_loss(
                    du, vu, beta=0.1, reduction="mean"
                )
            else:
                raise ValueError(
                    f"unknown trajectory_loss_type: {trajectory_loss_type!r} "
                    f"(use cosine, smooth1mcos, huber_vec)"
                )
            tgf = float(trajectory_grad_floor)
            tgw = float(trajectory_grad_floor_weight)
            if tgw > 0.0 and tgf > 0.0:
                gn = v.norm(p=2, dim=1)
                trajectory_loss = trajectory_loss + tgw * (
                    F.relu(tgf - gn) ** 2
                ).mean()

    attention_ground_loss = torch.tensor(0.0, device=device)
    if (
        attention_ground_weight > 0.0
        and getattr(model, "variant", None) == "pnode_explicit"
        and has_potential
    ):
        dcnt = torch.bincount(
            data_t.edge_index.view(-1), minlength=model.num_nodes
        ).float()
        phi_1d = phi_z if phi_z.dim() == 1 else phi_z.squeeze(-1)
        A_pred = torch.exp(-phi_1d)
        am = data_t.active_mask
        n_left = int(num_corps)
        if n_left > 0 and am is not None and am.size(0) >= n_left:
            left_sel = am[:n_left]
            if bool(left_sel.any()):
                t_log = torch.log(1.0 + dcnt[:n_left])
                v_sub = t_log[left_sel]
                t_tgt = torch.zeros(n_left, device=device, dtype=t_log.dtype)
                t_tgt[left_sel] = (v_sub - v_sub.min()) / (
                    v_sub.max() - v_sub.min() + 1e-8
                )
                attention_ground_loss = F.mse_loss(
                    A_pred[:n_left][left_sel], t_tgt[left_sel]
                )

    if getattr(model, "pnode_energy_td", False):
        if y0 is None:
            raise ValueError("y0 is required for PNodeEnergyTD predict_future")
        z_t1_pred = model.predict_future(z_history_for_prediction, int(y0))
    elif getattr(model, "time_dependent_potential", False) and getattr(model, "variant", None) == "pnode":
        if y0 is None:
            raise ValueError("y0 is required for time-dependent P-NODE predict_future")
        z_t1_pred = model.predict_future(z_history_for_prediction, int(y0))
    elif getattr(model, "time_dependent_neural_ode", False) and getattr(model, "variant", None) == "neural_ode":
        if y0 is None:
            raise ValueError("y0 is required for time-dependent Neural ODE predict_future")
        z_t1_pred = model.predict_future(z_history_for_prediction, int(y0))
    else:
        z_t1_pred = model.predict_future(z_history_for_prediction)
    latent_pred_loss = F.mse_loss(
        z_t1_pred[data_t1.active_mask], mu_t1_actual[data_t1.active_mask]
    )

    future_link_loss = torch.tensor(0.0, device=device)
    pos_edge_index_t1 = data_t1.edge_index
    if pos_edge_index_t1.size(1) > 0:
        active_c_t1 = torch.unique(pos_edge_index_t1[0][pos_edge_index_t1[0] < num_corps])
        active_p_t1 = torch.unique(pos_edge_index_t1[1][pos_edge_index_t1[1] >= num_corps])
        pos_set_t1 = {tuple(p.tolist()) for p in pos_edge_index_t1.t()}

        neg_t1 = sample_hard_negatives_v2(
            model,
            z_t1_pred,
            active_c_t1,
            active_p_t1,
            pos_set_t1,
            historical_edges,
            num_samples=num_neg_future,
        )
        if neg_t1 is not None:
            future_link_loss = -torch.log(
                decode_bipartite(model, z_t1_pred, pos_edge_index_t1, y1) + 1e-15
            ).mean() * pos_weight - torch.log(
                1 - decode_bipartite(model, z_t1_pred, neg_t1, y1) + 1e-15
            ).mean()

    weighted_kl = beta * kl_loss
    weighted_latent = latent_pred_weight * latent_pred_loss
    weighted_future = future_link_weight * future_link_loss
    weighted_potential = potential_weight * potential_loss
    weighted_trajectory = trajectory_weight * trajectory_loss

    density_align_loss = torch.tensor(0.0, device=device)
    if density_align_weight > 0 and has_potential:
        from pnode_patent_runner.models import PopulationCoupledPotentialNet
        pn_da = model.temporal_predictor.potential_net
        if isinstance(pn_da, PopulationCoupledPotentialNet):
            kde_f = pn_da.kde_field
            if kde_f._anchors_t is not None and kde_f._anchors_t.size(0) > 0:
                anchors = kde_f._anchors_t
                n_sample = min(64, anchors.size(0))
                idx_da = torch.randperm(anchors.size(0), device=device)[:n_sample]
                z_sample = anchors[idx_da].detach().requires_grad_(False)
                phi_at_sample = pn_da(z_sample)
                if phi_at_sample.dim() > 1:
                    phi_at_sample = phi_at_sample.squeeze(-1)
                log_rho_at_sample = kde_f.log_density(z_sample)
                density_align_loss = ((phi_at_sample + log_rho_at_sample) ** 2).mean()
    weighted_density_align = density_align_weight * density_align_loss
    weighted_attention_ground = attention_ground_weight * attention_ground_loss

    total_loss = (
        recon_loss
        + weighted_kl
        + weighted_latent
        + weighted_future
        + weighted_potential
        + weighted_trajectory
        + weighted_density_align
        + weighted_attention_ground
    )

    breakdown = {
        "total": float(total_loss.item()),
        "recon": float(recon_loss.item()),
        "kl": float(weighted_kl.item()),
        "latent_pred": float(weighted_latent.item()),
        "future_link": float(weighted_future.item()),
        "potential": float(weighted_potential.item()),
        "trajectory": float(weighted_trajectory.item()),
        "density_align": float(weighted_density_align.item()),
        "attention_ground": float(weighted_attention_ground.item()),
        "grad_phi_l2": float(grad_phi_l2_for_breakdown),
    }
    return total_loss, breakdown


def _future_link_pos_perm(
    year_prev: int,
    year_next: int,
    max_pos: int,
    neg_ratio: int,
    n_pos_edges: int,
    device: torch.device,
) -> torch.Tensor:
    """
    future-link 評価の正例サブサンプルを、呼び出し順に依存せず
    ``(year_prev, year_next, max_pos, neg_ratio, 候補本数)`` で再現可能にする。
    """
    seed = (
        (int(year_prev) % 500_000) * 1_000_003
        + (int(year_next) % 500_000) * 917_521
        + int(max_pos) * 13
        + int(neg_ratio) * 29
        + int(n_pos_edges) * 97
    ) % (2**31 - 1)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return torch.randperm(n_pos_edges, device=device, generator=gen)


def rollout_z_pred_multistep(
    model: torch.nn.Module,
    z_hist: List[torch.Tensor],
    years_sorted: List[int],
    idx_start: int,
    n_steps: int,
) -> torch.Tensor:
    """
    ``z_hist`` は ``years_sorted[idx_start]`` 年までエンコードした潜在（末尾がその年）。
    キー列上で ``n_steps`` ステップ先まで ``predict_future`` を連鎖する（学習は 1 ステップでも評価は多ホライズン可）。
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if idx_start + n_steps >= len(years_sorted):
        raise ValueError("rollout exceeds year range")
    variant = getattr(model, "variant", None)

    if getattr(model, "pnode_energy_td", False):
        z = z_hist[-1]
        for k in range(n_steps):
            cal = int(years_sorted[idx_start + k])
            z = model.predict_future([z], cal)
        return z

    if model.__class__.__name__ == "UnifiedVGAETD":
        z = z_hist[-1]
        for k in range(n_steps):
            cal = int(years_sorted[idx_start + k])
            if k == 0:
                z = model.predict_future(z_hist, cal)
            else:
                z = model.predict_future([z], cal)
        return z

    if variant == "rnn":
        hist = list(z_hist)
        z_out: Optional[torch.Tensor] = None
        for k in range(n_steps):
            z_out = model.predict_future(hist)
            if k < n_steps - 1:
                hist = hist[1:] + [z_out]
        assert z_out is not None
        return z_out

    if variant == "pnode" and getattr(model, "time_dependent_potential", False):
        z = z_hist[-1]
        for k in range(n_steps):
            cal = int(years_sorted[idx_start + k])
            if k == 0:
                z = model.predict_future(z_hist, cal)
            else:
                z = model.predict_future([z], cal)
        return z

    if variant == "neural_ode" and getattr(model, "time_dependent_neural_ode", False):
        z = z_hist[-1]
        for k in range(n_steps):
            cal = int(years_sorted[idx_start + k])
            if k == 0:
                z = model.predict_future(z_hist, cal)
            else:
                z = model.predict_future([z], cal)
        return z

    if variant == "pnode_pc" and hasattr(model.temporal_predictor, "set_population"):
        model.temporal_predictor.set_population(z_hist[-1].detach())

    z = model.predict_future(z_hist)
    for _ in range(1, n_steps):
        z = model.predict_future([z])
    return z


def future_link_auc_scores(
    model: torch.nn.Module,
    graphs: Dict[int, Data],
    num_corps: int,
    device: torch.device,
    year_prev: int,
    year_next: int,
    max_pos: int = 1500,
    neg_ratio: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ``year_prev`` までの履歴から ``year_next`` まで **sorted(graphs.keys()) 上のインデックス差分**だけ
    ロールアウトし、``year_next`` の二部リンクでラベル・スコアを返す（差分 1 なら従来の 1 年先）。
    """
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
        z_pred = rollout_z_pred_multistep(model, z_hist, years_sorted, idx0, n_steps)

    ei = data_next.edge_index.to(device)
    mask = (ei[0] < num_corps) & (ei[1] >= num_corps)
    pos_ei = ei[:, mask]
    if pos_ei.size(1) == 0:
        return np.array([0, 1]), np.array([0.5, 0.5])

    n_pos = min(max_pos, pos_ei.size(1))
    perm = _future_link_pos_perm(
        year_prev, year_next, max_pos, neg_ratio, pos_ei.size(1), device
    )[:n_pos]
    pos_ei = pos_ei[:, perm]

    active_c = torch.unique(pos_ei[0])
    active_p = torch.unique(pos_ei[1])
    pos_set = {tuple(pos_ei[:, i].tolist()) for i in range(pos_ei.size(1))}

    neg_list = []
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
        if (c, p) in pos_set:
            continue
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


# future-link 指標の ECE は decode の logit に対する sigmoid 確率の等幅ビン期待較正誤差
FUTURE_LINK_ECE_N_BINS = 15


def binary_ece_from_logits(
    y_true: np.ndarray,
    logits: np.ndarray,
    n_bins: int = FUTURE_LINK_ECE_N_BINS,
) -> float:
    """二値ラベルとリンク logit に対する ECE（Guo et al. 風の等幅ビン）。"""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    logits = np.asarray(logits, dtype=np.float64).ravel()
    if y_true.size == 0 or logits.size != y_true.size:
        return float("nan")
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = float(y_true.size)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        conf = float(prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def future_link_metrics_from_scores(
    yt: np.ndarray,
    yscore: np.ndarray,
    n_bins: int = FUTURE_LINK_ECE_N_BINS,
) -> Dict[str, float]:
    """`future_link_auc_scores` が返す logit スコアから ROC-AUC / AP / ECE をまとめて返す。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    yt = np.asarray(yt).ravel()
    yscore = np.asarray(yscore).ravel()
    if yt.size == 0 or len(np.unique(yt)) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "ece": float("nan")}
    return {
        "auc": float(roc_auc_score(yt, yscore)),
        "ap": float(average_precision_score(yt, yscore)),
        "ece": float(binary_ece_from_logits(yt, yscore, n_bins=n_bins)),
    }


def train_one_epoch(
    model: torch.nn.Module,
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
    agg = {
        k: 0.0
        for k in (
            "total",
            "recon",
            "kl",
            "latent_pred",
            "future_link",
            "potential",
            "trajectory",
            "density_align",
            "attention_ground",
            "grad_phi_l2",
        )
    }
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
        pn = getattr(model.temporal_predictor, "potential_net", None)
        if pn is not None and hasattr(pn, "update_from_mu"):
            pn.update_from_mu(mu_t.detach(), getattr(data_t, "active_mask", None))
        if hasattr(model.temporal_predictor, "set_population"):
            mu_prev_for_pc: Optional[torch.Tensor] = None
            am_prev: Optional[torch.Tensor] = None
            if start_idx < idx0:
                d_prev = graphs[years[start_idx]].to(device)
                with torch.no_grad():
                    _, mu_prev_for_pc, _ = model.encode(d_prev.x, d_prev.edge_index)
                am_prev = getattr(d_prev, "active_mask", None)
            model.temporal_predictor.set_population(
                mu_t.detach(),
                mu_prev_for_pc,
                active_mask_t=getattr(data_t, "active_mask", None),
                active_mask_prev=am_prev,
            )
        optimizer.zero_grad()
        loss, br = compute_loss_standardized(
            model,
            data_t,
            data_t1,
            num_corps,
            z_hist,
            hist_edges,
            precomputed_z_mu_logvar=(z_t, mu_t, logvar_t),
            y0=y0,
            y1=y1,
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


def evaluate_val_future_link_metrics(
    model: torch.nn.Module,
    graphs: Dict[int, Data],
    num_corps: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    最終2年 (y_{T-1}→y_T) の future-link 予測について ROC-AUC・AP・ECE（logit→sigmoid 上の等幅ビン）を返す。
    不均衡二部リンクで PR 曲線の要約として AP が有用な場合がある。
    """
    years = sorted(graphs.keys())
    if len(years) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "ece": float("nan")}
    y0, y1 = years[-2], years[-1]
    yt, yscore = future_link_auc_scores(
        model, graphs, num_corps, device, year_prev=y0, year_next=y1
    )
    return future_link_metrics_from_scores(yt, yscore)


def evaluate_val_future_link_metrics_for_years(
    model: torch.nn.Module,
    graphs: Dict[int, Data],
    num_corps: int,
    device: torch.device,
    year_prev: int,
    year_next: int,
) -> Dict[str, float]:
    """
    指定した ``(year_prev, year_next)`` の future-link について ROC-AUC / AP / ECE を返す。
    ホールドアウト最終年の評価などに使う（``graphs`` に両方の年が含まれること）。
    """
    if year_prev not in graphs or year_next not in graphs:
        return {"auc": float("nan"), "ap": float("nan"), "ece": float("nan")}
    yt, yscore = future_link_auc_scores(
        model,
        graphs,
        num_corps,
        device,
        year_prev=year_prev,
        year_next=year_next,
    )
    return future_link_metrics_from_scores(yt, yscore)


def evaluate_future_link_metrics_for_horizon_gaps(
    model: torch.nn.Module,
    graphs: Dict[int, Data],
    num_corps: int,
    device: torch.device,
    horizon_gaps: List[int],
    year_next: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    ``sorted(graphs.keys())`` 上の **インデックス差** ``k`` ごとに future-link 指標を返す。

    - ``year_next`` を省略すると末尾キーを終端年とする。
    - 各 ``k`` は ``years_sorted[idx_T - k]`` から ``year_next``（インデックス ``idx_T``）までの
      ``k`` ステップロールアウトで予測し、終端年の観測二部リンクで評価する。
    """
    nanm = {"auc": float("nan"), "ap": float("nan"), "ece": float("nan")}
    years_sorted = sorted(graphs.keys())
    if not years_sorted or not horizon_gaps:
        return {str(k): dict(nanm) for k in horizon_gaps}
    yT = year_next if year_next is not None else years_sorted[-1]
    if yT not in graphs:
        return {str(k): dict(nanm) for k in horizon_gaps}
    idx_T = years_sorted.index(yT)
    out: Dict[str, Dict[str, float]] = {}
    for k in sorted(set(int(x) for x in horizon_gaps)):
        if k < 1 or idx_T < k:
            out[str(k)] = dict(nanm)
            continue
        y_prev = years_sorted[idx_T - k]
        yt, ys = future_link_auc_scores(
            model, graphs, num_corps, device, year_prev=y_prev, year_next=yT
        )
        out[str(k)] = future_link_metrics_from_scores(yt, ys)
    return out


def latent_rollout_tensors_for_year_gap(
    model: torch.nn.Module,
    graphs: Dict[int, Data],
    device: torch.device,
    year_prev: int,
    year_next: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    ``future_link_auc_scores`` と同じ履歴構築・``rollout_z_pred_multistep`` で ``year_prev`` から
    ``year_next`` までロールアウトした ``z_pred`` と、観測終端年の教師 ``mu_next`` を返す。

    教師はエンコーダ平均 ``mu``（評価時 ``reparameterize`` は決定的に ``mu``）。
    ``active_mask`` は終端年 ``graphs[year_next]`` のものを使用（学習の ``latent_pred`` と整合）。
    """
    model.eval()
    years_sorted = sorted(graphs.keys())
    if year_prev not in graphs or year_next not in graphs:
        return None
    idx0 = years_sorted.index(year_prev)
    idx1 = years_sorted.index(year_next)
    n_steps = idx1 - idx0
    if n_steps < 1:
        return None
    history_len = int(getattr(model, "temporal_history_len", 1))
    start_idx = max(0, idx0 - history_len + 1)
    data_prev = graphs[year_prev]
    data_next = graphs[year_next]
    with torch.no_grad():
        z_hist: List[torch.Tensor] = []
        for j in range(start_idx, idx0):
            d = graphs[years_sorted[j]].to(device)
            zj, _, _ = model.encode(d.x, d.edge_index)
            z_hist.append(zj)
        dp = data_prev.to(device)
        z, _, _ = model.encode(dp.x, dp.edge_index)
        z_hist.append(z)
        z_pred = rollout_z_pred_multistep(model, z_hist, years_sorted, idx0, n_steps)
        dn = data_next.to(device)
        _, mu_next, _ = model.encode(dn.x, dn.edge_index)
        _, mu_prev, _ = model.encode(dp.x, dp.edge_index)
        z_prev = z_hist[-1]
        mask = getattr(dn, "active_mask", None)
        if mask is None:
            mask = torch.ones(dn.num_nodes, dtype=torch.bool, device=device)
        else:
            mask = mask.to(device)
    return {
        "z_pred": z_pred,
        "mu_next": mu_next,
        "mu_prev": mu_prev,
        "z_prev": z_prev,
        "mask": mask,
    }


def latent_pointwise_metrics_from_tensors(
    z_pred: torch.Tensor,
    mu_next: torch.Tensor,
    mu_prev: torch.Tensor,
    z_prev: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """
    - **mse / mae**: ``active_mask`` 上のノードについて ``z_pred`` と ``mu_next`` の全次元平均。
    - **direction_agreement**: 各ノードで ``g=||·||_2`` とし、
      ``sign(g(mu_next)-g(mu_prev))`` と ``sign(g(z_pred)-g(z_prev))`` が一致する割合
      （``g(mu_next)-g(mu_prev)==0`` のノードは分母から除外）。
    - **n_pred_delta_zero**: 予測側差分が 0 のノード数（診断用、分母には含めない）。
    """
    if not mask.any():
        nanv = float("nan")
        return {
            "mse": nanv,
            "mae": nanv,
            "direction_agreement": nanv,
            "n_masked_nodes": 0.0,
            "n_direction_defined": 0.0,
            "n_pred_delta_zero": 0.0,
        }
    m = mask.bool()
    zp = z_pred[m]
    mu_t = mu_next[m]
    mu_p = mu_prev[m]
    z_p = z_prev[m]
    mse = float(torch.mean((zp - mu_t) ** 2).item())
    mae = float(F.l1_loss(zp, mu_t).item())
    g_mu_t = torch.norm(mu_t, p=2, dim=1)
    g_mu_p = torch.norm(mu_p, p=2, dim=1)
    g_z_t = torch.norm(zp, p=2, dim=1)
    g_z_p = torch.norm(z_p, p=2, dim=1)
    d_true = g_mu_t - g_mu_p
    d_pred = g_z_t - g_z_p
    defined = d_true != 0
    n_def = int(defined.sum().item())
    n_pred_zero = int((d_pred == 0).sum().item())
    if n_def == 0:
        d_acc = float("nan")
    else:
        agree = ((torch.sign(d_true) == torch.sign(d_pred)) & defined).sum().item()
        d_acc = float(agree) / float(n_def)
    return {
        "mse": mse,
        "mae": mae,
        "direction_agreement": d_acc,
        "n_masked_nodes": float(m.sum().item()),
        "n_direction_defined": float(n_def),
        "n_pred_delta_zero": float(n_pred_zero),
    }


def evaluate_latent_rollout_metrics_for_horizon_gaps(
    model: torch.nn.Module,
    graphs: Dict[int, Data],
    num_corps: int,
    device: torch.device,
    horizon_gaps: List[int],
    year_next: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    ``evaluate_future_link_metrics_for_horizon_gaps`` と同じ ``(y_prev, y_T)`` のペアで、
    **潜在ロールアウト vs 教師 μ** の MSE / MAE / トレンド方向一致率を返す。

    - ``year_next`` 省略時は ``sorted(graphs.keys())[-1]`` を終端年 ``y_T`` とする。
    - 各 ``k`` は ``years_sorted[idx_T-k]`` から ``y_T`` まで ``k`` ステップロールアウト（リンク評価と同型）。
    """
    nanm = {
        "mse": float("nan"),
        "mae": float("nan"),
        "direction_agreement": float("nan"),
        "n_masked_nodes": float("nan"),
        "n_direction_defined": float("nan"),
        "n_pred_delta_zero": float("nan"),
    }
    years_sorted = sorted(graphs.keys())
    if not years_sorted or not horizon_gaps:
        return {str(k): dict(nanm) for k in horizon_gaps}
    yT = year_next if year_next is not None else years_sorted[-1]
    if yT not in graphs:
        return {str(k): dict(nanm) for k in horizon_gaps}
    idx_T = years_sorted.index(yT)
    out: Dict[str, Dict[str, float]] = {}
    for k in sorted(set(int(x) for x in horizon_gaps)):
        if k < 1 or idx_T < k:
            out[str(k)] = dict(nanm)
            continue
        y_prev = years_sorted[idx_T - k]
        tdict = latent_rollout_tensors_for_year_gap(
            model, graphs, device, year_prev=y_prev, year_next=yT
        )
        if tdict is None:
            out[str(k)] = dict(nanm)
            continue
        out[str(k)] = latent_pointwise_metrics_from_tensors(
            tdict["z_pred"],
            tdict["mu_next"],
            tdict["mu_prev"],
            tdict["z_prev"],
            tdict["mask"],
        )
    return out


def evaluate_val_auc(
    model: torch.nn.Module,
    graphs: Dict[int, Data],
    num_corps: int,
    device: torch.device,
) -> float:
    return float(evaluate_val_future_link_metrics(model, graphs, num_corps, device)["auc"])


def train_model_improved(
    model: torch.nn.Module,
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
    loss_aux_warmup_epochs: int = 0,
    density_align_weight: float = README_DEFAULT_DENSITY_ALIGN_WEIGHT,
    attention_ground_weight: float = 0.0,
    potential_reg_mode: str = README_DEFAULT_POTENTIAL_REG_MODE,
    trajectory_delta_source: str = README_DEFAULT_TRAJECTORY_DELTA_SOURCE,
    trajectory_loss_type: str = README_DEFAULT_TRAJECTORY_LOSS_TYPE,
    trajectory_grad_floor: float = README_DEFAULT_TRAJECTORY_GRAD_FLOOR,
    trajectory_grad_floor_weight: float = README_DEFAULT_TRAJECTORY_GRAD_FLOOR_WEIGHT,
) -> Tuple[torch.nn.Module, float, float, Dict[str, Any]]:
    """
    README_COPE.md の損失既定で `train_one_epoch` を回す短い学習ループ。
    `run_interactive_landscape_cope_vector_field` の `--epochs` からも利用。
    戻り値の MRR スロットは未実装のため 0.0。
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_kw_base: Dict[str, Any] = dict(
        beta=beta,
        pos_weight=pos_weight,
        latent_pred_weight=latent_pred_weight,
        future_link_weight=future_link_weight,
        num_neg_recon=num_neg_recon,
        num_neg_future=num_neg_future,
        density_align_weight=density_align_weight,
        attention_ground_weight=float(attention_ground_weight),
        potential_reg_mode=str(potential_reg_mode),
        trajectory_delta_source=str(trajectory_delta_source),
        trajectory_loss_type=str(trajectory_loss_type),
        trajectory_grad_floor=float(trajectory_grad_floor),
        trajectory_grad_floor_weight=float(trajectory_grad_floor_weight),
    )
    history: Dict[str, Any] = {"loss": [], "val_auc": []}
    _comp_keys = (
        "recon",
        "kl",
        "latent_pred",
        "future_link",
        "potential",
        "trajectory",
        "density_align",
        "attention_ground",
        "grad_phi_l2",
    )
    history["train_components"] = {k: [] for k in _comp_keys}
    last_breakdown: Optional[Dict[str, float]] = None
    best_auc = 0.0
    for ep in range(num_epochs):
        r = loss_aux_ramp_factor(ep, int(loss_aux_warmup_epochs))
        loss_kw: Dict[str, Any] = {
            **loss_kw_base,
            "potential_weight": float(potential_weight) * r,
            "trajectory_weight": float(trajectory_weight) * r,
        }
        tr = train_one_epoch(
            model, graphs, num_corps, hist_edges, optimizer, device, loss_kw
        )
        last_breakdown = dict(tr)
        va = evaluate_val_auc(model, graphs, num_corps, device)
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
