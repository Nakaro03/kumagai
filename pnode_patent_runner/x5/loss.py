"""X5 4-term composite loss with LOTO masking.

   L = L_predict + λ_phys · L_phys + λ_geom · L_geom + λ_smooth · L_smooth

L_predict  : Sinkhorn between SDE rollout and observed marginals; the masked
             timepoint contributes with weight α (0 -> 1 over warmup).
L_phys     : Φ-anchor at centroids (Φ(c_kj, t=j) ≈ -g_norm_kj).
L_geom     : path energy ∫‖drift‖² dt approximated at the rollout snapshots.
L_smooth   : ‖∂Φ/∂t‖² at sampled (z, t) grid points (extrapolation regularizer).
"""
from __future__ import annotations

from typing import Dict

import torch


def sinkhorn_call(loss_fn, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Uniform-weight Sinkhorn wrapper."""
    a = torch.ones(x.shape[0], device=x.device); a = a / a.sum()
    b = torch.ones(y.shape[0], device=y.device); b = b / b.sum()
    a.requires_grad_(); b.requires_grad_()
    return loss_fn(a, x, b, y)


def loss_predict(
    *,
    ot_loss,
    rollout_z: list[torch.Tensor],     # rollout[i] corresponds to ts[i]; ts = [0]+train_t
    observed: dict[int, torch.Tensor], # observed[j] = batch sampled at timepoint j
    train_t: list[int],
    mask_t: int | None,
    alpha: float,
) -> torch.Tensor:
    """Σ_j w_j · Sinkhorn(rollout(j), observed(j)) with mask_t having weight α."""
    total = rollout_z[0].new_zeros(())
    for i, j in enumerate(train_t):
        x = rollout_z[i + 1][:, :-1]  # strip trailing zero column
        y = observed[j]
        w = float(alpha) if (mask_t is not None and j == mask_t) else 1.0
        total = total + w * sinkhorn_call(ot_loss, x, y)
    return total / max(len(train_t), 1)


def loss_phys(
    *,
    model,
    centroids: dict[int, torch.Tensor],
    growth_norm: dict[int, torch.Tensor],
    train_t: list[int],
    device,
) -> torch.Tensor:
    """Φ-anchor: Σ_{j,k}(Φ(c_kj, t=j) + g_norm_kj)^2."""
    accum = []
    for j in train_t:
        c = centroids[j].to(device)
        g = growth_norm[j].to(device)
        active = c.abs().sum(-1) > 1e-6
        if active.sum() < 2:
            continue
        c_a = c[active].detach()  # anchor target only — no grad through c
        g_a = g[active]
        t_col = torch.full((c_a.shape[0], 1), float(j), device=device)
        xt = torch.cat([c_a, t_col], dim=1)
        phi = model._func._pot(xt).squeeze(-1)
        accum.append(((phi + g_a) ** 2).mean())
    if not accum:
        return torch.zeros((), device=device)
    return torch.stack(accum).mean()


def loss_geom(
    *,
    model,
    rollout_z: list[torch.Tensor],
    train_t: list[int],
    device,
) -> torch.Tensor:
    """Path energy: mean ‖∇_z Φ‖² along the rollout."""
    accum = []
    for i, j in enumerate(train_t):
        z = rollout_z[i + 1][:, :-1].detach().requires_grad_(True)
        t_col = torch.full((z.shape[0], 1), float(j), device=device)
        xt = torch.cat([z, t_col], dim=1)
        phi = model._func._pot(xt)
        grad_z = torch.autograd.grad(phi, xt, torch.ones_like(phi), create_graph=True)[0][:, :-1]
        accum.append(grad_z.pow(2).sum(-1).mean())
    if not accum:
        return torch.zeros((), device=device)
    return torch.stack(accum).mean()


def loss_smooth(
    *,
    model,
    z_sample: torch.Tensor,        # (M, x_dim) latent sample
    t_dense: torch.Tensor,         # (T_dense,) timepoints
    device,
) -> torch.Tensor:
    """(M·T_dense) grid of Φ(z, t); penalize finite-diff in t."""
    M = z_sample.shape[0]
    Td = t_dense.shape[0]
    z_rep = z_sample.unsqueeze(1).expand(-1, Td, -1).reshape(M * Td, -1)
    t_rep = t_dense.unsqueeze(0).expand(M, -1).reshape(M * Td, 1)
    xt = torch.cat([z_rep, t_rep], dim=1)
    phi = model._func._pot(xt).reshape(M, Td)
    return phi.diff(dim=1).pow(2).mean()


def composite_loss(
    *,
    model,
    ot_loss,
    rollout_z: list[torch.Tensor],
    observed: dict[int, torch.Tensor],
    centroids: dict[int, torch.Tensor],
    growth_norm: dict[int, torch.Tensor],
    train_t: list[int],
    mask_t: int | None,
    alpha: float,
    lam_phys: float,
    lam_geom: float,
    lam_smooth: float,
    z_sample: torch.Tensor,
    t_dense: torch.Tensor,
    device,
) -> tuple[torch.Tensor, Dict[str, float]]:
    L_pred = loss_predict(ot_loss=ot_loss, rollout_z=rollout_z,
                          observed=observed, train_t=train_t,
                          mask_t=mask_t, alpha=alpha)
    L_phys = loss_phys(model=model, centroids=centroids, growth_norm=growth_norm,
                       train_t=train_t, device=device)
    L_geom = loss_geom(model=model, rollout_z=rollout_z, train_t=train_t, device=device)
    L_smooth = loss_smooth(model=model, z_sample=z_sample, t_dense=t_dense, device=device)

    L = L_pred + lam_phys * L_phys + lam_geom * L_geom + lam_smooth * L_smooth
    comp = {
        "predict": float(L_pred.item()),
        "phys":    float(L_phys.item()),
        "geom":    float(L_geom.item()),
        "smooth":  float(L_smooth.item()),
        "alpha":   float(alpha),
        "mask_t":  -1 if mask_t is None else int(mask_t),
    }
    return L, comp
