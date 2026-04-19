"""
時間依存ポテンシャル Φ(z, year)（年インデックス埋め込み）。

既存 CoPE の PotentialNet とは別系統。学習は unified_training_td / run_train_unified_vgae_td。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def calendar_year_to_index(year: int, year_min: int, year_max: int) -> int:
    y = int(year)
    if y < year_min or y > year_max:
        raise ValueError(f"year {y} out of [{year_min}, {year_max}]")
    return y - year_min


class TimeDependentPotentialNet(nn.Module):
    """
    Φ(z, year) ≈ MLP([sin(zB), cos(zB), emb(year_idx)])。
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        year_min: int,
        year_max: int,
        year_emb_dim: int = 16,
    ):
        super().__init__()
        if year_max < year_min:
            raise ValueError("year_max < year_min")
        self.latent_dim = int(latent_dim)
        self.year_min = int(year_min)
        self.year_max = int(year_max)
        self.num_years = self.year_max - self.year_min + 1
        self.year_emb_dim = int(year_emb_dim)

        self.B = nn.Parameter(torch.randn(latent_dim, hidden_dim // 2) * 3.0, requires_grad=False)
        in_feat = hidden_dim + self.year_emb_dim
        self.year_emb = nn.Embedding(self.num_years, self.year_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(in_feat, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def year_tensor(self, calendar_year: int, n: int, device: torch.device) -> torch.Tensor:
        idx = calendar_year_to_index(calendar_year, self.year_min, self.year_max)
        return torch.full((n,), idx, dtype=torch.long, device=device)

    def forward(self, z: torch.Tensor, year_idx: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.size(1) != self.latent_dim:
            raise ValueError(f"z must be (N, {self.latent_dim})")
        if year_idx.dim() != 1 or year_idx.size(0) != z.size(0):
            raise ValueError("year_idx must be (N,) matching z")
        proj = torch.matmul(z, self.B)
        trig = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        e = self.year_emb(year_idx.clamp(0, self.num_years - 1))
        x = torch.cat([trig, e], dim=-1)
        return self.net(x)

    def compute_potential_grid(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        calendar_year: int,
        resolution: int = 50,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = torch.linspace(x_range[0], x_range[1], resolution)
        y = torch.linspace(y_range[0], y_range[1], resolution)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
        n = grid_points.size(0)
        idx = calendar_year_to_index(calendar_year, self.year_min, self.year_max)
        year_idx = torch.full((n,), idx, dtype=torch.long, device=device)
        model_device = next(self.parameters()).device
        with torch.no_grad():
            potentials = self.forward(grid_points.to(model_device), year_idx.to(model_device))
        pot = potentials.view(resolution, resolution)
        return (
            X.cpu().numpy(),
            Y.cpu().numpy(),
            pot.cpu().numpy(),
        )


class GradientODEFuncTime(nn.Module):
    """dz/dτ = -tanh(scale) ∇_z Φ(z, year_fixed)。"""

    def __init__(self, potential_net_time: TimeDependentPotentialNet):
        super().__init__()
        self.potential_net = potential_net_time
        self.scale = nn.Parameter(torch.tensor(0.1))
        # odeint_adjoint の逆伝播でも forward が再呼ばれるため、テンソルを保持してはならない。
        # カレンダー年スカラーのみ保持し、各 forward で z のバッチ長に合わせて year_idx を再構築する。
        self._ode_calendar_year: Optional[int] = None

    def set_ode_calendar_year(self, calendar_year: int) -> None:
        self._ode_calendar_year = int(calendar_year)

    def forward(self, t, z):
        if self._ode_calendar_year is None:
            raise RuntimeError("call set_ode_calendar_year before odeint")
        pn = self.potential_net
        idx = calendar_year_to_index(self._ode_calendar_year, pn.year_min, pn.year_max)
        year_idx = torch.full(
            (z.shape[0],),
            idx,
            dtype=torch.long,
            device=z.device,
        )
        with torch.set_grad_enabled(True):
            z_in = z.detach().requires_grad_(True)
            phi = self.potential_net(z_in, year_idx)
            grad_z = torch.autograd.grad(phi.sum(), z_in, create_graph=True)[0]
        return -torch.tanh(self.scale) * grad_z

    def compute_gradient_field(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        calendar_year: int,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        pn = self.potential_net
        idx = calendar_year_to_index(calendar_year, pn.year_min, pn.year_max)
        grid_points = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1),
            dtype=torch.float32,
            device=device,
        )
        n = grid_points.size(0)
        year_idx = torch.full((n,), idx, dtype=torch.long, device=device)
        grid_points = grid_points.requires_grad_(True)
        phi = pn(grid_points, year_idx)
        gradients = torch.autograd.grad(phi.sum(), grid_points, create_graph=False)[0]
        grad_x = gradients[:, 0].view(X.shape).cpu().numpy()
        grad_y = gradients[:, 1].view(Y.shape).cpu().numpy()
        return grad_x, grad_y


class GradientODEFuncTimePure(nn.Module):
    """
    保守力場の最急降下: dz/dτ = -∇_z Φ(z, year_fixed)。
    `GradientODEFuncTime` の tanh ゲートを付けない版（摩擦なし・単位質量の勾配流）。
    """

    def __init__(self, potential_net_time: TimeDependentPotentialNet):
        super().__init__()
        self.potential_net = potential_net_time
        self._ode_calendar_year: Optional[int] = None

    def set_ode_calendar_year(self, calendar_year: int) -> None:
        self._ode_calendar_year = int(calendar_year)

    def forward(self, t, z):
        if self._ode_calendar_year is None:
            raise RuntimeError("call set_ode_calendar_year before odeint")
        pn = self.potential_net
        idx = calendar_year_to_index(self._ode_calendar_year, pn.year_min, pn.year_max)
        year_idx = torch.full(
            (z.shape[0],),
            idx,
            dtype=torch.long,
            device=z.device,
        )
        with torch.set_grad_enabled(True):
            z_in = z.detach().requires_grad_(True)
            phi = self.potential_net(z_in, year_idx)
            grad_z = torch.autograd.grad(phi.sum(), z_in, create_graph=True)[0]
        return -grad_z

    def compute_gradient_field(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        calendar_year: int,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """∇_z Φ(z, year)（`interactive_landscape_vector_field_td` 用。可視化は −∇Φ を表示）。"""
        pn = self.potential_net
        idx = calendar_year_to_index(calendar_year, pn.year_min, pn.year_max)
        grid_points = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1),
            dtype=torch.float32,
            device=device,
        )
        n = grid_points.size(0)
        year_idx = torch.full((n,), idx, dtype=torch.long, device=device)
        grid_points = grid_points.requires_grad_(True)
        phi = pn(grid_points, year_idx)
        gradients = torch.autograd.grad(phi.sum(), grid_points, create_graph=False)[0]
        grad_x = gradients[:, 0].view(X.shape).cpu().numpy()
        grad_y = gradients[:, 1].view(Y.shape).cpu().numpy()
        return grad_x, grad_y


class GradientNeuralODEPredictorEnergy(nn.Module):
    """
    Φ(z, y) に沿った純勾配流 ODE（`GradientODEFuncTimePure`）。
    P-NODE Energy 変種で、力学とリンク尤度で同一 Φ を共有するために用いる。
    """

    def __init__(self, latent_dim: int, hidden_dim: int, year_min: int, year_max: int):
        super().__init__()
        self.potential_net = TimeDependentPotentialNet(
            latent_dim, hidden_dim, year_min, year_max
        )
        self.ode_func = GradientODEFuncTimePure(self.potential_net)
        self.year_min = year_min
        self.year_max = year_max

    def forward(self, z_current: torch.Tensor, year_calendar_start: int, delta_t: float = 1.0):
        from torchdiffeq import odeint_adjoint as odeint

        device = z_current.device
        self.ode_func.set_ode_calendar_year(int(year_calendar_start))
        t_span = torch.tensor([0.0, delta_t], device=device)
        return odeint(self.ode_func, z_current, t_span, method="dopri5", rtol=1e-3, atol=1e-3)[-1]


class GradientNeuralODEPredictorTime(nn.Module):
    """Φ(z, y_start) に沿った 1 年先シフト。"""

    def __init__(self, latent_dim: int, hidden_dim: int, year_min: int, year_max: int):
        super().__init__()
        self.potential_net = TimeDependentPotentialNet(
            latent_dim, hidden_dim, year_min, year_max
        )
        self.ode_func = GradientODEFuncTime(self.potential_net)
        self.year_min = year_min
        self.year_max = year_max

    def forward(self, z_current: torch.Tensor, year_calendar_start: int, delta_t: float = 1.0):
        from torchdiffeq import odeint_adjoint as odeint

        device = z_current.device
        self.ode_func.set_ode_calendar_year(int(year_calendar_start))
        t_span = torch.tensor([0.0, delta_t], device=device)
        # 逆伝播で adjoint が ODE を再評価するため、年は ode_func 内スカラー保持（finally で消さない）
        return odeint(self.ode_func, z_current, t_span, method="dopri5", rtol=1e-3, atol=1e-3)[-1]
