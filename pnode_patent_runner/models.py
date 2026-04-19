"""SharedVGAEEncoder / PotentialNet / 勾配流 ODE 予測子（UnifiedVGAE から利用）。"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SharedVGAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels * 2, heads=2, concat=False)
        self.conv2 = GATConv(hidden_channels * 2, hidden_channels, heads=2, concat=False)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=1, concat=False)
        self.conv_mu = GATConv(hidden_channels, out_channels, heads=1, concat=False)
        self.conv_logvar = GATConv(hidden_channels, out_channels, heads=1, concat=False)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels * 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


class PotentialNet(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.B = nn.Parameter(torch.randn(latent_dim, hidden_dim // 2) * 3.0, requires_grad=False)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        proj = torch.matmul(z, self.B)
        x = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(x)

    def compute_potential_grid(self, x_range, y_range, resolution=50, device="cpu"):
        x = torch.linspace(x_range[0], x_range[1], resolution)
        y = torch.linspace(y_range[0], y_range[1], resolution)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
        model_device = next(self.parameters()).device
        with torch.no_grad():
            potentials = self.forward(grid_points.to(model_device))
        return (
            X.cpu().numpy(),
            Y.cpu().numpy(),
            potentials.view(resolution, resolution).cpu().numpy(),
        )


class HistoricalDiagonalLogProb(nn.Module):
    """
    エンコーダ出力 μ の EMA 統計に基づく対角ガウス密度の log p(z)。

    案1（密度校準ポテンシャル）の最小実装: 履歴的に「よく現れる」潜在方向は分散が小さく
    log p が高くなり、Φ に -log p を足すと盆地として解釈しやすい。
    """

    def __init__(self, latent_dim: int, momentum: float = 0.05, min_var: float = 1e-3):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.momentum = float(momentum)
        self.min_var = float(min_var)
        self.register_buffer("running_mean", torch.zeros(self.latent_dim))
        self.register_buffer("running_var", torch.ones(self.latent_dim))
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, mu: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        """バッチの μ（encoder の平均）で移動平均を更新。mask があればそのノードのみ。"""
        if mu.dim() != 2 or mu.size(1) != self.latent_dim:
            return
        x = mu
        if mask is not None:
            if mask.dim() != 1 or mask.size(0) != mu.size(0):
                return
            if not mask.any():
                return
            x = mu[mask]
        if x.size(0) == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False).clamp_min(self.min_var)
        if int(self.num_updates.item()) == 0:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var)
        else:
            m = self.momentum
            self.running_mean.mul_(1.0 - m).add_(batch_mean * m)
            self.running_var.mul_(1.0 - m).add_(batch_var * m)
        self.running_var.clamp_(min=self.min_var)
        self.num_updates += 1

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """log p(z) under diagonal Gaussian N(running_mean, diag(running_var)). Shape (N,)."""
        if z.dim() != 2 or z.size(1) != self.latent_dim:
            raise ValueError(f"z must be (N, {self.latent_dim}), got {tuple(z.shape)}")
        if int(self.num_updates.item()) == 0:
            return torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        diff = z - self.running_mean.to(device=z.device, dtype=z.dtype)
        inv_var = 1.0 / self.running_var.to(device=z.device, dtype=z.dtype)
        # log N(z | μ, diag(σ²)) = -1/2 Σ_d [(z_d-μ_d)²/σ_d² + log(2πσ_d²)]
        logp = -0.5 * (
            (diff ** 2 * inv_var).sum(dim=-1)
            + torch.log(2.0 * math.pi * self.running_var.to(device=z.device, dtype=z.dtype)).sum()
        )
        return logp


class CalibratedPotentialNet(nn.Module):
    """
    Φ(z) ≈ φ_nn(z) - w * log p_hist(z)  （学習可能な φ_nn と履歴密度項）。

    ODE・デコーダは従来どおり ``potential_net`` 経由で同一 Φ を共有する。
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        log_density_weight_init: float = 1.0,
        momentum: float = 0.05,
        min_var: float = 1e-3,
    ):
        super().__init__()
        self.nn_pot = PotentialNet(latent_dim, hidden_dim)
        self.log_density = HistoricalDiagonalLogProb(
            latent_dim, momentum=momentum, min_var=min_var
        )
        self.log_density_weight = nn.Parameter(torch.tensor(float(log_density_weight_init)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.nn_pot(z)
        if raw.dim() > 1:
            raw = raw.squeeze(-1)
        logp = self.log_density(z)
        phi = raw - self.log_density_weight * logp
        return phi.unsqueeze(-1)

    @torch.no_grad()
    def update_from_mu(self, mu: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        self.log_density.update(mu, mask)

    def compute_potential_grid(self, x_range, y_range, resolution=50, device="cpu"):
        """latent_dim==2 または 2 次元グリッド + 残り次元は running_mean で固定。"""
        ld = int(self.nn_pot.B.shape[0])
        x = torch.linspace(x_range[0], x_range[1], resolution)
        y = torch.linspace(y_range[0], y_range[1], resolution)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid2 = torch.stack([X.flatten(), Y.flatten()], dim=1)
        if ld == 2:
            grid_points = grid2
        else:
            rest = self.log_density.running_mean[2:].to(device=device).unsqueeze(0).expand(
                grid2.size(0), -1
            )
            grid_points = torch.cat([grid2.to(device), rest], dim=1)
        model_device = next(self.parameters()).device
        with torch.no_grad():
            potentials = self.forward(grid_points.to(dtype=torch.float32, device=model_device))
        pot = potentials.view(resolution, resolution)
        return (
            X.cpu().numpy(),
            Y.cpu().numpy(),
            pot.cpu().numpy(),
        )


class GradientODEFunc(nn.Module):
    def __init__(self, potential_net):
        super().__init__()
        self.potential_net = potential_net
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, t, z):
        with torch.set_grad_enabled(True):
            z_in = z.detach().requires_grad_(True)
            phi = self.potential_net(z_in)
            grad_z = torch.autograd.grad(phi.sum(), z_in, create_graph=True)[0]
        return -torch.tanh(self.scale) * grad_z

    def compute_gradient_field(self, X, Y, device="cpu"):
        grid_points = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1),
            dtype=torch.float32,
            device=device,
        )
        grid_points.requires_grad_(True)
        phi = self.potential_net(grid_points)
        gradients = torch.autograd.grad(phi.sum(), grid_points, create_graph=False)[0]
        grad_x = gradients[:, 0].view(X.shape).cpu().numpy()
        grad_y = gradients[:, 1].view(Y.shape).cpu().numpy()
        return grad_x, grad_y


class GradientNeuralODEPredictor(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        density_calibrated: bool = False,
        density_log_weight: float = 1.0,
        density_momentum: float = 0.05,
    ):
        super().__init__()
        if density_calibrated:
            self.potential_net = CalibratedPotentialNet(
                latent_dim,
                hidden_dim,
                log_density_weight_init=density_log_weight,
                momentum=density_momentum,
            )
        else:
            self.potential_net = PotentialNet(latent_dim, hidden_dim)
        self.ode_func = GradientODEFunc(self.potential_net)

    def forward(self, z_current, delta_t=1.0):
        from torchdiffeq import odeint_adjoint as odeint

        t_span = torch.tensor([0.0, delta_t], device=z_current.device)
        z_future = odeint(self.ode_func, z_current, t_span, method="dopri5", rtol=1e-3, atol=1e-3)[-1]
        return z_future


class StaticLatentPredictor(nn.Module):
    """時間発展なし（前年潜在をそのまま次年予測に使用）。"""

    def forward(self, z_current: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
        return z_current


class StandardODEFunc(nn.Module):
    """素の Neural ODE 用ベクトル場 f(z)（ポテンシャルの勾配ではない）。"""

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.scale) * self.net(z)


class NeuralODEPredictor(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.ode_func = StandardODEFunc(latent_dim, hidden_dim)

    def forward(self, z_current: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
        from torchdiffeq import odeint_adjoint as odeint

        t_span = torch.tensor([0.0, delta_t], device=z_current.device)
        return odeint(
            self.ode_func,
            z_current,
            t_span,
            method="dopri5",
            rtol=1e-3,
            atol=1e-3,
        )[-1]


class RNNLatentPredictor(nn.Module):
    """ノードごとに LSTM で潜在系列を畳み、次時点の z を予測（VGRNN 系）。"""

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_history: List[torch.Tensor]) -> torch.Tensor:
        if len(z_history) == 1:
            z = z_history[-1]
            seq = z.unsqueeze(1)
        else:
            seq = torch.stack(z_history, dim=1)
        out, _ = self.lstm(seq)
        return self.fc_out(out[:, -1, :])
