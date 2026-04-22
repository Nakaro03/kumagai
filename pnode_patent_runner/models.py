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
    """
    スカラーポテンシャル Φ(z)。

    - ``feature_mode="rff"``: ランダム Fourier 風の固定次元射影 + sin/cos（従来）。
      ``rff_learnable_basis=True`` で射影行列 B を学習可能にする。
    - ``feature_mode="mlp"``: z を MLP で hidden に写像してから同形状の head へ（データ適応）。
    """

    def __init__(
        self,
        latent_dim,
        hidden_dim,
        feature_mode: str = "rff",
        rff_learnable_basis: bool = False,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.feature_mode = str(feature_mode).lower()
        if self.feature_mode not in ("rff", "mlp"):
            raise ValueError("feature_mode must be 'rff' or 'mlp'")
        self.rff_learnable_basis = bool(rff_learnable_basis)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        if self.feature_mode == "rff":
            B = torch.randn(self.latent_dim, self.hidden_dim // 2) * 3.0
            self.B = nn.Parameter(B, requires_grad=self.rff_learnable_basis)
            self.in_proj = None
        else:
            self.B = None  # type: ignore[assignment]
            self.in_proj = nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
            )

    def forward(self, z):
        if self.feature_mode == "rff":
            assert self.B is not None
            proj = torch.matmul(z, self.B)
            x = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        else:
            assert self.in_proj is not None
            x = self.in_proj(z)
        return self.head(x)

    def compute_potential_grid(self, x_range, y_range, resolution=50, device="cpu"):
        if self.latent_dim != 2:
            raise ValueError("compute_potential_grid は latent_dim==2 のときのみサポート")
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
        potential_feature_mode: str = "rff",
        rff_learnable_basis: bool = False,
    ):
        super().__init__()
        self.nn_pot = PotentialNet(
            latent_dim,
            hidden_dim,
            feature_mode=potential_feature_mode,
            rff_learnable_basis=rff_learnable_basis,
        )
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
        ld = int(self.nn_pot.latent_dim)
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


class KDEDensityField(nn.Module):
    """
    ガウスカーネルによる経験密度場 ρ̂_t(z) と対数密度変化率 Δ_t(z)。

    アンカー点 {μ_t,i} は各年のエンコーダ出力から stop-gradient で設定し、
    ODE 積分中は固定。ノード数が大きい場合は ``max_anchors`` でサブサンプリング
    してメモリを O(B * max_anchors) に抑える（2D KDE では 256–512 点で十分）。
    """

    MAX_ANCHORS_DEFAULT = 512

    def __init__(self, latent_dim: int, log_bandwidth_init: float = 0.0,
                 min_density: float = 1e-8, max_anchors: int = 512):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.log_bandwidth = nn.Parameter(torch.tensor(float(log_bandwidth_init)))
        self.min_density = float(min_density)
        self.max_anchors = int(max_anchors)
        self._anchors_t: Optional[torch.Tensor] = None
        self._anchors_prev: Optional[torch.Tensor] = None

    @property
    def bandwidth(self) -> torch.Tensor:
        return self.log_bandwidth.exp()

    @staticmethod
    def _subsample(mu: torch.Tensor, max_n: int) -> torch.Tensor:
        if mu.size(0) <= max_n:
            return mu
        idx = torch.randperm(mu.size(0), device=mu.device)[:max_n]
        return mu[idx]

    @torch.no_grad()
    def set_anchors(
        self,
        mu_t: torch.Tensor,
        mu_prev: Optional[torch.Tensor] = None,
        active_mask_t: Optional[torch.Tensor] = None,
        active_mask_prev: Optional[torch.Tensor] = None,
    ) -> None:
        if active_mask_t is not None:
            mu_t = mu_t[active_mask_t]
        self._anchors_t = self._subsample(mu_t.detach(), self.max_anchors)
        if mu_prev is not None:
            if active_mask_prev is not None:
                mu_prev = mu_prev[active_mask_prev]
            self._anchors_prev = self._subsample(mu_prev.detach(), self.max_anchors)
        else:
            self._anchors_prev = None

    def _kde(self, z: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Gaussian KDE: ρ̂(z) = (1/M) Σ_i κ_h(z - μ_i)。shape (B,)"""
        h = self.bandwidth
        diff = z.unsqueeze(1) - anchors.unsqueeze(0)  # (B, M, D)  M <= max_anchors
        exponent = -0.5 * (diff ** 2).sum(dim=-1) / (h ** 2)  # (B, M)
        kernel_vals = torch.exp(exponent)  # (B, M)
        density = kernel_vals.mean(dim=1)  # (B,)
        return density.clamp(min=self.min_density)

    def log_density(self, z: torch.Tensor) -> torch.Tensor:
        if self._anchors_t is None:
            return torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        return torch.log(self._kde(z, self._anchors_t))

    def density_change(self, z: torch.Tensor) -> torch.Tensor:
        if self._anchors_t is None or self._anchors_prev is None:
            return torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        log_rho_t = torch.log(self._kde(z, self._anchors_t))
        log_rho_prev = torch.log(self._kde(z, self._anchors_prev))
        return log_rho_t - log_rho_prev


class PopulationCoupledPotentialNet(nn.Module):
    """
    密度結合ポテンシャル:
      Φ_t(z) = ψ_θ(z) − w_ρ log ρ̂_t(z) − w_Δ Δ_t(z)

    既存の GradientODEFunc / GradientNeuralODEPredictor と互換の
    forward(z) → (B,1) インターフェースを持つ。
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        potential_feature_mode: str = "mlp",
        rff_learnable_basis: bool = False,
        w_rho_init: float = 0.1,
        w_delta_init: float = 0.1,
        log_bandwidth_init: float = 0.0,
        max_anchors: int = 512,
    ):
        super().__init__()
        self.nn_pot = PotentialNet(
            latent_dim, hidden_dim,
            feature_mode=potential_feature_mode,
            rff_learnable_basis=rff_learnable_basis,
        )
        self.kde_field = KDEDensityField(
            latent_dim, log_bandwidth_init=log_bandwidth_init,
            max_anchors=max_anchors,
        )
        self.w_rho = nn.Parameter(torch.tensor(float(w_rho_init)))
        self.w_delta = nn.Parameter(torch.tensor(float(w_delta_init)))

    def set_population(
        self,
        mu_t: torch.Tensor,
        mu_prev: Optional[torch.Tensor] = None,
        active_mask_t: Optional[torch.Tensor] = None,
        active_mask_prev: Optional[torch.Tensor] = None,
    ) -> None:
        self.kde_field.set_anchors(mu_t, mu_prev, active_mask_t, active_mask_prev)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        psi = self.nn_pot(z)  # (B, 1)
        if psi.dim() > 1:
            psi = psi.squeeze(-1)
        log_rho = self.kde_field.log_density(z)
        delta = self.kde_field.density_change(z)
        phi = psi - self.w_rho * log_rho - self.w_delta * delta
        return phi.unsqueeze(-1)

    def compute_potential_grid(self, x_range, y_range, resolution=50, device="cpu"):
        ld = self.nn_pot.latent_dim
        if ld != 2:
            raise ValueError("compute_potential_grid は latent_dim==2 のみ")
        x = torch.linspace(x_range[0], x_range[1], resolution)
        y = torch.linspace(y_range[0], y_range[1], resolution)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
        model_device = next(self.parameters()).device
        with torch.no_grad():
            pot = self.forward(grid.to(model_device))
        return (
            X.cpu().numpy(),
            Y.cpu().numpy(),
            pot.view(resolution, resolution).cpu().numpy(),
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


def integrate_gradient_flow_ode(
    ode_func,
    z0: torch.Tensor,
    delta_t: float,
    method: str = "dopri5",
    n_steps: int = 4,
) -> torch.Tensor:
    """
    勾配流 ODE の 1 ステップ積分。`dopri5`（適応）または `rk4` / `euler`（等間隔格子）。
    `n_steps` は固定法で区間 [0, delta_t] を何分割するか（>=1）。
    """
    from torchdiffeq import odeint_adjoint as odeint

    device = z0.device
    dtype = z0.dtype
    m = (method or "dopri5").lower()
    if m == "dopri5":
        t_span = torch.tensor([0.0, float(delta_t)], device=device, dtype=dtype)
        return odeint(ode_func, z0, t_span, method="dopri5", rtol=1e-3, atol=1e-3)[-1]
    n_steps = max(1, int(n_steps))
    t_span = torch.linspace(0.0, float(delta_t), n_steps + 1, device=device, dtype=dtype)
    step_size = float(delta_t) / float(n_steps)
    if m == "euler":
        return odeint(ode_func, z0, t_span, method="euler", options={"step_size": step_size})[-1]
    if m == "rk4":
        return odeint(ode_func, z0, t_span, method="rk4", options={"step_size": step_size})[-1]
    raise ValueError(f"unknown ode method: {method!r} (use dopri5, rk4, euler)")


class ExplicitAttentionPotentialNet(nn.Module):
    r"""
    **明示的に定義**したスカラーポテンシャル :math:`\Phi:\mathbb{R}^D\to\mathbb{R}_+` と注目度。

    - 実装: :math:`\Phi(z)=\mathrm{softplus}(g_\theta)\ge 0`。
      ``feature_mode="mlp"`` では 3 層 MLP が :math:`g`。
      ``feature_mode="rff"`` では ``PotentialNet`` と同型の sin/cos 特徴＋Tanh なしヘッドが :math:`g`。
    - **注目度（attention）** :math:`A(z):=\exp(-\Phi(z))\in(0,1]`.
    - 解釈: **:math:`\Phi` が小さい** ほど **:math:`A` は大きい**（「谷底」= 高い注目度・関与度に対応する**設計**）。
    - 力学: 勾配流 :math:`\frac{dz}{d\tau}=-\nabla\Phi` は :math:`\Phi` を減らす＝:math:`A` を**増やす**方向（谷底＝高注目へ）。

    学習時、``unified_training`` の補助損失で当該年グラフ上の**次数に基づく正規化目標**に
    :math:`A(z_i)`（左パーティション＝企業/著者）を近づけ、**データで意味づけ**を与える。

    注: 従来の ``PotentialNet``（Tanh 出力）とは別系統。可視化は同じ
    ``compute_potential_grid`` 形状でヒートマップ化可能（min-max 正規化は可視化側）。
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        feature_mode: str = "mlp",
        rff_learnable_basis: bool = False,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.feature_mode = str(feature_mode).lower()
        if self.feature_mode not in ("rff", "mlp"):
            raise ValueError("feature_mode must be 'rff' or 'mlp'")
        self.rff_learnable_basis = bool(rff_learnable_basis)
        if self.feature_mode == "rff":
            B = torch.randn(self.latent_dim, self.hidden_dim // 2) * 3.0
            self.B = nn.Parameter(B, requires_grad=self.rff_learnable_basis)
            self._g_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(self.hidden_dim, 1),
            )
            self.mlp = None  # type: ignore[assignment]
        else:
            self.B = None  # type: ignore[assignment]
            self.mlp = nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(self.hidden_dim, 1),
            )
            self._g_head = None  # type: ignore[assignment]

    def _g_pre_softplus(self, z: torch.Tensor) -> torch.Tensor:
        if self.feature_mode == "rff":
            assert self.B is not None and self._g_head is not None
            proj = torch.matmul(z, self.B)
            x = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
            return self._g_head(x)
        assert self.mlp is not None
        return self.mlp(z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        g = self._g_pre_softplus(z)
        return F.softplus(g)

    def attention(self, z: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.forward(z))

    def compute_potential_grid(
        self, x_range, y_range, resolution: int = 50, device: str = "cpu"
    ):
        if self.latent_dim != 2:
            raise ValueError("compute_potential_grid は latent_dim==2 のときのみサポート")
        x = torch.linspace(x_range[0], x_range[1], resolution)
        y = torch.linspace(y_range[0], y_range[1], resolution)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        model_device = next(self.parameters()).device
        with torch.no_grad():
            potentials = self.forward(grid_points.to(model_device))
        return (
            X.cpu().numpy(),
            Y.cpu().numpy(),
            potentials.view(resolution, resolution).cpu().numpy(),
        )


class GradientNeuralODEPredictorExplicitAttention(nn.Module):
    """
    P-NODE (explicit attention) 用: ``ExplicitAttentionPotentialNet`` + 勾配流 ODE 積分。
    従来の ``GradientNeuralODEPredictor`` と同形状の API（density 校正式なし）。
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        ode_method: str = "dopri5",
        ode_n_steps: int = 4,
        feature_mode: str = "mlp",
        rff_learnable_basis: bool = False,
    ):
        super().__init__()
        self.potential_net = ExplicitAttentionPotentialNet(
            latent_dim,
            hidden_dim,
            feature_mode=feature_mode,
            rff_learnable_basis=rff_learnable_basis,
        )
        self.ode_func = GradientODEFunc(self.potential_net)
        self.ode_method = str(ode_method or "dopri5").lower()
        self.ode_n_steps = int(ode_n_steps)

    def forward(self, z_current: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
        return integrate_gradient_flow_ode(
            self.ode_func,
            z_current,
            float(delta_t),
            method=self.ode_method,
            n_steps=self.ode_n_steps,
        )


class GradientNeuralODEPredictor(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        density_calibrated: bool = False,
        density_log_weight: float = 1.0,
        density_momentum: float = 0.05,
        potential_feature_mode: str = "rff",
        rff_learnable_basis: bool = False,
        ode_method: str = "dopri5",
        ode_n_steps: int = 4,
    ):
        super().__init__()
        if density_calibrated:
            self.potential_net = CalibratedPotentialNet(
                latent_dim,
                hidden_dim,
                log_density_weight_init=density_log_weight,
                momentum=density_momentum,
                potential_feature_mode=potential_feature_mode,
                rff_learnable_basis=rff_learnable_basis,
            )
        else:
            self.potential_net = PotentialNet(
                latent_dim,
                hidden_dim,
                feature_mode=potential_feature_mode,
                rff_learnable_basis=rff_learnable_basis,
            )
        self.ode_func = GradientODEFunc(self.potential_net)
        self.ode_method = str(ode_method or "dopri5").lower()
        self.ode_n_steps = int(ode_n_steps)

    def forward(self, z_current, delta_t=1.0):
        return integrate_gradient_flow_ode(
            self.ode_func,
            z_current,
            float(delta_t),
            method=self.ode_method,
            n_steps=self.ode_n_steps,
        )


class GradientPlusResidualODEFunc(nn.Module):
    """
    dz/dτ = -tanh(s_g) ∇Φ(z) + tanh(s_r) h_θ(z)。
    勾配流に学習可能な残差場を足し、非保守成分を近似する（PNODE 拡張 D）。
    """

    def __init__(self, potential_net: nn.Module, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.potential_net = potential_net
        self.scale_grad = nn.Parameter(torch.tensor(0.1))
        self.scale_res = nn.Parameter(torch.tensor(0.05))
        self.residual_mlp = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
        )

    def forward(self, t, z):
        with torch.set_grad_enabled(True):
            z_in = z.detach().requires_grad_(True)
            phi = self.potential_net(z_in)
            grad_z = torch.autograd.grad(phi.sum(), z_in, create_graph=True)[0]
        h = self.residual_mlp(z)
        return -torch.tanh(self.scale_grad) * grad_z + torch.tanh(self.scale_res) * h


class GradientResidualNeuralODEPredictor(nn.Module):
    """PotentialNet + 勾配項 + 残差 MLP（`GradientPlusResidualODEFunc`）。"""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        density_calibrated: bool = False,
        density_log_weight: float = 1.0,
        density_momentum: float = 0.05,
        potential_feature_mode: str = "rff",
        rff_learnable_basis: bool = False,
        ode_method: str = "dopri5",
        ode_n_steps: int = 4,
    ):
        super().__init__()
        if density_calibrated:
            self.potential_net = CalibratedPotentialNet(
                latent_dim,
                hidden_dim,
                log_density_weight_init=density_log_weight,
                momentum=density_momentum,
                potential_feature_mode=potential_feature_mode,
                rff_learnable_basis=rff_learnable_basis,
            )
        else:
            self.potential_net = PotentialNet(
                latent_dim,
                hidden_dim,
                feature_mode=potential_feature_mode,
                rff_learnable_basis=rff_learnable_basis,
            )
        self.ode_func = GradientPlusResidualODEFunc(self.potential_net, latent_dim, hidden_dim)
        self.ode_method = str(ode_method or "dopri5").lower()
        self.ode_n_steps = int(ode_n_steps)

    def forward(self, z_current, delta_t=1.0):
        return integrate_gradient_flow_ode(
            self.ode_func,
            z_current,
            float(delta_t),
            method=self.ode_method,
            n_steps=self.ode_n_steps,
        )


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


class StandardODEFuncTime(nn.Module):
    """素の Neural ODE 用: f(t,z) は暦年で条件付け（年埋め込みを z に連結）。"""

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
        self.year_emb = nn.Embedding(self.num_years, self.year_emb_dim)
        in_dim = self.latent_dim + self.year_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))
        self._calendar_year: Optional[int] = None

    def set_calendar_year(self, calendar_year: int) -> None:
        self._calendar_year = int(calendar_year)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self._calendar_year is None:
            raise RuntimeError("call set_calendar_year before odeint")
        y = self._calendar_year
        idx = y - self.year_min
        idx = max(0, min(self.num_years - 1, idx))
        year_idx = torch.full((z.size(0),), idx, dtype=torch.long, device=z.device)
        e = self.year_emb(year_idx)
        xz = torch.cat([z, e], dim=-1)
        return torch.tanh(self.scale) * self.net(xz)


class NeuralODEPredictorTime(nn.Module):
    """Neural ODE のベクトル場を暦年で条件付け（P-NODE の Φ(z,y) と対称な比較用）。"""

    def __init__(self, latent_dim: int, hidden_dim: int, year_min: int, year_max: int):
        super().__init__()
        self.ode_func = StandardODEFuncTime(latent_dim, hidden_dim, year_min, year_max)
        self.year_min = int(year_min)
        self.year_max = int(year_max)

    def forward(
        self,
        z_current: torch.Tensor,
        year_calendar_start: int,
        delta_t: float = 1.0,
    ) -> torch.Tensor:
        from torchdiffeq import odeint_adjoint as odeint

        self.ode_func.set_calendar_year(int(year_calendar_start))
        t_span = torch.tensor([0.0, float(delta_t)], device=z_current.device)
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
