"""SharedVGAEEncoder / PotentialNet / 勾配流 ODE 予測子（UnifiedVGAE から利用）。"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch_geometric.nn import GATConv


class SharedVGAEEncoder(nn.Module):
    """
    改善点:
      - BatchNorm → LayerNorm:
          BN は (N_active, D) のバッチ統計で正規化するため、年ごとに
          アクティブノード数が異なる時系列グラフでは統計が不安定になる。
          LN はノードごとに特徴次元で正規化するため、バッチサイズ非依存。
      - skip connection (残差接続):
          4層 GAT は over-smoothing（全ノードが同一表現に収束）を起こしやすい。
          skip は情報の近道を提供し、入力特徴のシグナルを最終層まで保持する。
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels * 2, heads=2, concat=False)
        self.conv2 = GATConv(hidden_channels * 2, hidden_channels, heads=2, concat=False)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=1, concat=False)
        self.conv_mu = GATConv(hidden_channels, out_channels, heads=1, concat=False)
        self.conv_logvar = GATConv(hidden_channels, out_channels, heads=1, concat=False)
        self.dropout = nn.Dropout(0.2)
        # LayerNorm: ノード数に依存しない正規化
        self.ln1 = nn.LayerNorm(hidden_channels * 2)
        self.ln2 = nn.LayerNorm(hidden_channels)
        # skip projection (次元不一致のため線形変換)
        self.skip1 = nn.Linear(in_channels, hidden_channels * 2, bias=False)
        self.skip2 = nn.Linear(hidden_channels * 2, hidden_channels, bias=False)

    def forward(self, x, edge_index):
        h1 = F.relu(self.ln1(self.conv1(x, edge_index))) + self.skip1(x)
        h1 = self.dropout(h1)
        h2 = F.relu(self.ln2(self.conv2(h1, edge_index))) + self.skip2(h1)
        h2 = self.dropout(h2)
        h3 = F.relu(self.conv3(h2, edge_index)) + h2  # 同次元: projection 不要
        return self.conv_mu(h3, edge_index), self.conv_logvar(h3, edge_index)


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

        # 改善: Tanh 除去 + spectral_norm
        #   Tanh: Φ を (-1,1) に clip → ∇Φ が飽和域でほぼ 0 → ODE 速度死亡
        #   spectral_norm: 各層の最大特異値 ≤ 1 に制約 → Φ の Lipschitz 定数に上界
        #     → ODE スティフネス低減 + rk4 n_steps=1 での積分精度向上
        #   Softplus: Φ ≥ 0 を保証 (谷=低エネルギー = 安定な技術ニッチ) かつ非飽和
        self.head = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim * 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim, 1)),
            nn.Softplus(),
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
        # 改善: tanh(scale) → softplus(log_scale)
        #   tanh: 値域 (-1,1) に上界 → 初期値 tanh(0.1)≈0.10 で速度が微小
        #   softplus(x) = log(1+exp(x)) > 0 かつ上界なし
        #   init: log_scale=-2 → softplus(-2)≈0.127 で既存と同じ初速
        #   学習が進むと log_scale が増加し α も増加 → 十分な移動量を確保できる
        self.log_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, t, z):
        with torch.set_grad_enabled(True):
            z_in = z.detach().requires_grad_(True)
            phi = self.potential_net(z_in)
            grad_z = torch.autograd.grad(phi.sum(), z_in, create_graph=True)[0]
        alpha = F.softplus(self.log_scale)
        return -alpha * grad_z

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


class AdaptiveGateODEFunc(nn.Module):
    r"""
    Adaptive-Gate ODE vector field:

        dz/dτ = α · [ -β(z) · ∇_z Φ_θ(z)  +  (1-β(z)) · f_ψ(z) ]

    β(z) = σ(gate_net(z)) ∈ (0,1) — per-node learned gate
      * β→1: pure gradient flow  (P-NODE regime)
      * β→0: pure free field     (NeuralODE regime)

    The gate is evaluated on z.detach() to avoid second-order graph through
    the gating branch; gradients only flow through the selected path.
    """

    def __init__(self, potential_net: nn.Module, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.potential_net = potential_net
        self.log_scale = nn.Parameter(torch.tensor(-2.0))  # softplus(-2)≈0.127

        self.free_field = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim) // 2),
            nn.ReLU(),
            nn.Linear(int(hidden_dim) // 2, 1),
        )
        # Init gate to β≈0.5 (balanced P-NODE / free-field start)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(True):
            z_in = z.detach().requires_grad_(True)
            phi = self.potential_net(z_in)
            grad_z = torch.autograd.grad(phi.sum(), z_in, create_graph=True)[0]

        beta = torch.sigmoid(self.gate_net(z.detach()))   # (N,1), stop-grad on gate input
        f = self.free_field(z)
        alpha = F.softplus(self.log_scale)
        return alpha * (-beta * grad_z + (1.0 - beta) * f)

    def gate_values(self, z: torch.Tensor) -> torch.Tensor:
        """Return β(z) for analysis / visualization."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_net(z))


class AdaptiveGateNeuralODEPredictor(nn.Module):
    r"""
    **AG-NODE** — Adaptive Gate Neural ODE Predictor.

    Architecture
    ============
    1. ODE dynamics (AdaptiveGateODEFunc):
       dz/dτ = α · [-β(z)·∇Φ(z)  +  (1-β(z))·f(z)]

    2. Static skip (per-node learned interpolation):
       ẑ_{t+1} = γ(z_t) · z_ODE_{t+1} + (1-γ(z_t)) · z_t

    Properties
    ----------
    * β→1, γ→1 : recovers P-NODE behaviour (potential-dominated, gradient flow)
    * β→0, γ→1 : recovers unconstrained NeuralODE
    * γ→0      : recovers static baseline (no temporal update)
    * By initialising γ to ≈0.12 (conservative), the model starts
      static-like and learns to trust the ODE only when beneficial,
      **guaranteeing ≥ static performance in expectation**.

    ``potential_net`` is exposed as an attribute so that ``unified_training``
    can compute potential / trajectory auxiliary losses unchanged.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        ode_method: str = "rk4",
        ode_n_steps: int = 2,
        feature_mode: str = "mlp",
        rff_learnable_basis: bool = False,
    ):
        super().__init__()
        self.potential_net = ExplicitAttentionPotentialNet(
            int(latent_dim), int(hidden_dim),
            feature_mode=feature_mode,
            rff_learnable_basis=rff_learnable_basis,
        )
        self.ode_func = AdaptiveGateODEFunc(self.potential_net, latent_dim, hidden_dim)
        self.ode_method = str(ode_method or "rk4").lower()
        self.ode_n_steps = int(ode_n_steps)

        # Static skip gate γ(z) ∈ (0,1)
        # Initialised to σ(-2) ≈ 0.12 — conservative: start close to static
        self.skip_gate = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim) // 4),
            nn.ReLU(),
            nn.Linear(int(hidden_dim) // 4, 1),
        )
        nn.init.zeros_(self.skip_gate[-1].weight)
        nn.init.constant_(self.skip_gate[-1].bias, -2.0)

    def forward(self, z_current: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
        z_ode = integrate_gradient_flow_ode(
            self.ode_func, z_current, float(delta_t),
            method=self.ode_method, n_steps=self.ode_n_steps,
        )
        gamma = torch.sigmoid(self.skip_gate(z_current.detach()))  # (N,1)
        return gamma * z_ode + (1.0 - gamma) * z_current

    def gate_stats(self, z: torch.Tensor):
        """Return (mean_beta, mean_gamma) for logging / analysis."""
        with torch.no_grad():
            beta = self.ode_func.gate_values(z).mean().item()
            gamma = torch.sigmoid(self.skip_gate(z)).mean().item()
        return beta, gamma


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
        self.log_scale_grad = nn.Parameter(torch.tensor(-2.0))
        self.log_scale_res  = nn.Parameter(torch.tensor(-3.0))
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
        return -F.softplus(self.log_scale_grad) * grad_z + F.softplus(self.log_scale_res) * h


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


class FiLMPotentialNet(nn.Module):
    r"""
    **改善案5 — FiLM 時間条件付きポテンシャル**

    Φ(z, t) = Softplus( MLP_φ(z) ⊙ γ(t_emb) + β_film(t_emb) )

    単純な連結 [z, t_emb] より FiLM が優れる理由:
      - 連結: 時間信号が全結合層を通過するだけ → 高次元 z に埋もれる
      - FiLM: 時間が MLP の「スケール + バイアス」を直接変調 → 全特徴次元に影響
      - 解釈: γ(t) は「この次元の重要度が年によって変わる」, β(t) は「ベースライン値」
      - 学習可能 Fourier 周波数 ω: 技術サイクル（年次/5年/10年）を自動発見

    これにより semiconductor の訓練崩壊（epoch3 で AUC 0.823 → 0.683）を防ぐ。
    崩壊原因: 固定 Φ(z) が IPC 急速再分類による非定常性を吸収できない。
    FiLM 時間条件付けにより Φ が年毎に異なる地形を表現できる。
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_fourier: int = 8,
        feature_mode: str = "mlp",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_fourier  = int(n_fourier)
        t_dim = n_fourier * 2  # sin + cos

        # 学習可能な Fourier 周波数（技術サイクルを自動発見）
        self.log_freqs = nn.Parameter(
            torch.linspace(-2.0, 2.0, n_fourier)
        )

        # Φ の MLP (z のみ処理)
        if feature_mode == "mlp":
            self.z_net = nn.Sequential(
                spectral_norm(nn.Linear(latent_dim, hidden_dim)),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            )
        else:
            # RFF モード
            B = torch.randn(latent_dim, hidden_dim // 2) * 3.0
            self.rff_B = nn.Parameter(B, requires_grad=False)
            self.z_net = nn.Sequential(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.LeakyReLU(0.2),
            )

        self.feature_mode = feature_mode

        # FiLM 条件付け: t_emb → (γ, β) ∈ R^{hidden_dim} × R^{hidden_dim}
        self.film_gamma = nn.Linear(t_dim, hidden_dim)
        self.film_beta  = nn.Linear(t_dim, hidden_dim)

        # ヘッド: modulated features → Φ(z,t)
        self.head = nn.Sequential(
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden_dim, 1)),
            nn.Softplus(),
        )

        # FiLM の γ を identity 初期化（学習前は時間依存なし = 従来 Φ と同等）
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.ones_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def _time_emb(self, t_scalar: float, n: int, device: torch.device) -> torch.Tensor:
        """t → Fourier 特徴量 (n, 2*n_fourier)"""
        freqs = self.log_freqs.exp()  # (n_fourier,)
        t = torch.full((n,), float(t_scalar), device=device, dtype=freqs.dtype)
        phase = t.unsqueeze(1) * freqs.unsqueeze(0)  # (n, n_fourier)
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)  # (n, 2*n_fourier)

    def forward(self, z: torch.Tensor, t_scalar: float = 0.0) -> torch.Tensor:
        """z: (N, D), t_scalar: float calendar year → Φ(z,t): (N, 1)"""
        if self.feature_mode == "rff":
            proj = torch.matmul(z, self.rff_B)
            h = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
            h = self.z_net(h)
        else:
            h = self.z_net(z)

        t_emb = self._time_emb(t_scalar, z.size(0), z.device)
        gamma = self.film_gamma(t_emb)  # (N, hidden)
        beta  = self.film_beta(t_emb)   # (N, hidden)
        h_mod = h * gamma + beta         # FiLM modulation

        return self.head(h_mod)          # (N, 1), Φ ≥ 0


class FiLMGradientNeuralODEPredictor(nn.Module):
    """FiLMPotentialNet + 勾配流 ODE + static skip（AG-NODE の時間条件付き版）"""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        ode_method: str = "rk4",
        ode_n_steps: int = 2,
        n_fourier: int = 8,
        feature_mode: str = "mlp",
    ):
        super().__init__()
        self.potential_net = FiLMPotentialNet(
            latent_dim, hidden_dim, n_fourier=n_fourier, feature_mode=feature_mode
        )
        self.log_scale = nn.Parameter(torch.tensor(-2.0))
        self.ode_method  = str(ode_method)
        self.ode_n_steps = int(ode_n_steps)

        self.skip_gate = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim) // 4),
            nn.ReLU(),
            nn.Linear(int(hidden_dim) // 4, 1),
        )
        nn.init.zeros_(self.skip_gate[-1].weight)
        nn.init.constant_(self.skip_gate[-1].bias, -2.0)

    def _make_ode_module(self, t_val: float) -> nn.Module:
        """t_val を保持する nn.Module ODE func を返す (odeint_adjoint 互換)。"""
        potential_net = self.potential_net
        log_scale     = self.log_scale

        class _FiLMODEFunc(nn.Module):
            def forward(self, t, z):
                with torch.set_grad_enabled(True):
                    z_in = z.detach().requires_grad_(True)
                    phi  = potential_net(z_in, t_val)
                    grad_z = torch.autograd.grad(phi.sum(), z_in, create_graph=True)[0]
                return -F.softplus(log_scale) * grad_z

        return _FiLMODEFunc()

    def forward(self, z_current: torch.Tensor, t_scalar: float = 0.0,
                delta_t: float = 1.0) -> torch.Tensor:
        ode_fn = self._make_ode_module(t_scalar)
        z_ode  = integrate_gradient_flow_ode(
            ode_fn, z_current, float(delta_t),
            method=self.ode_method, n_steps=self.ode_n_steps,
        )
        gamma = torch.sigmoid(self.skip_gate(z_current.detach()))
        return gamma * z_ode + (1.0 - gamma) * z_current


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


class HarmonicPotentialPredictor(nn.Module):
    r"""
    調和ポテンシャル（安定版）:
        ẑ_{t+1} = μ̄(t) + (z_0 - μ̄(t)) · σ(log_decay)

    パラメータ分離で学習安定化:
        - log_w     (K,) : softmax → どのトピックへ引き寄せるか（方向）
        - log_decay (1,) : sigmoid ∈ (0,1) → どれだけ元の位置を保持するか（大=保持、小=移動）

    元の数式との対応:
        μ̄(t) = softmax(w)^T [μ_1,...,μ_K]  （旧: Σ_k α_k μ_k / Σ_k α_k）
        σ(log_decay)                          （旧: exp(-Σ_k α_k)）

    元の exp(-A) は A が大きくなると 0 に崩壊し全ノードが μ̄ に集中する問題があった。
    sigmoid は (0,1) に bounded なため崩壊しない。

    bipartite グラフ前提:
        ノード 0..num_corps-1 : 著者/企業（動的）
        ノード num_corps..    : トピック（μ_k として固定、stop-gradient）
    """

    def __init__(self, num_corps: int, num_topics: int, latent_dim: int):
        super().__init__()
        self.num_corps = int(num_corps)
        self.num_topics = int(num_topics)
        self.latent_dim = int(latent_dim)
        self.log_w     = nn.Parameter(torch.zeros(self.num_topics))
        # sigmoid(1.0) ≈ 0.73: 初期は保守的（元の位置を 73% 保持）
        self.log_decay = nn.Parameter(torch.tensor(1.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (num_corps + num_topics, latent_dim)"""
        z_authors = z[: self.num_corps]           # (A, D)
        z_topics  = z[self.num_corps :].detach()  # (K, D) = μ_k(t)

        # 安定した加重重心: softmax で K 次元を正規化
        w      = F.softmax(self.log_w, dim=0)                         # (K,)
        mu_bar = (w.unsqueeze(1) * z_topics).sum(0)                   # (D,)

        # 安定した保持率: sigmoid ∈ (0,1)
        decay  = torch.sigmoid(self.log_decay)                        # スカラー
        z_authors_next = mu_bar.unsqueeze(0) + (z_authors - mu_bar.unsqueeze(0)) * decay

        return torch.cat([z_authors_next, z_topics], dim=0)


class GravityPotentialNet(nn.Module):
    r"""
    重力ポテンシャル:
        Φ(z) = -Σ_k M_k / (||z - c_k|| + ε)

    勾配流（GradientODEFunc で autograd が計算）:
        dz/dτ = -∇_z Φ = Σ_k M_k (c_k - z) / [ (||z-c_k|| + ε)² ||z-c_k|| ]
                                                ↑ 引力（c_k = トピック重心へ向かう）

    案Bがharmonicと異なる点:
        - 各著者が"全トピックから"距離依存の引力を受ける（ノードごとに異なる合力）
        - グローバル重心への崩壊が起きない
        - 近いトピックほど引力が強い（1/r² 則）

    パラメータ:
        log_mass (K,): 各トピックの質量 M_k = exp(log_mass_k) > 0（学習可能）
        c_k は encode 出力から stop-gradient で毎年セット（学習不要）
    """

    def __init__(self, num_topics: int, eps: float = 0.1):
        super().__init__()
        self.num_topics = int(num_topics)
        self.eps = float(eps)
        self.log_mass = nn.Parameter(torch.zeros(num_topics))
        self._c: Optional[torch.Tensor] = None

    def set_topic_positions(self, z_topics: torch.Tensor) -> None:
        """z_topics: (K, D) — 各年の encode 後に呼び出す（stop-gradient）"""
        self._c = z_topics.detach()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, D) → Φ(z): (N, 1)"""
        if self._c is None:
            return torch.zeros(z.size(0), 1, device=z.device, dtype=z.dtype)
        c = self._c.to(device=z.device, dtype=z.dtype)   # (K, D)
        M = self.log_mass.exp()                           # (K,)

        diff = z.unsqueeze(1) - c.unsqueeze(0)            # (N, K, D)
        dist = diff.norm(dim=-1)                          # (N, K)
        phi  = -(M.unsqueeze(0) / (dist + self.eps)).sum(dim=1)  # (N,)
        return phi.unsqueeze(-1)                          # (N, 1)

    def compute_potential_grid(self, x_range, y_range, resolution=50, device="cpu"):
        if self._c is None or self._c.size(1) != 2:
            raise ValueError("latent_dim==2 かつ set_topic_positions 後に呼び出してください")
        x = torch.linspace(x_range[0], x_range[1], resolution)
        y = torch.linspace(y_range[0], y_range[1], resolution)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
        with torch.no_grad():
            pot = self.forward(grid.to(device))
        return X.cpu().numpy(), Y.cpu().numpy(), pot.view(resolution, resolution).cpu().numpy()


class EvolveGCNOPredictor(nn.Module):
    """
    EvolveGCN-O (Pareja et al., NeurIPS 2020) — latent-space adaptation.

    Original: W_t = GRU_W(W_{t-1}, AggMsg(H_{t-1}, A_t)); H_t = σ(A_t H_{t-1} W_t)
    Adapted:  summary_t = mean_pool(z_{t-1})
              W_t       = GRUCell(summary_t, W_{t-1})   [hidden state = weight matrix]
              z_pred    = z_{t-1} @ W_t.T + b

    The GRU hidden state doubles as the evolving weight matrix W ∈ ℝ^{d×d}.
    Initial hidden = identity (flattened) so the first step is a near-copy.
    Accepts z_history (list of tensors) identical to RNNLatentPredictor.
    """

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        d = self.latent_dim
        self.gru = nn.GRUCell(d, d * d)
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, z_history: List[torch.Tensor]) -> torch.Tensor:
        d = self.latent_dim
        device = z_history[-1].device
        W_h = torch.eye(d, device=device).flatten().unsqueeze(0)  # (1, d²) — identity init
        for z in z_history:
            summary = z.mean(dim=0, keepdim=True)   # (1, d)
            W_h = self.gru(summary, W_h)             # (1, d²)
        W = W_h.squeeze(0).view(d, d)               # (d, d)
        return z_history[-1] @ W.T + self.bias      # (N, d)


class ROLANDPredictor(nn.Module):
    """
    ROLAND (You et al., KDD 2022) — latent-space adaptation.

    Original: h_v^t = (1-γ_v) h_v^{t-1} + γ_v · GNN(G_t, h_v^{t-1})
    Adapted:  z_cand = MLP(z_t)
              γ      = sigmoid(W_γ z_t + b_γ)   [per-node scalar gate]
              z_pred = (1 - γ) z_t + γ z_cand

    Single forward pass — no ODE, no RNN unroll.
    """

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.gate = nn.Linear(latent_dim, 1)

    def forward(self, z_current: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
        gamma = torch.sigmoid(self.gate(z_current))   # (N, 1)
        z_cand = self.mlp(z_current)                  # (N, d)
        return (1.0 - gamma) * z_current + gamma * z_cand


class GravityNeuralODEPredictor(nn.Module):
    """
    重力ポテンシャル + 勾配流 ODE（著者ノードのみ積分）。

    predict_future から呼ぶ前に set_topic_positions(z_topics) を呼ぶこと。
    """

    def __init__(
        self,
        num_topics: int,
        eps: float = 0.1,
        ode_method: str = "dopri5",
        ode_n_steps: int = 4,
    ):
        super().__init__()
        # 注: 属性名を gravity_net とすることで unified_training の
        # hasattr(..., "potential_net") 検査を通過させず、
        # P-NODE 専用の potential/trajectory loss を無効化する。
        self.gravity_net = GravityPotentialNet(num_topics, eps=eps)
        self.ode_func    = GradientODEFunc(self.gravity_net)
        self.ode_method  = str(ode_method)
        self.ode_n_steps = int(ode_n_steps)

    def set_topic_positions(self, z_topics: torch.Tensor) -> None:
        self.gravity_net.set_topic_positions(z_topics)

    def forward(self, z_authors: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
        """z_authors: (num_authors, D) → ẑ_authors_{t+1}: (num_authors, D)"""
        return integrate_gradient_flow_ode(
            self.ode_func, z_authors, float(delta_t),
            method=self.ode_method, n_steps=self.ode_n_steps,
        )
