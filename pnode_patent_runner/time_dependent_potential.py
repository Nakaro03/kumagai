"""
時間依存ポテンシャル Φ(z, year)（年インデックス埋め込み）。

【変更点】
  - GradientODEFuncContinuous: state=(z, t_cont) を一緒に積分し、
    ODE途中で時刻を固定せず連続的にΦ(z, t)を変化させる。
  - GradientNeuralODEPredictorContinuous: 上記を使う予測子。
  - 既存クラス（GradientODEFuncTime 等）は後方互換のため残置。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 補助：暦年 → インデックス変換
# ---------------------------------------------------------------------------

def calendar_year_to_index(year: int, year_min: int, year_max: int) -> int:
    y = int(year)
    if y < year_min or y > year_max:
        raise ValueError(f"year {y} out of [{year_min}, {year_max}]")
    return y - year_min


# ---------------------------------------------------------------------------
# 補助：Fourier 時刻埋め込み（連続時間用）
# ---------------------------------------------------------------------------

def fourier_time_embed(t_scalar: torch.Tensor, K: int, T: float) -> torch.Tensor:
    """
    スカラーテンソル t_scalar (0-dim or shape()) → shape (2K,)。
    t は [0, T] の範囲を想定した実連続時刻。
    """
    ks = torch.arange(1, K + 1, device=t_scalar.device, dtype=t_scalar.dtype)
    arg = 2.0 * torch.pi * ks * t_scalar / T
    return torch.cat([torch.sin(arg), torch.cos(arg)], dim=0)  # (2K,)


# ---------------------------------------------------------------------------
# TimeDependentPotentialNet（既存・後方互換）
# ---------------------------------------------------------------------------

class TimeDependentPotentialNet(nn.Module):
    """
    Φ(z, year) ≈ MLP([sin(zB), cos(zB), emb(year_idx)])。

    入力: z (N, latent_dim), year_idx (N,) — 整数インデックス
    出力: (N, 1)
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

        self.B = nn.Parameter(
            torch.randn(latent_dim, hidden_dim // 2) * 3.0, requires_grad=False
        )
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
        return X.cpu().numpy(), Y.cpu().numpy(), pot.cpu().numpy()


# ---------------------------------------------------------------------------
# 【新規】ContinuousTimePotentialNet
# ---------------------------------------------------------------------------

class ContinuousTimePotentialNet(nn.Module):
    """
    Φ(z, t_cont) — 連続実数時刻版。

    入力:
        z       : (N, latent_dim)
        t_cont  : スカラーテンソル（0-dim）or shape (1,)  — 積分擬時間 τ∈[y_start, y_end]
    出力: (N, 1)

    設計:
        Fourier 時刻埋め込み γ(t) = [sin(2πk·t/T), cos(2πk·t/T)]_{k=1..K}
        を z の RFF 特徴と concat して MLP に通す。

    なぜ既存の year_emb（整数 Embedding）ではなく Fourier か？
        - odeint の途中で t は実連続値 (e.g. 2014.37) を取る。
        - 整数インデックスでは補間不可。Fourier は微分可能かつ周期性を表現。
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        year_min: int,
        year_max: int,
        time_fourier_K: int = 8,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.year_min = float(year_min)
        self.year_max = float(year_max)
        self.T = float(year_max - year_min) if year_max > year_min else 1.0
        self.time_fourier_K = int(time_fourier_K)

        self.B = nn.Parameter(
            torch.randn(latent_dim, hidden_dim // 2) * 3.0, requires_grad=False
        )
        rff_dim = hidden_dim          # sin + cos 各 hidden_dim//2
        t_emb_dim = 2 * time_fourier_K
        in_feat = rff_dim + t_emb_dim

        self.net = nn.Sequential(
            nn.Linear(in_feat, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def _t_emb(self, t_scalar: torch.Tensor) -> torch.Tensor:
        """t_scalar: 0-dim → (2K,)"""
        return fourier_time_embed(t_scalar, self.time_fourier_K, self.T)

    def forward(self, z: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        """
        z       : (N, latent_dim)
        t_scalar: 0-dim or (1,) — 全ノード共通の実連続時刻
        -> (N, 1)
        """
        if z.dim() != 2 or z.size(1) != self.latent_dim:
            raise ValueError(f"z must be (N, {self.latent_dim})")
        proj = torch.matmul(z, self.B)
        rff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, hidden_dim)

        t = t_scalar.squeeze()
        t_emb = self._t_emb(t)                      # (2K,)
        t_emb_exp = t_emb.unsqueeze(0).expand(z.size(0), -1)  # (N, 2K)

        x = torch.cat([rff, t_emb_exp], dim=-1)     # (N, hidden_dim + 2K)
        return self.net(x)                            # (N, 1)

    def compute_potential_grid(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        calendar_year: float,
        resolution: int = 50,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = torch.linspace(x_range[0], x_range[1], resolution)
        y = torch.linspace(y_range[0], y_range[1], resolution)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
        model_device = next(self.parameters()).device
        t_scalar = torch.tensor(
            float(calendar_year) - self.year_min,
            dtype=torch.float32, device=model_device,
        )
        with torch.no_grad():
            potentials = self.forward(grid_points.to(model_device), t_scalar)
        pot = potentials.view(resolution, resolution)
        return X.cpu().numpy(), Y.cpu().numpy(), pot.cpu().numpy()


# ---------------------------------------------------------------------------
# 【新規】GradientODEFuncContinuous
# ---------------------------------------------------------------------------

class GradientODEFuncContinuous(nn.Module):
    """
    連続時間 ODE: d(z,t)/dτ = (-s·∇_z Φ(z,t), 1)。

    状態 state = (z, t_cont) をまとめて渡す。
        z      : (N, latent_dim)
        t_cont : (1,) — 積分擬時間（= 暦年オフセット）

    torchdiffeq は state をテンソルのタプルとして受け取れる。
    τ (odeint の独立変数) と t_cont（暦年）は共に 1ステップ進む。

    なぜこれが重要か？
        既存実装は ODE 積分中に暦年を定数固定していた。
        本クラスでは積分が進むにつれ Φ(z, t) の t も連続的に変化し、
        「2014年から2015年への移行中に地形が少しずつ変わる」を表現できる。
    """

    def __init__(self, potential_net: ContinuousTimePotentialNet):
        super().__init__()
        self.potential_net = potential_net
        self.scale = nn.Parameter(torch.tensor(0.3))  # tanh(0.3)≈0.29 で適度なスケール

    def forward(self, tau: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        """
        tau  : スカラー（odeint の擬時間）— Φ 計算には使わない（t_cont を使う）
        state: (z, t_cont)
            z      : (N, latent_dim)
            t_cont : (1,) — 現在の暦年オフセット（[0, T]）
        戻り値: (dz/dτ, dt/dτ) = (-s∇Φ, 1) のタプル
        """
        z, t_cont = state
        t_scalar = t_cont.squeeze()

        with torch.set_grad_enabled(True):
            z_in = z.detach().requires_grad_(True)
            phi = self.potential_net(z_in, t_scalar)         # (N, 1)
            grad_z = torch.autograd.grad(
                phi.sum(), z_in, create_graph=True
            )[0]                                              # (N, latent_dim)

        dz = -torch.tanh(self.scale) * grad_z
        dt = torch.ones_like(t_cont)                         # dt/dτ = 1
        return dz, dt

    def compute_gradient_field(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        calendar_year: float,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        pn = self.potential_net
        t_offset = float(calendar_year) - pn.year_min
        grid_points = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1),
            dtype=torch.float32, device=device,
        )
        t_scalar = torch.tensor(t_offset, dtype=torch.float32, device=device)
        grid_points = grid_points.requires_grad_(True)
        phi = pn(grid_points, t_scalar)
        gradients = torch.autograd.grad(phi.sum(), grid_points, create_graph=False)[0]
        grad_x = gradients[:, 0].view(X.shape).cpu().numpy()
        grad_y = gradients[:, 1].view(Y.shape).cpu().numpy()
        return grad_x, grad_y


# ---------------------------------------------------------------------------
# 【新規】GradientNeuralODEPredictorContinuous
# ---------------------------------------------------------------------------

class GradientNeuralODEPredictorContinuous(nn.Module):
    """
    連続時間 ODE で 1年先の z を予測する予測子。

    使い方:
        predictor = GradientNeuralODEPredictorContinuous(latent_dim, hidden_dim, year_min, year_max)
        z_next = predictor(z_current, year_start=2014)
        # → Φ(z, t) を t=2014→2015 の間連続的に変化させながら積分
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        year_min: int,
        year_max: int,
        time_fourier_K: int = 8,
    ):
        super().__init__()
        self.potential_net = ContinuousTimePotentialNet(
            latent_dim, hidden_dim, year_min, year_max, time_fourier_K
        )
        self.ode_func = GradientODEFuncContinuous(self.potential_net)
        self.year_min = int(year_min)
        self.year_max = int(year_max)

    def forward(
        self,
        z_current: torch.Tensor,
        year_start: int,
        delta_t: float = 1.0,
    ) -> torch.Tensor:
        """
        z_current : (N, latent_dim)
        year_start: 出発暦年（例: 2014）
        delta_t   : 積分幅（通常 1.0 = 1年）
        -> z_next : (N, latent_dim)

        ODE の初期状態:
            z0     = z_current
            t_cont = t_offset_start = year_start - year_min
        τ=0 → τ=delta_t で積分し、最終状態の z を返す。
        t_cont も delta_t だけ増加するが（dt/dτ=1）、z_next のみ返す。
        """
        from torchdiffeq import odeint_adjoint as odeint

        device = z_current.device
        t_offset_start = float(year_start) - float(self.year_min)
        t_cont0 = torch.tensor([t_offset_start], dtype=torch.float32, device=device)

        tau_span = torch.tensor([0.0, delta_t], device=device)
        state0 = (z_current, t_cont0)

        # torchdiffeq はタプル state をサポート（adjoint も可）
        state_T = odeint(
            self.ode_func,
            state0,
            tau_span,
            method="dopri5",
            rtol=1e-3,
            atol=1e-3,
        )
        # state_T: タプル (z_traj, t_traj), 各 shape (2, N, dim) or (2, 1)
        z_next = state_T[0][-1]  # τ=delta_t 時点の z
        return z_next


# ---------------------------------------------------------------------------
# 既存クラス（後方互換のため残置）
# ---------------------------------------------------------------------------

class GradientODEFuncTime(nn.Module):
    """dz/dτ = -tanh(scale) ∇_z Φ(z, year_fixed)。【旧実装・後方互換用】"""

    def __init__(self, potential_net_time: TimeDependentPotentialNet):
        super().__init__()
        self.potential_net = potential_net_time
        self.scale = nn.Parameter(torch.tensor(0.1))
        self._ode_calendar_year: Optional[int] = None

    def set_ode_calendar_year(self, calendar_year: int) -> None:
        self._ode_calendar_year = int(calendar_year)

    def forward(self, t, z):
        if self._ode_calendar_year is None:
            raise RuntimeError("call set_ode_calendar_year before odeint")
        pn = self.potential_net
        idx = calendar_year_to_index(self._ode_calendar_year, pn.year_min, pn.year_max)
        year_idx = torch.full((z.shape[0],), idx, dtype=torch.long, device=z.device)
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
            dtype=torch.float32, device=device,
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
    """純勾配流（tanh ゲートなし）。【旧実装・後方互換用】"""

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
        year_idx = torch.full((z.shape[0],), idx, dtype=torch.long, device=z.device)
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
        pn = self.potential_net
        idx = calendar_year_to_index(calendar_year, pn.year_min, pn.year_max)
        grid_points = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1),
            dtype=torch.float32, device=device,
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
    """純勾配流 ODE 予測子（旧実装・後方互換用）。"""

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
        return odeint(
            self.ode_func, z_current, t_span, method="dopri5", rtol=1e-3, atol=1e-3
        )[-1]


class GradientNeuralODEPredictorTime(nn.Module):
    """tanh ゲート付き勾配流 ODE 予測子（旧実装・後方互換用）。"""

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
        return odeint(
            self.ode_func, z_current, t_span, method="dopri5", rtol=1e-3, atol=1e-3
        )[-1]
