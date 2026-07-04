"""X5 model — Φ(z, t) with Fourier time embedding, SDE with drift = -∇_z Φ.

Wraps /tmp/PI-SDE/src/model.AutoGenerator's interface so the existing
src.sde.sdeint_adjoint can be reused unchanged.

Differences vs base AutoGenerator:
  * `_pot(xt)` interprets the last column of xt as a scalar t and replaces it
    with a Fourier embedding of shape (B, 2K) before feeding the MLP.
  * Drift along the t-axis is taken as 0 (we are not learning an HJB residual).
"""
from __future__ import annotations

import sys
import math
import torch
from torch import nn
from collections import OrderedDict


sys.path.insert(0, "/tmp/PI-SDE")
import src.sde as sde  # noqa: E402


class FourierTime(nn.Module):
    """t in R -> [sin(2π k t / T_max), cos(2π k t / T_max)]_{k=1..K}.

    T_max is registered as a buffer so the same model can be reused across
    domains with different T after `set_t_max`.
    """

    def __init__(self, K: int = 8, t_max: float = 12.0):
        super().__init__()
        self.K = K
        self.register_buffer("t_max", torch.tensor(float(t_max)))
        ks = torch.arange(1, K + 1, dtype=torch.float32)
        self.register_buffer("ks", ks)

    def set_t_max(self, t_max: float) -> None:
        self.t_max.fill_(float(t_max))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, 1)
        scaled = 2.0 * math.pi * (t / self.t_max) * self.ks    # (B, K)
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # (B, 2K)


class X5Generator(nn.Module):
    """Drop-in replacement for AutoGenerator with Fourier-embedded t.

    Compatible interface for sdeint_adjoint(method='euler'):
      f(t, x_r) -> drift (B, x_dim+1)
      g(t, x_r) -> diffusion (B, x_dim+1)
    Plus exposed: `_pot(xt)` and `_drift(xt)` for anchor / landscape evaluation.
    """

    def __init__(self, x_dim: int, fourier_K: int = 8, t_max: float = 12.0,
                 hidden: int = 400, n_layers: int = 2, sigma_const: float = 0.1,
                 activation: str = "softplus"):
        super().__init__()
        self.dim = x_dim
        self.K = fourier_K
        # K=0 disables Fourier embedding (A5 ablation): use raw scalar t.
        self.use_fourier = fourier_K > 0
        if self.use_fourier:
            self.fourier = FourierTime(K=fourier_K, t_max=t_max)
            in_dim = x_dim + 2 * fourier_K
        else:
            self.fourier = None
            in_dim = x_dim + 1
        if activation == "softplus":
            act = nn.Softplus
        elif activation == "relu":
            act = nn.LeakyReLU
        elif activation == "tanh":
            act = nn.Tanh
        else:
            raise ValueError(activation)

        layers = []
        prev = in_dim
        for i in range(n_layers):
            layers.append((f"linear{i+1}", nn.Linear(prev, hidden)))
            layers.append((f"act{i+1}", act()))
            prev = hidden
        layers.append(("linear_out", nn.Linear(prev, 1, bias=False)))
        self.net = nn.Sequential(OrderedDict(layers))
        # init final layer to zero so initial drift ≈ 0 (parity with base PI-SDE)
        with torch.no_grad():
            self.net[-1].weight.zero_()

        self.register_buffer("sigma", torch.full((1, x_dim), float(sigma_const)))
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    # ── potential

    def _pot(self, xt: torch.Tensor) -> torch.Tensor:
        """xt: (B, x_dim+1) where last column is scalar t."""
        xt = xt.requires_grad_()
        z = xt[:, :self.dim]
        t = xt[:, self.dim:self.dim + 1]
        if self.use_fourier:
            emb = self.fourier(t)
            inp = torch.cat([z, emb], dim=-1)
        else:
            inp = torch.cat([z, t], dim=-1)
        return self.net(inp)

    def _drift(self, xt: torch.Tensor) -> torch.Tensor:
        """Drift in z-direction only: -∂Φ/∂z."""
        pot = self._pot(xt)
        grad = torch.autograd.grad(pot, xt, torch.ones_like(pot), create_graph=True)[0]
        return -grad[:, :self.dim]

    # ── SDE drift / diffusion (compatible with sdeint_adjoint)

    def f(self, t, x_r):
        x = x_r[:, :self.dim]
        t_col = torch.full((x.shape[0], 1), float(t), device=x.device)
        xt = torch.cat([x, t_col], dim=1)
        pot = self._pot(xt)
        grad = torch.autograd.grad(pot, xt, torch.ones_like(pot), create_graph=True)[0]
        drift_x = -grad[:, :self.dim]
        # the trailing channel was used in base PI-SDE for an HJB residual; we
        # keep it as a zero column so the rollout tensor layout matches.
        zero_col = torch.zeros(x.shape[0], 1, device=x.device)
        return torch.cat([drift_x, zero_col], dim=1)

    def g(self, t, x_r):
        x = x_r[:, :self.dim]
        sig = self.sigma.expand(x.shape[0], -1)
        zero_col = torch.zeros(x.shape[0], 1, device=x.device)
        return torch.cat([sig, zero_col], dim=1)


class X5SDE(nn.Module):
    """Wrap X5Generator with sdeint_adjoint rollout (drop-in for ForwardSDE)."""

    def __init__(self, x_dim: int, fourier_K: int = 8, t_max: float = 12.0,
                 hidden: int = 400, n_layers: int = 2, sigma_const: float = 0.1,
                 activation: str = "softplus"):
        super().__init__()
        self._func = X5Generator(x_dim, fourier_K, t_max, hidden, n_layers,
                                 sigma_const, activation)

    def forward(self, ts, x_r_0):
        return sde.sdeint_adjoint(
            self._func, x_r_0, ts,
            method="euler", dt=0.1, dt_min=1e-4,
            adjoint_method="euler",
            names={"drift": "f", "diffusion": "g"},
        )
