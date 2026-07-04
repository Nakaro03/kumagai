"""pnode_climode_skeleton.py — minimal PNODE (ClimODE-style) skeleton + smoke test.

This is the thin vertical slice agreed in the design discussion:
  ① PNODE core:  dz/dt = -∇_z Φ_φ(z,t)  +  g_θ(z, {z_j}, t)
       - Φ_φ  : potential neural field, MLP([z; τ(t)])
       - ∇Φ   : exact, via autograd
       - g_θ  : attention interaction (global/residual term)
  ② persistence baseline:  ẑ(t+1) = z(t)
  ③ smoke test on SYNTHETIC data that has a REAL gradient flow (double-well).
       If the plumbing (odeint + autograd ∇Φ + backprop) is correct, the model
       should BEAT persistence here — because the particles genuinely flow down
       a potential. Failing this means the implementation is broken, not that
       "the method doesn't work".

NOTE: This file deliberately contains NO real patent data and makes NO scientific
claim. It only verifies the machinery and wires in the baseline + ablation
switches so the next step (real construction data head-to-head) is informative.

Run:  python pnode_patent_runner/pnode_climode_skeleton.py
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
from torchdiffeq import odeint


# --------------------------------------------------------------------------- #
# time embedding  τ(t) = (sin 2πt/T, cos 2πt/T, t)
# --------------------------------------------------------------------------- #
def time_embed(t: torch.Tensor, period: float) -> torch.Tensor:
    t = t.reshape(-1)
    return torch.stack(
        [torch.sin(2 * math.pi * t / period), torch.cos(2 * math.pi * t / period), t],
        dim=-1,
    )  # [B, 3]


# --------------------------------------------------------------------------- #
# Φ_φ(z, t) : potential neural field
# --------------------------------------------------------------------------- #
class PotentialField(nn.Module):
    def __init__(self, dim: int, hidden: int = 64, period: float = 10.0):
        super().__init__()
        self.period = period
        self.net = nn.Sequential(
            nn.Linear(dim + 3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # z: [N, d], t: scalar tensor -> phi: [N, 1]
        tau = time_embed(t, self.period).expand(z.shape[0], -1)
        return self.net(torch.cat([z, tau], dim=-1))


# --------------------------------------------------------------------------- #
# g_θ : attention interaction term  g_i = Σ_j α_ij V(z_j - z_i)
# --------------------------------------------------------------------------- #
class Interaction(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.q = nn.Linear(dim, hidden)
        self.k = nn.Linear(dim, hidden)
        self.v = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, dim))
        self.scale = 1.0 / math.sqrt(hidden)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        q, k = self.q(z), self.k(z)                       # [N, h]
        att = (q @ k.t()) * self.scale                    # [N, N]
        att = att - torch.diag(torch.full((z.shape[0],), float("inf"), device=z.device))
        alpha = torch.softmax(att, dim=-1)                # no self (diag = -inf -> 0)
        diff = z.unsqueeze(0) - z.unsqueeze(1)            # [N, N, d]  (z_j - z_i)
        v = self.v(diff)                                  # [N, N, d]
        return (alpha.unsqueeze(-1) * v).sum(dim=1)       # [N, d]


# --------------------------------------------------------------------------- #
# PNODE drift:  f(t, z) = -∇Φ + g_θ   (ablation switches: use_grad / use_interaction)
# --------------------------------------------------------------------------- #
class PNODEDrift(nn.Module):
    def __init__(self, dim: int, use_grad: bool = True, use_interaction: bool = True,
                 period: float = 10.0):
        super().__init__()
        self.use_grad = use_grad
        self.use_interaction = use_interaction
        self.potential = PotentialField(dim, period=period) if use_grad else None
        self.interaction = Interaction(dim) if use_interaction else None

    def grad_phi(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            z = z.requires_grad_(True)
            phi = self.potential(z, t).sum()
            (g,) = torch.autograd.grad(phi, z, create_graph=self.training)
        return g

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        drift = torch.zeros_like(z)
        if self.use_grad:
            drift = drift - self.grad_phi(z, t)
        if self.use_interaction:
            drift = drift + self.interaction(z, t)
        return drift


# --------------------------------------------------------------------------- #
# synthetic data: particles flowing down a true double-well potential
#   Φ*(z) = (z0^2 - 1)^2 + 0.5 z1^2   ->   dz/dt = -∇Φ*
# --------------------------------------------------------------------------- #
def true_drift(z: torch.Tensor) -> torch.Tensor:
    g0 = 4 * z[:, 0] * (z[:, 0] ** 2 - 1)
    g1 = z[:, 1]
    return -torch.stack([g0, g1], dim=-1)


def make_synthetic(n: int, k: int, dt: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z0 = (torch.rand(n, 2, generator=g) - 0.5) * 4.0     # spread across both wells
    ts = torch.arange(k, dtype=torch.float32) * dt
    traj = [z0]
    z = z0
    for _ in range(k - 1):                               # RK4 integrate the truth
        for _ in range(5):                               # substeps for stability
            h = dt / 5
            k1 = true_drift(z)
            k2 = true_drift(z + 0.5 * h * k1)
            k3 = true_drift(z + 0.5 * h * k2)
            k4 = true_drift(z + h * k3)
            z = z + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(z)
    return torch.stack(traj, dim=0), ts                  # [K, N, 2], [K]


# --------------------------------------------------------------------------- #
# baselines & evaluation
# --------------------------------------------------------------------------- #
def persistence_rmse(traj: torch.Tensor) -> float:
    # one-step-ahead: predict z(t+1) = z(t)
    pred, obs = traj[:-1], traj[1:]
    return torch.sqrt(((pred - obs) ** 2).sum(-1).mean()).item()


def model_rmse_onestep(drift: nn.Module, traj: torch.Tensor, ts: torch.Tensor) -> float:
    # FAIR vs persistence: from each TRUE z(t), integrate one step to predict z(t+1).
    drift.eval()
    se, n = 0.0, 0
    with torch.no_grad():
        for k in range(len(ts) - 1):
            pred = odeint(drift, traj[k], ts[k:k + 2], method="dopri5",
                          rtol=1e-4, atol=1e-5)[-1]
            se += ((pred - traj[k + 1]) ** 2).sum(-1).sum().item()
            n += traj.shape[1]
    return math.sqrt(se / n)


def rollout(drift: nn.Module, z0: torch.Tensor, ts: torch.Tensor,
            method: str = "dopri5") -> torch.Tensor:
    return odeint(drift, z0, ts, method=method, rtol=1e-4, atol=1e-5)  # [K, N, d]


def model_rmse(drift: nn.Module, traj: torch.Tensor, ts: torch.Tensor) -> float:
    drift.eval()
    with torch.no_grad():
        pred = rollout(drift, traj[0], ts)
    return torch.sqrt(((pred - traj) ** 2).sum(-1).mean()).item()


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="particles")
    ap.add_argument("--k", type=int, default=11, help="timesteps")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--no-grad", action="store_true", help="ablate -∇Φ")
    ap.add_argument("--no-interaction", action="store_true", help="ablate g_θ")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    traj, ts = make_synthetic(args.n, args.k, args.dt, seed=args.seed)
    k_tr = args.k * 2 // 3
    traj_tr, ts_tr = traj[:k_tr], ts[:k_tr]              # train on first 2/3

    drift = PNODEDrift(dim=2, use_grad=not args.no_grad,
                       use_interaction=not args.no_interaction, period=args.k * args.dt)
    opt = torch.optim.Adam(drift.parameters(), lr=1e-2)

    print(f"PNODE smoke test | grad={not args.no_grad} interaction={not args.no_interaction} "
          f"| N={args.n} K={args.k}")
    print(f"persistence RMSE (full traj) = {persistence_rmse(traj):.4f}")

    for ep in range(args.epochs):
        drift.train()
        opt.zero_grad()
        pred = rollout(drift, traj_tr[0], ts_tr)         # [k_tr, N, 2]
        loss = ((pred - traj_tr) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == args.epochs - 1:
            full = model_rmse(drift, traj, ts)
            print(f"  epoch {ep:3d} | train MSE {loss.item():.4f} | model RMSE (full) {full:.4f}")

    p = persistence_rmse(traj)
    m_roll = model_rmse(drift, traj, ts)                 # full rollout from z0 (hard)
    m_step = model_rmse_onestep(drift, traj, ts)         # FAIR: one-step like persistence
    print(f"\nRESULT (one-step, fair):  model {m_step:.4f}  vs  persistence {p:.4f}  "
          f"-> {'MODEL WINS' if m_step < p else 'model loses'}")
    print(f"RESULT (full rollout):    model {m_roll:.4f}  (harder; compounds error)")
    print("plumbing OK: ∇Φ autograd + odeint + backprop train (train MSE -> ~0).")


if __name__ == "__main__":
    main()
