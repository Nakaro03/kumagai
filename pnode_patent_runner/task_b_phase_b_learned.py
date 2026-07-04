"""task_b_phase_b_learned.py — Phase B: add the learned correction ε·Φ_θ.

Phase A potential was  U(z) = Φ_rel(z)        (α=0, ε=0, no learning).
Phase B turns on a SMALL learned scalar field Φ_θ(z):

    U(z) = Φ_rel(z) + ε · Φ_θ(z)

Prediction = one flow step along the drift -∇U of the learned part:
    z̃_i = z_i - ε · ∇_z Φ_θ(z_i)            (the SDE drift, made discrete)
    π_ip = softmax_p[ -‖z̃_i - e_p‖²/2σ² + b_p ]

  • ε = 0  ⇒  z̃ = z  ⇒  EXACTLY Phase A  (Proposition 1 null, H0).
  • ε ≠ 0 lets the model warp the relatedness landscape from data.
  • ‖∇Φ_θ‖² penalty keeps the correction small / non-degenerate.

Success criterion (BOTH required — in-sample ℓ gain alone is a mirage):
  (1) held-out MRR(B) > MRR(A) = 0.197 on test-first new CPCs, AND
  (2) likelihood-ratio  Λ = 2(ℓ_B - ℓ_A)  significant vs χ²(df=#θ+1).

Run:  python pnode_patent_runner/task_b_phase_b_learned.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from task_b_phase_a_poisson import (
    build_cooccurrence_embeddings,
    build_count_sequences,
    compute_relatedness_weights,
    load,
    log_pi,
    phi_rel,
    poisson_loglik,
    rank_eval,
    relatedness_centroid,
)


class PhiTheta(nn.Module):
    """Small learned scalar field Φ_θ(z) on latent space (the ε-correction)."""

    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # start near zero so Φ_θ≈0 → Phase A at init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, z):
        return self.net(z).squeeze(-1)            # (N,)

    def grad(self, z, create_graph):
        """∇_z Φ_θ(z).  (N,d)"""
        z = z.detach().requires_grad_(True)
        phi = self.forward(z).sum()
        (g,) = torch.autograd.grad(phi, z, create_graph=create_graph)
        return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--granularity", default="maingroup")
    ap.add_argument("--test-start", type=int, default=2019)
    ap.add_argument("--test-end", type=int, default=2023)
    ap.add_argument("--embedding-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--lr-theta", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--h", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=1.0, help="Gibbs prior on Φ_rel")
    ap.add_argument("--lam", type=float, default=1e-2, help="‖∇Φ_θ‖² penalty")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df, train, test = load(args.domain, args.granularity, args.test_start, args.test_end)
    ipcs = sorted(df.c.unique())

    fy_train = defaultdict(dict)
    for (u, y), g in train.groupby(["u", "year"]):
        fy_train[u][y] = set(g.c)
    pre = {u: set().union(*y.values()) if y.values() else set()
           for u, y in fy_train.items()}
    test_first = {}
    for u, g in test.groupby("u"):
        new = set(g.c) - pre.get(u, set())
        if new and pre.get(u):
            test_first[u] = new
    firms = [u for u in test_first if pre.get(u)]
    print(f"domain={args.domain} granularity={args.granularity} "
          f"CPCs={len(ipcs)} | eval firms={len(firms):,}\n")

    n_uct, N_ut, years_grid = build_count_sequences(train, firms, ipcs, args.test_start)
    omega_np = compute_relatedness_weights(train, ipcs)
    e_p = build_cooccurrence_embeddings(train, ipcs, args.embedding_dim, seed=42).to(dev)
    omega = torch.tensor(omega_np, device=dev)
    b_p = torch.log(omega + 1e-12)
    n_ip = torch.tensor(n_uct.sum(axis=1), dtype=torch.float32, device=dev)

    z_h0 = relatedness_centroid(pre, firms, ipcs, e_p.cpu(), omega.cpu()).to(dev)

    # ── Phase A anchor (ε=0): re-fit z under Φ_rel prior, no learned field ──
    z = nn.Parameter(z_h0.clone())
    optA = torch.optim.Adam([z], lr=args.lr)
    for _ in range(args.epochs):
        optA.zero_grad()
        ll = poisson_loglik(z, e_p, b_p, n_ip, args.sigma)
        prior = args.beta * phi_rel(z, e_p, omega, args.h).sum()
        (-(ll - prior) / (n_ip.sum() + 1e-9)).backward()
        optA.step()
    with torch.no_grad():
        ll_a = poisson_loglik(z, e_p, b_p, n_ip, args.sigma).item()
    res_a = rank_eval(z.detach(), e_p, b_p, firms, ipcs, test_first, pre, args.sigma)
    print("Phase A (ε=0):  "
          + "  ".join(f"{k}={v:.3f}" for k, v in res_a.items())
          + f"  | ℓ_A={ll_a:,.1f}")

    # ── Phase B: jointly learn z, Φ_θ, ε ───────────────────────────────────
    zB = nn.Parameter(z.detach().clone())
    phi_theta = PhiTheta(args.embedding_dim).to(dev)
    raw_eps = nn.Parameter(torch.zeros((), device=dev))   # ε = softplus(raw); 0 init→Phase A
    optB = torch.optim.Adam(
        [{"params": [zB], "lr": args.lr},
         {"params": list(phi_theta.parameters()) + [raw_eps], "lr": args.lr_theta}])

    print(f"\nPhase B: learn ε·Φ_θ  (λ‖∇Φ_θ‖²={args.lam}, σ={args.sigma})")
    for ep in range(args.epochs):
        optB.zero_grad()
        eps = torch.nn.functional.softplus(raw_eps)
        g = phi_theta.grad(zB, create_graph=True)         # ∇Φ_θ(z)  (N,d)
        z_tilde = zB - eps * g                            # flow one step along -∇U_θ
        ll = poisson_loglik(z_tilde, e_p, b_p, n_ip, args.sigma)
        prior = args.beta * phi_rel(zB, e_p, omega, args.h).sum()
        pen = args.lam * (g ** 2).sum(dim=1).mean() * len(firms)
        loss = -(ll - prior - pen) / (n_ip.sum() + 1e-9)
        loss.backward()
        optB.step()
        if ep % 80 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d} | -logpost/N {loss.item():.4f} "
                  f"| ℓ={ll.item():,.0f} | ε={eps.item():.4f}")

    eps = torch.nn.functional.softplus(raw_eps)
    g = phi_theta.grad(zB, create_graph=False)
    z_tilde = (zB - eps * g).detach()
    with torch.no_grad():
        ll_b = poisson_loglik(z_tilde, e_p, b_p, n_ip, args.sigma).item()
    res_b = rank_eval(z_tilde, e_p, b_p, firms, ipcs, test_first, pre, args.sigma)
    print("\nPhase B (ε·Φ_θ): "
          + "  ".join(f"{k}={v:.3f}" for k, v in res_b.items())
          + f"  | ℓ_B={ll_b:,.1f}  | ε={eps.item():.4f}")

    # ── Verdict: BOTH criteria required ────────────────────────────────────
    n_theta = sum(p.numel() for p in phi_theta.parameters()) + 1
    lam_stat = 2 * (ll_b - ll_a)
    # χ²(df) upper-5% critical via Wilson–Hilferty (no scipy dep here)
    df_ = n_theta
    chi2_crit = df_ * (1 - 2/(9*df_) + 1.6449 * (2/(9*df_)) ** 0.5) ** 3
    d_mrr = res_b["MRR"] - res_a["MRR"]
    print(f"\nVerdict (Phase B must satisfy BOTH):")
    print(f"  (1) held-out  ΔMRR = {d_mrr:+.4f}   "
          f"({'PASS' if d_mrr > 0.005 else 'fail'}; need >+0.005)")
    print(f"  (2) LR  Λ=2(ℓ_B−ℓ_A) = {lam_stat:,.1f}  vs χ²₀.₀₅(df={df_})≈{chi2_crit:,.0f}  "
          f"({'PASS' if lam_stat > chi2_crit else 'fail'})")
    print(f"  learned ε = {eps.item():.4f}   (ε→0 ⇒ relatedness is everything)")
    both = d_mrr > 0.005 and lam_stat > chi2_crit
    print(f"  ⇒ {'★ ε≠0 beats relatedness (held-out)' if both else '— relatedness ceiling holds (simple-beats-complex)'}")


if __name__ == "__main__":
    main()
