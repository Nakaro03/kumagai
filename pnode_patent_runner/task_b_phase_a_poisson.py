"""task_b_phase_a_poisson.py — Phase A: Poisson observation + fixed Φ_rel.

New design (PI-SDE, Phase A — α=0, ε=0 baseline):

  Latent state z_i(t) ∈ ℝ^d (NOT observed). Observation counts:
    n_ip(t_k) ~ Poisson(N_i(t_k) · π_ip(t_k))
    π_ip(z) = softmax_p[ -‖z_i - e_p‖²/(2σ²) + b_p ]          (emission)

  Fixed relatedness potential (the SDE drift when α=ε=0):
    U(z) = Φ_rel(z) = -log Σ_p ω_p k_h(z - e_p),  k_h(u)=exp(-‖u‖²/2h²)
  Stationary law of  dz = -∇U dt + √(2τ) dW  is Gibbs  ρ_∞ ∝ exp(-U/τ),
  i.e. z concentrates at relatedness minima → z* is the MAP under
  Poisson likelihood + Gibbs(Φ_rel) prior.

Faithful to the design (fixes over the earlier draft):
  • e_p are co-occurrence PPMI+SVD embeddings (NOT random) → Φ_rel is a
    geometrically meaningful relatedness landscape.
  • Φ_rel is actually computed and used as the Gibbs prior on z.
  • Proposition 1 anchor: we report the H0 (relatedness-centroid, ε=0)
    log-likelihood ℓ_H0 and the Phase-A MLE log-likelihood ℓ_A so the
    future likelihood-ratio test  Λ = 2(ℓ_H1 - ℓ_H0)  has its null anchor.

Proposition 1 (degenerate limit): with α=ε=0 the fitted model must
reproduce the relatedness ranking (MRR_A ≈ MRR_baseline) and ℓ_A ≈ ℓ_H0.

Run:  python pnode_patent_runner/task_b_phase_a_poisson.py
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

ROOT = Path("/home/nakamuraroi/kumagai")


def coarsen(code: str, level: str) -> str:
    if level == "subclass":
        m = re.match(r"^[A-Z]\d{2}[A-Z]", code)
        return m.group(0) if m else code
    if level == "maingroup":
        return code.split("/")[0]
    return code


def load(domain, granularity, test_start, test_end):
    df = pd.read_csv(ROOT / f"data/processed/bipartite_{domain}_firm.csv",
                     dtype={"u": str, "i": str})
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    df["c"] = df["i"].map(lambda x: coarsen(x, granularity))
    df = df[["year", "u", "c"]].drop_duplicates()
    train = df[df.year < test_start]
    test = df[(df.year >= test_start) & (df.year <= test_end)]
    return df, train, test


def build_count_sequences(df, firms, ipcs, test_start, window=5):
    """Build count matrix n_ip(t) for each firm over time window."""
    idx_c = {c: j for j, c in enumerate(ipcs)}
    idx_u = {u: i for i, u in enumerate(firms)}

    years = sorted(df.year.unique())
    year_min = max(years[0], test_start - window)
    year_max = test_start - 1
    years_grid = list(range(year_min, year_max + 1))
    grid_pos = {y: t for t, y in enumerate(years_grid)}

    n_uct = np.zeros((len(firms), len(years_grid), len(ipcs)), dtype=np.int32)
    N_ut = np.zeros((len(firms), len(years_grid)), dtype=np.int32)

    for (u, y, c), count in df.groupby(["u", "year", "c"]).size().items():
        if u not in idx_u or c not in idx_c or y not in grid_pos:
            continue
        i, t, j = idx_u[u], grid_pos[y], idx_c[c]
        n_uct[i, t, j] = count
        N_ut[i, t] += count

    return n_uct, N_ut, years_grid


def compute_relatedness_weights(train, ipcs):
    """ω_p = popularity of CPC p (# firm-years that entered it in train)."""
    idx_c = {c: j for j, c in enumerate(ipcs)}
    omega = np.zeros(len(ipcs), dtype=np.float64)
    for (u, y), g in train.groupby(["u", "year"]):
        for c in set(g.c):
            omega[idx_c[c]] += 1.0
    omega = omega / (omega.sum() + 1e-12)
    return omega.astype(np.float32)


def build_cooccurrence_embeddings(train, ipcs, dim=16, seed=42):
    """e_p = PPMI + Truncated-SVD embedding of the CPC co-occurrence graph.

    Co-occurrence basket = (firm, year). PPMI(p,q) then SVD to dim d,
    L2-normalized rows. CPCs that co-occur in firm portfolios sit nearby,
    so the relatedness potential Φ_rel(z)=-log Σ ω_p k_h(z-e_p) has minima
    along the firm's technological neighborhood.
    """
    idx_c = {c: j for j, c in enumerate(ipcs)}
    P = len(ipcs)

    # Sparse basket-incidence: rows = baskets, cols = cpc
    rows, cols = [], []
    for b, ((u, y), g) in enumerate(train.groupby(["u", "year"])):
        for c in set(g.c):
            rows.append(b)
            cols.append(idx_c[c])
    n_baskets = (rows[-1] + 1) if rows else 1
    B = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          shape=(n_baskets, P))

    # Co-occurrence counts C = BᵀB (diagonal = marginal counts)
    C = (B.T @ B).tocoo()
    total = C.data.sum()
    marg = np.asarray(B.sum(axis=0)).ravel() + 1e-9  # per-cpc occurrence

    # PPMI on off-diagonal structure
    pmi_rows, pmi_cols, pmi_vals = [], [], []
    for r, c, v in zip(C.row, C.col, C.data):
        p_pq = v / total
        p_p = marg[r] / total
        p_q = marg[c] / total
        pmi = math.log((p_pq + 1e-12) / (p_p * p_q + 1e-12))
        if pmi > 0:
            pmi_rows.append(r); pmi_cols.append(c); pmi_vals.append(pmi)
    PPMI = sparse.csr_matrix((pmi_vals, (pmi_rows, pmi_cols)), shape=(P, P))

    d = min(dim, P - 1)
    svd = TruncatedSVD(n_components=d, random_state=seed)
    e = svd.fit_transform(PPMI)            # (P, d)
    if d < dim:                             # pad if granularity tiny
        e = np.pad(e, ((0, 0), (0, dim - d)))
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    return torch.tensor(e, dtype=torch.float32)


def phi_rel(z, e_p, omega, h):
    """Φ_rel(z) = -log Σ_p ω_p exp(-‖z-e_p‖²/2h²).  z:(N,d) → (N,)"""
    d2 = torch.cdist(z, e_p) ** 2                      # (N, P)
    log_k = -d2 / (2.0 * h ** 2) + torch.log(omega + 1e-12)
    return -torch.logsumexp(log_k, dim=1)             # (N,)


def log_pi(z, e_p, b_p, sigma):
    """log π_ip(z) = log_softmax_p[ -‖z_i-e_p‖²/2σ² + b_p ].  (N,P)"""
    d2 = torch.cdist(z, e_p) ** 2
    logits = -d2 / (2.0 * sigma ** 2) + b_p.unsqueeze(0)
    return torch.log_softmax(logits, dim=1)


def poisson_loglik(z, e_p, b_p, n_ip, sigma):
    """ℓ = Σ_ip n_ip log π_ip(z_i).  Multinomial-equiv to Poisson (N_i fixed)."""
    return torch.sum(n_ip * log_pi(z, e_p, b_p, sigma))


def relatedness_centroid(pre, firms, ipcs, e_p, omega):
    """c_i = Σ_{p∈port_i} ω_p e_p / Σ ω_p  — the H0 (relatedness) anchor for z_i."""
    idx_c = {c: j for j, c in enumerate(ipcs)}
    z0 = torch.zeros(len(firms), e_p.shape[1], dtype=torch.float32)
    for i, u in enumerate(firms):
        port = [idx_c[c] for c in pre.get(u, set()) if c in idx_c]
        if not port:
            continue
        w = omega[port]
        z0[i] = (w.unsqueeze(1) * e_p[port]).sum(0) / (w.sum() + 1e-9)
    return z0


def rank_eval(z, e_p, b_p, firms, ipcs, test_first, pre, sigma):
    """Rank CPCs by π_ip, mask portfolio, score Hit@k / MRR on test-first."""
    idx_c = {c: j for j, c in enumerate(ipcs)}
    lp = log_pi(z, e_p, b_p, sigma).detach().cpu().numpy()   # (firms, P)
    ks = [5, 10, 20]
    hits = {k: 0 for k in ks}
    rr, n_pairs = 0.0, 0
    for i, u in enumerate(firms):
        masked = lp[i].copy()
        for c in pre.get(u, set()):
            if c in idx_c:
                masked[idx_c[c]] = -np.inf
        order = np.argsort(-masked)
        rank = np.empty(len(ipcs), dtype=np.int64)
        rank[order] = np.arange(1, len(ipcs) + 1)
        for true_c in test_first[u]:
            r = rank[idx_c[true_c]]
            rr += 1.0 / r
            for k in ks:
                hits[k] += int(r <= k)
            n_pairs += 1
    res = {f"Hit@{k}": hits[k] / n_pairs if n_pairs else 0.0 for k in ks}
    res["MRR"] = rr / n_pairs if n_pairs else 0.0
    res["n"] = n_pairs
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--granularity", default="maingroup")
    ap.add_argument("--test-start", type=int, default=2019)
    ap.add_argument("--test-end", type=int, default=2023)
    ap.add_argument("--embedding-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--h", type=float, default=0.3, help="Φ_rel kernel bandwidth")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="Gibbs prior weight (1/τ) on Φ_rel")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df, train, test = load(args.domain, args.granularity, args.test_start, args.test_end)
    ipcs = sorted(df.c.unique())

    # Evaluation set: firms with a pre-portfolio and ≥1 genuinely new test CPC
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
    print(f"Time grid: {years_grid[0]}–{years_grid[-1]} ({len(years_grid)} yrs) | "
          f"counts {n_uct.shape}")

    # ── Fixed components (α=0, ε=0) ────────────────────────────────────────
    omega_np = compute_relatedness_weights(train, ipcs)
    e_p = build_cooccurrence_embeddings(train, ipcs, args.embedding_dim, seed=42).to(dev)
    omega = torch.tensor(omega_np, device=dev)
    # popularity emission bias  b_p = log ω_p  (fixed, observation-derived)
    b_p = torch.log(omega + 1e-12)
    print(f"e_p: co-occurrence PPMI+SVD  shape={tuple(e_p.shape)} | "
          f"ω support={int((omega_np>0).sum())}/{len(ipcs)}\n")

    # Counts aggregated over the window (Poisson sufficient statistic)
    n_ip = torch.tensor(n_uct.sum(axis=1), dtype=torch.float32, device=dev)  # (firms,P)

    # ── H0 anchor: z fixed at relatedness centroid (Proposition 1 null) ─────
    z_h0 = relatedness_centroid(pre, firms, ipcs, e_p.cpu(), omega.cpu()).to(dev)
    ll_h0 = poisson_loglik(z_h0, e_p, b_p, n_ip, args.sigma).item()
    base = rank_eval(z_h0, e_p, b_p, firms, ipcs, test_first, pre, args.sigma)
    print("H0 (relatedness centroid, ε=0): "
          + "  ".join(f"{k}={v:.3f}" for k, v in base.items())
          + f"  | ℓ_H0={ll_h0:,.1f}")

    # ── Phase A: MLE of z under Poisson likelihood + Gibbs(Φ_rel) prior ─────
    z = nn.Parameter(z_h0.clone())
    opt = torch.optim.Adam([z], lr=args.lr)
    print(f"\nPhase A: fit z_i  (Poisson NLL + {args.beta}·Φ_rel prior, "
          f"h={args.h}, σ={args.sigma})")
    for ep in range(args.epochs):
        opt.zero_grad()
        ll = poisson_loglik(z, e_p, b_p, n_ip, args.sigma)
        prior = args.beta * phi_rel(z, e_p, omega, args.h).sum()
        loss = -(ll - prior) / (n_ip.sum() + 1e-9)
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d} | -logpost/N {loss.item():.4f} "
                  f"| ℓ={ll.item():,.0f} | Φ_rel={prior.item():,.0f}")

    with torch.no_grad():
        ll_a = poisson_loglik(z, e_p, b_p, n_ip, args.sigma).item()
    res = rank_eval(z.detach(), e_p, b_p, firms, ipcs, test_first, pre, args.sigma)
    print("\nPhase A (Poisson MLE):          "
          + "  ".join(f"{k}={v:.3f}" for k, v in res.items())
          + f"  | ℓ_A={ll_a:,.1f}")

    # ── Proposition 1 verification ─────────────────────────────────────────
    drift = (z.detach() - z_h0).norm(dim=1).mean().item()
    print(f"\nProposition 1 (α=ε=0 ⇒ relatedness degeneracy):")
    print(f"  ΔMRR (A − H0)         = {res['MRR'] - base['MRR']:+.3f}")
    print(f"  mean ‖z_A − z_H0‖     = {drift:.3f}")
    print(f"  Λ-anchor 2(ℓ_A−ℓ_H0)  = {2*(ll_a - ll_h0):,.1f}  "
          f"(LR-test reference; H1 ε≠0 in Phase B)")
    ok = abs(res["MRR"] - base["MRR"]) < 0.03
    print(f"  → {'✓ degenerates to relatedness' if ok else '✗ diverges from relatedness'} "
          f"(|ΔMRR|<0.03)")


if __name__ == "__main__":
    main()
