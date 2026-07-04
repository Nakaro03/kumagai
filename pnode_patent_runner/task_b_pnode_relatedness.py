"""task_b_pnode_relatedness.py — Neural ODE with relatedness feature for Task B.

Simplified architecture:
  - Firm latent state z_i(t) ∈ ℝ^d (learned embedding)
  - Relatedness score r_i ∈ [0, 1] (static from training co-occurrence)
  - ODE: dz/dt = f(z, r, t; θ)  where f is a learned MLP
  - Emission: p(c | z) = softmax(z · W_c) predicts which CPC firm enters

Task: Predict which new CPC a firm first-enters in test window [2019, 2023],
given portfolio pre-2019. Compare to pure relatedness baseline.

Run:  python pnode_patent_runner/task_b_pnode_relatedness.py
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
from torchdiffeq import odeint

ROOT = Path("/home/nakamuraroi/kumagai")


def coarsen(code: str, level: str) -> str:
    if level == "subclass":
        m = re.match(r"^[A-Z]\d{2}[A-Z]", code)
        return m.group(0) if m else code
    if level == "maingroup":
        return code.split("/")[0]
    return code


# --------------------------------------------------------------------------- #
# Load and prepare data
# --------------------------------------------------------------------------- #
def load(domain, granularity, test_start, test_end):
    df = pd.read_csv(ROOT / f"data/processed/bipartite_{domain}_firm.csv",
                     dtype={"u": str, "i": str})
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    df["c"] = df["i"].map(lambda x: coarsen(x, granularity))
    df = df[["year", "u", "c"]].drop_duplicates()
    train = df[df.year < test_start]
    test = df[(df.year >= test_start) & (df.year <= test_end)]
    return df, train, test


def build_eval_sets(train, test):
    fy = defaultdict(dict)
    for (u, y), g in train.groupby(["u", "year"]):
        fy[u][y] = set(g.c)
    pre = {u: set().union(*y.values()) if y.values() else set()
           for u, y in fy.items()}
    test_first = {}
    for u, g in test.groupby("u"):
        new = set(g.c) - pre.get(u, set())
        if new and pre.get(u):
            test_first[u] = new
    return fy, pre, test_first


def relatedness_scores(train, ipcs):
    """Compute co-occurrence scores and scalar summary per firm."""
    cooc = defaultdict(lambda: defaultdict(int))
    for (u, y), g in train.groupby(["u", "year"]):
        cl = list(g.c)
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                cooc[cl[a]][cl[b]] += 1
                cooc[cl[b]][cl[a]] += 1

    idx = {c: j for j, c in enumerate(ipcs)}
    firm_rel_vec = {}
    firm_rel_scalar = {}
    fy = defaultdict(set)
    for (u, y), g in train.groupby(["u", "year"]):
        fy[u] |= set(g.c)

    for u in fy:
        port = fy[u]
        rel_vec = np.zeros(len(ipcs), dtype=np.float32)
        for inc in port:
            for c, w in cooc[inc].items():
                rel_vec[idx[c]] += w
        # Normalize to [0, 1]
        max_w = np.max(rel_vec) if np.any(rel_vec > 0) else 1.0
        rel_vec = rel_vec / (max_w + 1e-9)
        # Scalar: average of non-zero entries
        rel_scalar = np.mean(rel_vec[rel_vec > 0]) if np.any(rel_vec > 0) else 0.0
        firm_rel_vec[u] = rel_vec
        firm_rel_scalar[u] = float(rel_scalar)

    return firm_rel_vec, firm_rel_scalar


def evaluate(score_of_firm, firms, pre, test_first, ipcs, ks):
    idx = {c: j for j, c in enumerate(ipcs)}
    hits = {k: 0 for k in ks}
    rr, n = 0.0, 0
    for u in firms:
        scores = score_of_firm(u)
        port = pre[u]
        masked = scores.copy()
        for c in port:
            masked[idx[c]] = -np.inf
        order = np.argsort(-masked)
        rank = np.empty(len(ipcs), dtype=np.int64)
        rank[order] = np.arange(1, len(ipcs) + 1)
        for true_c in test_first[u]:
            r = rank[idx[true_c]]
            rr += 1.0 / r
            for k in ks:
                hits[k] += int(r <= k)
            n += 1
    res = {f"Hit@{k}": hits[k] / n for k in ks}
    res["MRR"] = rr / n
    res["n"] = n
    return res


# --------------------------------------------------------------------------- #
# Learned ODE drift conditioned on relatedness
# --------------------------------------------------------------------------- #
class LearnedDrift(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        # Input: [z (dim), r (1), t (1)] -> drift (dim)
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # z: [N, d], r: [N, 1], t: scalar
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(z.shape[0])
        else:
            t = t.expand(z.shape[0])
        t_input = t.unsqueeze(1)  # [N, 1]
        inp = torch.cat([z, r, t_input], dim=-1)  # [N, d+2]
        return self.net(inp)  # [N, d]


# --------------------------------------------------------------------------- #
# Emission model
# --------------------------------------------------------------------------- #
class EmissionModel(nn.Module):
    def __init__(self, embedding_dim: int, n_ipc: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, n_ipc)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.linear(z), dim=-1)


# --------------------------------------------------------------------------- #
# Full model: ODE + Emission
# --------------------------------------------------------------------------- #
class ODEWithRelatednessModel(nn.Module):
    def __init__(self, dim: int, n_ipc: int, hidden: int = 64):
        super().__init__()
        self.drift = LearnedDrift(dim, hidden)
        self.emission = EmissionModel(dim, n_ipc)

    def forward(self, z0: torch.Tensor, r: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        # z0: [N, d], r: [N, 1], ts: [T]
        # Returns: [T, N, d]
        def drift_wrapper(t, z):
            return self.drift(t, z, r)

        traj = odeint(drift_wrapper, z0, ts, method="dopri5", rtol=1e-3, atol=1e-4)
        return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--granularity", default="maingroup")
    ap.add_argument("--test-start", type=int, default=2019)
    ap.add_argument("--test-end", type=int, default=2023)
    ap.add_argument("--embedding-dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df, train, test = load(args.domain, args.granularity, args.test_start, args.test_end)
    ipcs = sorted(df.c.unique())
    fy, pre, test_first = build_eval_sets(train, test)
    firms = [u for u in test_first if pre.get(u)]
    print(f"granularity={args.granularity} IPCs={len(ipcs)} | eval firms={len(firms):,}")

    # Compute relatedness scores (static)
    rel_vec_map, rel_scalar_map = relatedness_scores(train, ipcs)

    # Baseline: pure relatedness
    def rel_score(u):
        return rel_vec_map.get(u, np.zeros(len(ipcs), dtype=np.float32))

    rel = evaluate(rel_score, firms, pre, test_first, ipcs, args.ks)
    print(f"relatedness : " + "  ".join(f"{k}={v:.3f}" for k, v in rel.items()))

    # ---------- Neural ODE with relatedness ----------
    # Initialize firm embeddings
    z0_map = {}
    for u in firms:
        z0 = torch.randn(args.embedding_dim, device=dev, dtype=torch.float32) * 0.1
        z0_map[u] = z0

    r_map = {}
    for u in firms:
        r_val = rel_scalar_map.get(u, 0.0)
        r = torch.tensor([[r_val]], device=dev, dtype=torch.float32)
        r_map[u] = r

    model = ODEWithRelatednessModel(args.embedding_dim, len(ipcs), args.hidden).to(dev)
    z0_params = nn.ParameterList([nn.Parameter(z0_map[u]) for u in firms])
    opt = torch.optim.Adam(list(model.parameters()) + list(z0_params), lr=5e-3)

    # Time grid: t=0 to t=1 (one-step forecast)
    ts = torch.linspace(0.0, 1.0, 3, device=dev, dtype=torch.float32)

    print(f"Training ODE with relatedness feature...")
    for ep in range(args.epochs):
        model.train()
        tot_loss = 0.0
        for i, u in enumerate(firms):
            opt.zero_grad()
            z0 = z0_params[i].unsqueeze(0)  # [1, d]
            r = r_map[u]  # [1, 1]
            try:
                traj = model(z0, r, ts)  # [3, 1, d]
                z_final = traj[-1, 0]  # [d]
                logp = model.emission(z_final)  # [n_ipc]

                # Multi-hot loss
                true_entries = test_first[u]
                target = torch.zeros(len(ipcs), device=dev)
                for c in true_entries:
                    idx_c = ipcs.index(c)
                    target[idx_c] = 1.0

                loss = -torch.sum(target * logp) / max(len(true_entries), 1)
                loss.backward()
                tot_loss += loss.item()
            except Exception as e:
                print(f"    Error on firm {u}: {e}")

        opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d} | loss {tot_loss / len(firms):.4f}")

    # Evaluate on test set
    model.eval()
    pnode_score_map = {}
    with torch.no_grad():
        for i, u in enumerate(firms):
            z0 = z0_params[i].unsqueeze(0)
            r = r_map[u]
            try:
                traj = model(z0, r, ts)
                z_final = traj[-1, 0]
                logp = model.emission(z_final)
                pnode_score_map[u] = torch.exp(logp).cpu().numpy()
            except:
                pnode_score_map[u] = np.ones(len(ipcs)) / len(ipcs)

    pnode = evaluate(lambda u: pnode_score_map[u], firms, pre, test_first, ipcs, args.ks)
    print(f"ODE+rel     : " + "  ".join(f"{k}={v:.3f}" for k, v in pnode.items()))

    print("\nHEAD-TO-HEAD (maingroup Task B):")
    for k in args.ks + ["MRR"]:
        key = f"Hit@{k}" if isinstance(k, int) else k
        win = "ODE+rel" if pnode[key] > rel[key] else "relatedness"
        print(f"  {key:8s}  relatedness {rel[key]:.3f}  vs  ODE+rel {pnode[key]:.3f}   -> {win}")


if __name__ == "__main__":
    main()
