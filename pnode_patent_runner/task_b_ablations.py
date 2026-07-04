"""task_b_ablations.py — Ablate architecture: ODE (no r) vs ODE+r vs PNODE+r.

Test hypothesis: does relatedness feature carry signal, and does explicit
potential structure (PNODE) beat implicit MLP drift?

Three models:
  1. ODE-only: dz/dt = f(z, t) [NO relatedness input]
  2. ODE+r:   dz/dt = f(z, r, t) [relatedness as feature]
  3. PNODE+r: dz/dt = -∇Φ(z,r,t) + g(z,r,t) [potential + interaction]

Run:  python pnode_patent_runner/task_b_ablations.py
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
        max_w = np.max(rel_vec) if np.any(rel_vec > 0) else 1.0
        rel_vec = rel_vec / (max_w + 1e-9)
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
# Model variants
# --------------------------------------------------------------------------- #

class ODEDriftNoR(nn.Module):
    """ODE-only: dz/dt = f(z, t) - no relatedness input."""
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor, r: torch.Tensor = None) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(z.shape[0])
        t_input = t.unsqueeze(-1)
        inp = torch.cat([z, t_input], dim=-1)
        return self.net(inp)


class ODEDriftWithR(nn.Module):
    """ODE+r: dz/dt = f(z, r, t) - relatedness as feature."""
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(z.shape[0])
        t_input = t.unsqueeze(-1)
        inp = torch.cat([z, r, t_input], dim=-1)
        return self.net(inp)


class PotentialField(nn.Module):
    """Potential Φ(z, r, t) -> scalar."""
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(z.shape[0])
        t_input = t.unsqueeze(-1)
        inp = torch.cat([z, r, t_input], dim=-1)
        return self.net(inp)


class Interaction(nn.Module):
    """Interaction term g(z, r, t)."""
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.q = nn.Linear(dim + 1, hidden)
        self.k = nn.Linear(dim + 1, hidden)
        self.v = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.Tanh(), nn.Linear(hidden, dim)
        )
        self.scale = 1.0 / math.sqrt(hidden)

    def forward(self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        zr = torch.cat([z, r], dim=-1)
        q, k = self.q(zr), self.k(zr)
        att = (q @ k.t()) * self.scale
        att = att - torch.diag(torch.full((z.shape[0],), float("inf"), device=z.device))
        alpha = torch.softmax(att, dim=-1)
        diff = z.unsqueeze(0) - z.unsqueeze(1)
        v = self.v(zr.unsqueeze(1))
        return (alpha.unsqueeze(-1) * v).sum(dim=1)


class PNODEDriftWithR(nn.Module):
    """PNODE+r: dz/dt = -∇Φ(z,r,t) + g(z,r,t)."""
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.potential = PotentialField(dim, hidden)
        self.interaction = Interaction(dim, hidden)

    def grad_phi(self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            z = z.requires_grad_(True)
            phi = self.potential(z, r, t).sum()
            (g,) = torch.autograd.grad(phi, z, create_graph=self.training)
        return g

    def forward(self, t: torch.Tensor, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        drift = -self.grad_phi(z, r, t) + self.interaction(z, r, t)
        return drift


class ODEModel(nn.Module):
    def __init__(self, dim: int, n_ipc: int, drift: nn.Module):
        super().__init__()
        self.drift = drift
        self.emission = nn.Linear(dim, n_ipc)

    def forward(self, z0: torch.Tensor, r: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        def drift_wrapper(t, z):
            return self.drift(t, z, r)
        # For PNODE: use multistep solver for stiff systems
        if isinstance(self.drift, PNODEDriftWithR):
            traj = odeint(drift_wrapper, z0, ts, method="implicit_adams", rtol=5e-2, atol=1e-3)
        else:
            traj = odeint(drift_wrapper, z0, ts, method="dopri5", rtol=1e-3, atol=1e-4)
        return traj


def train_model(model, z_params, r_map, firms, test_first, ipcs, ts, dev,
                epochs=100, batch_size=128, lr=5e-3):
    opt = torch.optim.Adam(list(model.parameters()) + list(z_params), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(firms))
        tot_loss = 0.0
        for i in range(0, len(firms), batch_size):
            b = perm[i:min(i+batch_size, len(firms))]

            z0_batch = torch.stack([z_params[j] for j in b])
            r_batch = torch.stack([r_map[firms[j]] for j in b])

            opt.zero_grad()
            try:
                traj = model(z0_batch, r_batch, ts)
                z_final = traj[-1]
                logits = model.emission(z_final)

                target = torch.zeros(len(b), len(ipcs), device=dev)
                for bi, j in enumerate(b):
                    u = firms[j]
                    for c in test_first[u]:
                        target[bi, ipcs.index(c)] = 1.0

                loss = loss_fn(logits, target)
                loss.backward()
                opt.step()
                tot_loss += loss.item() * len(b)
            except RuntimeError as e:
                pass

        if ep % 20 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3d} | loss {tot_loss / len(firms):.4f}")

    return model


def eval_model(model, z_params, r_map, firms, test_first, ipcs, ts, dev):
    model.eval()
    score_map = {}
    with torch.no_grad():
        for u in firms:
            z0 = z_params[firms.index(u)].unsqueeze(0)
            r = r_map[u].unsqueeze(0)
            traj = model(z0, r, ts)
            z_final = traj[-1, 0]
            logits = model.emission(z_final)
            score_map[u] = torch.sigmoid(logits).cpu().numpy()
    return score_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--granularity", default="maingroup")
    ap.add_argument("--test-start", type=int, default=2019)
    ap.add_argument("--test-end", type=int, default=2023)
    ap.add_argument("--embedding-dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
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
    print(f"granularity={args.granularity} IPCs={len(ipcs)} | eval firms={len(firms):,}\n")

    rel_vec_map, rel_scalar_map = relatedness_scores(train, ipcs)

    def rel_score(u):
        return rel_vec_map.get(u, np.zeros(len(ipcs), dtype=np.float32))

    rel = evaluate(rel_score, firms, pre, test_first, ipcs, args.ks)
    print(f"BASELINE relatedness: " + "  ".join(f"{k}={v:.3f}" for k, v in rel.items()))

    ts = torch.linspace(0.0, 1.0, 2, device=dev, dtype=torch.float32)

    # Initialize embeddings and relatedness
    def make_params():
        z_map = {u: torch.randn(args.embedding_dim, device=dev, dtype=torch.float32) * 0.1
                 for u in firms}
        r_map = {u: torch.tensor([rel_scalar_map.get(u, 0.0)], device=dev, dtype=torch.float32)
                 for u in firms}
        z_params = nn.ParameterList([nn.Parameter(z_map[u]) for u in firms])
        return z_params, r_map

    # ---------- Model 1: ODE-only (no r) ----------
    print("\n=== ODE-only (no relatedness input) ===")
    z_params_1, r_map_1 = make_params()
    model_1 = ODEModel(args.embedding_dim, len(ipcs), ODEDriftNoR(args.embedding_dim, args.hidden)).to(dev)
    model_1 = train_model(model_1, z_params_1, r_map_1, firms, test_first, ipcs, ts, dev, args.epochs)
    score_map_1 = eval_model(model_1, z_params_1, r_map_1, firms, test_first, ipcs, ts, dev)
    ode_only = evaluate(lambda u: score_map_1[u], firms, pre, test_first, ipcs, args.ks)
    print(f"ODE-only:           " + "  ".join(f"{k}={v:.3f}" for k, v in ode_only.items()))

    # ---------- Model 2: ODE+r ----------
    print("\n=== ODE+r (relatedness as feature) ===")
    z_params_2, r_map_2 = make_params()
    model_2 = ODEModel(args.embedding_dim, len(ipcs), ODEDriftWithR(args.embedding_dim, args.hidden)).to(dev)
    model_2 = train_model(model_2, z_params_2, r_map_2, firms, test_first, ipcs, ts, dev, args.epochs)
    score_map_2 = eval_model(model_2, z_params_2, r_map_2, firms, test_first, ipcs, ts, dev)
    ode_r = evaluate(lambda u: score_map_2[u], firms, pre, test_first, ipcs, args.ks)
    print(f"ODE+r:              " + "  ".join(f"{k}={v:.3f}" for k, v in ode_r.items()))

    # ---------- Model 3: PNODE+r ----------
    print("\n=== PNODE+r (potential + interaction + relatedness) ===")
    z_params_3, r_map_3 = make_params()
    model_3 = ODEModel(args.embedding_dim, len(ipcs), PNODEDriftWithR(args.embedding_dim, args.hidden)).to(dev)
    model_3 = train_model(model_3, z_params_3, r_map_3, firms, test_first, ipcs, ts, dev, args.epochs)
    score_map_3 = eval_model(model_3, z_params_3, r_map_3, firms, test_first, ipcs, ts, dev)
    pnode_r = evaluate(lambda u: score_map_3[u], firms, pre, test_first, ipcs, args.ks)
    print(f"PNODE+r:            " + "  ".join(f"{k}={v:.3f}" for k, v in pnode_r.items()))

    # ---------- Summary ----------
    print("\n" + "="*80)
    print("ABLATION SUMMARY (maingroup Task B)")
    print("="*80)
    for k in args.ks + ["MRR"]:
        key = f"Hit@{k}" if isinstance(k, int) else k
        print(f"{key:8s}  relatedness {rel[key]:.3f}  ODE-only {ode_only[key]:.3f}  "
              f"ODE+r {ode_r[key]:.3f}  PNODE+r {pnode_r[key]:.3f}")

    print("\nKEY FINDINGS:")
    r_lift = (ode_r["MRR"] - ode_only["MRR"]) / ode_only["MRR"] * 100
    pnode_lift = (pnode_r["MRR"] - ode_r["MRR"]) / ode_r["MRR"] * 100
    print(f"  Relatedness contribution (ODE+r vs ODE-only): +{r_lift:.1f}% MRR")
    print(f"  PNODE structure contribution (PNODE+r vs ODE+r): {pnode_lift:+.1f}% MRR")
    models = [('relatedness', rel['MRR']), ('ODE-only', ode_only['MRR']),
              ('ODE+r', ode_r['MRR']), ('PNODE+r', pnode_r['MRR'])]
    best = max(models, key=lambda x: x[1])
    print(f"  Best model: {best[0]}")


if __name__ == "__main__":
    main()
