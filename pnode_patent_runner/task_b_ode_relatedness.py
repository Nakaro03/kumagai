"""task_b_ode_relatedness.py — Neural ODE with relatedness feature (fast variant).

Quick test: does time-evolving latent state (ODE) beat static fusion model?
Uses batch ODE solving to avoid per-firm overhead.

Architecture:
  - dz/dt = f(z, r, t; θ)  learned MLP drift
  - Emission: p(c | z(t)) = softmax(z · W_c)
  - Train on [0, 1] time grid; evaluate at t=1

Run:  python pnode_patent_runner/task_b_ode_relatedness.py
"""
from __future__ import annotations

import argparse
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


class ODEDrift(nn.Module):
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


class ODEModel(nn.Module):
    def __init__(self, dim: int, n_ipc: int, hidden: int = 64):
        super().__init__()
        self.drift = ODEDrift(dim, hidden)
        self.emission = nn.Linear(dim, n_ipc)

    def forward(self, z0: torch.Tensor, r: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
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
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
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

    rel_vec_map, rel_scalar_map = relatedness_scores(train, ipcs)

    def rel_score(u):
        return rel_vec_map.get(u, np.zeros(len(ipcs), dtype=np.float32))

    rel = evaluate(rel_score, firms, pre, test_first, ipcs, args.ks)
    print(f"relatedness : " + "  ".join(f"{k}={v:.3f}" for k, v in rel.items()))

    # ODE model with relatedness conditioning
    z_map = {u: torch.randn(args.embedding_dim, device=dev, dtype=torch.float32) * 0.1
             for u in firms}
    r_map = {u: torch.tensor([rel_scalar_map.get(u, 0.0)], device=dev, dtype=torch.float32)
             for u in firms}

    model = ODEModel(args.embedding_dim, len(ipcs), args.hidden).to(dev)
    z_params = nn.ParameterList([nn.Parameter(z_map[u]) for u in firms])
    opt = torch.optim.Adam(list(model.parameters()) + list(z_params), lr=5e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    ts = torch.linspace(0.0, 1.0, 2, device=dev, dtype=torch.float32)

    print(f"Training ODE with relatedness conditioning...")
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(firms))
        tot_loss = 0.0
        for i in range(0, len(firms), args.batch_size):
            b = perm[i:min(i+args.batch_size, len(firms))]

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
                print(f"    Batch error: {e}, skipping")

        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d} | loss {tot_loss / len(firms):.4f}")

    # Evaluate
    model.eval()
    ode_score_map = {}
    with torch.no_grad():
        for u in firms:
            z0 = z_params[firms.index(u)].unsqueeze(0)
            r = r_map[u].unsqueeze(0)
            traj = model(z0, r, ts)
            z_final = traj[-1, 0]
            logits = model.emission(z_final)
            ode_score_map[u] = torch.sigmoid(logits).cpu().numpy()

    ode = evaluate(lambda u: ode_score_map[u], firms, pre, test_first, ipcs, args.ks)
    print(f"ODE+rel     : " + "  ".join(f"{k}={v:.3f}" for k, v in ode.items()))

    print("\nHEAD-TO-HEAD (maingroup Task B):")
    for k in args.ks + ["MRR"]:
        key = f"Hit@{k}" if isinstance(k, int) else k
        best = max(rel[key], ode[key])
        win = "ODE+rel" if ode[key] > rel[key] else "relatedness"
        print(f"  {key:8s}  relatedness {rel[key]:.3f}  vs  ODE+rel {ode[key]:.3f}   -> {win}")


if __name__ == "__main__":
    main()
