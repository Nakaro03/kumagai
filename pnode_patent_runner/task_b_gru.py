"""task_b_gru.py — GRU vs relatedness on Task B (firm CPC-entry), head-to-head.

Tier-2 sequence model (no structure) against the static relatedness ceiling,
on the IDENTICAL eval set (same time split, pre-portfolio, test_first, candidate
set, n_pairs) so the comparison is airtight.

GRU: firm history = sequence of yearly multi-hot CPC vectors over the last L
years (recency window). GRU encodes -> linear -> logit per CPC. Trained as
next-year new-entry multi-label prediction (BCEWithLogits). At test, score all
CPCs, rank the not-yet-entered ones, evaluate Hit@k / MRR against the CPCs the
firm actually first-enters in the test window.

Default granularity = maingroup (the non-saturated battleground; static ceiling
relatedness MRR 0.213 / Hit@10 0.473).

Run:  python pnode_patent_runner/task_b_gru.py
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/home/nakamuraroi/kumagai")


def coarsen(code: str, level: str) -> str:
    if level == "subclass":
        m = re.match(r"^[A-Z]\d{2}[A-Z]", code)
        return m.group(0) if m else code
    if level == "maingroup":
        return code.split("/")[0]
    return code


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
    # firm -> {year -> set(cpc)} on train; pre-portfolio; test first-entries
    fy = defaultdict(dict)
    for (u, y), g in train.groupby(["u", "year"]):
        fy[u][y] = set(g.c)
    pre = {u: set().union(* y.values()) for u, y in fy.items()}
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
    return cooc


def evaluate(score_of_firm, firms, pre, test_first, ipcs, ks):
    idx = {c: j for j, c in enumerate(ipcs)}
    hits = {k: 0 for k in ks}
    rr, n = 0.0, 0
    for u in firms:
        scores = score_of_firm(u)                       # np.array over ipcs
        port = pre[u]
        masked = scores.copy()
        for c in port:
            masked[idx[c]] = -np.inf                     # candidates = not-yet-entered
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
class GRURec(nn.Module):
    def __init__(self, n_ipc, hidden=128):
        super().__init__()
        self.gru = nn.GRU(n_ipc, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_ipc)

    def forward(self, x):                                # x: [B, T, n_ipc]
        h, _ = self.gru(x)
        return self.out(h)                              # [B, T, n_ipc] logits


def build_sequences(fy, ipcs, years_grid, horizon=5):
    """Per firm: [T, n_ipc] multi-hot input over years_grid, and new-entry targets
    over the NEXT `horizon` years (to match the multi-year test window). Returns
    dict u -> (X, Y)."""
    idx = {c: j for j, c in enumerate(ipcs)}
    seqs = {}
    for u, yd in fy.items():
        X = np.zeros((len(years_grid), len(ipcs)), dtype=np.float32)
        Y = np.zeros((len(years_grid), len(ipcs)), dtype=np.float32)
        cum = set()
        for y in sorted(yd):
            if y < years_grid[0]:
                cum |= yd[y]
        for ti, y in enumerate(years_grid):
            cur = yd.get(y, set())
            for c in cur:
                X[ti, idx[c]] = 1.0
            future = set()
            for dy in range(1, horizon + 1):             # next `horizon` years
                future |= yd.get(y + dy, set())
            for c in future - cum - cur:                 # first entries in the window
                Y[ti, idx[c]] = 1.0
            cum |= cur
        seqs[u] = (X, Y)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--granularity", default="maingroup")
    ap.add_argument("--test-start", type=int, default=2019)
    ap.add_argument("--test-end", type=int, default=2023)
    ap.add_argument("--window", type=int, default=20, help="recency window (years)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--target-horizon", type=int, default=5,
                    help="train target = new entries within next H years (match test window)")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df, train, test = load(args.domain, args.granularity, args.test_start, args.test_end)
    ipcs = sorted(df.c.unique())
    fy, pre, test_first = build_eval_sets(train, test)
    firms = [u for u in test_first]
    print(f"granularity={args.granularity} IPCs={len(ipcs)} | eval firms={len(firms):,}")

    years_grid = list(range(args.test_start - args.window, args.test_start))  # input years
    idx = {c: j for j, c in enumerate(ipcs)}

    # ---------- baseline: relatedness ----------
    cooc = relatedness_scores(train, ipcs)

    def rel_score(u):
        s = np.zeros(len(ipcs), dtype=np.float32)
        for inc in pre[u]:
            for c, w in cooc[inc].items():
                s[idx[c]] += w
        return s

    rel = evaluate(rel_score, firms, pre, test_first, ipcs, args.ks)
    print(f"relatedness : " + "  ".join(f"{k}={v:.3f}" for k, v in rel.items()))

    # ---------- GRU ----------
    seqs = build_sequences(fy, ipcs, years_grid, horizon=args.target_horizon)
    train_firms = [u for u in fy if (seqs[u][1].sum() > 0)]   # has at least one target
    Xtr = torch.tensor(np.stack([seqs[u][0] for u in train_firms])).to(dev)
    Ytr = torch.tensor(np.stack([seqs[u][1] for u in train_firms])).to(dev)
    print(f"GRU train firms={len(train_firms):,}  X={tuple(Xtr.shape)}")

    model = GRURec(len(ipcs), args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = torch.full((len(ipcs),), 20.0, device=dev)   # rare positives
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bs = 512
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_firms))
        tot = 0.0
        for i in range(0, len(train_firms), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            logit = model(Xtr[b])
            loss = lossf(logit, Ytr[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:2d} | BCE {tot / len(train_firms):.4f}")

    # GRU score for eval firms = last-timestep logits from their input window
    model.eval()
    Xev = torch.tensor(np.stack([seqs[u][0] for u in firms])).to(dev)
    with torch.no_grad():
        logits = model(Xev)[:, -1, :].cpu().numpy()      # [F, n_ipc] last step
    gru_score_map = {u: logits[j] for j, u in enumerate(firms)}
    gru = evaluate(lambda u: gru_score_map[u], firms, pre, test_first, ipcs, args.ks)
    print(f"GRU         : " + "  ".join(f"{k}={v:.3f}" for k, v in gru.items()))

    print("\nHEAD-TO-HEAD (maingroup Task B):")
    for k in args.ks + ["MRR"]:
        key = f"Hit@{k}" if isinstance(k, int) else k
        win = "GRU" if gru[key] > rel[key] else "relatedness"
        print(f"  {key:8s}  relatedness {rel[key]:.3f}  vs  GRU {gru[key]:.3f}   -> {win}")


if __name__ == "__main__":
    import pandas as pd  # noqa: E402  (kept local so torch import errors surface first)
    globals()["pd"] = pd
    main()
