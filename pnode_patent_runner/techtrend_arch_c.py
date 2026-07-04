"""techtrend_arch_c.py — Arch C: 二因子 CPCトレンド潜在 ODE × 企業関連性ゲート.

主成果物路線(plan: refactored-pondering-gizmo.md)。
  - CPCトレンド潜在 m_p(t): 各 maingroup の年次 entrant 数系列を共有 ODE-RNN で
    符号化し、Y+H へ外挿。decode して trend_p(= Prophet の機構置換、技術トレンド)。
  - 企業参入: logit_{i,p} = a*log1p(rel) + b*trend_p + c*log1p(rel)*trend_p
    「伸びている技術」×「参入できる企業」の二因子分解。

検証: trend_p ランキングが momentum/persistence を上回るか(主)、
      参入 logit が rel-only を上回るか(副)、交互作用 c=0 アブレーション。

Run:  python pnode_patent_runner/techtrend_arch_c.py --domain construction
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

import techtrend_common as tc
from techtrend_arch_a import ODERNN


def entrant_series(df, years_grid, cidx):
    """[L,n_c]: 各 maingroup の年次「新規 entrant 企業数」(リーク無し)。"""
    L, n_c = len(years_grid), len(cidx)
    yi = {y: j for j, y in enumerate(years_grid)}
    out = np.zeros((L, n_c), dtype=np.float32)
    ymin = years_grid[0]
    seen = {}                                  # firm -> set of codes seen before
    for (u, y), g in df[df.year <= years_grid[-1]].sort_values("year").groupby(
            ["u", "year"], sort=False):
        prev = seen.setdefault(u, set())
        for c in g.i:
            if c not in prev:
                prev.add(c)
                if y in yi and c in cidx:
                    out[yi[y], cidx[c]] += 1.0
    return out


class ArchC(nn.Module):
    def __init__(self, m_dim=16):
        super().__init__()
        self.enc = ODERNN(1, m_dim)            # CPC をバッチに、entrant数系列を符号化
        self.dec = nn.Linear(m_dim, 1)         # trend_p スカラ
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))
        self.c = nn.Parameter(torch.tensor(0.0))

    def trend(self, S, ts, t_future):
        """S[L,n_c,1] -> trend[n_c]."""
        h = self.enc(S, ts)
        hf = self.enc._flow(h, float(ts[-1]), t_future)
        return self.dec(hf).squeeze(-1)

    def forward(self, S, ts, t_future, logrel, use_interaction=True):
        tr = self.trend(S, ts, t_future)                 # [n_c]
        logit = self.a * logrel + self.b * tr[None, :]   # [N,n_c]
        if use_interaction:
            logit = logit + self.c * logrel * tr[None, :]
        return logit, tr


def run(domain, Y, H, args, use_interaction=True):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = tc.prepare(domain, args.y0, args.y1, Y, H, args.max_firms, args.seed,
                   embed_dim=args.embed_dim)
    ipcs, cidx, firms = s["ipcs"], s["cidx"], s["firms"]
    prior, test_first, rel = s["prior"], s["test_first"], s["rel"]
    n_c = len(ipcs)
    tag = "ArchC" if use_interaction else "ArchC(c=0)"
    print(f"[{domain}] Y={Y} H={H} maingroups={n_c} eval_firms={len(firms)}  {tag}")

    years_grid = list(range(Y - args.seq_len + 1, Y + 1))
    S_np = np.log1p(entrant_series(s["df"], years_grid, cidx))      # [L,n_c]
    span = (Y + H) - years_grid[0]
    ts = torch.tensor([(y - years_grid[0]) / span for y in years_grid],
                      dtype=torch.float32, device=dev)
    t_future = float(((Y + H) - years_grid[0]) / span)
    S = torch.tensor(S_np[:, :, None], device=dev)                 # [L,n_c,1]
    relmat = np.stack([rel.get(u, np.zeros(n_c, np.float32)) for u in firms])
    logrel = torch.tensor(np.log1p(relmat), device=dev)

    fi = {u: j for j, u in enumerate(firms)}
    target = torch.zeros(len(firms), n_c, device=dev)
    lossmask = torch.ones(len(firms), n_c, device=dev)
    for u in firms:
        for c in test_first[u]:
            if c in cidx:
                target[fi[u], cidx[c]] = 1.0
        for c in prior[u]:
            if c in cidx:
                lossmask[fi[u], cidx[c]] = 0.0

    model = ArchC(args.m_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        logit, _ = model(S, ts, t_future, logrel, use_interaction)
        loss = (bce(logit, target) * lossmask).sum() / lossmask.sum()
        loss.backward(); opt.step()
        if ep % 30 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d} | loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logit, tr = model(S, ts, t_future, logrel, use_interaction)
        proba = torch.sigmoid(logit).cpu().numpy()
        trend = tr.cpu().numpy()

    # baselines
    rel_base = tc.evaluate_entry(lambda u: rel.get(u, np.zeros(n_c)),
                                 firms, prior, test_first, ipcs, cidx, args.ks)
    ode = tc.evaluate_entry(lambda u: proba[fi[u]], firms, prior, test_first,
                            ipcs, cidx, args.ks)
    print(f"  [副] 参入 MRR  rel-only {rel_base['MRR']:.3f}  {tag} {ode['MRR']:.3f}  "
          f"({(ode['MRR']-rel_base['MRR'])/(rel_base['MRR']+1e-9)*100:+.0f}%)  "
          f"a={model.a.item():.2f} b={model.b.item():.2f} c={model.c.item():.2f}")

    actual = tc.actual_trend_counts(s["df"], Y, H, ipcs, cidx)
    df = s["df"]
    momentum = np.zeros(n_c); persist = np.zeros(n_c)
    for c, v in Counter(df[df.year == Y]["i"]).items():
        if c in cidx:
            momentum[cidx[c]] = v
    for c, v in Counter(df[(df.year > Y - 3) & (df.year <= Y)]["i"]).items():
        if c in cidx:
            persist[cidx[c]] = v / 3.0
    print(f"  [主] トレンド NDCG@10 / Spearman:")
    for name, pr in [(tag, trend), ("momentum", momentum), ("persistence", persist)]:
        print(f"    {name:<12} NDCG@10={tc.ndcg_at_k(pr, actual, 10):.3f}  "
              f"NDCG@20={tc.ndcg_at_k(pr, actual, 20):.3f}  "
              f"P@10={tc.precision_at_k(pr, actual, 10):.2f}  "
              f"Spearman={tc.spearman(pr, actual):.3f}")
    return dict(rel=rel_base, ode=ode, trend_ndcg=tc.ndcg_at_k(trend, actual, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--y0", type=int, default=2000)
    ap.add_argument("--y1", type=int, default=2020)
    ap.add_argument("--train-end", type=int, default=2017)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--m-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--max-firms", type=int, default=1500)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    run(args.domain, args.train_end, args.horizon, args, use_interaction=True)
    print("\n--- アブレーション(交互作用 c=0)---")
    run(args.domain, args.train_end, args.horizon, args, use_interaction=False)


if __name__ == "__main__":
    main()
