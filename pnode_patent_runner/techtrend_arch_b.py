"""techtrend_arch_b.py — Arch B: Neural Jump-ODE 時間点過程(参入タイミング).

plan: refactored-pondering-gizmo.md。
  - h_i(t) を小 MLP ドリフトで連続発展、出願年に GRU 更新(ODE-RNN = Jump-ODE)。
  - 条件付き強度 λ_{i,p}(t) = softplus( g(h_i(t))·Cemb_p + β·log1p(rel_{i,p}) )。
  - 離散時間 TPP/生存 NLL(年次ビン):
      NLL = -Σ_{first-entry events} log(1-exp(-λ)) + Σ_{non-event bins} λ   (保有/既参入は除外)
  - 参入確率(副) = 1-exp(-Σ_{horizon} λ);  トレンド(主) = Σ_i Σ_{horizon} λ_{i,p}。

Run:  python pnode_patent_runner/techtrend_arch_b.py --domain construction
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

import techtrend_common as tc
from techtrend_arch_a import ODERNN


def first_entry_year(df, firms, Y, H, cidx):
    """firm -> {year in (Y,Y+H]: set(新規参入 maingroup)}(prior 及び早い年を除外)。"""
    _, prior, _ = tc.build_sets(df, Y, H)
    out = {u: defaultdict(set) for u in firms}
    fset = set(firms)
    test = df[(df.year > Y) & (df.year <= Y + H)]
    for u, g in test.groupby("u"):
        if u not in fset:
            continue
        acc = set(prior.get(u, set()))
        for y in sorted(g.year.unique()):
            for c in set(g[g.year == y].i) - acc:
                if c in cidx:
                    out[u][y].add(c)
                acc.add(c)
    return out, prior


class ArchB(nn.Module):
    def __init__(self, in_dim, emb_k, hidden=64):
        super().__init__()
        self.enc = ODERNN(in_dim, hidden)
        self.g = nn.Linear(hidden, emb_k)
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def intensity(self, h, Cemb, logrel):
        """λ[N,n_c] = softplus(g(h)·Cemb + β·logrel + bias)。"""
        q = self.g(h)
        z = q @ Cemb.t() + self.beta * logrel + self.bias
        return torch.nn.functional.softplus(z)

    def rollout(self, X, ts, Cemb, logrel, fut_ts):
        """符号化後 horizon 各年へ flow し λ を返す: list over fut years of [N,n_c]。"""
        h = self.enc(X, ts)
        lam, t_prev = [], float(ts[-1])
        for t in fut_ts:
            h = self.enc._flow(h, t_prev, t); t_prev = t
            lam.append(self.intensity(h, Cemb, logrel))
        return lam


def run(domain, Y, H, args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = tc.prepare(domain, args.y0, args.y1, Y, H, args.max_firms, args.seed,
                   embed_dim=args.embed_dim)
    ipcs, cidx, firms = s["ipcs"], s["cidx"], s["firms"]
    prior, test_first, Cemb, rel = s["prior"], s["test_first"], s["Cemb"], s["rel"]
    n_c, k = len(ipcs), Cemb.shape[1]
    print(f"[{domain}] Y={Y} H={H} maingroups={n_c} eval_firms={len(firms)}  ArchB(Jump-ODE TPP)")

    years_grid = list(range(Y - args.seq_len + 1, Y + 1))
    X_np = tc.firm_year_features(s["df"], firms, cidx, Cemb, years_grid)
    span = (Y + H) - years_grid[0]
    ts = torch.tensor([(y - years_grid[0]) / span for y in years_grid],
                      dtype=torch.float32, device=dev)
    fut_years = list(range(Y + 1, Y + H + 1))
    fut_ts = [((y - years_grid[0]) / span) for y in fut_years]
    X = torch.tensor(X_np, device=dev)
    Cemb_t = torch.tensor(Cemb, device=dev)
    relmat = np.stack([rel.get(u, np.zeros(n_c, np.float32)) for u in firms])
    logrel = torch.tensor(np.log1p(relmat), device=dev)

    fi = {u: j for j, u in enumerate(firms)}
    fey, _ = first_entry_year(s["df"], firms, Y, H, cidx)
    # 年次イベントマスク event[y][N,n_c]=1 if 新規参入, alive[y]=該当ビンが評価対象(未参入/非保有)
    owned = torch.zeros(len(firms), n_c, device=dev)
    for u in firms:
        for c in prior[u]:
            if c in cidx:
                owned[fi[u], cidx[c]] = 1.0
    event = []
    for y in fut_years:
        e = torch.zeros(len(firms), n_c, device=dev)
        for u in firms:
            for c in fey[u].get(y, ()):
                e[fi[u], cidx[c]] = 1.0
        event.append(e)

    model = ArchB(k, k, args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        lam = model.rollout(X, ts, Cemb_t, logrel, fut_ts)
        nll = 0.0
        entered = owned.clone()              # 既に参入済み(保有)は対象外、参入後も以降除外
        for y in range(len(fut_years)):
            l = lam[y].clamp(1e-6, 50.0)
            active = (1.0 - entered)         # まだ参入していない (i,p)
            ev = event[y]
            # 生存 NLL: イベント -log(1-e^-λ), 非イベント active ビン +λ
            nll = nll + (-(torch.log1p(-torch.exp(-l)) * ev)
                         + l * active * (1.0 - ev)).sum()
            entered = torch.clamp(entered + ev, max=1.0)
        nll = nll / len(firms)
        nll.backward(); opt.step()
        if ep % 30 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d} | nll {nll.item():.3f}")

    model.eval()
    with torch.no_grad():
        lam = model.rollout(X, ts, Cemb_t, logrel, fut_ts)
        Lam = torch.stack(lam, 0).sum(0)                  # Σ_horizon λ  [N,n_c]
        proba = (1.0 - torch.exp(-Lam)).cpu().numpy()
        trend = Lam.sum(0).cpu().numpy()

    rel_base = tc.evaluate_entry(lambda u: rel.get(u, np.zeros(n_c)),
                                 firms, prior, test_first, ipcs, cidx, args.ks)
    ode = tc.evaluate_entry(lambda u: proba[fi[u]], firms, prior, test_first,
                            ipcs, cidx, args.ks)
    print(f"  [副] 参入 MRR  rel-only {rel_base['MRR']:.3f}  ArchB {ode['MRR']:.3f}  "
          f"({(ode['MRR']-rel_base['MRR'])/(rel_base['MRR']+1e-9)*100:+.0f}%)")

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
    for name, pr in [("ArchB", trend), ("momentum", momentum), ("persistence", persist)]:
        print(f"    {name:<12} NDCG@10={tc.ndcg_at_k(pr, actual, 10):.3f}  "
              f"NDCG@20={tc.ndcg_at_k(pr, actual, 20):.3f}  "
              f"P@10={tc.precision_at_k(pr, actual, 10):.2f}  "
              f"Spearman={tc.spearman(pr, actual):.3f}")
    return dict(rel=rel_base, ode=ode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--y0", type=int, default=2000)
    ap.add_argument("--y1", type=int, default=2020)
    ap.add_argument("--train-end", type=int, default=2017)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--max-firms", type=int, default=1500)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    run(args.domain, args.train_end, args.horizon, args)


if __name__ == "__main__":
    main()
