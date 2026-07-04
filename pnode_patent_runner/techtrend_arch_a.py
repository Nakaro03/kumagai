"""techtrend_arch_a.py — Arch A: ODE-RNN 帰納エンコーダ + 関連性ゲート参入ヘッド.

Step 0 の go/no-go 本体(plan: refactored-pondering-gizmo.md)。
現行 task_b_ode_relatedness.py の自由 z0(トランスダクティブ)を、企業の年次
ポートフォリオ系列を読む ODE-RNN(観測年に GRU 更新、年間は小 MLP ドリフトで
連続発展)に置換し、未知の将来年へ odeint で外挿する帰納モデル。

  logit_{i,p} = q(h_i(Y+H)) · Cemb_p  +  beta * log(1 + rel_{i,p})

検証する問い:
  (副) 帰納 ODE が rel-only(関連性ベクトル)を企業参入 MRR で上回るか
       = +126% がトランスダクティブ artifact でないか
  (主) 参入確率の企業横断集約による技術トレンドランキングが
       momentum/persistence 外挿を上回るか

Run:  python pnode_patent_runner/techtrend_arch_a.py --domain construction --gonogo
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

import techtrend_common as tc


# ----------------------------------------------------------- model ------------
class Drift(nn.Module):
    """dh/dt = MLP(h)  時間不変の小ドリフト(明示 Φ は入れない)。"""

    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, t, h):
        return self.net(h)


class ODERNN(nn.Module):
    """観測年に GRUCell 更新、年間は odeint(rk4) で連続発展する帰納エンコーダ。"""

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.drift = Drift(hidden)
        self.cell = nn.GRUCell(in_dim, hidden)
        self.h0 = nn.Parameter(torch.zeros(hidden))

    def _flow(self, h, t0, t1):
        if abs(t1 - t0) < 1e-9:
            return h
        ts = torch.tensor([t0, t1], dtype=h.dtype, device=h.device)
        return odeint(self.drift, h, ts, method="rk4")[-1]

    def forward(self, X, ts):
        """X[L,N,in], ts[L](正規化時刻) -> h[N,hidden](最終観測年 Y の状態)。"""
        L, N, _ = X.shape
        h = self.h0.unsqueeze(0).expand(N, -1).contiguous()
        for k in range(L):
            if k > 0:
                h = self._flow(h, float(ts[k - 1]), float(ts[k]))
            h = self.cell(X[k], h)
        return h


class ArchA(nn.Module):
    def __init__(self, in_dim, emb_k, hidden=64):
        super().__init__()
        self.enc = ODERNN(in_dim, hidden)
        self.query = nn.Linear(hidden, emb_k)
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, X, ts, t_future, Cemb, logrel):
        """logits[N,n_c]。"""
        h = self.enc(X, ts)
        hf = self.enc._flow(h, float(ts[-1]), t_future)
        q = self.query(hf)                       # [N,k]
        logits = q @ Cemb.t() + self.beta * logrel
        return logits


# ----------------------------------------------------------- train/eval -------
def run(domain, Y, H, args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = tc.prepare(domain, args.y0, args.y1, Y, H, args.max_firms, args.seed,
                   embed_dim=args.embed_dim)
    ipcs, cidx, firms = s["ipcs"], s["cidx"], s["firms"]
    prior, test_first, Cemb, rel = s["prior"], s["test_first"], s["Cemb"], s["rel"]
    n_c, k = len(ipcs), Cemb.shape[1]
    print(f"[{domain}] Y={Y} H={H} maingroups={n_c} eval_firms={len(firms)} k={k}")

    # --- baseline: relatedness-only ---
    rel_base = tc.evaluate_entry(lambda u: rel.get(u, np.zeros(n_c)),
                                 firms, prior, test_first, ipcs, cidx, args.ks)
    print("rel-only   : " + "  ".join(f"{kk}={vv:.3f}" for kk, vv in rel_base.items()))

    # --- 帰納エンコーダ入力(共通の年グリッド [Y-L+1 .. Y]) ---
    years_grid = list(range(Y - args.seq_len + 1, Y + 1))
    X_np = tc.firm_year_features(s["df"], firms, cidx, Cemb, years_grid)
    span = (Y + H) - years_grid[0]
    ts = torch.tensor([(y - years_grid[0]) / span for y in years_grid],
                      dtype=torch.float32, device=dev)
    t_future = float(((Y + H) - years_grid[0]) / span)
    X = torch.tensor(X_np, device=dev)                              # [L,N,k]
    Cemb_t = torch.tensor(Cemb, device=dev)                         # [n_c,k]
    relmat = np.stack([rel.get(u, np.zeros(n_c, np.float32)) for u in firms])
    logrel = torch.tensor(np.log1p(relmat), device=dev)            # [N,n_c]

    # multi-hot target + owned mask
    fi = {u: j for j, u in enumerate(firms)}
    target = torch.zeros(len(firms), n_c, device=dev)
    lossmask = torch.ones(len(firms), n_c, device=dev)
    for u in firms:
        for c in test_first[u]:
            if c in cidx:
                target[fi[u], cidx[c]] = 1.0
        for c in prior[u]:                      # 保有は損失から除外
            if c in cidx:
                lossmask[fi[u], cidx[c]] = 0.0

    model = ArchA(k, k, args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        logits = model(X, ts, t_future, Cemb_t, logrel)
        loss = (bce(logits, target) * lossmask).sum() / lossmask.sum()
        loss.backward(); opt.step()
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d} | loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X, ts, t_future, Cemb_t, logrel)
        proba = torch.sigmoid(logits).cpu().numpy()
    ode = tc.evaluate_entry(lambda u: proba[fi[u]], firms, prior, test_first,
                            ipcs, cidx, args.ks)
    print("ArchA(ODE) : " + "  ".join(f"{kk}={vv:.3f}" for kk, vv in ode.items()))

    print("\n[副] 企業参入 HEAD-TO-HEAD (rel-only vs ArchA):")
    for kk in [f"Hit@{x}" for x in args.ks] + ["MRR"]:
        win = "ArchA" if ode[kk] > rel_base[kk] else "rel-only"
        lift = (ode[kk] - rel_base[kk]) / (rel_base[kk] + 1e-9) * 100
        print(f"  {kk:8s} rel-only {rel_base[kk]:.3f}  ArchA {ode[kk]:.3f}  "
              f"({lift:+.0f}%) -> {win}")

    # --- 主: 技術トレンドランキング ---
    actual = tc.actual_trend_counts(s["df"], Y, H, ipcs, cidx)
    # ArchA トレンドスコア = 参入確率の企業横断和(保有は除外済みの proba をそのまま和)
    ode_trend = proba.sum(0)
    rel_trend = np.stack([rel.get(u, np.zeros(n_c, np.float32)) for u in firms]).sum(0)
    # momentum(直近年 Y の出願企業数), persistence(直近3年平均), popularity(≤Y entrant 累積)
    df = s["df"]
    momentum = np.zeros(n_c)
    for c, nn_ in __import__("collections").Counter(df[df.year == Y]["i"]).items():
        if c in cidx:
            momentum[cidx[c]] = nn_
    persist = np.zeros(n_c)
    rec = df[(df.year > Y - 3) & (df.year <= Y)]
    for c, nn_ in __import__("collections").Counter(rec["i"]).items():
        if c in cidx:
            persist[cidx[c]] = nn_ / 3.0
    print("\n[主] 技術トレンドランキング (target = entrant数 in Y+1..Y+H):")
    hdr = f"  {'method':<12}{'NDCG@10':>9}{'NDCG@20':>9}{'P@10':>7}{'P@20':>7}{'Spearman':>10}"
    print(hdr)
    for name, pr in [("ArchA(ODE)", ode_trend), ("rel-sum", rel_trend),
                     ("momentum", momentum), ("persistence", persist)]:
        print(f"  {name:<12}{tc.ndcg_at_k(pr, actual, 10):>9.3f}"
              f"{tc.ndcg_at_k(pr, actual, 20):>9.3f}"
              f"{tc.precision_at_k(pr, actual, 10):>7.2f}"
              f"{tc.precision_at_k(pr, actual, 20):>7.2f}"
              f"{tc.spearman(pr, actual):>10.3f}")
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
    ap.add_argument("--gonogo", action="store_true",
                    help="construction で go/no-go を1回実行")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    run(args.domain, args.train_end, args.horizon, args)


if __name__ == "__main__":
    main()
