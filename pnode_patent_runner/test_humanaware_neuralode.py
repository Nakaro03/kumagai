"""test_humanaware_neuralode.py — does CONTINUOUS-TIME Neural ODE beat the static
human-aware embedding on jumps?

human-aware so far = a STATIC PPMI+SVD embedding of the inventor x CPC bipartite
at year Y. The open question: would modelling its TEMPORAL DRIFT with a Neural ODE
(like the PISDE/X5 machinery) crack the jumps that the static embedding leaves at
chance? We test three readouts of the SAME bipartite embedding on the jump task
(AA==0 new CPC-CPC convergences):

  STATIC     : cos(C_Y[i], C_Y[j])
  VELOCITY   : extrapolate C(Y+1) ~ C_Y + (C_Y - C_{Y-1}); cos of extrapolated  (poor-man's ODE)
  NEURAL-ODE : learn dz/dt = MLP(z) on the aligned yearly embedding trajectory
               (torchdiffeq), integrate C_Y -> C(Y+1); cos of extrapolated

Per-year embeddings are Procrustes-aligned to year Y (SVD rotation is gauge). If
VELOCITY already adds nothing over STATIC, a Neural ODE (a flexible fit of the
same drift) is almost certainly futile — confirming the limit is the task, not
the architecture.

Run:  python pnode_patent_runner/test_humanaware_neuralode.py --domain construction
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torchdiffeq import odeint

from diagnose_convergence_signal import ROOT, CSV, yearly_cooc_edges, build_adj
from diagnose_entry_humanaware import bipartite_embed


def procrustes(A, B):
    """orthogonal R minimizing ||A R - B|| (align A onto B). A,B: [n,d]."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


def common_cpc_embeddings(df, years, cidx, dim=64):
    """Per-year CPC embeddings on a shared code set, Procrustes-aligned to last year."""
    embs, codesets = {}, []
    for t in years:
        _, _, Cemb = bipartite_embed(df, t, cidx, dim=dim)
        embs[t] = Cemb
        codesets.append({c for c in cidx if (Cemb[cidx[c]] != 0).any()})
    common = sorted(set.intersection(*codesets))
    idx = np.array([cidx[c] for c in common])
    ref = embs[years[-1]][idx]
    aligned = {}
    for t in years:
        E = embs[t][idx]
        aligned[t] = E @ procrustes(E, ref)
    cpos = {c: k for k, c in enumerate(common)}
    return common, cpos, aligned       # aligned[t]: [n_common, d]


def neural_ode_extrapolate(aligned, years, device, epochs=400):
    """Fit dz/dt=MLP(z) to consecutive aligned snapshots; extrapolate last->+1."""
    Z = {t: torch.tensor(aligned[t], dtype=torch.float32, device=device) for t in years}
    f = nn.Sequential(nn.Linear(Z[years[0]].shape[1], 128), nn.Tanh(),
                      nn.Linear(128, Z[years[0]].shape[1])).to(device)

    class ODEF(nn.Module):
        def forward(self, t, z): return f(z)
    func = ODEF()
    opt = torch.optim.Adam(f.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad(); loss = 0.0
        for a, b in zip(years[:-1], years[1:]):
            pred = odeint(func, Z[a], torch.tensor([0.0, 1.0], device=device),
                          method="rk4", options={"step_size": 0.5})[-1]
            loss = loss + ((pred - Z[b]) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        ext = odeint(func, Z[years[-1]], torch.tensor([0.0, 1.0], device=device),
                     method="rk4", options={"step_size": 0.5})[-1]
    return ext.cpu().numpy()


def jump_pairs(df, Y, rng, max_pos=800, neg_ratio=5):
    cum = set()
    for y in range(Y - 8, Y + 1):
        cum |= yearly_cooc_edges(df, y)
    adj = build_adj(cum); present = set(adj)
    def aa(i, j): return len(adj.get(i, set()) & adj.get(j, set()))
    nxt = yearly_cooc_edges(df, Y + 1)
    jumps = [e for e in nxt if e not in cum and set(e) <= present and aa(*tuple(e)) == 0]
    rng.shuffle(jumps); pos = jumps[:max_pos]
    pl = list(present); negs = set(); tries = 0
    while len(negs) < len(pos) * neg_ratio and tries < len(pos) * neg_ratio * 80:
        a, b = rng.choice(pl, 2, replace=False); e = frozenset((a, b)); tries += 1
        if e not in cum and e not in nxt and aa(a, b) == 0:
            negs.add(e)
    return pos, list(negs)


def score(pairs, emb, cpos):
    out = []
    for e in pairs:
        i, j = tuple(e)
        if i in cpos and j in cpos:
            a, b = emb[cpos[i]], emb[cpos[j]]
            out.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        else:
            out.append(np.nan)
    return np.array(out)


def auc_valid(y, s):
    m = ~np.isnan(s)
    try: return roc_auc_score(y[m], s[m]), int(m.sum())
    except ValueError: return float("nan"), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--Y", type=int, default=2018)
    ap.add_argument("--hist", type=int, default=3, help="years of history before Y")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(ROOT / CSV[args.domain]); df["year"] = pd.to_datetime(df["ts"]).dt.year
    z = np.load(ROOT / f"data/processed/cpc_content_{args.domain}.npz", allow_pickle=True)
    cidx = {c: k for k, c in enumerate(list(z["codes"]))}
    years = list(range(args.Y - args.hist, args.Y + 1))
    print(f"domain={args.domain} Y={args.Y} snapshots={years}")

    common, cpos, aligned = common_cpc_embeddings(df, years, cidx)
    print(f"common CPCs across snapshots: {len(common)}")

    static = aligned[args.Y]
    velocity = aligned[args.Y] + (aligned[args.Y] - aligned[args.Y - 1])
    print("training Neural ODE on aligned trajectory ...")
    ode = neural_ode_extrapolate(aligned, years, device)

    pos, negs = jump_pairs(df, args.Y, rng)
    pairs = pos + negs
    y = np.array([1] * len(pos) + [0] * len(negs))
    print(f"\nJUMP test: pos(AA==0 convergences)={len(pos)} negs={len(negs)}")
    print("\nAUC on jumps (same bipartite embedding, three temporal readouts):")
    for nm, E in [("STATIC      C_Y", static),
                  ("VELOCITY    extrapolation", velocity),
                  ("NEURAL-ODE  extrapolation", ode)]:
        a, n = auc_valid(y, score(pairs, E, cpos))
        print(f"  {nm:28s} = {a:.3f}  (n={n})")

    s_stat = score(pairs, static, cpos)
    s_velo = score(pairs, velocity, cpos)
    s_ode = score(pairs, ode, cpos)
    a_stat = auc_valid(y, s_stat)[0]; a_velo = auc_valid(y, s_velo)[0]; a_ode = auc_valid(y, s_ode)[0]
    print("\n" + "=" * 64)
    print(f"  velocity gain over static  = {a_velo - a_stat:+.3f}")
    print(f"  neural-ODE gain over static= {a_ode - a_stat:+.3f}")
    if max(a_velo, a_ode) - a_stat >= 0.03:
        print("  VERDICT: temporal drift (ODE) DOES add on jumps — worth a full PISDE study.")
    else:
        print("  VERDICT: temporal drift adds ~nothing on jumps => continuous-time Neural")
        print("           ODE/SDE does NOT beat the static embedding. The limit is the task.")
    print("=" * 64)


if __name__ == "__main__":
    main()
