"""prototype_bipartite_sde.py — Minimal Viable Prototype: Bipartite Latent SDE
for firm-CPC link prediction. Feasibility check (2-3 weeks compressed into one
script).

Architecture (kept minimal):
  - Each node (firm or CPC) has latent z_v(0) ∈ R^d
  - SDE: dz = μ_θ(z) dt + σ dW   (Euler-Maruyama, T steps)
  - μ_θ: small MLP per node type
  - σ: scalar learnable (aleatoric)
  - Link score: p(f -> g) = sigmoid(z_f · z_g)
  - epistemic uncertainty: variance across MC SDE samples

Trained on link-existence at year T_train. Evaluated on:
  - AUC on held-out new edges at T_test (predictive accuracy)
  - ECE on calibrated predictions
  - Comparison vs Adamic-Adar (the baseline we keep finding hard to beat)
  - Comparison vs static SVD embeddings (no SDE)

Go/no-go criterion: SDE adds AUC > 0.02 over AA on hard negatives.

Run:  python pnode_patent_runner/prototype_bipartite_sde.py --domain construction
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from diagnose_convergence_signal import ROOT
import recommender_firm as R


class BipartiteLatentSDE(nn.Module):
    def __init__(self, n_f, n_g, d=32, n_steps=8, sigma_init=0.10, drift_hidden=64):
        super().__init__()
        self.n_f, self.n_g, self.d, self.n_steps = n_f, n_g, d, n_steps
        self.z0_f = nn.Embedding(n_f, d); self.z0_g = nn.Embedding(n_g, d)
        nn.init.normal_(self.z0_f.weight, std=0.05)
        nn.init.normal_(self.z0_g.weight, std=0.05)
        self.drift_f = nn.Sequential(nn.Linear(d, drift_hidden), nn.Tanh(),
                                     nn.Linear(drift_hidden, d))
        self.drift_g = nn.Sequential(nn.Linear(d, drift_hidden), nn.Tanh(),
                                     nn.Linear(drift_hidden, d))
        self.log_sigma = nn.Parameter(torch.tensor(float(np.log(sigma_init))))

    def init_from(self, Cemb_init=None, Femb_init=None):
        with torch.no_grad():
            if Femb_init is not None:
                self.z0_f.weight.copy_(torch.tensor(Femb_init).float())
            if Cemb_init is not None:
                self.z0_g.weight.copy_(torch.tensor(Cemb_init).float())

    def trajectory(self, ids_f, ids_g, n_samples=1):
        """Euler-Maruyama forward, return per-sample final states."""
        dt = 1.0 / self.n_steps
        sigma = self.log_sigma.exp()
        zf0 = self.z0_f(ids_f); zg0 = self.z0_g(ids_g)
        outs_f, outs_g = [], []
        for _ in range(n_samples):
            zf = zf0; zg = zg0
            for _ in range(self.n_steps):
                zf = zf + self.drift_f(zf) * dt + sigma * np.sqrt(dt) * torch.randn_like(zf)
                zg = zg + self.drift_g(zg) * dt + sigma * np.sqrt(dt) * torch.randn_like(zg)
            outs_f.append(zf); outs_g.append(zg)
        return torch.stack(outs_f), torch.stack(outs_g)            # [S, B, d] each

    def predict(self, ids_f, ids_g, n_samples=4):
        zf_s, zg_s = self.trajectory(ids_f, ids_g, n_samples)
        logits = (zf_s * zg_s).sum(-1)                              # [S, B]
        return torch.sigmoid(logits).mean(0), logits.var(0)         # mean, epistemic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--n-firms", type=int, default=300)
    ap.add_argument("--n-cpcs", type=int, default=80)
    ap.add_argument("--train-year", type=int, default=2010)
    ap.add_argument("--test-year", type=int, default=2013)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--neg-ratio", type=int, default=5)
    ap.add_argument("--n-samples-eval", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed); R.LEVEL = "group"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  scale: firms<={args.n_firms} cpcs<={args.n_cpcs}")

    # --- load + subset ---
    df = pd.read_csv(ROOT / f"data/processed/bipartite_{args.domain}_firm.csv")
    df["year"] = pd.to_datetime(df["ts"], errors="coerce").dt.year
    df = df.dropna(subset=["year"]); df["year"] = df["year"].astype(int)
    df = df[(df.year >= 2000) & (df.year <= 2020)]
    df["i"] = df["i"].map(R.coarsen); df = df.dropna(subset=["i"]).drop_duplicates(["u", "i", "year"])

    # top CPCs by total activity
    top_cpcs = df.i.value_counts().head(args.n_cpcs).index.tolist()
    df = df[df.i.isin(top_cpcs)]
    # top firms by portfolio size up to train year, with new entries in horizon
    prior = df[df.year <= args.train_year].groupby("u")["i"].agg(set)
    nextf = df[(df.year > args.train_year) & (df.year <= args.train_year + args.horizon)].groupby("u")["i"].agg(set)
    cand_firms = [u for u in prior.index if u in nextf.index
                  and 3 <= len(prior[u]) <= 30 and len(nextf[u] - prior[u]) >= 1]
    cand_firms = sorted(cand_firms, key=lambda u: -len(prior[u]))[:args.n_firms]
    df = df[df.u.isin(cand_firms)]
    firms = sorted(df.u.unique()); cpcs = sorted(df.i.unique())
    fidx = {u: k for k, u in enumerate(firms)}; cidx = {c: k for k, c in enumerate(cpcs)}
    nF, nG = len(firms), len(cpcs)
    print(f"subset: {nF} firms × {nG} CPCs, {len(df)} edge-rows")

    # warm start: PPMI+SVD on train-time bipartite (firm x cpc binary)
    print("PPMI-SVD warm start ...")
    tr_pairs = df[df.year <= args.train_year][["u", "i"]].drop_duplicates()
    rows = tr_pairs.u.map(fidx).to_numpy(); cols = tr_pairs.i.map(cidx).to_numpy()
    M = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nF, nG))
    ru = np.asarray(M.sum(1)).ravel(); cc = np.asarray(M.sum(0)).ravel(); tot = M.sum()
    Mco = M.tocoo()
    with np.errstate(divide="ignore"):
        pmi = np.log((Mco.data * tot) / (ru[Mco.row] * cc[Mco.col] + 1e-12) + 1e-12)
    pmi = np.maximum(pmi, 0.0)
    P = csr_matrix((pmi, (Mco.row, Mco.col)), shape=(nF, nG))
    k = min(args.d, min(nF, nG) - 2)
    U, S, Vt = svds(P, k=k); s = np.sqrt(np.maximum(S, 0))
    Uemb = U * s; Cemb = Vt.T * s
    # static-SVD baseline: link score = U_f · V_g
    z_f_static = Uemb; z_g_static = Cemb

    # --- training set: positives = (f, g) edges at train_year, hard negatives via co-occurrence ---
    pos_pairs = tr_pairs.drop_duplicates().values
    pos_set = set((fidx[u], cidx[c]) for u, c in pos_pairs)
    train_pos = list(pos_set)
    print(f"train positives = {len(train_pos)}")
    train_neg = set()
    while len(train_neg) < args.neg_ratio * len(train_pos):
        f, g = rng.integers(0, nF), rng.integers(0, nG)
        if (f, g) not in pos_set and (f, g) not in train_neg:
            train_neg.add((f, g))
    train_neg = list(train_neg)
    train_pairs = train_pos + train_neg
    train_labels = np.array([1] * len(train_pos) + [0] * len(train_neg), dtype=np.float32)
    perm = rng.permutation(len(train_pairs))
    train_pairs = [train_pairs[i] for i in perm]; train_labels = train_labels[perm]

    # --- test set: NEW edges at test_year..test_year+horizon (not in prior) ---
    test_edges = df[(df.year > args.test_year) & (df.year <= args.test_year + args.horizon)][["u","i"]].drop_duplicates()
    seen_train = set((fidx[u], cidx[c]) for u, c in pos_pairs)
    test_pos = []
    for u, c in test_edges.values:
        if u not in fidx or c not in cidx:
            continue
        p = (fidx[u], cidx[c])
        if p not in seen_train:
            test_pos.append(p)
    test_pos = list(set(test_pos))
    # hard negatives for test: random non-edges of same firms
    test_firm_ids = list({f for f, _ in test_pos})
    test_neg = set()
    while len(test_neg) < args.neg_ratio * len(test_pos):
        f = rng.choice(test_firm_ids); g = rng.integers(0, nG)
        p = (int(f), int(g))
        if p not in seen_train and p not in set(test_pos) and p not in test_neg:
            test_neg.add(p)
    test_pairs = test_pos + list(test_neg)
    test_labels = np.array([1] * len(test_pos) + [0] * len(test_neg))
    print(f"test positives={len(test_pos)} negatives={len(test_neg)}")

    # --- baseline 1: Adamic-Adar (cooc graph from train edges) ---
    # build CPC-CPC cooc from co-firm pairs in train
    cooc = defaultdict(lambda: defaultdict(int))
    deg = defaultdict(int)
    for u, csu in df[df.year <= args.train_year].groupby("u")["i"].agg(set).items():
        cs = list(csu); cs = [c for c in cs if c in cidx]
        for a in cs:
            deg[a] = max(deg[a], 1)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                cooc[cidx[cs[i]]][cidx[cs[j]]] += 1
                cooc[cidx[cs[j]]][cidx[cs[i]]] += 1
    deg_g = {g: max(len(cooc[g]), 1) for g in cooc}
    # firm's prior CPCs (train)
    firm_prior = {fidx[u]: [cidx[c] for c in s if c in cidx] for u, s in
                  df[df.year <= args.train_year].groupby("u")["i"].agg(set).items() if u in fidx}
    def aa(f, g):
        prior_g = firm_prior.get(f, [])
        return sum(cooc[g].get(j, 0) / np.log(deg_g.get(j, 2) + 1) for j in prior_g)
    aa_scores = np.array([aa(f, g) for f, g in test_pairs])

    # --- baseline 2: static SVD score ---
    svd_scores = np.array([z_f_static[f] @ z_g_static[g] for f, g in test_pairs])

    # --- train SDE ---
    sde = BipartiteLatentSDE(nF, nG, d=args.d, n_steps=args.n_steps).to(device)
    sde.init_from(Cemb_init=Cemb, Femb_init=Uemb)
    opt = torch.optim.Adam(sde.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    f_ids_tr = torch.tensor([p[0] for p in train_pairs], dtype=torch.long, device=device)
    g_ids_tr = torch.tensor([p[1] for p in train_pairs], dtype=torch.long, device=device)
    y_tr = torch.tensor(train_labels, device=device)
    f_ids_te = torch.tensor([p[0] for p in test_pairs], dtype=torch.long, device=device)
    g_ids_te = torch.tensor([p[1] for p in test_pairs], dtype=torch.long, device=device)

    t0 = time.time()
    print("training Bipartite Latent SDE ...")
    for ep in range(1, args.epochs + 1):
        sde.train(); opt.zero_grad()
        zf_s, zg_s = sde.trajectory(f_ids_tr, g_ids_tr, n_samples=1)
        logits = (zf_s[0] * zg_s[0]).sum(-1)
        loss = bce(logits, y_tr)
        loss.backward(); opt.step()
        if ep % 20 == 0 or ep == 1:
            with torch.no_grad():
                sde.eval()
                p_mean, epi = sde.predict(f_ids_te, g_ids_te, n_samples=args.n_samples_eval)
                auc = roc_auc_score(test_labels, p_mean.cpu().numpy())
                print(f"  ep {ep:3d}  loss {loss.item():.4f}  sigma {sde.log_sigma.exp().item():.3f}  test AUC {auc:.3f}")
    print(f"training done in {time.time()-t0:.1f}s")

    # --- evaluate ---
    sde.eval()
    with torch.no_grad():
        p_mean, epi = sde.predict(f_ids_te, g_ids_te, n_samples=args.n_samples_eval)
    p_mean = p_mean.cpu().numpy(); epi = epi.cpu().numpy()

    aa_auc = roc_auc_score(test_labels, aa_scores)
    svd_auc = roc_auc_score(test_labels, svd_scores)
    sde_auc = roc_auc_score(test_labels, p_mean)

    # calibration ECE for SDE (isotonic on half of test, eval on other)
    rng2 = np.random.default_rng(0)
    idx = rng2.permutation(len(test_labels)); sp = len(idx) // 2
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_mean[idx[:sp]], test_labels[idx[:sp]])
    p_cal = iso.predict(p_mean[idx[sp:]]); y_h = test_labels[idx[sp:]]
    edges = np.quantile(p_cal, np.linspace(0, 1, 9)); edges[-1] += 1e-9
    ece = 0.0
    for i in range(len(edges) - 1):
        m = (p_cal >= edges[i]) & (p_cal < edges[i + 1])
        if m.sum() >= 5:
            ece += m.mean() * abs(p_cal[m].mean() - y_h[m].mean())

    print("\n" + "=" * 70)
    print(f"  TEST AUC (predict NEW edges in {args.test_year+1}-{args.test_year+args.horizon}):")
    print(f"    Adamic-Adar (relatedness baseline)  = {aa_auc:.3f}")
    print(f"    Static SVD embeddings (no SDE)       = {svd_auc:.3f}")
    print(f"    Bipartite Latent SDE (this)          = {sde_auc:.3f}")
    print(f"    SDE − AA gain                         = {sde_auc - aa_auc:+.3f}")
    print(f"    SDE − Static SVD gain                 = {sde_auc - svd_auc:+.3f}")
    print(f"  SDE calibration ECE (isotonic, held-out half) = {ece:.3f}")
    print(f"  Mean epistemic variance (SDE uncertainty)     = {epi.mean():.4f}")
    print("=" * 70)
    if sde_auc - aa_auc >= 0.02:
        print("  VERDICT: SDE adds signal vs AA -> proceed to scale up (Phase 2)")
    elif abs(sde_auc - aa_auc) < 0.01:
        print("  VERDICT: SDE ties AA -> pivot to UNCERTAINTY narrative (LGNSDE-light)")
    else:
        print("  VERDICT: SDE < AA -> proximity-bound pattern extends; do not invest more")
    print("=" * 70)


if __name__ == "__main__":
    main()
