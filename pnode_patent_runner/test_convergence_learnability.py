"""test_convergence_learnability.py — sharpened go/no-go for PISDE-Converge.

Phase-0 (diagnose_convergence_timing.py) showed that on the SPARSE construction
graph, topological scores collapse under hard negatives (Adamic-Adar AUC
0.89 -> 0.68). But a baseline DROPPING only proves topology fails — NOT that a
learned model recovers it (the hard-negative task could just be hard for
everyone). This script tests the missing half:

  Does a LEARNED node embedding beat Adamic-Adar's hard-negative AUC?

Embedding = PPMI of the co-occurrence adjacency + truncated SVD (the
Levy-Goldberg closed form that node2vec/DeepWalk implicitly factorize). No extra
deps (scipy only). Two learned predictors, both on the SAME hard-negative pairs
that AA scores 0.68 on, under a TRUE temporal split:

  (L1) unsupervised: cosine similarity of embeddings.
  (L2) supervised: logistic regression on [emb hadamard, |emb diff|, AA, CN,
       Jaccard, deg_a, deg_b], weights learned on a TRAIN year, applied to a
       held-out TEST year.

Verdict:
  learned hard-neg AUC >> AA hard-neg AUC  => headroom is REAL & learnable
                                              => build PISDE-Converge.
  learned ~ AA (both ~0.68)                => task is just hard, not headroom
                                              => prefer descriptive / neg-results.

Run (main env: torch/scipy/sklearn present):
  python pnode_patent_runner/test_convergence_learnability.py --domain construction
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from diagnose_convergence_signal import ROOT, CSV, yearly_cooc_edges, build_adj, score_pairs
from diagnose_convergence_timing import sample_present_nonedges


def ppmi_svd_embeddings(cum_edges: set, dim: int):
    """PPMI(co-occurrence adjacency) + truncated SVD -> {node: vec}."""
    nodes = sorted({n for e in cum_edges for n in e})
    idx = {n: k for k, n in enumerate(nodes)}
    rows, cols = [], []
    for e in cum_edges:
        a, b = tuple(e)
        rows += [idx[a], idx[b]]
        cols += [idx[b], idx[a]]
    n = len(nodes)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(1)).ravel()
    vol = deg.sum()
    # PPMI on the nonzeros: log(vol / (deg_a deg_b)), clipped at 0
    A = A.tocoo()
    with np.errstate(divide="ignore"):
        pmi = np.log(vol / (deg[A.row] * deg[A.col] + 1e-12))
    pmi = np.maximum(pmi, 0.0)
    P = csr_matrix((pmi, (A.row, A.col)), shape=(n, n))
    k = min(dim, n - 2)
    if k < 2:
        return {}, idx
    U, S, _ = svds(P, k=k)
    emb = U * np.sqrt(np.maximum(S, 0))
    return {nodes[i]: emb[i] for i in range(n)}, idx


def emb_feats(pairs, emb):
    """Cosine sim + raw [hadamard, |diff|] features for a list of pairs."""
    cos, had, dif, mask = [], [], [], []
    d = len(next(iter(emb.values()))) if emb else 0
    for e in pairs:
        a, b = tuple(e)
        if a in emb and b in emb:
            va, vb = emb[a], emb[b]
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            cos.append(float(va @ vb / (na * nb + 1e-12)))
            had.append(va * vb)
            dif.append(np.abs(va - vb))
            mask.append(True)
        else:
            cos.append(0.0); had.append(np.zeros(d)); dif.append(np.zeros(d)); mask.append(False)
    return np.array(cos), np.array(had), np.array(dif), np.array(mask)


def build_split(df, y0, args, rng):
    """Hard-negative binary set predicting convergences at y0+1 from graph<=y0."""
    cum = set()
    for y in range(args.year_start - 5, y0 + 1):
        cum |= yearly_cooc_edges(df, y)
    adj = build_adj(cum)
    present = set(adj.keys())
    present_list = list(present)
    nxt = yearly_cooc_edges(df, y0 + 1)
    pos = [e for e in nxt if e not in cum and set(e) <= present]
    if len(pos) > args.max_pos:
        pos = [pos[k] for k in rng.choice(len(pos), args.max_pos, replace=False)]
    if len(pos) < 20:
        return None
    neg = sample_present_nonedges(present_list, cum | nxt, len(pos) * args.neg_ratio,
                                  rng, require_common=True, adj=adj)
    pairs = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg))
    return {"cum": cum, "adj": adj, "pairs": pairs, "labels": labels}


def topo_matrix(pairs, adj):
    s = score_pairs(pairs, adj)
    deg = {n: len(adj[n]) for n in adj}
    da = np.array([deg.get(tuple(e)[0], 0) for e in pairs], float)
    dbb = np.array([deg.get(tuple(e)[1], 0) for e in pairs], float)
    return np.column_stack([s["CommonNeighbors"], s["AdamicAdar"], s["Jaccard"], da, dbb]), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--year-start", type=int, default=2010)
    ap.add_argument("--train-year", type=int, default=2015)
    ap.add_argument("--test-year", type=int, default=2018)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--neg-ratio", type=int, default=5)
    ap.add_argument("--max-pos", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(ROOT / CSV[args.domain])
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    df = df[(df.year >= args.year_start - 5) & (df.year <= args.test_year + 1)]
    print(f"domain={args.domain} train_year={args.train_year} test_year={args.test_year} "
          f"dim={args.dim} (TRUE temporal split, HARD negatives)")

    tr = build_split(df, args.train_year, args, rng)
    te = build_split(df, args.test_year, args, rng)
    if tr is None or te is None:
        print("insufficient positives — abort"); return

    emb_tr, _ = ppmi_svd_embeddings(tr["cum"], args.dim)
    emb_te, _ = ppmi_svd_embeddings(te["cum"], args.dim)

    # ---- baseline: AA alone on the TEST hard-negative set ----
    Xte_topo, ste = topo_matrix(te["pairs"], te["adj"])
    aa_auc = roc_auc_score(te["labels"], ste["AdamicAdar"])

    # ---- L1: unsupervised embedding cosine on TEST ----
    cos_te, had_te, dif_te, _ = emb_feats(te["pairs"], emb_te)
    l1_auc = roc_auc_score(te["labels"], cos_te)

    # ---- L2: supervised LR (train on TRAIN year, eval on TEST year) ----
    Xtr_topo, _ = topo_matrix(tr["pairs"], tr["adj"])
    cos_tr, had_tr, dif_tr, _ = emb_feats(tr["pairs"], emb_tr)
    Xtr = np.column_stack([had_tr, dif_tr, cos_tr, Xtr_topo])
    Xte = np.column_stack([had_te, dif_te, cos_te, Xte_topo])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), tr["labels"])
    l2_auc = roc_auc_score(te["labels"], clf.predict_proba(sc.transform(Xte))[:, 1])

    # LR on topology features only — isolates whether the EMBEDDING (not just
    # combining topo features) is what helps.
    sc2 = StandardScaler().fit(Xtr_topo)
    clf2 = LogisticRegression(max_iter=2000).fit(sc2.transform(Xtr_topo), tr["labels"])
    topo_lr_auc = roc_auc_score(te["labels"], clf2.predict_proba(sc2.transform(Xte_topo))[:, 1])

    print(f"\nTEST-year hard-negative AUC (n_pos={int(te['labels'].sum())}, "
          f"n_neg={len(te['labels'])-int(te['labels'].sum())}):")
    print(f"  Adamic-Adar (baseline)            = {aa_auc:.3f}")
    print(f"  Topo-features LogReg              = {topo_lr_auc:.3f}")
    print(f"  L1 embedding cosine (unsupervised)= {l1_auc:.3f}")
    print(f"  L2 embedding+topo LogReg (sup.)   = {l2_auc:.3f}")

    best_learned = max(l1_auc, l2_auc)
    gain = best_learned - aa_auc
    print("\n" + "=" * 60)
    print(f"  best learned AUC = {best_learned:.3f}  vs AA = {aa_auc:.3f}  (gain {gain:+.3f})")
    if gain >= 0.05:
        print("  VERDICT: GO. A learned embedding RECOVERS what AA loses on hard")
        print("           negatives => headroom is real & learnable. Build PISDE-Converge.")
    elif gain >= 0.02:
        print("  VERDICT: MARGINAL. Some learnable headroom; SDE must earn its keep via")
        print("           interpretability/uncertainty, not raw AUC.")
    else:
        print("  VERDICT: NO-GO on accuracy. Learned model ~ AA => the hard-negative task")
        print("           is just hard, not headroom. Prefer descriptive / neg-results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
