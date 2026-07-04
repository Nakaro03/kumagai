"""diagnose_entry_content_vs_structure.py — go/no-go for Human-Aware Tech
Emergence Forecasting (the angle-shifted proposal).

Question: on inventor->new-CPC ENTRY, does a SEMANTIC content signal (patent-title
embeddings, independent of co-occurrence) beat the structural Adamic-Adar
relatedness and the popularity/persistence baseline — ESPECIALLY on INDUCTIVE
entries (inventor enters a CPC that never co-occurred with their prior CPCs, so
structure is blind)? Sourati-Evans (NHB 2023) predicts content+human helps most
exactly here, in the sparse/inductive regime.

Task: at anchor year Y, for inventors with >=3 prior CPCs, predict which NEW CPC
they file at Y+1 (not in their prior set). Scores for candidate (u, i):
  popularity : # inventors filing i in year Y          (persistence/EdgeBank)
  struct(AA) : sum_{j in prior(u)} AdamicAdar(i, j)    (relatedness, structure)
  content    : cos(content[i], mean_j content[prior(u)]) (semantic, AA-blind)
Hard negatives = CPCs that DO co-occur with prior(u) (struct>0) but were not
entered — stresses structure. Inductive positives = struct(i)==0.

Verdict: content adds AUC over [pop+struct] AND beats struct on inductive
entries => the content/human-aware angle escapes structure-saturation => build.

Run:  python pnode_patent_runner/diagnose_entry_content_vs_structure.py --domain energy
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/nakamuraroi/kumagai")
CSV = {"energy": "data/processed/bipartite_energy.csv",
       "construction": "data/processed/bipartite_construction.csv",
       "computing": "data/processed/bipartite_computing.csv"}


def cooc_graph(df, upto_year):
    """CPC-CPC co-occurrence counts (same inventor, same year) up to upto_year."""
    sub = df[df.year <= upto_year]
    cooc = defaultdict(Counter)
    deg = Counter()
    for (_, _), g in sub.groupby(["u", "year"]):
        cs = sorted(set(g["i"]))
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                cooc[cs[a]][cs[b]] += 1
                cooc[cs[b]][cs[a]] += 1
    for c in cooc:
        deg[c] = len(cooc[c])
    return cooc, deg


def build_eval(df, Y, content, cidx, rng, max_inv=1500, neg_ratio=5, max_pos=5):
    cooc, deg = cooc_graph(df, Y)
    pop = Counter(df[df.year == Y]["i"])                       # popularity at Y
    prior = df[df.year <= Y].groupby("u")["i"].agg(set)
    nextf = df[df.year == Y + 1].groupby("u")["i"].agg(set)
    all_codes = [c for c in cidx if (content[cidx[c]] != 0).any()]
    code_set = set(all_codes)

    def aa(i, Su):
        s = 0.0
        ci = cooc.get(i, {})
        for j in Su:
            w = ci.get(j, 0)
            if w:
                s += w / np.log(deg[j] + 2)
        return s

    rows = []  # (label, pop, struct, content, inductive)
    invs = [u for u in prior.index if u in nextf.index and len(prior[u]) >= 3]
    rng.shuffle(invs)
    for u in invs[:max_inv]:
        Su = [c for c in prior[u] if c in code_set]
        if len(Su) < 3:
            continue
        new = [c for c in (nextf[u] - prior[u]) if c in code_set]
        if not new:
            continue
        exp_vec = content[[cidx[c] for c in Su]].mean(0)
        exp_vec = exp_vec / (np.linalg.norm(exp_vec) + 1e-9)
        pos = new[:max_pos]
        # hard negatives: co-occur with Su (struct>0) but not entered
        hard_pool = set()
        for j in Su:
            hard_pool |= set(cooc.get(j, {}).keys())
        hard_pool = [c for c in hard_pool if c in code_set and c not in prior[u]
                     and c not in nextf[u]]
        rng.shuffle(hard_pool)
        negs = hard_pool[:len(pos) * neg_ratio]
        if len(negs) < 2:
            continue
        for c, lab in [(c, 1) for c in pos] + [(c, 0) for c in negs]:
            st = aa(c, Su)
            cs = float(content[cidx[c]] @ exp_vec)
            rows.append((lab, pop.get(c, 0), st, cs, 1 if st == 0 else 0))
    return np.array(rows, float)


def auc(y, s):
    try:
        return roc_auc_score(y, s)
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="energy")
    ap.add_argument("--train-year", type=int, default=2015)
    ap.add_argument("--test-year", type=int, default=2018)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    import random; random.seed(args.seed)

    df = pd.read_csv(ROOT / CSV[args.domain])
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    z = np.load(ROOT / f"data/processed/cpc_content_{args.domain}.npz", allow_pickle=True)
    codes, emb = list(z["codes"]), z["emb"]
    cidx = {c: k for k, c in enumerate(codes)}
    print(f"domain={args.domain} rows={len(df)} codes={len(codes)} "
          f"content_dim={emb.shape[1]}  train_year={args.train_year} test_year={args.test_year}")

    import builtins
    rs = np.random.RandomState(args.seed)
    class Sh:  # rng with .shuffle for lists
        @staticmethod
        def shuffle(x): rs.shuffle(x)
    tr = build_eval(df, args.train_year, emb, cidx, Sh)
    te = build_eval(df, args.test_year, emb, cidx, Sh)
    if len(tr) < 50 or len(te) < 50:
        print("insufficient eval rows — abort"); return

    yte = te[:, 0]
    pop_s, st_s, ct_s, ind = te[:, 1], te[:, 2], te[:, 3], te[:, 4]
    print(f"\nTEST set: n={len(te)} (pos={int(yte.sum())}), "
          f"inductive positives (struct==0): {int(((ind == 1) & (yte == 1)).sum())}")

    print("\nHard-negative AUC (whole test set):")
    print(f"  popularity (persistence) = {auc(yte, pop_s):.3f}")
    print(f"  struct  Adamic-Adar      = {auc(yte, st_s):.3f}")
    print(f"  content (semantic)       = {auc(yte, ct_s):.3f}")

    # fusion LR trained on train-year, tested on test-year
    def fuse(cols):
        Xtr, Xte = tr[:, cols], te[:, cols]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), tr[:, 0])
        return auc(yte, clf.predict_proba(sc.transform(Xte))[:, 1])
    a_ps = fuse([1, 2])         # pop + struct
    a_psc = fuse([1, 2, 3])     # pop + struct + content
    print(f"\nFusion LR (temporal train->test):")
    print(f"  pop + struct            = {a_ps:.3f}")
    print(f"  pop + struct + content  = {a_psc:.3f}   (content gain {a_psc - a_ps:+.3f})")

    # INDUCTIVE slice: pure struct==0 positives vs hard negatives
    mask = (ind == 1) | (yte == 0)
    n_ind = int((yte[mask] == 1).sum())
    if n_ind >= 5 and (yte[mask] == 0).sum() >= 5:
        print(f"\nPURE-INDUCTIVE entries (struct==0 positives vs negatives, n_pos={n_ind}):")
        print(f"  struct  (blind here)    = {auc(yte[mask], st_s[mask]):.3f}")
        print(f"  popularity              = {auc(yte[mask], pop_s[mask]):.3f}")
        print(f"  content (semantic)      = {auc(yte[mask], ct_s[mask]):.3f}")
    else:
        print(f"\n(only {n_ind} pure-inductive positives — using weak-structure slice instead)")

    # WEAK-STRUCTURE slice: rows in the bottom tertile of struct (structure barely
    # informative) — where content should matter most. Robust when pure-inductive is rare.
    thr = np.quantile(st_s, 0.33)
    wmask = st_s <= thr
    ind_content = float("nan")
    if (yte[wmask] == 1).sum() >= 5 and (yte[wmask] == 0).sum() >= 5:
        ind_content = auc(yte[wmask], ct_s[wmask])
        print(f"\nWEAK-STRUCTURE slice (struct<=33th pct, n={int(wmask.sum())}, "
              f"pos={int(yte[wmask].sum())}):")
        print(f"  struct                  = {auc(yte[wmask], st_s[wmask]):.3f}")
        print(f"  popularity              = {auc(yte[wmask], pop_s[wmask]):.3f}")
        print(f"  content (semantic)      = {ind_content:.3f}")

    gain = a_psc - a_ps
    print("\n" + "=" * 64)
    print(f"  content gain over pop+struct = {gain:+.3f};  inductive content AUC = {ind_content:.3f}")
    if gain >= 0.03 and (ind_content == ind_content and ind_content >= 0.60):
        print("  VERDICT: GO. Content carries signal structure cannot — strongest on")
        print("           inductive entries. Build Human-Aware Emergence Forecasting.")
    elif gain >= 0.02 or (ind_content == ind_content and ind_content >= 0.60):
        print("  VERDICT: PARTIAL. Some content signal; promising but not decisive.")
    else:
        print("  VERDICT: NO-GO. Content does not beat structure here — saturation")
        print("           extends even to semantics. Strengthens the limits paper.")
    print("=" * 64)


if __name__ == "__main__":
    main()
