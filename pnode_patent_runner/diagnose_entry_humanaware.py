"""diagnose_entry_humanaware.py — the CORE go/no-go for the proposal.

The content-only diagnostic (diagnose_entry_content_vs_structure.py) lifted the
fusion +0.031 in sparse construction but failed pure-inductive (content AUC
0.413). It tested only "semantic similarity to the inventor's OWN prior CPCs."
The proposal's actual engine — Sourati-Evans's HUMAN-AWARE signal (who is
positioned in the expertise network) — was untested.

Here we add it. Human-aware score = cosine in a node2vec-equivalent (PPMI+SVD)
embedding of the inventor x CPC bipartite graph up to year Y. This captures
TRANSITIVE expertise position (technologies reached via u's peers / 2+-hop
neighbourhood) that direct CPC-CPC co-occurrence (struct AA) cannot. We then ask:
does human-aware beat content+struct, ESPECIALLY on pure-inductive entries
(struct==0), where similarity-to-own-prior was blind?

  human-aware lifts fusion AND cracks pure-inductive (AUC>0.6) => proposal GO.
  no lift / still <0.5 inductive => structure-saturation extends to content+human
  => the limits paper is the honest output.

Run:  python pnode_patent_runner/diagnose_entry_humanaware.py --domain construction
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/nakamuraroi/kumagai")
CSV = {"energy": "data/processed/bipartite_energy.csv",
       "construction": "data/processed/bipartite_construction.csv",
       "computing": "data/processed/bipartite_computing.csv"}


def cooc_graph(df, Y):
    sub = df[df.year <= Y]
    cooc = defaultdict(Counter); deg = Counter()
    for (_, _), g in sub.groupby(["u", "year"]):
        cs = sorted(set(g["i"]))
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                cooc[cs[a]][cs[b]] += 1; cooc[cs[b]][cs[a]] += 1
    for c in cooc:
        deg[c] = len(cooc[c])
    return cooc, deg


def bipartite_embed(df, Y, cidx, dim=64):
    """node2vec-equivalent PPMI+SVD embedding of inventor x CPC graph up to Y."""
    sub = df[df.year <= Y][["u", "i"]].drop_duplicates()
    sub = sub[sub.i.isin(cidx)]
    invs = sorted(sub.u.unique())
    uidx = {u: k for k, u in enumerate(invs)}
    r = sub.u.map(uidx).to_numpy()
    c = sub.i.map(cidx).to_numpy()
    n_u, n_c = len(invs), len(cidx)
    M = csr_matrix((np.ones(len(r)), (r, c)), shape=(n_u, n_c))
    # PPMI on the bipartite counts
    ru = np.asarray(M.sum(1)).ravel(); cc = np.asarray(M.sum(0)).ravel()
    tot = M.sum()
    Mco = M.tocoo()
    with np.errstate(divide="ignore"):
        pmi = np.log((Mco.data * tot) / (ru[Mco.row] * cc[Mco.col] + 1e-12) + 1e-12)
    pmi = np.maximum(pmi, 0.0)
    P = csr_matrix((pmi, (Mco.row, Mco.col)), shape=(n_u, n_c))
    k = min(dim, min(n_u, n_c) - 2)
    U, S, Vt = svds(P, k=k)
    s = np.sqrt(np.maximum(S, 0))
    Uemb = U * s                      # [n_u,k] inventor embeddings
    Cemb = (Vt.T) * s                 # [n_c,k] CPC embeddings
    # normalize for cosine
    Uemb /= (np.linalg.norm(Uemb, axis=1, keepdims=True) + 1e-9)
    Cemb /= (np.linalg.norm(Cemb, axis=1, keepdims=True) + 1e-9)
    return uidx, Uemb, Cemb


def build_eval(df, Y, content, cidx, rng, max_inv=1500, neg_ratio=5, max_pos=5):
    cooc, deg = cooc_graph(df, Y)
    uidx, Uemb, Cemb = bipartite_embed(df, Y, cidx)
    pop = Counter(df[df.year == Y]["i"])
    prior = df[df.year <= Y].groupby("u")["i"].agg(set)
    nextf = df[df.year == Y + 1].groupby("u")["i"].agg(set)
    code_set = {c for c in cidx if (content[cidx[c]] != 0).any()}

    def aa(i, Su):
        ci = cooc.get(i, {})
        return sum(ci.get(j, 0) / np.log(deg[j] + 2) for j in Su if ci.get(j, 0))

    rows = []
    invs = [u for u in prior.index if u in nextf.index and len(prior[u]) >= 3 and u in uidx]
    rng.shuffle(invs)
    for u in invs[:max_inv]:
        Su = [c for c in prior[u] if c in code_set]
        if len(Su) < 3:
            continue
        new = [c for c in (nextf[u] - prior[u]) if c in code_set]
        if not new:
            continue
        exp_vec = content[[cidx[c] for c in Su]].mean(0)
        exp_vec /= (np.linalg.norm(exp_vec) + 1e-9)
        u_emb = Uemb[uidx[u]]
        pos = new[:max_pos]
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
            ha = float(u_emb @ Cemb[cidx[c]])
            rows.append((lab, pop.get(c, 0), st, cs, ha, 1 if st == 0 else 0))
    return np.array(rows, float)


def auc(y, s):
    try:
        return roc_auc_score(y, s)
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="construction")
    ap.add_argument("--train-year", type=int, default=2015)
    ap.add_argument("--test-year", type=int, default=2018)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rs = np.random.RandomState(args.seed)
    class Sh:
        @staticmethod
        def shuffle(x): rs.shuffle(x)

    df = pd.read_csv(ROOT / CSV[args.domain])
    df["year"] = pd.to_datetime(df["ts"]).dt.year
    z = np.load(ROOT / f"data/processed/cpc_content_{args.domain}.npz", allow_pickle=True)
    codes, emb = list(z["codes"]), z["emb"]
    cidx = {c: k for k, c in enumerate(codes)}
    print(f"domain={args.domain} rows={len(df)} codes={len(codes)} "
          f"train={args.train_year} test={args.test_year}")

    tr = build_eval(df, args.train_year, emb, cidx, Sh)
    te = build_eval(df, args.test_year, emb, cidx, Sh)
    if len(tr) < 50 or len(te) < 50:
        print("insufficient — abort"); return

    y = te[:, 0]; pop_s, st_s, ct_s, ha_s, ind = te[:, 1], te[:, 2], te[:, 3], te[:, 4], te[:, 5]
    print(f"\nTEST n={len(te)} pos={int(y.sum())} pure-inductive pos={int(((ind==1)&(y==1)).sum())}")
    print("\nHard-negative AUC (whole test):")
    for nm, s in [("popularity", pop_s), ("struct AA", st_s), ("content", ct_s),
                  ("human-aware (bipartite emb)", ha_s)]:
        print(f"  {nm:28s} = {auc(y, s):.3f}")

    def fuse(cols):
        sc = StandardScaler().fit(tr[:, cols])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(tr[:, cols]), tr[:, 0])
        return auc(y, clf.predict_proba(sc.transform(te[:, cols]))[:, 1])
    a_ps = fuse([1, 2]); a_psc = fuse([1, 2, 3]); a_psch = fuse([1, 2, 3, 4]); a_psh = fuse([1, 2, 4])
    print("\nFusion LR (temporal train->test):")
    print(f"  pop+struct                 = {a_ps:.3f}")
    print(f"  pop+struct+content         = {a_psc:.3f}")
    print(f"  pop+struct+human           = {a_psh:.3f}")
    print(f"  pop+struct+content+human   = {a_psch:.3f}   (full vs pop+struct {a_psch-a_ps:+.3f})")

    # pure-inductive & weak-structure slices
    def slice_report(m, tag):
        if (y[m] == 1).sum() < 5 or (y[m] == 0).sum() < 5:
            print(f"\n({tag}: too few)"); return None
        print(f"\n{tag} (n={int(m.sum())}, pos={int(y[m].sum())}):")
        for nm, s in [("struct", st_s), ("popularity", pop_s), ("content", ct_s),
                      ("human-aware", ha_s)]:
            print(f"  {nm:14s} = {auc(y[m], s[m]):.3f}")
        return auc(y[m], ha_s[m])
    ind_ha = slice_report((ind == 1) | (y == 0), "PURE-INDUCTIVE (struct==0 pos vs negs)")
    slice_report(st_s <= np.quantile(st_s, 0.33), "WEAK-STRUCTURE slice")

    print("\n" + "=" * 64)
    ha_gain = a_psch - a_psc
    print(f"  human-aware gain over content fusion = {ha_gain:+.3f}; "
          f"pure-inductive human-aware AUC = {ind_ha if ind_ha is not None else float('nan'):.3f}")
    if ha_gain >= 0.02 and ind_ha is not None and ind_ha >= 0.60:
        print("  VERDICT: GO. Human-aware position adds signal AND cracks pure-inductive")
        print("           => the proposal's engine works. Build it.")
    elif (ha_gain >= 0.02) or (ind_ha is not None and ind_ha >= 0.58):
        print("  VERDICT: PARTIAL. Human-aware helps somewhat; promising, not decisive.")
    else:
        print("  VERDICT: NO-GO. Human-aware does not crack inductive => saturation extends")
        print("           to content+human. The limits paper is the honest output.")
    print("=" * 64)


if __name__ == "__main__":
    main()
