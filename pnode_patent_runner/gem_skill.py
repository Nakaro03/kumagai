#!/usr/bin/env python3
"""
GEM (Gated Entry Model) v0 + Ceiling-Aware Skill Score（計画 Phase 1–3）。

- 天井（義務ベースライン）: popularity / seen_before / relatedness(共起コサイン) / seen+pop。
  Skill = (AUC_model − AUC_maxceiling) / (1 − AUC_maxceiling)
- GEM: logit = offset[log1p deg_j(t)]（固定・ceiling-anchored） + α·rel + γ0·mom + γ1·mom·burst + δ·seen + b
  学習は天井が説明できない残差のみ。unanchored 版（全特徴自由）も比較用に学習。
- 特徴は常に遷移元年 t 以前の情報のみ（rel/seen は年 ≤ t で再構築）→ リークなし。
- 評価ペアは `predictability_popularity_auc.eval_pairs`（学習モデル holdout と同一）。
- シードは学習ペアのサンプリングにのみ作用（評価ペアは決定的）。

例:
  python -m pnode_patent_runner.gem_skill \\
    --output-json pnode_patent_runner/outputs/predictability_map/gem_skill.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from scipy import sparse, stats
from sklearn.metrics import roc_auc_score

from pnode_patent_runner.predictability_popularity_auc import eval_pairs

BURST_PCT = 80.0


# ---------------- 年次構造（すべて年 ≤ t の情報のみ） ----------------

def bip_edges(graphs, y: int, num_corps: int) -> Tuple[np.ndarray, np.ndarray]:
    ei = graphs[y].edge_index.cpu()
    m = (ei[0] < num_corps) & (ei[1] >= num_corps)
    return ei[0, m].numpy(), (ei[1, m].numpy() - num_corps)


class YearContext:
    """遷移元年 t 時点の特徴計算コンテキスト。"""

    def __init__(self, graphs, years: List[int], t: int, num_corps: int, n_items: int):
        self.t = t
        self.n_items = n_items
        # 次数と momentum
        u_t, i_t = bip_edges(graphs, t, num_corps)
        self.deg = np.bincount(i_t, minlength=n_items).astype(float)
        prev = [y for y in years if y < t]
        if prev:
            _, i_p = bip_edges(graphs, prev[-1], num_corps)
            deg_p = np.bincount(i_p, minlength=n_items).astype(float)
            self.mom = np.log1p(self.deg) - np.log1p(deg_p)
        else:
            self.mom = np.zeros(n_items)
        pos_mom = self.mom[self.mom > 0]
        thr = np.percentile(pos_mom, BURST_PCT) if pos_mom.size else np.inf
        self.burst = (self.mom >= thr).astype(float)

        # 年 ≤ t の (u, i) 履歴 → seen / relatedness
        us, is_ = [], []
        for y in [y for y in years if y <= t]:
            uy, iy = bip_edges(graphs, y, num_corps)
            us.append(uy)
            is_.append(iy)
        ua = np.concatenate(us)
        ia = np.concatenate(is_)
        keys = ua.astype(np.int64) * n_items + ia
        self.seen_keys = np.unique(keys)
        B = sparse.csr_matrix(
            (np.ones(len(ua)), (ua, ia)), shape=(num_corps, n_items)
        )
        B.data[:] = 1.0
        C = (B.T @ B).tocsr()  # item×item 共起
        d = np.sqrt(C.diagonal())
        d[d == 0] = 1.0
        Dinv = sparse.diags(1.0 / d)
        self.sim = (Dinv @ C @ Dinv).tocsr()  # コサイン類似
        self.B = B.tocsr()

    def seen(self, u: np.ndarray, j: np.ndarray) -> np.ndarray:
        k = u.astype(np.int64) * self.n_items + j
        return np.isin(k, self.seen_keys, kind="sort").astype(float)

    def rel(self, u: np.ndarray, j: np.ndarray) -> np.ndarray:
        """score(u, j) = Σ_{j' ∈ items_t(u)} sim(j', j)。ユーザー単位でまとめて計算。

        j' = j 自身の寄与 sim(j,j)=1 は seen 特徴と重複するため除く。
        """
        out = np.zeros(len(u), dtype=float)
        order = np.argsort(u, kind="stable")
        us = u[order]
        indptr, indices = self.B.indptr, self.B.indices
        start = 0
        while start < len(us):
            uu = us[start]
            end = start
            while end < len(us) and us[end] == uu:
                end += 1
            items_u = indices[indptr[uu]:indptr[uu + 1]]
            if items_u.size:
                v = np.asarray(self.sim[items_u].sum(axis=0)).ravel()
                idx = order[start:end]
                jj = j[idx]
                out[idx] = v[jj]
                # 自己類似の除去（既出ペアのみ sim(j,j)=1 が混入）
                out[idx] -= np.isin(jj, items_u).astype(float)
            start = end
        return out


# ---------------- ペア生成 ----------------

def sample_train_pairs(graphs, t1: int, num_corps: int, cap: int, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_p, i_p = bip_edges(graphs, t1, num_corps)
    keys = np.unique(u_p.astype(np.int64) * 10**9 + i_p)
    n_pos = min(cap, len(u_p))
    sel = rng.choice(len(u_p), size=n_pos, replace=False)
    up, ip = u_p[sel], i_p[sel]
    au, ai = np.unique(u_p), np.unique(i_p)
    negs_u, negs_i = [], []
    tries = 0
    while len(negs_u) < n_pos and tries < n_pos * 50:
        tries += 1
        c, p = rng.choice(au), rng.choice(ai)
        if np.searchsorted(keys, int(c) * 10**9 + int(p)) < len(keys) and keys[
            np.searchsorted(keys, int(c) * 10**9 + int(p))
        ] == int(c) * 10**9 + int(p):
            continue
        negs_u.append(c)
        negs_i.append(p)
    u = np.concatenate([up, np.array(negs_u)])
    i = np.concatenate([ip, np.array(negs_i)])
    y = np.concatenate([np.ones(n_pos), np.zeros(len(negs_u))])
    return u, i, y


# ---------------- モデル ----------------

FEATS = ["rel", "mom", "mom_burst", "seen"]


def features(ctx: YearContext, u: np.ndarray, j: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rel = ctx.rel(u, j)
    mom = ctx.mom[j]
    mb = mom * ctx.burst[j]
    seen = ctx.seen(u, j)
    X = np.stack([rel, mom, mb, seen], axis=1)
    off = np.log1p(ctx.deg[j])
    return X, off


def fit_logistic(X, off, y, feat_idx, anchored: bool, l2=1e-4, steps=500):
    Xt = torch.tensor(X[:, feat_idx], dtype=torch.float32)
    ot = torch.tensor(off, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    n_f = Xt.shape[1] + (0 if anchored else 1)
    w = torch.zeros(n_f, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=l2)
    for _ in range(steps):
        opt.zero_grad()
        if anchored:
            logit = ot + Xt @ w + b
        else:
            logit = torch.cat([Xt, ot.unsqueeze(1)], dim=1) @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt)
        loss.backward()
        opt.step()
    return w.detach().numpy(), float(b.item())


def predict(X, off, w, b, feat_idx, anchored: bool):
    Xs = X[:, feat_idx]
    if anchored:
        return off + Xs @ w + b
    return np.concatenate([Xs, off[:, None]], axis=1) @ w + b


# ---------------- ドメイン実行 ----------------

def run_domain(graphs, num_corps: int, years: List[int], eval_prev: int, eval_next: int,
               n_seeds: int, pos_cap: int) -> Dict:
    n_items = max(int(graphs[y].edge_index[1].max()) for y in years) - num_corps + 1
    train_years = [y for y in years if y < eval_next]
    # momentum が定義できる遷移のみ学習に使う（遷移元 t に前年が必要）
    train_trans = [(t, t1) for t, t1 in zip(train_years[:-1], train_years[1:]) if any(y < t for y in years)]

    ctx_by_t = {t: YearContext(graphs, years, t, num_corps, n_items) for t, _ in train_trans}
    ctx_eval = YearContext(graphs, years, eval_prev, num_corps, n_items)

    pos_ei, neg_ei = eval_pairs(graphs, num_corps, eval_prev, eval_next)
    ue = np.concatenate([pos_ei[0].numpy(), neg_ei[0].numpy()])
    je = np.concatenate([pos_ei[1].numpy(), neg_ei[1].numpy()]) - num_corps
    ye = np.concatenate([np.ones(pos_ei.size(1)), np.zeros(neg_ei.size(1))])
    Xe, offe = features(ctx_eval, ue, je)

    # 天井
    ceil = {
        "popularity": roc_auc_score(ye, offe),
        "seen_before": roc_auc_score(ye, Xe[:, 3]),
        "relatedness": roc_auc_score(ye, Xe[:, 0]),
        "seen_plus_pop": roc_auc_score(ye, Xe[:, 3] * 1e6 + offe),
    }
    ceil_max = max(ceil.values())
    skill = lambda auc: (auc - ceil_max) / (1.0 - ceil_max)

    variants = {
        "gem_rel": ([0], True),
        "gem_rel_seen": ([0, 3], True),
        "gem_rel_seen_mom": ([0, 1, 3], True),
        "gem_full": ([0, 1, 2, 3], True),
        "unanchored_full": ([0, 1, 2, 3], False),
    }
    res = {k: {"auc": [], "coef": []} for k in variants}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        Xs, offs, ys = [], [], []
        for t, t1 in train_trans:
            u, i, y = sample_train_pairs(graphs, t1, num_corps, pos_cap, rng)
            X, off = features(ctx_by_t[t], u, i)
            Xs.append(X)
            offs.append(off)
            ys.append(y)
        Xtr = np.concatenate(Xs)
        otr = np.concatenate(offs)
        ytr = np.concatenate(ys)
        # rel の標準化（train 統計で fit、eval に適用）
        mu, sd = Xtr[:, 0].mean(), Xtr[:, 0].std() + 1e-9
        Xtr_n = Xtr.copy()
        Xtr_n[:, 0] = (Xtr[:, 0] - mu) / sd
        Xe_n = Xe.copy()
        Xe_n[:, 0] = (Xe[:, 0] - mu) / sd

        for name, (fi, anch) in variants.items():
            w, b = fit_logistic(Xtr_n, otr, ytr, fi, anch)
            s = predict(Xe_n, offe, w, b, fi, anch)
            res[name]["auc"].append(roc_auc_score(ye, s))
            res[name]["coef"].append([float(x) for x in w])

    out = {"ceilings": {k: float(v) for k, v in ceil.items()}, "ceiling_max": float(ceil_max),
           "n_items": int(n_items), "train_transitions": train_trans,
           "eval_transition": [eval_prev, eval_next], "variants": {}}
    for name in variants:
        a = np.array(res[name]["auc"])
        out["variants"][name] = {
            "auc_mean": float(a.mean()),
            "auc_std": float(a.std()),
            "skill_mean": float(skill(a.mean())),
            "coef_mean": np.array(res[name]["coef"]).mean(axis=0).tolist(),
            "feat_names": [FEATS[i] for i in variants[name][0]] + ([] if variants[name][1] else ["log_pop"]),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="GEM v0 + ceiling-aware skill score")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--pos-cap", type=int, default=8000)
    p.add_argument("--patent-domains", type=str, nargs="+",
                   default=["agrifood", "construction", "energy"])
    p.add_argument("--skip-author-topic", action="store_true")
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()

    results = {}
    if not args.skip_author_topic:
        from pnode_patent_runner.dual_force_data import load_dual_force_bundle
        b = load_dual_force_bundle(
            "data/processed/arxiv_cs_embedded_2020-2026_full.csv", topic_column="topic", min_papers=5
        )
        graphs = {y: g for y, g in b.graphs.items() if 2022 <= y <= 2025}  # 検証 A と同一範囲
        results["author_topic"] = run_domain(
            graphs, b.num_corps, sorted(graphs), 2024, 2025, args.n_seeds, args.pos_cap
        )
        print("author_topic done")

    from pnode_patent_runner.cope_experiment import load_bipartite_domain_graph_bundle
    for dom in args.patent_domains:
        # year_range は検証 A/C と同一（ノード索引・評価ペアの同一性を保つ）。
        # momentum は範囲内 2 年目以降の遷移でのみ定義される。
        bb = load_bipartite_domain_graph_bundle(
            f"data/processed/bipartite_{dom}.csv", year_range=(2017, 2021)
        )
        graphs = {y: g for y, g in bb.graphs.items() if 2017 <= y <= 2021}
        results[dom] = run_domain(
            graphs, bb.num_corps, sorted(graphs), 2020, 2021, args.n_seeds, args.pos_cap
        )
        print(f"{dom} done")

    out = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
           "protocol": {"burst_percentile": BURST_PCT, "n_seeds": args.n_seeds,
                        "pos_cap": args.pos_cap,
                        "note": "eval pairs identical to learned-model holdout eval; features use info <= year_prev only"},
           "results": results}
    oj = Path(args.output_json)
    oj.parent.mkdir(parents=True, exist_ok=True)
    with open(oj, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {oj}")
    return 0


if __name__ == "__main__":
    main()
