"""techtrend_common.py — shared基盤 for the Neural-ODE tech-trend experiments.

帰納エンコーダ(企業履歴→状態)・固定関連性ゲート・評価(企業参入 + 技術トレンド)
を提供する。Arch A/B/C はここを共有する。現行 task_b_ode_relatedness.py の
"自由 z0"(トランスダクティブ)を、共起埋め込み系列を読む帰納エンコーダで置換する
ことが目的(plan: refactored-pondering-gizmo.md の Step 0)。

データ:  data/processed/bipartite_{domain}_firm.csv  (列 ts,u,i)
粒度  :  maingroup (code.split("/")[0])
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/nakamuraroi/kumagai")

# diagnose_entry_humanaware.py を同ディレクトリから import(bipartite_embed, cooc_graph 再利用)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_entry_humanaware import bipartite_embed, cooc_graph  # noqa: E402


# ---------------------------------------------------------------- data --------
def coarsen(code: str) -> str:
    return str(code).split("/")[0]


def load_firm(domain: str, y0: int, y1: int) -> pd.DataFrame:
    """firm-level bipartite を maingroup 粒度でロード(列 year,u,i)。"""
    df = pd.read_csv(ROOT / f"data/processed/bipartite_{domain}_firm.csv",
                     dtype={"u": str, "i": str})
    df["year"] = pd.to_datetime(df["ts"], errors="coerce").dt.year
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[(df.year >= y0) & (df.year <= y1)]
    df["i"] = df["i"].map(coarsen)
    df = df[["year", "u", "i"]].dropna(subset=["i"]).drop_duplicates()
    return df


def build_sets(df: pd.DataFrame, Y: int, H: int):
    """train_end=Y, horizon=H。prior(≤Y のポートフォリオ)/ test_first(Y<year≤Y+H の
    新規参入で prior 非空のもの)/ firm-year 集合 fy を返す。リーク無し。"""
    train = df[df.year <= Y]
    test = df[(df.year > Y) & (df.year <= Y + H)]
    fy = defaultdict(dict)
    for (u, y), g in train.groupby(["u", "year"]):
        fy[u][y] = set(g.i)
    prior = {u: set().union(*d.values()) if d else set() for u, d in fy.items()}
    test_first = {}
    for u, g in test.groupby("u"):
        new = set(g.i) - prior.get(u, set())
        if new and prior.get(u):
            test_first[u] = new
    return fy, prior, test_first


# ------------------------------------------------- relatedness (固定ゲート) ----
def relatedness_vectors(df: pd.DataFrame, Y: int, ipcs, cidx):
    """leak-free(≤Y)cooc に基づく企業ごとの関連性ベクトル rel_vec[u] over ipcs。
    task_b_ode_relatedness.relatedness_scores と同一ロジック(ベクトル版)。"""
    train = df[df.year <= Y]
    cooc = defaultdict(lambda: defaultdict(int))
    for (u, y), g in train.groupby(["u", "year"]):
        cl = list(g.i)
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                cooc[cl[a]][cl[b]] += 1
                cooc[cl[b]][cl[a]] += 1
    port = defaultdict(set)
    for (u, y), g in train.groupby(["u", "year"]):
        port[u] |= set(g.i)
    rel = {}
    for u, ps in port.items():
        v = np.zeros(len(ipcs), dtype=np.float32)
        for inc in ps:
            for c, w in cooc[inc].items():
                if c in cidx:
                    v[cidx[c]] += w
        rel[u] = v
    return rel


# ------------------------------------------------- 帰納エンコーダ入力特徴 -------
def firm_year_features(df: pd.DataFrame, firms, cidx, Cemb, years_grid):
    """X[L,N,k]: 企業 i・年 y の出願 maingroup の Cemb 加重(出願数)平均。
    共起ジオメトリに統一(content と混ぜない)。活動の無い年はゼロベクトル。"""
    L, N, k = len(years_grid), len(firms), Cemb.shape[1]
    yi = {y: j for j, y in enumerate(years_grid)}
    fi = {u: j for j, u in enumerate(firms)}
    X = np.zeros((L, N, k), dtype=np.float32)
    cnt = np.zeros((L, N), dtype=np.float32)
    sub = df[df.year.isin(yi)]
    for (u, y), g in sub.groupby(["u", "year"]):
        if u not in fi:
            continue
        cols = [cidx[c] for c in g.i if c in cidx]
        if not cols:
            continue
        X[yi[y], fi[u]] += Cemb[cols].sum(0)
        cnt[yi[y], fi[u]] += len(cols)
    nz = cnt > 0
    X[nz] /= cnt[nz][:, None]
    return X


# ----------------------------------------------------------- 評価:企業参入 ----
def evaluate_entry(score_of_firm, firms, prior, test_first, ipcs, cidx, ks=(5, 10, 20)):
    """MRR / Hit@k。保有 maingroup は -inf でマスク(task_b.evaluate と同規約)。"""
    hits = {k: 0 for k in ks}
    rr, n = 0.0, 0
    for u in firms:
        scores = np.asarray(score_of_firm(u), dtype=np.float64).copy()
        for c in prior[u]:
            if c in cidx:
                scores[cidx[c]] = -np.inf
        order = np.argsort(-scores)
        rank = np.empty(len(ipcs), dtype=np.int64)
        rank[order] = np.arange(1, len(ipcs) + 1)
        for true_c in test_first[u]:
            if true_c not in cidx:
                continue
            r = int(rank[cidx[true_c]])
            rr += 1.0 / r
            for k in ks:
                hits[k] += int(r <= k)
            n += 1
    res = {f"Hit@{k}": hits[k] / max(n, 1) for k in ks}
    res["MRR"] = rr / max(n, 1)
    res["n"] = n
    return res


# --------------------------------------------------- 評価:技術トレンド --------
def actual_trend_counts(df, Y, H, ipcs, cidx):
    """各 maingroup p の Y<year≤Y+H の新規 entrant 企業数(prior に無い企業の参入)。"""
    _, prior, _ = build_sets(df, Y, H)
    test = df[(df.year > Y) & (df.year <= Y + H)]
    cnt = np.zeros(len(ipcs), dtype=np.float64)
    for u, g in test.groupby("u"):
        for c in (set(g.i) - prior.get(u, set())):
            if c in cidx:
                cnt[cidx[c]] += 1
    return cnt


def ndcg_at_k(pred, actual, k):
    order = np.argsort(-np.asarray(pred))[:k]
    gains = np.asarray(actual)[order]
    disc = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * disc).sum())
    ideal = np.sort(actual)[::-1][:k]
    idcg = float((ideal * (1.0 / np.log2(np.arange(2, len(ideal) + 2)))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(pred, actual, k):
    """top-k 予測のうち実測 entrant>0 の割合(rising tech のヒット率)。"""
    order = np.argsort(-np.asarray(pred))[:k]
    return float((np.asarray(actual)[order] > 0).mean())


def spearman(pred, actual):
    from scipy.stats import spearmanr
    rho, _ = spearmanr(pred, actual)
    return float(rho) if rho == rho else 0.0


# ----------------------------------------------------------- 較正(ECE) -------
def ece_buckets(P, Y, nb=8):
    """recommender_firm.ece_buckets と同一。"""
    P = np.asarray(P); Y = np.asarray(Y)
    edges = np.quantile(P, np.linspace(0, 1, nb + 1)); edges[-1] += 1e-9
    rows, ece = [], 0.0
    for i in range(nb):
        m = (P >= edges[i]) & (P < edges[i + 1])
        if m.sum() < 10:
            continue
        ece += m.mean() * abs(P[m].mean() - Y[m].mean())
        rows.append((float(P[m].mean()), float(Y[m].mean()), int(m.sum())))
    return rows, float(ece)


# ------------------------------------------------------ setup ヘルパ -----------
def prepare(domain, y0, y1, Y, H, max_firms, seed, embed_dim=64):
    """全アーキ共通のセットアップ。"""
    rng = np.random.default_rng(seed)
    df = load_firm(domain, y0, y1)
    ipcs = sorted(df.i.unique())
    cidx = {c: j for j, c in enumerate(ipcs)}
    fy, prior, test_first = build_sets(df, Y, H)
    firms = [u for u in test_first if prior.get(u)]
    rng.shuffle(firms)
    firms = firms[:max_firms]
    # 共起埋め込み(≤Y、firm×CPC PPMI+SVD)
    df_le = df[df.year <= Y].rename(columns={})  # bipartite_embed は year,u,i を使用
    uidx, Uemb, Cemb = bipartite_embed(df_le, Y, cidx, dim=embed_dim)
    rel = relatedness_vectors(df, Y, ipcs, cidx)
    return dict(df=df, ipcs=ipcs, cidx=cidx, prior=prior, test_first=test_first,
                firms=firms, Cemb=Cemb.astype(np.float32), rel=rel, rng=rng)
