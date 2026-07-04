"""
USPTO Patent (energy domain) を X1 形式で準備。

論文版 (prepare_pisde_paper_x1.py) と同じ構造を持つ:
  data_pt = {
      'xp':          list of (N_t, D) state vectors
      'y':           list of time values
      'topics':      list of (N_t,) topic ID per cell
      'topic_names': list of IPC codes
      'centroids':   list of (T, D) per-year topic centroids
      'growth':      list of (T,) per-year growth
      'growth_norm': standardized growth
      'n_topics':    int
  }

各「細胞」= bipartite の 1 row (inventor + IPC event)
状態 x = IPC one-hot を PCA 50D に削減
topic_j = その IPC コード

出力: data/PNode_Patent_Energy_X1/alltime/fate_train.pt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

DOMAIN  = os.environ.get("PNODE_DOMAIN", "energy")
SRC_CSV = f"data/processed/bipartite_{DOMAIN}.csv"
OUT_DIR = Path(f"data/PNode_Patent_{DOMAIN.capitalize()}_X1/alltime")
OUT_PT  = OUT_DIR / "fate_train.pt"
YEARS   = list(range(2010, 2022))    # 12 年
PCA_DIM = 50
MAX_PER_YEAR = 4000


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Domain: {DOMAIN}  CSV: {SRC_CSV}")

    df = pd.read_csv(SRC_CSV)
    df["year"] = pd.to_datetime(df["ts"]).dt.year.astype(int)
    df = df[df["year"].isin(YEARS)].reset_index(drop=True)
    print(f"  rows in {YEARS[0]}-{YEARS[-1]}: {len(df)}")

    # IPC を「トピック」として扱う (右パーティション)
    ipc_list = sorted(df["i"].dropna().unique())
    if len(ipc_list) > 1000:
        # 高頻度 top-300 IPC のみに削減 (n_topics 制御)
        top_ipc = df["i"].value_counts().head(300).index.tolist()
        df = df[df["i"].isin(top_ipc)].reset_index(drop=True)
        ipc_list = sorted(df["i"].dropna().unique())
        print(f"  filtered to top-300 IPCs by frequency: {len(ipc_list)}")
    n_ipc = len(ipc_list)
    ipc_to_idx = {ipc: i for i, ipc in enumerate(ipc_list)}
    df["ipc_idx"] = df["i"].map(ipc_to_idx)
    print(f"  unique IPCs (topics): {n_ipc}")

    # 各 row を one-hot 化 → PCA 削減
    print("Building one-hot vectors...")
    rows_per_year = {}
    topics_per_year_raw = {}
    for yr in YEARS:
        sub = df[df["year"] == yr].copy()
        if len(sub) == 0:
            continue
        if len(sub) > MAX_PER_YEAR:
            sub = sub.sample(n=MAX_PER_YEAR, random_state=42)
        oh = np.zeros((len(sub), n_ipc), dtype=np.float32)
        oh[np.arange(len(sub)), sub["ipc_idx"].values] = 1.0
        rows_per_year[yr] = oh
        topics_per_year_raw[yr] = sub["ipc_idx"].values
        print(f"  year {yr}: {oh.shape[0]} rows")

    all_X = np.vstack(list(rows_per_year.values()))
    print(f"\nPCA fit on combined {all_X.shape}...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(PCA_DIM, n_ipc - 1), random_state=42)
    pca.fit(all_X)
    print(f"  PCA dim: {pca.n_components_}, explained var: {pca.explained_variance_ratio_.sum():.4f}")
    actual_dim = pca.n_components_

    xp, topics_per_year, y_list = [], [], []
    centroids_per_year = []
    counts_per_year = {}    # {(year, ipc_idx): count}
    for k, yr in enumerate(YEARS):
        oh = rows_per_year.get(yr)
        if oh is None: continue
        X_yr = pca.transform(oh).astype(np.float32)
        X_yr = (X_yr - X_yr.mean(axis=0)) / (X_yr.std(axis=0) + 1e-8)
        xp.append(torch.tensor(X_yr, dtype=torch.float32))
        topics_per_year.append(torch.tensor(topics_per_year_raw[yr], dtype=torch.long))
        y_list.append(float(k))

        # トピック centroid を計算
        cents = torch.zeros(n_ipc, actual_dim, dtype=torch.float32)
        for j in range(n_ipc):
            sub = X_yr[topics_per_year_raw[yr] == j]
            if len(sub) > 0:
                cents[j] = torch.tensor(sub.mean(axis=0), dtype=torch.float32)
        centroids_per_year.append(cents)

        # IPC ごとの count
        for j in range(n_ipc):
            counts_per_year[(yr, j)] = int((topics_per_year_raw[yr] == j).sum())

    # 成長率 g_j(t)
    growth_per_year = []
    for k, yr in enumerate(YEARS):
        g = torch.zeros(n_ipc, dtype=torch.float32)
        if k == 0:
            growth_per_year.append(g); continue
        prev_yr = YEARS[k - 1]
        for j in range(n_ipc):
            c_now = float(counts_per_year[(yr, j)])
            c_prev = float(counts_per_year[(prev_yr, j)])
            g[j] = (c_now - c_prev) / (c_prev + 1.0)
        growth_per_year.append(g)

    # 標準化
    all_g = torch.stack([g for g in growth_per_year if g.abs().sum() > 0])
    g_mean = all_g.mean().item()
    g_std = all_g.std().item() + 1e-8
    growth_norm = [(g - g_mean) / g_std for g in growth_per_year]

    data_pt = {
        "xp":          xp,
        "y":           y_list,
        "topics":      topics_per_year,
        "topic_names": ipc_list,
        "centroids":   centroids_per_year,
        "growth":      growth_per_year,
        "growth_norm": growth_norm,
        "n_topics":    n_ipc,
    }
    torch.save(data_pt, OUT_PT)
    print(f"\n✅ Saved -> {OUT_PT}")

    # leaveout 用 (最終年 t = len(y_list) - 1 = 11)
    leaveout_idx = len(y_list) - 1
    for lo in [leaveout_idx - 2, leaveout_idx - 1, leaveout_idx]:
        out_lo = Path(f"data/PNode_Patent_{DOMAIN.capitalize()}_X1/leaveout{lo}/fate_train.pt")
        out_lo.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_pt, out_lo)
        print(f"  Saved -> {out_lo}")

    print(f"\n[統計]")
    print(f"  topics: {n_ipc}, time points: {len(y_list)}, x_dim: {actual_dim}")
    g_last = growth_per_year[-1].numpy()
    order = np.argsort(-g_last)
    print(f"  成長率 TOP 5 (year {YEARS[-1]}):")
    for i in order[:5]:
        if abs(g_last[i]) > 1e-6:
            print(f"    {ipc_list[i]:<20}  g={g_last[i]:+.3f}")
    print(f"  下位 5:")
    for i in order[-5:][::-1]:
        if abs(g_last[i]) > 1e-6:
            print(f"    {ipc_list[i]:<20}  g={g_last[i]:+.3f}")


if __name__ == "__main__":
    main()
