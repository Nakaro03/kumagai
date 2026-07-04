"""
X1 (Topic-Anchor) 用に拡張した PI-SDE データ準備。

元の準備 (prepare_pisde_paper_data.py) に加えて、トピックラベル・centroid・成長率を保存:

data_pt = {
    'xp':        [Tensor(N_t, D)]                  # 各時点の状態 (50D)
    'y':         [t_0, t_1, ...]                   # 時刻値
    'topics':    [Tensor(N_t,)]                    # 各点のトピック ID (X1 用)
    'topic_names': List[str]                       # トピック名 (cs.AI etc.)
    'centroids': [Tensor(T, D)]                    # 各時点のトピック centroid (X1 用)
    'growth':    [Tensor(T,)]                      # 各時点の成長率 g_j (X1 用)
}

出力:
  data/PNode_Paper_X1/alltime/fate_train.pt
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

SRC_CSV  = "data/processed/arxiv_cs_embedded_2020-2026_full.csv"
OUT_DIR  = Path("data/PNode_Paper_X1/alltime")
OUT_PT   = OUT_DIR / "fate_train.pt"
YEARS    = [2022, 2023, 2024, 2025]
PCA_DIM  = 50


def parse_embedding(s):
    if isinstance(s, (list, np.ndarray)):
        return np.array(s, dtype=np.float32)
    s = str(s).strip()
    if not s or s == "nan":
        return None
    s_clean = s.strip("[]")
    arr = np.fromstring(s_clean, sep=" ", dtype=np.float32)
    if arr.size <= 1:
        arr = np.fromstring(s_clean, sep=",", dtype=np.float32)
    return arr if arr.size > 1 else None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {SRC_CSV}...")
    df = pd.read_csv(SRC_CSV)
    df = df[df["year"].isin(YEARS)].reset_index(drop=True)
    print(f"  rows: {len(df)}  topic uniques: {df['topic'].nunique()}")

    # embedding パース
    embeds, valid_mask = [], np.zeros(len(df), dtype=bool)
    for i, s in enumerate(df["description_embedding"]):
        e = parse_embedding(s)
        if e is not None and e.size > 100:
            embeds.append(e); valid_mask[i] = True
    df = df[valid_mask].reset_index(drop=True)
    embeds = np.stack(embeds)
    print(f"  valid: {embeds.shape}")

    # PCA
    print(f"PCA reduction → {PCA_DIM}D ...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    X = pca.fit_transform(embeds).astype(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    print(f"  explained var sum: {pca.explained_variance_ratio_.sum():.4f}")

    # トピック ID マッピング
    topic_names = sorted(df["topic"].dropna().unique())
    topic_to_id = {t: i for i, t in enumerate(topic_names)}
    n_topics = len(topic_names)
    print(f"  topics: {n_topics}")

    # 年ごとに分割 + centroid 計算
    xp, topics_per_year, centroids_per_year, growth_per_year = [], [], [], []
    counts_per_year = {}   # {(year, topic_id): count}

    y_list = []
    for k, yr in enumerate(YEARS):
        mask = df["year"].values == yr
        X_yr = X[mask]
        topics_yr = df.loc[mask, "topic"].map(topic_to_id).values

        xp.append(torch.tensor(X_yr, dtype=torch.float32))
        topics_per_year.append(torch.tensor(topics_yr, dtype=torch.long))
        y_list.append(float(k))

        # centroid (各トピックの平均座標)
        cents = torch.zeros(n_topics, PCA_DIM, dtype=torch.float32)
        for j in range(n_topics):
            sub = X_yr[topics_yr == j]
            if len(sub) > 0:
                cents[j] = torch.tensor(sub.mean(axis=0), dtype=torch.float32)
            # 空のトピックは前年の centroid を引き継ぐ (default は zero)
        centroids_per_year.append(cents)

        # トピック数 count
        for j in range(n_topics):
            counts_per_year[(yr, j)] = int((topics_yr == j).sum())

        print(f"  year {yr} (t={k}): N={mask.sum()}  active topics={int((cents.abs().sum(-1) > 0).sum())}")

    # 成長率 g_j(t) = (count(t) - count(t-1)) / (count(t-1) + 1)
    for k, yr in enumerate(YEARS):
        g_yr = torch.zeros(n_topics, dtype=torch.float32)
        if k == 0:
            growth_per_year.append(g_yr); continue
        prev_yr = YEARS[k - 1]
        for j in range(n_topics):
            c_now = float(counts_per_year[(yr, j)])
            c_prev = float(counts_per_year[(prev_yr, j)])
            g_yr[j] = (c_now - c_prev) / (c_prev + 1.0)
        growth_per_year.append(g_yr)

    # 標準化 (全期間の g を集めて mean/std を計算)
    all_g = torch.stack([g for g in growth_per_year if g.abs().sum() > 0])
    g_mean, g_std = all_g.mean().item(), all_g.std().item() + 1e-8
    growth_per_year_norm = [(g - g_mean) / g_std for g in growth_per_year]

    data_pt = {
        "xp":          xp,
        "y":           y_list,
        "topics":      topics_per_year,
        "topic_names": topic_names,
        "centroids":   centroids_per_year,
        "growth":      growth_per_year,           # raw
        "growth_norm": growth_per_year_norm,      # 標準化済み
        "n_topics":    n_topics,
    }
    torch.save(data_pt, OUT_PT)
    print(f"\n✅ Saved -> {OUT_PT}")

    # leaveout1, 2, 3 用に同じデータをコピー
    for lo in [1, 2, 3]:
        out_lo = Path(f"data/PNode_Paper_X1/leaveout{lo}/fate_train.pt")
        out_lo.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_pt, out_lo)
        print(f"  Saved -> {out_lo}")

    print(f"\n[統計]")
    print(f"  topics: {n_topics}")
    print(f"  growth 平均成長率上位5:")
    g_2024 = growth_per_year[2].numpy()
    order = np.argsort(-g_2024)
    for i in order[:5]:
        print(f"    {topic_names[i]:<10}  g_{YEARS[2]}={g_2024[i]:+.3f}  g_norm={growth_per_year_norm[2][i]:+.3f}")
    print(f"  下位5:")
    for i in order[-5:][::-1]:
        print(f"    {topic_names[i]:<10}  g_{YEARS[2]}={g_2024[i]:+.3f}  g_norm={growth_per_year_norm[2][i]:+.3f}")


if __name__ == "__main__":
    main()
