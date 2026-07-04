"""
ArXiv 論文データを PI-SDE 形式に変換。

PI-SDE 形式:
  data_pt = {
      'xp': [Tensor(N_0, D), Tensor(N_1, D), ...],  # 各時刻の状態
      'y':  [t_0, t_1, ...]                          # 時刻値 (連続)
  }

変換ルール:
  - 各年の各論文を 1 「細胞」として扱う (細胞 = paper instance)
  - 状態 x_i = 論文の埋め込み (PCA 50次元に削減)
  - 時刻 y = [0, 1, 2, 3] (年 2022-2025 の正規化)

出力:
  data/PNode_Paper/alltime/fate_train.pt
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
OUT_DIR  = Path("data/PNode_Paper/alltime")
OUT_PT   = OUT_DIR / "fate_train.pt"
YEARS    = [2022, 2023, 2024, 2025]
PCA_DIM  = 50    # PI-SDE 論文と同じ次元数 (~50)


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
    print(f"  rows: {len(df)}")
    print(f"  year range: {df['year'].min()}–{df['year'].max()}")

    # year フィルタ
    df = df[df["year"].isin(YEARS)].reset_index(drop=True)
    print(f"  after year filter [{YEARS}]: {len(df)} rows")

    # embedding parse
    print("Parsing embeddings...")
    embeds_list = []
    valid_mask = np.zeros(len(df), dtype=bool)
    for i, s in enumerate(df["description_embedding"]):
        e = parse_embedding(s)
        if e is not None and e.size > 100:
            embeds_list.append(e)
            valid_mask[i] = True
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(df)}")

    df = df[valid_mask].reset_index(drop=True)
    embeds = np.stack(embeds_list)
    print(f"  valid: {embeds.shape}")

    # PCA reduction to PCA_DIM
    print(f"PCA reduction {embeds.shape[1]} → {PCA_DIM}...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    X = pca.fit_transform(embeds).astype(np.float32)
    print(f"  explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")

    # 標準化 (ゼロ平均・単位分散) PI-SDE は前処理として scaling 想定
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # 年ごとに分割
    xp = []
    y = []
    for k, yr in enumerate(YEARS):
        mask = df["year"].values == yr
        X_yr = torch.tensor(X[mask], dtype=torch.float32)
        xp.append(X_yr)
        y.append(float(k))
        print(f"  year {yr} (t={k}): {X_yr.shape[0]} papers")

    data_pt = {"xp": xp, "y": y}
    torch.save(data_pt, OUT_PT)
    print(f"\nSaved -> {OUT_PT}")

    # Leaveout 版 (最終年=t=3 を holdout)
    OUT_PT_LO = OUT_DIR / "../leaveout3/fate_train.pt"
    OUT_PT_LO.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_pt, OUT_PT_LO)
    print(f"Saved -> {OUT_PT_LO}")


if __name__ == "__main__":
    main()
