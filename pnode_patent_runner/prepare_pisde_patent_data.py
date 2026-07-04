"""
USPTO 特許データを PI-SDE 形式に変換。

入力: data/processed/bipartite_energy.csv  (ts, u=inventor, i=IPC)
出力: data/PNode_Patent_Energy/alltime/fate_train.pt

「細胞 = 特許 instance」 (invener × IPC ペア = 1 row)
状態: IPC の one-hot (PCA で 50 次元に削減)
時刻: 年 2010-2021 を t=0..11
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
OUT_DIR = Path(f"data/PNode_Patent_{DOMAIN.capitalize()}/alltime")
OUT_PT  = OUT_DIR / "fate_train.pt"
YEARS   = list(range(2010, 2022))     # 2010-2021 (12 年)
PCA_DIM = 50
MAX_PER_YEAR = 4000                    # 各年の最大 row 数 (計算コスト管理)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Domain: {DOMAIN}  CSV: {SRC_CSV}")

    df = pd.read_csv(SRC_CSV)
    df["year"] = pd.to_datetime(df["ts"]).dt.year.astype(int)
    df = df[df["year"].isin(YEARS)].reset_index(drop=True)
    print(f"  rows in {YEARS[0]}-{YEARS[-1]}: {len(df)}")

    # IPC を整数 ID 化 → one-hot
    ipc_list = sorted(df["i"].dropna().unique())
    ipc_to_idx = {ipc: i for i, ipc in enumerate(ipc_list)}
    n_ipc = len(ipc_list)
    print(f"  unique IPCs: {n_ipc}")
    if n_ipc > 4000:
        print(f"  WARNING: too many IPCs ({n_ipc}); limiting to top-1000 by freq")
        top_ipc = df["i"].value_counts().head(1000).index.tolist()
        df = df[df["i"].isin(top_ipc)].reset_index(drop=True)
        ipc_list = sorted(df["i"].dropna().unique())
        ipc_to_idx = {ipc: i for i, ipc in enumerate(ipc_list)}
        n_ipc = len(ipc_list)

    df["ipc_idx"] = df["i"].map(ipc_to_idx)

    # 各 row を one-hot 化
    print(f"Building one-hot vectors ({n_ipc}-dim)...")
    rows_per_year = {}
    for yr in YEARS:
        sub = df[df["year"] == yr]
        if len(sub) == 0:
            continue
        if len(sub) > MAX_PER_YEAR:
            sub = sub.sample(n=MAX_PER_YEAR, random_state=42)
        oh = np.zeros((len(sub), n_ipc), dtype=np.float32)
        oh[np.arange(len(sub)), sub["ipc_idx"].values] = 1.0
        rows_per_year[yr] = oh
        print(f"  year {yr}: {oh.shape[0]} rows")

    # 全期間データを vstack して PCA fit
    all_X = np.vstack(list(rows_per_year.values()))
    print(f"\nPCA fit on combined {all_X.shape}...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(PCA_DIM, n_ipc-1), random_state=42)
    pca.fit(all_X)
    print(f"  PCA dim: {pca.n_components_}, explained var: {pca.explained_variance_ratio_.sum():.4f}")

    xp = []
    y = []
    for k, yr in enumerate(YEARS):
        oh = rows_per_year.get(yr)
        if oh is None or len(oh) == 0:
            continue
        X_yr = pca.transform(oh).astype(np.float32)
        X_yr = (X_yr - X_yr.mean(axis=0)) / (X_yr.std(axis=0) + 1e-8)
        xp.append(torch.tensor(X_yr, dtype=torch.float32))
        y.append(float(k))

    data_pt = {"xp": xp, "y": y}
    torch.save(data_pt, OUT_PT)
    print(f"\nSaved -> {OUT_PT}")

    # leaveout (最終年)
    leaveout_idx = len(y) - 1
    OUT_PT_LO = OUT_DIR / f"../leaveout{leaveout_idx}/fate_train.pt"
    OUT_PT_LO.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_pt, OUT_PT_LO)
    print(f"Saved -> {OUT_PT_LO}")


if __name__ == "__main__":
    main()
