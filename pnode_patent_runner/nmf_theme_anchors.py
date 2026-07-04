"""
NMF theme extraction on inventor×technology bipartite patent data.

Builds an inventor × CPC-subclass count matrix (prolific inventors only),
factorizes it with NMF into K interpretable technology themes, and reports
the top CPC subclasses and top inventors per theme.

This is the data-side foundation for Φ-NBD (NMF-anchored basis-decomposed
potential P-NODE): each NMF theme k provides an interpretable anchor whose
latent center μ_k seeds a basis of the potential field.

Usage:
  python -m pnode_patent_runner.nmf_theme_anchors \
      --domain construction --min-patents 20 --year-min 2012 --year-max 2021 --num-themes 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF


def _subclass(ipc: str) -> str:
    """'H01L21/338' -> 'H01L' (4-char CPC subclass)."""
    return ipc[:4] if len(ipc) >= 4 else ipc


def load_events(
    csv_path: Path, year_min: int, year_max: int, min_patents: int
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["ts", "u", "i"], dtype=str)
    df["year"] = pd.to_datetime(df["ts"], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]
    df["sub"] = df["i"].map(_subclass)
    # prolific filter on distinct (year, sub) activity count per inventor
    counts = df.groupby("u").size()
    active = counts[counts >= min_patents].index
    df = df[df["u"].isin(active)].reset_index(drop=True)
    return df


def build_matrix(df: pd.DataFrame):
    inventors = sorted(df["u"].unique())
    subs = sorted(df["sub"].unique())
    inv_idx = {a: i for i, a in enumerate(inventors)}
    sub_idx = {s: j for j, s in enumerate(subs)}
    rows = df["u"].map(inv_idx).to_numpy()
    cols = df["sub"].map(sub_idx).to_numpy()
    data = np.ones(len(df), dtype=np.float32)
    X = csr_matrix((data, (rows, cols)), shape=(len(inventors), len(subs)))
    X.sum_duplicates()
    return X, inventors, subs


def build_node_theme_matrix(
    csv_path,
    corps,
    num_nodes: int,
    year_min: int,
    year_max: int,
    num_themes: int,
    seed: int = 42,
):
    """Build a (num_nodes, K) row-normalized NMF theme-loading matrix aligned
    to the bundle's node indexing (left partition = inventors = corps order).

    Returns (node_theme_W: np.ndarray (num_nodes, K), H: (K, n_sub), subs: list).
    Right-partition (CPC) rows and inventors absent from the data stay zero.
    """
    import numpy as np

    corps = list(corps)
    corp_set = set(corps)
    df = pd.read_csv(csv_path, usecols=["ts", "u", "i"], dtype=str)
    df["year"] = pd.to_datetime(df["ts"], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]
    df = df[df["u"].isin(corp_set)]

    # NMF テーマは「フル CPC グループ」を特徴次元にする（サブクラスでは粗すぎ、
    # 例えば computing は G06 配下のサブクラスが 12 種しか無く K を大きくできない）。
    subs = sorted(df["i"].unique())
    sub_idx = {s: j for j, s in enumerate(subs)}
    corp_pos = {c: i for i, c in enumerate(corps)}
    rows = df["u"].map(corp_pos).to_numpy()
    cols = df["i"].map(sub_idx).to_numpy()
    data = np.ones(len(df), dtype=np.float32)
    X = csr_matrix((data, (rows, cols)), shape=(len(corps), len(subs)))
    X.sum_duplicates()

    model = NMF(
        n_components=num_themes,
        init="nndsvda",
        random_state=seed,
        max_iter=400,
        beta_loss="frobenius",
    )
    W = model.fit_transform(X)          # (len(corps), K)
    H = model.components_               # (K, n_sub)

    # row-normalize each inventor's loadings to sum to 1
    row_sum = W.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    W = W / row_sum

    node_W = np.zeros((num_nodes, num_themes), dtype=np.float32)
    node_W[: len(corps)] = W            # inventor nodes occupy indices 0..A-1
    return node_W, H, subs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="construction")
    p.add_argument("--data", default=None, help="override CSV path")
    p.add_argument("--year-min", type=int, default=2012)
    p.add_argument("--year-max", type=int, default=2021)
    p.add_argument("--min-patents", type=int, default=20)
    p.add_argument("--num-themes", type=int, default=12)
    p.add_argument("--top-cpc", type=int, default=10)
    p.add_argument("--top-inv", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    csv_path = (
        Path(args.data)
        if args.data
        else repo / f"data/processed/bipartite_{args.domain}.csv"
    )
    if not csv_path.is_file():
        raise SystemExit(f"data not found: {csv_path}")

    print(f"Loading {csv_path} ...", flush=True)
    df = load_events(csv_path, args.year_min, args.year_max, args.min_patents)
    X, inventors, subs = build_matrix(df)
    print(
        f"matrix: inventors={X.shape[0]:,}  cpc_subclass={X.shape[1]:,}  "
        f"nnz={X.nnz:,}  density={X.nnz / (X.shape[0]*X.shape[1]):.4f}",
        flush=True,
    )

    print(f"Running NMF (K={args.num_themes}) ...", flush=True)
    model = NMF(
        n_components=args.num_themes,
        init="nndsvda",
        random_state=args.seed,
        max_iter=400,
        beta_loss="frobenius",
    )
    W = model.fit_transform(X)          # (inventors, K)
    H = model.components_               # (K, cpc)
    print(f"reconstruction err: {model.reconstruction_err_:.2f}", flush=True)

    subs_arr = np.array(subs)
    inv_arr = np.array(inventors)
    print("\n" + "=" * 70)
    print(f"THEMES (domain={args.domain}, K={args.num_themes})")
    print("=" * 70)
    for k in range(args.num_themes):
        top_c = np.argsort(H[k])[::-1][: args.top_cpc]
        top_i = np.argsort(W[:, k])[::-1][: args.top_inv]
        cpc_str = ", ".join(f"{subs_arr[c]}({H[k][c]:.1f})" for c in top_c)
        n_inv = int((W[:, k] > W[:, k].mean()).sum())
        print(f"\n[Theme {k:2d}]  (~{n_inv} inventors loaded)")
        print(f"   top CPC : {cpc_str}")
        print(f"   top inv : {', '.join(inv_arr[top_i])}")

    # theme assignment distribution
    assign = W.argmax(axis=1)
    print("\n" + "-" * 70)
    print("inventors per dominant theme:")
    for k in range(args.num_themes):
        print(f"   theme {k:2d}: {int((assign == k).sum()):,}")


if __name__ == "__main__":
    main()
