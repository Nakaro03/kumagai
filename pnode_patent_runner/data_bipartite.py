"""
Loader for simple bipartite event CSVs extracted from PatentsView.

CSV format:  ts (YYYY-MM-DD), u (inventor_id), i (cpc_group e.g. "H01L21/338")

Node features:
  - IPC nodes  (right):  one-hot over CPC subclass (4-char prefix, e.g. "H01L")
  - Actor nodes (left):  zero vector of same dimension
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# ── helpers ──────────────────────────────────────────────────────────────────

def _subclass(ipc: str) -> str:
    """'H01L21/338' → 'H01L'  (4-char CPC subclass)."""
    return ipc[:4] if len(ipc) >= 4 else ipc


def _build_ipc_features(
    all_ipc: List[str],
    subclass_vocab: List[str],
) -> Dict[str, np.ndarray]:
    sub_to_idx = {s: i for i, s in enumerate(subclass_vocab)}
    dim = len(subclass_vocab)
    features: Dict[str, np.ndarray] = {}
    for code in all_ipc:
        v = np.zeros(dim, dtype=np.float32)
        idx = sub_to_idx.get(_subclass(code))
        if idx is not None:
            v[idx] = 1.0
        features[code] = v
    return features


# ── public API ────────────────────────────────────────────────────────────────

def preprocess_bipartite_events(
    file_path: str,
    *,
    year_min: int = 2000,
    year_max: int = 2021,
    min_events: int = 2,
) -> pd.DataFrame:
    """
    Read a ts/u/i event CSV, filter by year, drop low-activity actors.

    Returns a clean DataFrame with columns: year, u, i
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Not found: {path}")

    df = pd.read_csv(path, usecols=["ts", "u", "i"], dtype=str)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    df["year"] = df["ts"].dt.year.astype(int)
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]

    # drop actors with fewer than min_events across whole period
    counts = df["u"].value_counts()
    active = set(counts[counts >= min_events].index)
    df = df[df["u"].isin(active)]

    df = df[["year", "u", "i"]].drop_duplicates().reset_index(drop=True)
    return df


def build_bipartite_event_graphs(
    df: pd.DataFrame,
) -> Tuple[Dict[int, Data], List[str], List[str], int, Set[Tuple[int, int]], int]:
    """
    Build annual bipartite graphs from a preprocessed ts/u/i DataFrame.

    Returns
    -------
    graphs       : {year: Data}  (x, edge_index, active_mask, num_nodes)
    all_actors   : list of actor IDs   (indices 0 .. A-1)
    all_ipc      : list of IPC codes   (indices A .. A+K-1)
    total_n      : A + K
    hist_edges   : set of (actor_idx, ipc_idx)
    in_dim       : feature dimension
    """
    if len(df) == 0:
        return {}, [], [], 0, set(), 0

    all_actors: List[str] = sorted(df["u"].unique().tolist())
    all_ipc:    List[str] = sorted(df["i"].unique().tolist())

    actor_to_idx = {a: i for i, a in enumerate(all_actors)}
    ipc_to_idx   = {c: i + len(all_actors) for i, c in enumerate(all_ipc)}
    total_nodes  = len(all_actors) + len(all_ipc)

    # subclass vocabulary (from the actual IPC codes in this dataset)
    subclass_vocab = sorted({_subclass(c) for c in all_ipc})
    ipc_features = _build_ipc_features(all_ipc, subclass_vocab)
    in_dim = len(subclass_vocab)

    global_graph_dict: Dict[int, Data] = {}
    hist_edges: Set[Tuple[int, int]] = set()

    for year, group in df.groupby("year"):
        a_ids = group["u"].map(actor_to_idx).dropna().astype(int)
        c_ids = group["i"].map(ipc_to_idx).dropna().astype(int)
        # align on common index
        idx   = a_ids.index.intersection(c_ids.index)
        a_arr = a_ids.loc[idx].to_numpy()
        c_arr = c_ids.loc[idx].to_numpy()
        if len(a_arr) == 0:
            continue
        for a, c in zip(a_arr, c_arr):
            hist_edges.add((int(a), int(c)))
        fwd = np.stack([a_arr, c_arr], axis=1)
        bwd = np.stack([c_arr, a_arr], axis=1)
        edges_arr = np.concatenate([fwd, bwd], axis=0)

        x = torch.zeros(total_nodes, in_dim)
        for code, feat in ipc_features.items():
            idx = ipc_to_idx.get(code)
            if idx is not None:
                x[idx] = torch.tensor(feat, dtype=torch.float32)

        edge_index = torch.tensor(edges_arr, dtype=torch.long).t().contiguous()
        active_mask = torch.zeros(total_nodes, dtype=torch.bool)
        active_mask[torch.unique(edge_index)] = True
        global_graph_dict[int(year)] = Data(
            x=x,
            edge_index=edge_index,
            active_mask=active_mask,
            num_nodes=total_nodes,
        )

    return global_graph_dict, all_actors, all_ipc, total_nodes, hist_edges, in_dim


def calculate_initial_actor_vectors(
    df: pd.DataFrame,
    num_actors: int,
    in_dim: int,
    all_actors: List[str],
) -> torch.Tensor:
    """Average IPC one-hot (by subclass) over each actor's events — fully vectorized."""
    all_ipc        = sorted(df["i"].unique().tolist())
    subclass_vocab = sorted({_subclass(c) for c in all_ipc})
    sub_to_idx     = {s: i for i, s in enumerate(subclass_vocab)}
    actor_to_idx   = {a: i for i, a in enumerate(all_actors)}

    # map each row to (actor_idx, subclass_idx) — O(N) vectorized
    actor_ids  = df["u"].map(actor_to_idx).to_numpy(dtype=np.float32)
    sub_ids    = df["i"].map(lambda c: sub_to_idx.get(_subclass(c), -1)).to_numpy(dtype=np.float32)

    valid = (actor_ids >= 0) & (sub_ids >= 0)
    actor_ids = actor_ids[valid].astype(np.int64)
    sub_ids   = sub_ids[valid].astype(np.int64)

    vecs   = np.zeros((num_actors, in_dim), dtype=np.float32)
    counts = np.zeros(num_actors, dtype=np.float32)
    np.add.at(vecs,   (actor_ids, sub_ids), 1.0)
    np.add.at(counts,  actor_ids,           1.0)

    mask = counts > 0
    vecs[mask] /= counts[mask, None]
    rng = np.random.default_rng(42)
    vecs[~mask] = rng.normal(0, 0.01, (int((~mask).sum()), in_dim)).astype(np.float32)
    return torch.tensor(vecs, dtype=torch.float32)
