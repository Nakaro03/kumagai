"""X5 data loader — reads existing fate_train.pt format.

The bipartite scholarly-graph data already lives in
data/{DOMAIN}/alltime/fate_train.pt. We do not re-extract; we just wrap it.

Returned dict keys:
  xp           : list[Tensor[N_t, x_dim]]    per-timepoint particle samples
  y            : list[float]                  timepoint scalars (0,1,2,...)
  centroids    : list[Tensor[K, x_dim]]       per-t topic centroids
  growth       : list[Tensor[K]]              raw growth signal
  growth_norm  : list[Tensor[K]]              z-normalized growth signal
  topic_names  : list[str]
  n_topics     : int
"""
from __future__ import annotations

import torch
from typing import Dict, Any

from .config import X5Config


def load_bipartite(cfg: X5Config) -> Dict[str, Any]:
    cfg.resolve_paths()
    data = torch.load(cfg.data_path, weights_only=False)
    cfg.x_dim = int(data["xp"][0].shape[-1])
    cfg.n_topics = int(data["n_topics"])
    cfg.y = [float(yv) for yv in data["y"]]
    return data


def loto_train_test_split(train_t: list[int], leaveout_t: int | None) -> tuple[list[int], list[int]]:
    """Split train_t into (effective_train, test) given a leaveout index."""
    if leaveout_t is None:
        return list(train_t), []
    if leaveout_t not in train_t:
        raise ValueError(f"leaveout_t={leaveout_t} not in train_t={train_t}")
    eff = [t for t in train_t if t != leaveout_t]
    return eff, [leaveout_t]
