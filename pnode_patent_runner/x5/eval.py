"""X5 evaluation module — unified metrics for both X5 and baselines.

All metrics are computed per held-out timepoint t and returned as a dict; the
caller can aggregate across t. The module is intentionally light on
dependencies so that baseline scripts (baseline_all.py) can re-use it.

Implements:
  Primary
  -------
  W1_marginal   : Sinkhorn-W1 between sampled populations
  MMD_RBF       : RBF MMD between sampled populations
  Hits@K        : top-K topic overlap (rank by -Φ vs rank by g_norm)
  MRR           : mean reciprocal rank of true top-1
  AP            : binary high-growth precision-recall area
  NDCG@K        : ranking quality (g_raw as relevance)

  Secondary
  ---------
  Spearman_rho  : kept for backwards compat
  W1_centroid   : per-topic centroid prediction distance
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from scipy import stats


# ---------------------------------------------------------------------- W1

def _sinkhorn_loss(blur: float = 0.05):
    from geomloss import SamplesLoss
    return SamplesLoss("sinkhorn", p=1, blur=blur, scaling=0.7, debias=True)


def w1_marginal(samples_a: torch.Tensor, samples_b: torch.Tensor,
                blur: float = 0.05) -> float:
    """Symmetric Sinkhorn-W1 between two sample clouds (N×D)."""
    loss = _sinkhorn_loss(blur=blur)
    a = samples_a.detach().float()
    b = samples_b.detach().float()
    return float(loss(a, b).item())


# ---------------------------------------------------------------------- MMD

def mmd_rbf(samples_a: torch.Tensor, samples_b: torch.Tensor,
            sigma: Optional[float] = None) -> float:
    """Unbiased MMD with RBF kernel; sigma is set by the median heuristic."""
    a = samples_a.detach().float()
    b = samples_b.detach().float()
    n, m = a.shape[0], b.shape[0]

    if sigma is None:
        with torch.no_grad():
            sub = torch.cat([a[:min(500, n)], b[:min(500, m)]], dim=0)
            d2 = torch.cdist(sub, sub).pow(2)
            sigma = float(torch.sqrt(d2.median()) + 1e-8)

    gamma = 1.0 / (2.0 * sigma ** 2)

    Kxx = torch.exp(-gamma * torch.cdist(a, a).pow(2))
    Kyy = torch.exp(-gamma * torch.cdist(b, b).pow(2))
    Kxy = torch.exp(-gamma * torch.cdist(a, b).pow(2))

    Kxx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1) + 1e-12)
    Kyy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1) + 1e-12)
    Kxy = Kxy.mean()
    return float((Kxx + Kyy - 2 * Kxy).item())


# ---------------------------------------------------------------------- ranking

def hits_at_k(scores: np.ndarray, truth: np.ndarray, k: int = 10) -> float:
    """fraction of top-k(truth) recovered by top-k(scores)."""
    k = min(k, len(scores))
    top_pred = set(np.argsort(-scores)[:k].tolist())
    top_true = set(np.argsort(-truth)[:k].tolist())
    return len(top_pred & top_true) / float(k)


def mean_reciprocal_rank(scores: np.ndarray, truth: np.ndarray) -> float:
    """MRR: 1/rank of the true top-1 in the predicted ranking."""
    true_top1 = int(np.argmax(truth))
    pred_order = np.argsort(-scores)
    rank = int(np.where(pred_order == true_top1)[0][0]) + 1
    return 1.0 / rank


def average_precision(scores: np.ndarray, truth: np.ndarray,
                      threshold: str = "median") -> float:
    """Binary AP: positive class = truth > threshold."""
    if threshold == "median":
        thr = float(np.median(truth))
    else:
        thr = float(threshold)
    y_true = (truth > thr).astype(int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(average_precision_score(y_true, scores))


def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int = 10) -> float:
    """NDCG@K with raw relevance as gain."""
    k = min(k, len(scores))
    order = np.argsort(-scores)
    rel = np.asarray(relevance, dtype=float)
    rel_pos = np.clip(rel - rel.min(), 0.0, None)  # ensure non-negative gains
    dcg = sum(rel_pos[order[i]] / math.log2(i + 2) for i in range(k))
    ideal_order = np.argsort(-rel_pos)
    idcg = sum(rel_pos[ideal_order[i]] / math.log2(i + 2) for i in range(k))
    return float(dcg / (idcg + 1e-12))


def spearman_rho(scores: np.ndarray, truth: np.ndarray) -> float:
    if len(scores) < 3:
        return float("nan")
    rho, _ = stats.spearmanr(scores, truth)
    return float(rho) if rho is not None else float("nan")


# ---------------------------------------------------------------------- centroid

def w1_centroid(pred_centroids: torch.Tensor, obs_centroids: torch.Tensor) -> float:
    """Pairwise mean L2 distance — simple proxy for centroid W1."""
    return float((pred_centroids - obs_centroids).pow(2).sum(-1).sqrt().mean().item())


# ---------------------------------------------------------------------- bundle

def evaluate_timepoint(
    *,
    rollout_samples: torch.Tensor,
    observed_samples: torch.Tensor,
    phi_at_centroids: np.ndarray,
    growth_norm: np.ndarray,
    growth_raw: np.ndarray,
    pred_centroids: Optional[torch.Tensor] = None,
    obs_centroids: Optional[torch.Tensor] = None,
    k: int = 10,
    sinkhorn_blur: float = 0.05,
) -> Dict[str, float]:
    """Evaluate ONE held-out timepoint.

    phi_at_centroids is the model's score for each topic; we negate it (low Φ
    means dense/growing). Set `pred_centroids=None` to skip centroid metric.
    """
    scores = -np.asarray(phi_at_centroids, dtype=float)
    truth_norm = np.asarray(growth_norm, dtype=float)
    truth_raw = np.asarray(growth_raw, dtype=float)

    out: Dict[str, float] = {}
    out["w1_marginal"] = w1_marginal(rollout_samples, observed_samples, blur=sinkhorn_blur)
    out["mmd_rbf"]     = mmd_rbf(rollout_samples, observed_samples)
    out["hits_at_10"]  = hits_at_k(scores, truth_norm, k=k)
    out["mrr"]         = mean_reciprocal_rank(scores, truth_norm)
    out["ap"]          = average_precision(scores, truth_norm, threshold="median")
    out["ndcg_at_10"]  = ndcg_at_k(scores, truth_raw, k=k)
    out["spearman"]    = spearman_rho(scores, truth_norm)

    if pred_centroids is not None and obs_centroids is not None:
        out["w1_centroid"] = w1_centroid(pred_centroids, obs_centroids)
    return out


def evaluate_baseline_predictions(
    *,
    g_pred: np.ndarray,
    g_norm: np.ndarray,
    g_raw: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    """Lighter eval for baselines that only output a per-topic score (no SDE).

    Skips W1/MMD/centroid (no particle distribution). Used from baseline_all.py
    so the table is internally consistent with X5 columns.
    """
    out: Dict[str, float] = {}
    out["hits_at_10"]  = hits_at_k(g_pred, g_norm, k=k)
    out["mrr"]         = mean_reciprocal_rank(g_pred, g_norm)
    out["ap"]          = average_precision(g_pred, g_norm, threshold="median")
    out["ndcg_at_10"]  = ndcg_at_k(g_pred, g_raw, k=k)
    out["spearman"]    = spearman_rho(g_pred, g_norm)
    return out
