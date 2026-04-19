"""
学習済みエンコーダの μ（特許ノード）だけから、グリッド上の log p̂_t と D_t を計算するユーティリティ。

KDE は sklearn.neighbors.KernelDensity（ガウスカーネル）。帯域幅は Scott 風の簡易式または手動指定。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    from sklearn.neighbors import KernelDensity
except ImportError as e:  # pragma: no cover
    raise ImportError("offline_density_maps requires scikit-learn") from e


def scott_bandwidth_2d(points: np.ndarray) -> float:
    """2D 点列に対する Scott 風のスカラー帯域幅（各軸同一）。"""
    n, d = points.shape
    if n < 2 or d != 2:
        return 0.5
    sigma = np.std(points, axis=0)
    sig = float(np.clip(np.mean(sigma), 1e-6, None))
    return float(n ** (-1.0 / (d + 4)) * sig)


def build_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """形状 (ny, nx) のメッシュ。X, Y は各セル中心。"""
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return X, Y, np.stack([X.ravel(), Y.ravel()], axis=1)


def kde_log_density(
    points_xy: np.ndarray,
    grid_xy: np.ndarray,
    bandwidth: Optional[float] = None,
    min_points: int = 3,
) -> np.ndarray:
    """
    points_xy: (n, 2)
    grid_xy: (m, 2)
    Returns: log p̂ on grid, shape (m,)
    """
    if points_xy.shape[0] < min_points:
        return np.full(grid_xy.shape[0], -np.inf, dtype=np.float64)
    bw = float(bandwidth) if bandwidth is not None else scott_bandwidth_2d(points_xy)
    bw = max(bw, 1e-6)
    kde = KernelDensity(kernel="gaussian", bandwidth=bw)
    kde.fit(points_xy)
    return kde.score_samples(grid_xy)


def union_bounds_with_margin(
    all_points: List[np.ndarray], margin: float
) -> Tuple[float, float, float, float]:
    """複数年の μ をまとめた外接矩形 + margin。"""
    if not all_points:
        return -1.0, 1.0, -1.0, 1.0
    pts = np.vstack([p for p in all_points if len(p) > 0])
    if len(pts) == 0:
        return -1.0, 1.0, -1.0, 1.0
    x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
    return x0 - margin, x1 + margin, y0 - margin, y1 + margin


def resolve_ref_year(
    years_sorted: List[int], t: int, delta_years: int
) -> Optional[int]:
    """t より前で、少なくとも delta_years 年さかのぼった年に最も近い利用可能年。"""
    target = t - delta_years
    candidates = [y for y in years_sorted if y <= target]
    if not candidates:
        return None
    return max(candidates)


def log_density_grid(
    points_xy: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution: int,
    bandwidth: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X, Y : (ny, nx)
    logp : (ny, nx)
    """
    X, Y, grid = build_grid(x_min, x_max, y_min, y_max, resolution)
    logp = kde_log_density(points_xy, grid, bandwidth=bandwidth)
    ny, nx = X.shape
    return X, Y, logp.reshape(ny, nx)
