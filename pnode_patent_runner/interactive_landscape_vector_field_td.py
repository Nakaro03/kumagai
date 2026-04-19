"""
UnifiedVGAETD 用: Φ(z, calendar_year) のヒート・等高線・−∇Φ（年固定）。
interactive_landscape_vector_field.compute_vector_field_for_plotly と同型の JSON を返す。
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from pnode_patent_runner.interactive_landscape_vector_field import (
    _ARROW_COLORS,
    _ARROW_LABELS,
)


def compute_vector_field_for_plotly_td(
    model: torch.nn.Module,
    device: torch.device,
    z_np: np.ndarray,
    calendar_year: int,
    margin: float = 0.5,
    resolution: int = 42,
    quiver_stride: int = 3,
    quiver_length_mult: float = 1.75,
    n_mag_bins: int = 5,
    *,
    phi_heatmap_style: str = "default",
    arrow_style: str = "magnitude",
) -> Dict[str, Any]:
    """calendar_year に固定した Φ(z, year) と −∇Φ。"""
    x_min, x_max = float(z_np[:, 0].min()), float(z_np[:, 0].max())
    y_min, y_max = float(z_np[:, 1].min()), float(z_np[:, 1].max())

    pot = model.temporal_predictor.potential_net
    ode_f = model.temporal_predictor.ode_func
    pot.eval()
    ode_f.eval()

    X, Y, Z = pot.compute_potential_grid(
        (x_min - margin, x_max + margin),
        (y_min - margin, y_max + margin),
        calendar_year,
        resolution=resolution,
        device=str(device),
    )
    grad_x, grad_y = ode_f.compute_gradient_field(X, Y, calendar_year, device=str(device))
    gx = np.asarray(grad_x, dtype=np.float64)
    gy = np.asarray(grad_y, dtype=np.float64)
    Xn = np.asarray(X, dtype=np.float64)
    Yn = np.asarray(Y, dtype=np.float64)
    Zn = np.asarray(Z, dtype=np.float64)

    if phi_heatmap_style == "valley_red_peak_blue":
        z_lo = float(Zn.min())
        z_hi = float(Zn.max())
        if z_hi > z_lo + 1e-12:
            Zn = 2.0 * (Zn - z_lo) / (z_hi - z_lo) - 1.0
        else:
            Zn = np.zeros_like(Zn)

    z1_coords = Xn[:, 0].tolist()
    z2_coords = Yn[0, :].tolist()
    z_heatmap = Zn.T.tolist()

    u = -gx
    v = -gy
    ny, nx = Xn.shape
    if nx > 1 and ny > 1:
        dz1 = abs(float(Xn[1, 0] - Xn[0, 0]))
        dz2 = abs(float(Yn[0, 1] - Yn[0, 0]))
        cell = (dz1 + dz2) / 2.0
    else:
        cell = 1.0

    z_contour: List[List[float]] = [[float(Zn[i, j]) for i in range(ny)] for j in range(nx)]

    mag = np.sqrt(u * u + v * v)
    stride = max(1, int(quiver_stride))
    arrow_len = 0.36 * cell * float(quiver_length_mult)

    arrow_bins: List[Dict[str, Any]] = []
    n_bins = 1

    if arrow_style == "gray_grid":
        xl_all: List[Any] = []
        yl_all: List[Any] = []
        for i in range(0, ny, stride):
            for j in range(0, nx, stride):
                x0 = float(Xn[i, j])
                y0 = float(Yn[i, j])
                ui = float(u[i, j])
                vi = float(v[i, j])
                m = float(np.hypot(ui, vi))
                if m < 1e-14:
                    continue
                ux = ui / m
                uy = vi / m
                x1 = x0 + ux * arrow_len
                y1 = y0 + uy * arrow_len
                xl_all.extend([x0, x1, None])
                yl_all.extend([y0, y1, None])
        if xl_all:
            arrow_bins.append(
                {
                    "xl": xl_all,
                    "yl": yl_all,
                    "color": "rgba(145,148,158,0.88)",
                    "name": "−∇Φ",
                }
            )
    else:
        n_bins = max(2, min(8, int(n_mag_bins)))
        sampled = [
            float(mag[i, j])
            for i in range(0, ny, stride)
            for j in range(0, nx, stride)
            if float(mag[i, j]) > 1e-14
        ]
        if not sampled:
            sampled = [0.0]
        qs = list(np.quantile(sampled, np.linspace(0.0, 1.0, n_bins + 1)))
        qs[0] = 0.0
        qs[-1] = qs[-1] + 1e-9
        colors = (_ARROW_COLORS * (1 + n_bins // len(_ARROW_COLORS)))[:n_bins]
        labels = (_ARROW_LABELS * (1 + n_bins // len(_ARROW_LABELS)))[:n_bins]
        bins_xl: List[List[Any]] = [[] for _ in range(n_bins)]
        bins_yl: List[List[Any]] = [[] for _ in range(n_bins)]
        for i in range(0, ny, stride):
            for j in range(0, nx, stride):
                x0 = float(Xn[i, j])
                y0 = float(Yn[i, j])
                ui = float(u[i, j])
                vi = float(v[i, j])
                m = float(np.hypot(ui, vi))
                if m < 1e-14:
                    continue
                ux = ui / m
                uy = vi / m
                x1 = x0 + ux * arrow_len
                y1 = y0 + uy * arrow_len
                bi = int(np.searchsorted(qs, m, side="right") - 1)
                bi = min(n_bins - 1, max(0, bi))
                bins_xl[bi].extend([x0, x1, None])
                bins_yl[bi].extend([y0, y1, None])
        for b in range(n_bins):
            if not bins_xl[b]:
                continue
            arrow_bins.append(
                {
                    "xl": bins_xl[b],
                    "yl": bins_yl[b],
                    "color": colors[b],
                    "name": f"−∇Φ（{labels[b]}）",
                }
            )

    meta: Dict[str, Any] = {
        "resolution": resolution,
        "margin": margin,
        "quiverStride": stride,
        "quiverLengthMult": float(quiver_length_mult),
        "nMagBins": n_bins,
        "phiHeatmapStyle": phi_heatmap_style,
        "arrowStyle": arrow_style,
        "phiPotentialSource": "time_dependent_neural",
        "calendarYear": int(calendar_year),
        "heatmapColorbarTitle": (
            f"Φ(z, {calendar_year})"
            if phi_heatmap_style == "valley_red_peak_blue"
            else f"Φ(z, year={calendar_year})"
        ),
    }
    if phi_heatmap_style == "valley_red_peak_blue":
        meta["phiDisplayRange"] = [-1.0, 1.0]

    return {
        "heatmap": {"x": z1_coords, "y": z2_coords, "z": z_heatmap},
        "phiContour": {"x": z1_coords, "y": z2_coords, "z": z_contour},
        "arrowBins": arrow_bins,
        "meta": meta,
    }
