"""
Φ グリッド・等高線用 Z・疎矢印を JSON に載せ、HTML テンプレートに埋め込む。

- 矢印: ``arrow_style="magnitude"`` で |∇Φ| ビンごとの色、``"gray_grid"`` で単色グレー（グリッド上の矢印）。
- ``phi_heatmap_style="valley_red_peak_blue"`` のとき、グリッド上の Φ を **min–max で [−1, 1]** に線形正規化してからヒート・等高線に載せる（谷＝低＝赤、山＝高＝青はテンプレ側の colorscale + zmin/zmax）。
- ``arrow_style="gray_grid"`` で **グレー矢印**（−∇Φ 方向＝等高線に直交）。

既定テンプレート: ``interactive_vector_field_template.html``  
別レイアウト例: ``interactive_vector_field_alt_dark.html``（``template_path`` ＋ ``alt_dark_ui_labels(...)`` で企業–特許／著者–論文／著者–トピックの表記を切替）
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from pnode_patent_runner.offline_density_maps import log_density_grid

_ARROW_COLORS = [
    "rgba(49, 54, 149, 0.82)",
    "rgba(69, 117, 180, 0.84)",
    "rgba(116, 173, 209, 0.88)",
    "rgba(253, 174, 97, 0.9)",
    "rgba(215, 48, 39, 0.92)",
]
_ARROW_LABELS = ["ごく弱", "弱", "中", "強", "ごく強"]

_MULTI_PEAK_LINE_COLORS = [
    "rgba(251, 191, 36, 0.9)",
    "rgba(56, 189, 248, 0.9)",
    "rgba(167, 139, 250, 0.88)",
    "rgba(52, 211, 153, 0.88)",
    "rgba(251, 113, 133, 0.88)",
    "rgba(253, 224, 71, 0.86)",
    "rgba(125, 211, 252, 0.88)",
    "rgba(216, 180, 254, 0.86)",
    "rgba(165, 243, 252, 0.85)",
    "rgba(251, 146, 60, 0.88)",
    "rgba(244, 114, 182, 0.86)",
    "rgba(94, 234, 212, 0.86)",
]


def _find_logp_local_peaks(
    logp: np.ndarray,
    *,
    min_percentile: float,
    min_sep_cells: float,
    max_peaks: int,
) -> List[Tuple[int, int]]:
    """log p̂ グリッド上の局所最大（3×3）を、しきい値と最小間隔で間引く。"""
    from scipy import ndimage

    ny, nx = logp.shape
    if ny < 3 or nx < 3:
        return []
    interior = np.ones_like(logp, dtype=bool)
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
    mx = ndimage.maximum_filter(logp, size=3, mode="nearest")
    pth = float(np.percentile(logp, min_percentile))
    lm = (logp == mx) & interior & (logp >= pth)
    coords = np.argwhere(lm)
    if len(coords) == 0:
        return []
    scores = np.array([logp[int(c[0]), int(c[1])] for c in coords])
    order = np.argsort(-scores)
    coords = coords[order]
    sep2 = float(min_sep_cells) ** 2
    kept: List[Tuple[int, int]] = []
    for iy, ix in coords:
        if len(kept) >= max_peaks:
            break
        iy, ix = int(iy), int(ix)
        ok = True
        for jy, jx in kept:
            if (iy - jy) ** 2 + (ix - jx) ** 2 < sep2:
                ok = False
                break
        if ok:
            kept.append((iy, ix))
    return kept


def _masks_for_multi_peak_contours(
    logp: np.ndarray,
    peaks: List[Tuple[int, int]],
    level_drop: float,
) -> List[np.ndarray]:
    """
    各頂点について ``logp >= logp(頂点) - level_drop`` の連結成分（まだ誰にも割当てられていないセル）を領域とする。
    高密度の「山」ごとに別々の等高線が閉じるようにする。
    """
    from scipy import ndimage

    claimed = np.zeros_like(logp, dtype=bool)
    masks: List[np.ndarray] = []
    for iy, ix in peaks:
        thr = logp[iy, ix] - float(level_drop)
        binm = (logp >= thr) & (~claimed)
        if not np.any(binm):
            continue
        labeled, _ = ndimage.label(binm)
        lab = int(labeled[iy, ix])
        if lab == 0:
            continue
        mask = (labeled == lab) & (~claimed)
        if int(np.sum(mask)) < 3:
            continue
        claimed |= mask
        masks.append(mask)
    return masks


def _zn_to_plotly_z_masked(Zn: np.ndarray, mask: np.ndarray) -> List[List[Any]]:
    """Plotly 用: マスク外は null（JSON None）。"""
    ny, nx = Zn.shape
    out: List[List[Any]] = []
    for j in range(nx):
        row: List[Any] = []
        for i in range(ny):
            if mask[i, j]:
                row.append(float(Zn[i, j]))
            else:
                row.append(None)
        out.append(row)
    return out


def compute_vector_field_for_plotly(
    model: torch.nn.Module,
    device: torch.device,
    z_np: np.ndarray,
    margin: float = 0.5,
    resolution: int = 42,
    quiver_stride: int = 3,
    quiver_length_mult: float = 1.75,
    n_mag_bins: int = 5,
    *,
    phi_heatmap_style: str = "default",
    arrow_style: str = "magnitude",
) -> Dict[str, Any]:
    x_min, x_max = float(z_np[:, 0].min()), float(z_np[:, 0].max())
    y_min, y_max = float(z_np[:, 1].min()), float(z_np[:, 1].max())

    pot = model.temporal_predictor.potential_net
    ode_f = model.temporal_predictor.ode_func
    pot.eval()
    ode_f.eval()

    X, Y, Z = pot.compute_potential_grid(
        (x_min - margin, x_max + margin),
        (y_min - margin, y_max + margin),
        resolution=resolution,
        device=device,
    )
    grad_x, grad_y = ode_f.compute_gradient_field(X, Y, device=device)
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
    }
    if phi_heatmap_style == "valley_red_peak_blue":
        meta["phiDisplayRange"] = [-1.0, 1.0]

    return {
        "heatmap": {"x": z1_coords, "y": z2_coords, "z": z_heatmap},
        "phiContour": {"x": z1_coords, "y": z2_coords, "z": z_contour},
        "arrowBins": arrow_bins,
        "meta": meta,
    }


def compute_vector_field_density_potential_for_plotly(
    z_np: np.ndarray,
    patent_xy: np.ndarray,
    margin: float = 0.5,
    resolution: int = 42,
    quiver_stride: int = 3,
    quiver_length_mult: float = 1.75,
    n_mag_bins: int = 5,
    *,
    phi_heatmap_style: str = "valley_red_peak_blue",
    arrow_style: str = "gray_grid",
    bandwidth: Optional[float] = None,
    phi_contour_mode: str = "multi_peak",
    peak_min_percentile: float = 82.0,
    peak_min_sep_cells: float = 2.5,
    peak_level_drop: float = 1.8,
    peak_max: int = 14,
    multi_peak_n_contours: int = 5,
) -> Dict[str, Any]:
    """
    ``PotentialNet`` の代わりに **Φ(z) = −log p̂(z)**（特許ノード μ の 2D ガウス KDE）をグリッド上に置き、
    ``−∇Φ`` は ``numpy.gradient`` の数値勾配で求める。それ以外の JSON 形状は
    ``compute_vector_field_for_plotly`` と同一（map_cope_alt_dark テンプレ互換）。

    ``phi_contour_mode="multi_peak"`` のとき、log p̂ の **複数の局所最大**を頂点とみなし、
    各頂点まわりのスーパーレベル集合だけにマスクした Φ で **別々の等高線**を描く（全域を1つの頂点に見ない）。

    Parameters
    ----------
    z_np
        その年の全ノード潜在座標（企業・特許）。表示範囲（margin 付き）の決定に使う。
    patent_xy
        KDE に使う点列 ``(n, 2)``。通常は当該年アクティブな特許ノードの μ のみ。
    """
    x_min = float(z_np[:, 0].min()) - margin
    x_max = float(z_np[:, 0].max()) + margin
    y_min = float(z_np[:, 1].min()) - margin
    y_max = float(z_np[:, 1].max()) + margin

    pts = np.asarray(patent_xy, dtype=np.float64)
    if pts.shape[0] < 3:
        pts = np.asarray(z_np, dtype=np.float64)

    X, Y, logp = log_density_grid(pts, x_min, x_max, y_min, y_max, resolution, bandwidth=bandwidth)
    Xn = np.asarray(X, dtype=np.float64)
    Yn = np.asarray(Y, dtype=np.float64)
    logp_a = np.asarray(logp, dtype=np.float64)
    finite = np.isfinite(logp_a)
    if np.any(finite):
        lp_floor = float(np.nanpercentile(logp_a[finite], 1.0))
        logp_a = np.where(finite, logp_a, lp_floor - 1.0)
        logp_a = np.maximum(logp_a, lp_floor)
    Phi_raw = -logp_a

    ny, nx = Phi_raw.shape
    if ny > 1 and nx > 1:
        dz1 = abs(float(Xn[1, 0] - Xn[0, 0]))
        dz2 = abs(float(Yn[0, 1] - Yn[0, 0]))
    else:
        dz1 = dz2 = 1.0
    g0, g1 = np.gradient(Phi_raw, dz1, dz2)
    gx = np.asarray(g0, dtype=np.float64)
    gy = np.asarray(g1, dtype=np.float64)
    u = -gx
    v = -gy

    Zn = np.asarray(Phi_raw, dtype=np.float64)
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
    z_contour: List[List[float]] = [[float(Zn[i, j]) for i in range(ny)] for j in range(nx)]

    multi_peak_traces: List[Dict[str, Any]] = []
    hide_primary_contour = False
    if phi_contour_mode == "multi_peak":
        peaks = _find_logp_local_peaks(
            logp_a,
            min_percentile=peak_min_percentile,
            min_sep_cells=peak_min_sep_cells,
            max_peaks=peak_max,
        )
        masks = _masks_for_multi_peak_contours(logp_a, peaks, peak_level_drop)
        for k, mask in enumerate(masks):
            z_mp = _zn_to_plotly_z_masked(Zn, mask)
            multi_peak_traces.append(
                {
                    "x": z1_coords,
                    "y": z2_coords,
                    "z": z_mp,
                    "ncontours": int(max(3, min(8, multi_peak_n_contours))),
                    "name": f"Φ 等高線（峰 {k + 1}）",
                    "color": _MULTI_PEAK_LINE_COLORS[k % len(_MULTI_PEAK_LINE_COLORS)],
                }
            )
        if multi_peak_traces:
            hide_primary_contour = True

    if nx > 1 and ny > 1:
        cell = (dz1 + dz2) / 2.0
    else:
        cell = 1.0

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
        "phiPotentialSource": "density_kde_neg_logp",
        "heatmapColorbarTitle": (
            "Φ = −log p̂（KDE・密度由来）"
            if phi_heatmap_style == "valley_red_peak_blue"
            else "Φ = −log p̂（KDE）"
        ),
    }
    if phi_heatmap_style == "valley_red_peak_blue":
        meta["phiDisplayRange"] = [-1.0, 1.0]
    if hide_primary_contour:
        meta["hidePrimaryContour"] = True
    if multi_peak_traces:
        meta["multiPeakContourTraces"] = multi_peak_traces
        meta["phiContourMode"] = "multi_peak"

    return {
        "heatmap": {"x": z1_coords, "y": z2_coords, "z": z_heatmap},
        "phiContour": {"x": z1_coords, "y": z2_coords, "z": z_contour},
        "arrowBins": arrow_bins,
        "meta": meta,
    }


def merge_payload_with_vector_field(
    base: Dict[str, Any],
    vector_field: Dict[str, Any],
) -> Dict[str, Any]:
    out = dict(base)
    out["vectorField"] = vector_field
    return out


def alt_dark_ui_labels(
    domain: Literal["patent", "author_paper", "author_topic"],
) -> Dict[str, str]:
    """
    ``interactive_vector_field_alt_dark.html`` 用の表記（企業–特許 / 著者–論文 / 著者–トピック）。
    JSON の ``patents`` / ``corporations`` キーは共通のまま、UI 文言だけ切り替える。
    """
    if domain == "patent":
        return {
            "domain": "patent",
            "yearSliderLabel": "年度",
            "leftSelectLabel": "企業",
            "rightTraceName": "特許",
            "leftTraceName": "企業",
            "layoutTitleTemplate": "表示年: {year} ／ 特許 {nRight} · 企業 {nLeft}",
            "corpOptionTemplate": "{name} ／ この年 {patentsYear} · 全期間 {patentsTotal}",
            "statusYearRightPrefix": " ／ この年の特許 ",
            "statusYearRightSuffix": " 件",
            "jointStatusPrefix": "共同出願特許 ",
            "idleHint": "企業・特許をクリックで強調。",
            "hintHtml": "左: 特許の色分け（単独/共同・Lead_IPC・FI・F-term）。右: クリックしたノードの詳細。ホバーは番号・分類・要約を強調表示。",
            "legendInnerHtml": (
                '<span><span class="swatch" style="background:rgba(248,113,113,0.9)"></span>単独出願</span>'
                '<span><span class="swatch" style="background:rgba(74,222,128,0.9)"></span>共同出願</span>'
                '<span><span class="swatch teal"></span>共同クリック時の関与企業</span>'
            ),
        }
    if domain == "author_paper":
        return {
            "domain": "author_paper",
            "yearSliderLabel": "論文の年（スライダー）",
            "leftSelectLabel": "著者",
            "rightTraceName": "論文",
            "leftTraceName": "著者",
            "layoutTitleTemplate": "論文年: {year} ／ 論文 {nRight} · 著者 {nLeft}",
            "corpOptionTemplate": "{name} ／ この年 {patentsYear} 本 · 全期間 {patentsTotal} 本",
            "statusYearRightPrefix": " ／ この年の論文 ",
            "statusYearRightSuffix": " 本",
            "jointStatusPrefix": "共著論文 ",
            "idleHint": "著者・論文をクリックで強調。",
            "hintHtml": "著者–論文二部グラフ。年は学習・可視化で arxiv_year_min/max と --year-range を揃えてください。",
            "legendInnerHtml": (
                '<span><span class="swatch" style="background:rgba(218,58,58,0.88)"></span>単著論文</span>'
                '<span><span class="swatch" style="background:rgba(42,160,85,0.88)"></span>共著論文</span>'
                '<span><span class="swatch teal"></span>共著クリック時の共著者（著者ノード）</span>'
            ),
        }
    # author_topic
    return {
        "domain": "author_topic",
        "yearSliderLabel": "年（スライダー）",
        "leftSelectLabel": "著者",
        "rightTraceName": "トピック",
        "leftTraceName": "著者",
        "layoutTitleTemplate": "年: {year} ／ トピック {nRight} · 著者 {nLeft}",
        "corpOptionTemplate": "{name} ／ この年 {patentsYear} トピック · 全期間 {patentsTotal} 本",
        "statusYearRightPrefix": " ／ この年のトピック ",
        "statusYearRightSuffix": " 件",
        "jointStatusPrefix": "選択: ",
        "idleHint": "著者・トピックをクリックで強調。",
        "hintHtml": "著者–トピック二部グラフ。チェックポイントは --data-domain author_topic で学習したものを使う。",
        "legendInnerHtml": (
            '<span><span class="swatch" style="background:rgba(218,58,58,0.88)"></span>トピック</span>'
            '<span><span class="swatch" style="background:rgba(42,160,85,0.88)"></span>著者ハイライト時</span>'
            '<span><span class="swatch teal"></span>関連著者</span>'
        ),
    }


def write_interactive_vector_field_html(
    by_year: Dict[str, Dict[str, Any]],
    out_path: Path,
    default_year: str,
    *,
    page_title: str = "Interactive latent map + vector field",
    heading: str = "潜在マップ ＋ Φ / −∇Φ",
    template_path: Optional[Path] = None,
    ui: Optional[Dict[str, str]] = None,
) -> None:
    base = Path(__file__).resolve().parent
    tpl_path = Path(template_path) if template_path is not None else base / "interactive_vector_field_template.html"
    if not tpl_path.is_file():
        raise FileNotFoundError(f"テンプレートが見つかりません: {tpl_path}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if str(default_year) not in by_year:
        raise ValueError(f"default_year={default_year} が by_year に無い: {list(by_year.keys())}")

    wrapper: Dict[str, Any] = {"byYear": by_year, "defaultYear": str(default_year)}
    if ui:
        wrapper["ui"] = ui
    json_str = json.dumps(wrapper, ensure_ascii=False)
    b64 = base64.b64encode(json_str.encode("utf-8")).decode("ascii")

    html_out = tpl_path.read_text(encoding="utf-8")
    html_out = (
        html_out.replace("__PAYLOAD_B64__", b64)
        .replace("__PAGE_TITLE__", page_title)
        .replace("__HEADING__", heading)
    )
    out_path.write_text(html_out, encoding="utf-8")
