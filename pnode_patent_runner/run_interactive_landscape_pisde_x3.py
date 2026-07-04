"""
PI-SDE X3 (Minimal EBM) — Interactive HTML landscape with time slider.

X3 の Φ_θ(x, t) を 2D-PCA 平面上で可視化する自己完結 Plotly HTML。

  ・PCA で潜在 2D 平面を固定 (時間方向に一貫した座標系)
  ・各 t ∈ {1..T} で:
      - 2D グリッド上に Φ_θ(PCA⁻¹(z₂), t) を heatmap
      - 同グリッド上に score field  -∇_z Φ を quiver で表示
        (PCA は線形射影なので gradient も同じ線形変換で 2D に落とせる)
      - 観測点を年色で散布
      - centroid を成長率 g_j で色分け
      - growth-anchor の per-topic 「Φ_θ(c_j,t) vs -g̃_j(t)」 比較

なぜ PCA か (UMAP ではなく):
  UMAP は非線形・非可逆なので 2D grid から高次元への lift 関数が定義できず、
  Φ や ∇Φ を grid 上で正しく計算できない。
  X3 の主張は "Φ = -log p" (score = -∇Φ) なので、ベクトル場の物理的意味を
  保つには線形射影が必須。

Usage:
  PNODE_DOMAIN_TARGET=paper PNODE_LAM_G=0.5 PNODE_SEED=42 \\
    python pnode_patent_runner/run_interactive_landscape_pisde_x3.py
"""
from __future__ import annotations

import os
import sys
import glob
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

from src.model import ForwardSDE


# ── 設定 ─────────────────────────────────────────────────────────────
DOMAIN  = os.environ.get("PNODE_DOMAIN_TARGET", "paper")
SEED    = int(os.environ.get("PNODE_SEED", 42))
LAM_G   = float(os.environ.get("PNODE_LAM_G", 0.5))
D_CTX   = int(os.environ.get("PNODE_D_CTX", 32))
GRID_N  = int(os.environ.get("PNODE_GRID_N", 60))
QUIVER_STRIDE = int(os.environ.get("PNODE_QUIVER_STRIDE", 4))
VARIANT = os.environ.get("PNODE_X3_VARIANT", "baseline")
assert VARIANT in {"baseline", "clean"}

DOMAIN_MAP = {
    "paper":              ("PNode_Paper_X1",                  "data/PNode_Paper_X1",                  2022),
    "patent_energy_top50":("PNode_Patent_Energy_X1_top50",    "data/PNode_Patent_Energy_X1_top50",    2010),
    "arxiv_construction": ("PNode_ArXiv_Construction_X1_v2",  "data/PNode_ArXiv_Construction_X1_v2",  2014),
    "jp_construction":    ("PNode_JP_Construction_X1",        "data/PNode_JP_Construction_X1",        2014),
}
if DOMAIN not in DOMAIN_MAP:
    raise ValueError(f"unknown DOMAIN={DOMAIN}")
DATA_NAME, DATA_DIR, YEAR_BASE = DOMAIN_MAP[DOMAIN]

if VARIANT == "baseline":
    X3_NAME = f"softplus-400_400-0.0-const-0.1-0.1-0.005-x3_g{LAM_G}"
    X3_DIR  = Path(f"RESULTS_X3/{DATA_NAME}/{X3_NAME}/seed_{SEED}/alltime")
else:
    X3_DIR  = Path(f"RESULTS_X3_ABLATION/{DATA_NAME}/mask/x3abl_mask_g{LAM_G}/seed_{SEED}/alltime")
DATA_PT = f"{DATA_DIR}/alltime/fate_train.pt"
_suffix = "x3clean" if VARIANT == "clean" else "x3"
OUT_HTML = X3_DIR / f"landscape_{_suffix}_interactive.html"


# ── MinimalPredictor (run_pisde_x3.py と同一定義) ───────────────────
class MinimalPredictor(nn.Module):
    def __init__(self, x_dim, d_ctx=32):
        super().__init__()
        self.embed = nn.Linear(x_dim + 2, d_ctx)
        self.attn = nn.MultiheadAttention(d_ctx, num_heads=2, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(d_ctx)
        self.out = nn.Linear(d_ctx + x_dim + 1, 1)

    def forward(self, c, g_norm, t_val):
        K = c.shape[0]
        t_col = torch.full((K, 1), float(t_val), device=c.device)
        g_col = g_norm.unsqueeze(-1)
        feat = torch.cat([c, g_col, t_col], dim=-1)
        emb = self.embed(feat).unsqueeze(0)
        att, _ = self.attn(emb, emb, emb)
        ctx = self.norm(emb + att).squeeze(0)
        inp = torch.cat([ctx, c, t_col], dim=-1)
        return self.out(inp).squeeze(-1)


# ── データロード ────────────────────────────────────────────────────
print(f"[X3 interactive] domain={DOMAIN}  seed={SEED}  λ_g={LAM_G}")
print(f"  RESULTS dir: {X3_DIR}")
if not X3_DIR.exists():
    raise SystemExit(
        f"\n✗ X3 checkpoint dir not found: {X3_DIR}\n"
        f"  → Run training first:\n"
        f"    PNODE_DOMAIN_TARGET={DOMAIN} PNODE_LAM_G={LAM_G} PNODE_SEED={SEED} "
        f"python pnode_patent_runner/run_pisde_x3.py"
    )

data = torch.load(DATA_PT, weights_only=False)
xp           = data["xp"]
y            = data["y"]
topics       = data["topics"]
topic_names  = data["topic_names"]
centroids    = data["centroids"]
growth_raw   = data["growth"]
growth_norm  = data["growth_norm"]
n_topics     = data["n_topics"]
T = len(y) - 1
print(f"  topics={n_topics}, time points={y}")

config = SimpleNamespace(**torch.load(X3_DIR / "config.pt", weights_only=False))
config.x_dim = xp[0].shape[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ForwardSDE(config).to(device)
predictor = MinimalPredictor(x_dim=config.x_dim, d_ctx=D_CTX).to(device)

ckpts = sorted(glob.glob(str(X3_DIR / "train.epoch_*.pt")))
ckpt_path = ckpts[-1] if ckpts else str(X3_DIR / "train.best.pt")
if not Path(ckpt_path).exists():
    raise SystemExit(f"✗ No checkpoint under {X3_DIR}")
print(f"  Loading ckpt: {ckpt_path}")
ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
predictor.load_state_dict(ckpt["predictor_state_dict"])
model.eval(); predictor.eval()


# ── PCA fit (時間一貫の 2D 座標系) ───────────────────────────────────
print("PCA fitting (linear, invertible for valid score field)...")
from sklearn.decomposition import PCA
x_all = torch.cat(xp).numpy()
x_mean = x_all.mean(axis=0, keepdims=True)
pca = PCA(n_components=2, random_state=42)
x_all_2d = pca.fit_transform(x_all - x_mean)   # (N, 2)
W = pca.components_                            # (2, x_dim)
years_all = np.concatenate([np.full(v.shape[0], y[k]) for k, v in enumerate(xp)])
explained = pca.explained_variance_ratio_.sum()
print(f"  PCA explained variance (first 2 PCs): {explained*100:.1f}%")


# ── grid 設定 ───────────────────────────────────────────────────────
pad = 0.20
x0, x1 = x_all_2d[:, 0].min(), x_all_2d[:, 0].max()
y0, y1 = x_all_2d[:, 1].min(), x_all_2d[:, 1].max()
rx, ry = x1 - x0, y1 - y0
xs = np.linspace(x0 - pad * rx, x1 + pad * rx, GRID_N)
ys = np.linspace(y0 - pad * ry, y1 + pad * ry, GRID_N)
XX, YY = np.meshgrid(xs, ys)                                       # (G, G)
grid_2d = np.stack([XX.ravel(), YY.ravel()], axis=1)               # (G², 2)

# 2D grid → 高次元 lift  (PCA 逆射影:  x_hd = z_2 · W + mean)
grid_hd = grid_2d @ W + x_mean                                     # (G², x_dim)
grid_hd_t = torch.tensor(grid_hd, dtype=torch.float32, device=device)


def compute_phi_and_score_at_t(t_val: float):
    """Φ と score (-∇_z Φ in 2D-PCA space) を grid 上で計算"""
    G2 = grid_hd_t.shape[0]
    t_col = torch.full((G2, 1), float(t_val), device=device)
    xt = torch.cat([grid_hd_t, t_col], dim=1).requires_grad_()
    phi = model._func._pot(xt).squeeze(-1)                          # (G²,)
    drift = torch.autograd.grad(phi.sum(), xt, create_graph=False)[0]
    grad_x = drift[:, :-1].detach().cpu().numpy()                   # (G², x_dim)  ∇_x Φ
    # 2D-PCA 空間への射影: PCA 線形なので chain rule で  ∇_z Φ = (∇_x Φ) · Wᵀ
    grad_2d = grad_x @ W.T                                          # (G², 2)
    score_2d = -grad_2d                                             # (G², 2)
    return phi.detach().cpu().numpy().reshape(GRID_N, GRID_N), score_2d


# ── 各 t について全計算 ─────────────────────────────────────────────
frames_data = []
phi_global_min, phi_global_max = np.inf, -np.inf
for t_idx in range(1, T + 1):
    t_val = float(y[t_idx])
    phi_grid, score_2d = compute_phi_and_score_at_t(t_val)
    phi_global_min = min(phi_global_min, float(phi_grid.min()))
    phi_global_max = max(phi_global_max, float(phi_grid.max()))

    # quiver 用に間引き
    idx_2d = np.arange(GRID_N * GRID_N).reshape(GRID_N, GRID_N)
    sel = idx_2d[::QUIVER_STRIDE, ::QUIVER_STRIDE].ravel()
    quiver_x  = grid_2d[sel, 0]; quiver_y  = grid_2d[sel, 1]
    quiver_u  = score_2d[sel, 0]; quiver_v = score_2d[sel, 1]
    # 矢印長をグリッド幅にスケール
    qmag = np.sqrt(quiver_u**2 + quiver_v**2) + 1e-12
    target_len = (xs[1] - xs[0]) * QUIVER_STRIDE * 0.45
    scale = target_len / np.percentile(qmag, 80)
    quiver_u_s = quiver_u * scale
    quiver_v_s = quiver_v * scale

    # その t の observation
    mask_t = (years_all.astype(int) == t_idx)
    obs_x = x_all_2d[mask_t, 0]; obs_y = x_all_2d[mask_t, 1]

    # centroid と growth (active のみ)
    cent_t = centroids[t_idx].numpy()
    am = cent_t.sum(axis=-1) != 0
    cent_active_hd = cent_t[am]
    cent_2d_t = (cent_active_hd - x_mean) @ W.T                     # (K, 2)
    g_t = growth_raw[t_idx].numpy()[am]
    g_n = growth_norm[t_idx].numpy()[am]
    names_t = [str(topic_names[i]) for i in range(n_topics) if am[i]]

    # Φ at centroids (anchor 検証)
    xt_c = torch.cat([torch.tensor(cent_active_hd, dtype=torch.float32),
                      torch.full((len(cent_active_hd), 1), t_val)], dim=1).to(device).requires_grad_()
    phi_c = model._func._pot(xt_c).squeeze(-1).detach().cpu().numpy()

    frames_data.append({
        "t_idx": t_idx,
        "t_val": t_val,
        "year_label": f"t={t_idx} (year ~{YEAR_BASE + t_idx})",
        "phi_grid": phi_grid.tolist(),
        "obs_x": obs_x.tolist(),
        "obs_y": obs_y.tolist(),
        "quiver_x": quiver_x.tolist(),
        "quiver_y": quiver_y.tolist(),
        "quiver_u": quiver_u_s.tolist(),
        "quiver_v": quiver_v_s.tolist(),
        "cent_x": cent_2d_t[:, 0].tolist(),
        "cent_y": cent_2d_t[:, 1].tolist(),
        "cent_g": g_t.tolist(),
        "cent_g_n": g_n.tolist(),
        "cent_phi": phi_c.tolist(),
        "cent_names": names_t,
    })

print(f"  frames computed: {len(frames_data)}  (Φ range: {phi_global_min:.3f}..{phi_global_max:.3f})")


# ── Plotly HTML 構築 ────────────────────────────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def make_frame(fd):
    traces = []
    # (1) Φ heatmap
    traces.append(go.Heatmap(
        x=xs, y=ys, z=fd["phi_grid"],
        colorscale="RdYlBu_r", zmin=phi_global_min, zmax=phi_global_max,
        colorbar=dict(title="Φ = -log p", x=0.43, len=0.85),
        hovertemplate="PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>Φ: %{z:.3f}<extra></extra>",
        name="Φ heatmap", showscale=True,
    ))
    # (2) score quiver (line segments)
    arrow_xs, arrow_ys = [], []
    for x0_, y0_, u_, v_ in zip(fd["quiver_x"], fd["quiver_y"], fd["quiver_u"], fd["quiver_v"]):
        arrow_xs += [x0_, x0_ + u_, None]
        arrow_ys += [y0_, y0_ + v_, None]
    traces.append(go.Scatter(
        x=arrow_xs, y=arrow_ys, mode="lines",
        line=dict(color="rgba(80,80,80,0.55)", width=1),
        name="score = -∇Φ", hoverinfo="skip", showlegend=True,
    ))
    # (3) observations
    traces.append(go.Scatter(
        x=fd["obs_x"], y=fd["obs_y"], mode="markers",
        marker=dict(size=3.5, color="rgba(255,255,255,0.55)", line=dict(width=0)),
        name=f"obs @ {fd['year_label']}",
        hovertemplate="obs<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
    ))
    # (4) centroids
    abs_max_g = max(0.5, float(np.max(np.abs(fd["cent_g"]))) if fd["cent_g"] else 0.5)
    hover = [
        f"<b>{nm}</b><br>g_j = {g:.3f}<br>Φ_θ(c_j,t) = {p:.3f}<br>"
        f"target -g̃_j = {-gn:.3f}<br>residual = {(p + gn):.3f}"
        for nm, g, gn, p in zip(fd["cent_names"], fd["cent_g"], fd["cent_g_n"], fd["cent_phi"])
    ]
    traces.append(go.Scatter(
        x=fd["cent_x"], y=fd["cent_y"], mode="markers+text",
        marker=dict(
            size=14, color=fd["cent_g"], colorscale="RdYlGn",
            cmin=-abs_max_g, cmax=abs_max_g,
            line=dict(width=1, color="black"),
            colorbar=dict(title="g_j", x=1.04, len=0.85),
        ),
        text=[nm.replace("cs.", "") for nm in fd["cent_names"]],
        textposition="top center", textfont=dict(size=8, color="black"),
        name="centroids", hovertext=hover, hoverinfo="text",
    ))
    # (5) anchor scatter (subplot 2): -g̃_j vs Φ(c_j,t)
    target = [-gn for gn in fd["cent_g_n"]]
    traces.append(go.Scatter(
        x=target, y=fd["cent_phi"], mode="markers+text",
        marker=dict(
            size=12, color=fd["cent_g"], colorscale="RdYlGn",
            cmin=-abs_max_g, cmax=abs_max_g,
            line=dict(width=1, color="black"),
            showscale=False,
        ),
        text=[nm.replace("cs.", "") for nm in fd["cent_names"]],
        textposition="top center", textfont=dict(size=7),
        name="anchor scatter",
        hovertext=hover, hoverinfo="text",
        xaxis="x2", yaxis="y2", showlegend=False,
    ))
    # (6) anchor reference line y = x
    if target:
        lo = min(min(target), min(fd["cent_phi"]))
        hi = max(max(target), max(fd["cent_phi"]))
        traces.append(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color="black", dash="dash", width=1),
            name="Φ = -g̃ (anchor target)",
            xaxis="x2", yaxis="y2", showlegend=False,
        ))
    return traces


# 初期フレーム = 最終時刻
init_idx = len(frames_data) - 1
init_traces = make_frame(frames_data[init_idx])

# layout (2 columns: landscape | anchor scatter)
layout = go.Layout(
    title=dict(
        text=(f"<b>PI-SDE {'X3-clean' if VARIANT == 'clean' else 'X3'} (Minimal EBM)</b>  |  {DOMAIN}, seed={SEED}, λ_g={LAM_G}<br>"
              f"<sup>Φ_θ(x,t) = -log p_data(x,t)  →  -∇Φ = score = SDE drift  |  "
              f"PCA-2D ({explained*100:.0f}% var)</sup>"),
        x=0.5, xanchor="center",
    ),
    width=1500, height=700,
    paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(domain=[0.0, 0.50], title="PC1", scaleanchor="y", scaleratio=1.0),
    yaxis=dict(domain=[0.0, 1.0], title="PC2"),
    xaxis2=dict(domain=[0.62, 1.0], title="Target: -g̃_j(t)"),
    yaxis2=dict(domain=[0.0, 1.0], title="Φ_θ(c_j, t)"),
    margin=dict(l=60, r=60, t=80, b=80),
    sliders=[dict(
        active=init_idx,
        currentvalue=dict(prefix="Time: ", font=dict(size=14)),
        pad=dict(t=40),
        steps=[
            dict(method="animate",
                 args=[[f"t{fd['t_idx']}"],
                       dict(mode="immediate", frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0))],
                 label=fd["year_label"])
            for fd in frames_data
        ],
    )],
    annotations=[
        dict(text="<b>Left:</b> Φ landscape (heatmap) + score field arrows (−∇Φ) + observations + centroids",
             xref="paper", yref="paper", x=0.0, y=1.06,
             showarrow=False, font=dict(size=10), align="left"),
        dict(text="<b>Right:</b> Anchor check — perfect = on y=x line",
             xref="paper", yref="paper", x=0.62, y=1.06,
             showarrow=False, font=dict(size=10), align="left"),
    ],
)

frames = [
    go.Frame(data=make_frame(fd), name=f"t{fd['t_idx']}")
    for fd in frames_data
]

fig = go.Figure(data=init_traces, layout=layout, frames=frames)
OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(OUT_HTML), include_plotlyjs="cdn", full_html=True)
print(f"\nSaved -> {OUT_HTML}")
print(f"  Open in browser; slider moves t ∈ {{1,…,{T}}}")
