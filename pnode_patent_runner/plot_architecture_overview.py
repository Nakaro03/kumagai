"""
NeuralSDE + X1 (Topic-Anchored Potential) 論文用 アーキテクチャ全体図。
1 枚で pipeline + loss 構成 + landscape を一望できる schematic。
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np

OUT = Path("RESULTS/architecture_overview.png")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 5, height_ratios=[0.9, 1.1, 0.9],
                      width_ratios=[1, 1, 1.4, 1.2, 1], hspace=0.45, wspace=0.35)

# ------------------- Row 0: data pipeline -------------------
ax_data = fig.add_subplot(gs[0, :])
ax_data.set_xlim(0, 10); ax_data.set_ylim(0, 1.2); ax_data.axis("off")
ax_data.set_title("(1) Data pipeline:  text → embedding → topic centroid + growth",
                  fontsize=12, fontweight="bold", loc="left")

def box(ax, x, y, w, h, text, color="#dde9f3", ec="#3a5a78"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       facecolor=color, edgecolor=ec, lw=1.5)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10)

def arrow(ax, x0, y0, x1, y1):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->",
                        mutation_scale=18, color="#444", lw=1.2)
    ax.add_patch(a)

box(ax_data, 0.05, 0.30, 1.6, 0.55, "Paper / Patent\n(title + abstract)\n× T timesteps", color="#fff4e6")
arrow(ax_data, 1.70, 0.58, 2.10, 0.58)
box(ax_data, 2.15, 0.30, 1.55, 0.55, "Sentence-BERT\n(384/1024 D)")
arrow(ax_data, 3.75, 0.58, 4.15, 0.58)
box(ax_data, 4.20, 0.30, 1.30, 0.55, "PCA → 50 D")
arrow(ax_data, 5.55, 0.58, 5.95, 0.58)
box(ax_data, 6.00, 0.30, 1.85, 0.55, "k-means topics\n→ centroid c_j(t)", color="#e6f4e6")
arrow(ax_data, 7.90, 0.58, 8.30, 0.58)
box(ax_data, 8.35, 0.30, 1.55, 0.55, "norm. growth\ng̃_j(t) ∈ R", color="#fce8e8")

# ------------------- Row 1 left: model body -------------------
ax_model = fig.add_subplot(gs[1, 0:2])
ax_model.set_xlim(0, 10); ax_model.set_ylim(0, 6); ax_model.axis("off")
ax_model.set_title("(2) Potential network  Φ_θ(x, t)",
                   fontsize=12, fontweight="bold", loc="left")

# NN layers
layer_x = [1.2, 3.0, 4.8, 6.6, 8.4]
layer_w = [1.1, 1.4, 1.4, 1.4, 1.1]
layer_h = [4.4, 3.2, 3.2, 3.2, 1.0]
layer_label = ["x ∈ R⁵⁰\n⊕ t ∈ R", "Linear\n(51→400)\nSoftplus", "Linear\n(400→400)\nSoftplus", "Linear\n(400→400)\nSoftplus", "Linear\n→ R\nΦ"]
layer_color = ["#fff4e6", "#dde9f3", "#dde9f3", "#dde9f3", "#fce8e8"]
for x, w, h, lab, c in zip(layer_x, layer_w, layer_h, layer_label, layer_color):
    y_c = 3.0
    p = FancyBboxPatch((x, y_c - h/2), w, h, boxstyle="round,pad=0.03",
                       facecolor=c, edgecolor="#3a5a78", lw=1.5)
    ax_model.add_patch(p)
    ax_model.text(x + w/2, y_c, lab, ha="center", va="center", fontsize=8.5)

# arrows between layers
for i in range(len(layer_x) - 1):
    x0 = layer_x[i] + layer_w[i]
    x1 = layer_x[i+1]
    arrow(ax_model, x0, 3.0, x1, 3.0)

ax_model.text(5, 0.4, "≈180k params, scalar potential field on R⁵⁰ × time",
              ha="center", fontsize=8, style="italic", color="#555")

# ------------------- Row 1 center: SDE equation + loss diagram -------------------
ax_sde = fig.add_subplot(gs[1, 2])
ax_sde.set_xlim(0, 10); ax_sde.set_ylim(0, 10); ax_sde.axis("off")
ax_sde.set_title("(3) Latent SDE + loss",
                 fontsize=12, fontweight="bold", loc="left")

ax_sde.text(5, 8.9, r"$dx = -\nabla_x \Phi_\theta(x, t)\, dt + \sigma\, dW$",
            ha="center", fontsize=12, color="#1f4f7f", fontweight="bold")
ax_sde.text(5, 8.05, "  (drift = -∇Φ,  σ = 0.1)", ha="center", fontsize=8.5, color="#444")

# 3 loss components
loss_y = [6.0, 4.3, 2.6, 0.9]
loss_text = [
    r"$\mathcal{L}_{\mathrm{Sinkhorn}}$: OT match SDE→obs at each t",
    r"$\mathcal{L}_{\mathrm{HJ}}$:  $|\partial_t\Phi - \frac{1}{2}|\nabla\Phi|^2|$",
    r"$\mathcal{L}_{\mathrm{val}}$:  $\Phi(c_j,t) \approx -\alpha\, \tilde g_j(t)$",
    r"$\mathcal{L}_{\mathrm{grad/basin}}$:  centroid = valley",
]
loss_colors = ["#dde9f3", "#dde9f3", "#fce8e8", "#fce8e8"]
loss_marker = ["PI-SDE", "PI-SDE", "X1 ✦", "X1 ✦"]
for y, txt, c, mk in zip(loss_y, loss_text, loss_colors, loss_marker):
    p = FancyBboxPatch((0.4, y), 8.8, 1.1, boxstyle="round,pad=0.02",
                       facecolor=c, edgecolor="#3a5a78", lw=1.2)
    ax_sde.add_patch(p)
    ax_sde.text(0.7, y + 0.55, txt, ha="left", va="center", fontsize=9)
    ax_sde.text(8.95, y + 0.55, mk, ha="right", va="center", fontsize=7.5,
                color="#7a0000" if "X1" in mk else "#1f4f7f", fontweight="bold")

# ------------------- Row 1 right: inputs/outputs box -------------------
ax_io = fig.add_subplot(gs[1, 3])
ax_io.set_xlim(0, 10); ax_io.set_ylim(0, 10); ax_io.axis("off")
ax_io.set_title("(4) Input / Output",
                fontsize=12, fontweight="bold", loc="left")

ax_io.text(0.4, 8.7, "Inputs (per topic, per t)", fontsize=10, fontweight="bold", color="#1f4f7f")
ax_io.text(0.4, 7.7, "• centroid  $c_j(t) \\in \\mathbb{R}^{50}$", fontsize=9.5)
ax_io.text(0.4, 6.7, "• growth   $\\tilde g_j(t) \\in \\mathbb{R}$", fontsize=9.5)
ax_io.text(0.4, 5.7, "• point cloud  $x_t \\in \\mathbb{R}^{N_t \\times 50}$", fontsize=9.5)

ax_io.text(0.4, 4.4, "Outputs", fontsize=10, fontweight="bold", color="#7a0000")
ax_io.text(0.4, 3.4, "• $\\Phi_\\theta(x, t)$  growth potential field", fontsize=9.5)
ax_io.text(0.4, 2.4, "• rank(topic) for $t = T{+}1$", fontsize=9.5)
ax_io.text(0.4, 1.4, "• topology of trends (UMAP)", fontsize=9.5)

# ------------------- Row 1 far right: evaluation -------------------
ax_ev = fig.add_subplot(gs[1, 4])
ax_ev.set_xlim(0, 10); ax_ev.set_ylim(0, 10); ax_ev.axis("off")
ax_ev.set_title("(5) Eval metrics",
                fontsize=12, fontweight="bold", loc="left")

ev_lines = [
    ("Spearman ρ",   "rank(Φ) ↔ rank(g)"),
    ("NDCG@10",      "top-10 ranking gain"),
    ("P@10",         "top-10 precision"),
    ("MSE / MAE",    "growth-head residual"),
]
for i, (lab, sub) in enumerate(ev_lines):
    y = 7.5 - i * 1.6
    p = FancyBboxPatch((0.3, y - 0.35), 9.4, 1.1, boxstyle="round,pad=0.02",
                       facecolor="#f0f0f0", edgecolor="#888", lw=1.0)
    ax_ev.add_patch(p)
    ax_ev.text(0.55, y + 0.4, lab, fontsize=9.5, fontweight="bold", color="#1f4f7f")
    ax_ev.text(0.55, y + 0.0, sub, fontsize=8, color="#444")

# ------------------- Row 2: trained Φ landscape (preview) -------------------
ax_land = fig.add_subplot(gs[2, :])
ax_land.set_title("(6) Trained Φ landscape  (Paper / arXiv CS,  t = 2025,  seed=42)",
                  fontsize=12, fontweight="bold", loc="left")

# Load the landscape png we just generated and embed as raster
try:
    import matplotlib.image as mpimg
    img = mpimg.imread("RESULTS/PNode_Paper_X1/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01/seed_42/alltime/trajectories_x1_umap.png")
    ax_land.imshow(img)
    ax_land.axis("off")
except Exception as e:
    ax_land.text(0.5, 0.5, f"(landscape preview missing: {e})",
                 ha="center", va="center", fontsize=10, color="#888",
                 transform=ax_land.transAxes)
    ax_land.axis("off")

plt.suptitle("Neural-SDE Technology-Trend Forecasting  (PI-SDE + X1 Topic-Anchored Potential)  —  Architecture overview",
             fontsize=13, fontweight="bold", y=0.995)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
print(f"Saved -> {OUT}")
