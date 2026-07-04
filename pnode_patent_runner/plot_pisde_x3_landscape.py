"""
PI-SDE X3 (Minimal EBM) — Landscape visualization (6-panel paper figure).

X3 の理論主張を「視覚的に証明」する 6 パネル構成。

  Top row  (既存の解釈):
    [A] 観測点 (年色分け)
    [B] Φ heatmap — "Estimated negative log-density"
    [C] トピック centroid 配置 + 実成長率 g_j

  Bottom row (X3 の理論を視覚的に検証):
    [D] Φ ランキング vs 実 g ランキング (相関)
    [E] **EBM proof**: Φ_θ vs empirical -log p̂ (KDE)
        → 線形な正相関なら "Φ = -log p" 主張が経験的に成立
    [F] **Anchor proof**: Φ_θ(c_j, t) vs -g̃_j(t)
        → y=x 線上に乗れば L_val が anchor 制約を満たしている

X3 の checkpoint パス (run_pisde_x3.py 出力):
  RESULTS_X3/{DATA}/softplus-400_400-0.0-const-0.1-0.1-0.005-x3_g{LAM_G}/seed_{SEED}/alltime/

Usage:
  PNODE_DOMAIN_TARGET=paper PNODE_LAM_G=0.5 PNODE_SEED=42 PNODE_EVAL_T=3 \\
    python pnode_patent_runner/plot_pisde_x3_landscape.py
"""
from __future__ import annotations

import os
import sys
import glob
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

from src.model import ForwardSDE

# ── 設定 ─────────────────────────────────────────────────────────────
DOMAIN  = os.environ.get("PNODE_DOMAIN_TARGET", "paper")
SEED    = int(os.environ.get("PNODE_SEED", 42))
LAM_G   = float(os.environ.get("PNODE_LAM_G", 0.5))
EVAL_T  = int(os.environ.get("PNODE_EVAL_T", 3))
D_CTX   = int(os.environ.get("PNODE_D_CTX", 32))
VARIANT = os.environ.get("PNODE_X3_VARIANT", "baseline")   # baseline | clean (= mask)
assert VARIANT in {"baseline", "clean"}, f"unknown VARIANT={VARIANT}"

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
else:  # clean
    X3_DIR  = Path(f"RESULTS_X3_ABLATION/{DATA_NAME}/mask/x3abl_mask_g{LAM_G}/seed_{SEED}/alltime")
DATA_PT = f"{DATA_DIR}/alltime/fate_train.pt"
_suffix = "x3clean" if VARIANT == "clean" else "x3"
OUT_PNG = X3_DIR / f"landscape_{_suffix}_t{EVAL_T}.png"


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
print(f"[X3 landscape] domain={DOMAIN}  seed={SEED}  λ_g={LAM_G}  eval_t={EVAL_T}")
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
print(f"  topics={n_topics}, time points={y}, EVAL_T={EVAL_T} (year ~{YEAR_BASE+EVAL_T})")

# ── モデルロード ────────────────────────────────────────────────────
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


# ── UMAP fit ────────────────────────────────────────────────────────
print("UMAP fitting...")
import umap
x_all = torch.cat(xp).numpy()
um = umap.UMAP(n_components=2, n_neighbors=30, metric="euclidean",
               random_state=42, transform_seed=42)
x_all_2d = um.fit_transform(x_all)
y_all = np.concatenate([np.full(v.shape[0], y[k]) for k, v in enumerate(xp)])

# Φ(x, t=EVAL_T) を全観測点で計算 (時間を EVAL_T 固定にして等高線の意味を統一)
print(f"Computing Φ(x, t={EVAL_T}) for all observations...")
t_col_all = np.full(x_all.shape[0], float(y[EVAL_T]))
xt_all = torch.cat([torch.tensor(x_all, dtype=torch.float32),
                    torch.tensor(t_col_all, dtype=torch.float32).unsqueeze(1)], dim=1)
phi_all = model._func._pot(xt_all.to(device).requires_grad_()).squeeze(-1).detach().cpu().numpy()

# centroid → UMAP 投影 & Φ 計算
cent_t = centroids[EVAL_T].numpy()
active_mask = cent_t.sum(axis=-1) != 0
cent_active = cent_t[active_mask]
cent_2d = um.transform(cent_active)
g_t = growth_raw[EVAL_T].numpy()[active_mask]
g_n = growth_norm[EVAL_T].numpy()[active_mask]
topic_names_active = [topic_names[i] for i in range(n_topics) if active_mask[i]]

xt_cent = torch.cat([torch.tensor(cent_active, dtype=torch.float32),
                     torch.full((len(cent_active), 1), float(y[EVAL_T]))], dim=1)
phi_cent = model._func._pot(xt_cent.to(device).requires_grad_()).squeeze(-1).detach().cpu().numpy()

# Empirical density via 2D KDE in UMAP space  (Φ = -log p の経験的検証用)
print("Estimating empirical density p̂(x, t) via 2D-UMAP KDE...")
years_int = y_all.astype(int)
mask_t = years_int == EVAL_T
if mask_t.sum() >= 10:
    kde_pts = x_all_2d[mask_t].T   # (2, N_t)
    kde = gaussian_kde(kde_pts, bw_method="scott")
    p_hat_at_all = kde(x_all_2d.T)            # 全観測点の密度
    eps = 1e-12
    neg_log_p_all = -np.log(p_hat_at_all + eps)
    valid_kde = True
else:
    p_hat_at_all = np.zeros(len(x_all_2d))
    neg_log_p_all = np.zeros(len(x_all_2d))
    valid_kde = False

# ── プロット ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 3, width_ratios=[1, 1, 1.1], height_ratios=[1, 1],
              wspace=0.28, hspace=0.32)

# [A] 観測点 (年色分け)
ax0 = plt.subplot(gs[0, 0])
T_max = int(max(years_int))
for yt in range(T_max + 1):
    mask = years_int == yt
    ax0.scatter(x_all_2d[mask, 0], x_all_2d[mask, 1], s=3, alpha=0.45,
                color=plt.cm.viridis(yt / max(T_max, 1)),
                label=f"t={yt} (year {YEAR_BASE+yt})")
ax0.set_title("[A] Observations by year", fontsize=11, fontweight="bold")
ax0.set_xlabel("UMAP1"); ax0.set_ylabel("UMAP2")
ax0.legend(fontsize=8, loc="best"); ax0.set_xticks([]); ax0.set_yticks([])

# [B] Φ heatmap
ax1 = plt.subplot(gs[0, 1])
order_phi = np.argsort(phi_all)
sc1 = ax1.scatter(x_all_2d[order_phi, 0], x_all_2d[order_phi, 1],
                  c=phi_all[order_phi], s=3, cmap="RdYlBu_r")
ax1.set_title(f"[B] Estimated $\\Phi_\\theta(x, t={EVAL_T})$ = $-\\log\\,\\hat p(x,t)$\n"
              f"  (low Φ = valley = high density = expected growth)",
              fontsize=11, fontweight="bold")
ax1.set_xlabel("UMAP1"); ax1.set_ylabel("UMAP2")
ax1.set_xticks([]); ax1.set_yticks([])
plt.colorbar(sc1, ax=ax1, label="Φ", fraction=0.04)

# [C] centroid + g
ax2 = plt.subplot(gs[0, 2])
ax2.scatter(x_all_2d[:, 0], x_all_2d[:, 1], s=1, color="lightgray", alpha=0.3)
g_abs_max = max(0.5, float(np.abs(g_t).max()) if len(g_t) > 0 else 0.5)
sc2 = ax2.scatter(cent_2d[:, 0], cent_2d[:, 1], c=g_t, cmap="RdYlGn",
                  s=180, edgecolors="black", linewidths=1.0,
                  vmin=-g_abs_max, vmax=g_abs_max, zorder=5)
for i, name in enumerate(topic_names_active):
    fc = "white" if abs(g_t[i]) > 0.2 * g_abs_max else "black"
    fs = 8 if abs(g_t[i]) > 0.3 * g_abs_max else 6.5
    ax2.annotate(str(name).replace("cs.", ""), (cent_2d[i, 0], cent_2d[i, 1]),
                 ha="center", va="center", fontsize=fs, color=fc,
                 fontweight="bold" if abs(g_t[i]) > 0.4 * g_abs_max else "normal")
ax2.set_title(f"[C] Topic centroids @ t={EVAL_T} (year ~{YEAR_BASE+EVAL_T})\n"
              f"  Color = actual growth rate $g_j$", fontsize=11, fontweight="bold")
ax2.set_xlabel("UMAP1"); ax2.set_ylabel("UMAP2")
ax2.set_xticks([]); ax2.set_yticks([])
plt.colorbar(sc2, ax=ax2, label="$g_j$", fraction=0.04)

# [D] Φ-rank vs g-rank
ax3 = plt.subplot(gs[1, 0])
phi_rank = np.argsort(np.argsort(phi_cent))   # low Φ → rank 0
g_rank   = np.argsort(np.argsort(-g_t))       # high g → rank 0
r, p = stats.spearmanr(phi_cent, g_t)
ax3.scatter(phi_rank, g_rank, c=g_t, cmap="RdYlGn", s=100,
            edgecolors="black", linewidths=0.7, vmin=-g_abs_max, vmax=g_abs_max)
for i, name in enumerate(topic_names_active):
    if abs(g_t[i]) > 0.3 * g_abs_max or phi_rank[i] < 5 or phi_rank[i] >= len(phi_rank) - 5:
        ax3.annotate(str(name).replace("cs.", ""), (phi_rank[i], g_rank[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)
n_t = len(phi_rank)
ax3.plot([0, n_t - 1], [0, n_t - 1], "k--", lw=0.8, alpha=0.5,
         label="Perfect (Φ ↑ ⇔ g ↑)")
sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
ax3.set_title(f"[D] Φ-rank vs growth-rank (centroids)\n"
              f"  Spearman r = {r:+.3f}{sig} (p={p:.4f})",
              fontsize=11, fontweight="bold")
ax3.set_xlabel("Φ rank (1 = predicted growing)")
ax3.set_ylabel("g rank (1 = actual top growing)")
ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

# [E] EBM proof: Φ_θ vs empirical -log p̂  (X3 の理論主張の視覚的証明)
ax4 = plt.subplot(gs[1, 1])
if valid_kde:
    # 同時点 t=EVAL_T の観測点上でだけ比較
    phi_at_t  = phi_all[mask_t]
    nlogp_at_t = neg_log_p_all[mask_t]
    # 標準化して傾き比較しやすく
    phi_s   = (phi_at_t  - phi_at_t.mean())  / (phi_at_t.std()  + 1e-12)
    nlogp_s = (nlogp_at_t - nlogp_at_t.mean()) / (nlogp_at_t.std() + 1e-12)
    r_ebm, p_ebm = stats.pearsonr(phi_s, nlogp_s)
    sp_ebm, _    = stats.spearmanr(phi_at_t, nlogp_at_t)
    ax4.scatter(nlogp_s, phi_s, s=4, alpha=0.4, color="steelblue", rasterized=True)
    lo = min(nlogp_s.min(), phi_s.min()); hi = max(nlogp_s.max(), phi_s.max())
    ax4.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.6,
             label=r"$\Phi_\theta = -\log\hat p$ (perfect EBM)")
    sig_e = "***" if p_ebm < 0.001 else "**" if p_ebm < 0.01 else "*" if p_ebm < 0.05 else ""
    ax4.set_title(f"[E] EBM proof: $\\Phi_\\theta$ vs empirical $-\\log\\hat p(x,t)$\n"
                  f"  Pearson r={r_ebm:+.3f}{sig_e}, Spearman ρ={sp_ebm:+.3f}",
                  fontsize=11, fontweight="bold")
    ax4.set_xlabel("Empirical $-\\log\\hat p(x, t)$ (z-scored)")
    ax4.set_ylabel("$\\Phi_\\theta(x, t)$ (z-scored)")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)
else:
    ax4.text(0.5, 0.5, f"Not enough samples at t={EVAL_T}\nfor KDE estimation",
             ha="center", va="center", transform=ax4.transAxes, fontsize=11)
    ax4.set_title("[E] EBM proof — N/A", fontsize=11, fontweight="bold")
    ax4.set_xticks([]); ax4.set_yticks([])

# [F] Anchor proof: Φ_θ(c_j, t) vs -g̃_j(t)
ax5 = plt.subplot(gs[1, 2])
target = -g_n   # anchor: Φ(c_j, t) ≈ -g̃_j(t)
r_a, p_a = stats.pearsonr(phi_cent, target) if len(phi_cent) >= 3 else (0.0, 1.0)
sp_a, _  = stats.spearmanr(phi_cent, target) if len(phi_cent) >= 3 else (0.0, 1.0)
ax5.scatter(target, phi_cent, c=g_t, cmap="RdYlGn",
            s=120, edgecolors="black", linewidths=0.7,
            vmin=-g_abs_max, vmax=g_abs_max, zorder=3)
for i, name in enumerate(topic_names_active):
    if abs(g_t[i]) > 0.3 * g_abs_max:
        ax5.annotate(str(name).replace("cs.", ""), (target[i], phi_cent[i]),
                     textcoords="offset points", xytext=(5, 4), fontsize=7)
lo = min(target.min(), phi_cent.min()); hi = max(target.max(), phi_cent.max())
ax5.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.6,
         label=r"$\Phi_\theta(c_j,t) = -\tilde g_j(t)$ (perfect anchor)")
sig_a = "***" if p_a < 0.001 else "**" if p_a < 0.01 else "*" if p_a < 0.05 else ""
ax5.set_title(f"[F] Growth-anchor proof (L_val target)\n"
              f"  Pearson r={r_a:+.3f}{sig_a}, Spearman ρ={sp_a:+.3f}",
              fontsize=11, fontweight="bold")
ax5.set_xlabel(r"Target: $-\tilde g_j(t)$")
ax5.set_ylabel(r"$\Phi_\theta(c_j, t)$")
ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

_method = "X3-clean (predictor input g_n=0)" if VARIANT == "clean" else "X3 (Minimal EBM)"
fig.suptitle(
    f"PI-SDE {_method}  |  {DOMAIN}, seed={SEED}, λ_g={LAM_G}, t={EVAL_T}\n"
    f"Theory ⟶ Prediction ⟶ Visualization unified by one Φ_θ = $-\\log p_{{\\mathrm{{data}}}}(x,t)$",
    fontsize=12, fontweight="bold", y=1.00,
)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")
