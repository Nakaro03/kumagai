"""
PI-SDE X3 — Growth-anchor visual proof across all time points.

X3 の核となる anchor 制約:
    Φ_θ(c_j, t)  ≈  -g̃_j(t)               (Gibbs:  p(c_j,t) ∝ exp(g̃_j(t)))

これが各 t で実際に成立しているかを 1 図で見せる:

  Top row:    各 t ごとの scatter (Φ(c_j,t) vs -g̃_j(t)) + y=x reference
  Bottom L:   全 (j,t) を集約した scatter (色 = t)
  Bottom R:   per-t metrics の bar plot
              - Pearson r(Φ, -g̃)
              - residual RMSE  ||Φ + g̃||₂

「Φ_θ = -log p_data」という X3 の理論主張が成長率データに対して
学習で達成されたことの査読向け視覚的証拠。

Usage:
  PNODE_DOMAIN_TARGET=paper PNODE_LAM_G=0.5 PNODE_SEED=42 \\
    python pnode_patent_runner/plot_pisde_x3_anchor.py
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

warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

from src.model import ForwardSDE


DOMAIN  = os.environ.get("PNODE_DOMAIN_TARGET", "paper")
SEED    = int(os.environ.get("PNODE_SEED", 42))
LAM_G   = float(os.environ.get("PNODE_LAM_G", 0.5))
D_CTX   = int(os.environ.get("PNODE_D_CTX", 32))
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
OUT_PNG = X3_DIR / f"anchor_{_suffix}_all_t.png"


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


print(f"[X3 anchor proof] domain={DOMAIN}  seed={SEED}  λ_g={LAM_G}")
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
topic_names  = data["topic_names"]
centroids    = data["centroids"]
growth_raw   = data["growth"]
growth_norm  = data["growth_norm"]
n_topics     = data["n_topics"]
T = len(y) - 1

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


# ── 各 t で Φ(c_j,t) と target -g̃_j(t) を計算 ─────────────────────
per_t = []
all_targets, all_phis, all_t_idx, all_g = [], [], [], []
for t_idx in range(1, T + 1):
    cent_t = centroids[t_idx].numpy()
    am = cent_t.sum(axis=-1) != 0
    if am.sum() < 2:
        per_t.append(None)
        continue
    cent_active = cent_t[am]
    g_t = growth_raw[t_idx].numpy()[am]
    g_n = growth_norm[t_idx].numpy()[am]
    names_t = [str(topic_names[i]) for i in range(n_topics) if am[i]]

    xt_c = torch.cat([torch.tensor(cent_active, dtype=torch.float32),
                      torch.full((len(cent_active), 1), float(y[t_idx]))], dim=1)
    phi_c = model._func._pot(xt_c.to(device).requires_grad_()).squeeze(-1).detach().cpu().numpy()
    target = -g_n

    r_p, p_p = stats.pearsonr(phi_c, target) if len(phi_c) >= 3 else (0.0, 1.0)
    r_s, _   = stats.spearmanr(phi_c, target) if len(phi_c) >= 3 else (0.0, 1.0)
    rmse = float(np.sqrt(((phi_c - target) ** 2).mean()))

    per_t.append(dict(
        t_idx=t_idx, year=YEAR_BASE + t_idx,
        target=target, phi=phi_c, g=g_t, names=names_t,
        pearson_r=r_p, pearson_p=p_p, spearman_r=r_s, rmse=rmse,
    ))
    all_targets.extend(target.tolist())
    all_phis.extend(phi_c.tolist())
    all_t_idx.extend([t_idx] * len(phi_c))
    all_g.extend(g_t.tolist())

valid = [d for d in per_t if d is not None]
T_valid = len(valid)
if T_valid == 0:
    raise SystemExit("✗ No valid time points with active centroids")

# ── プロット ────────────────────────────────────────────────────────
n_top = T_valid
fig = plt.figure(figsize=(max(16, 4 * n_top), 10))
gs = GridSpec(2, max(n_top, 3), height_ratios=[1, 1.0], hspace=0.45, wspace=0.30)

# Top row: per-t scatter
for i, d in enumerate(valid):
    ax = plt.subplot(gs[0, i])
    g_abs = max(0.5, float(np.abs(d["g"]).max()) if len(d["g"]) > 0 else 0.5)
    sc = ax.scatter(d["target"], d["phi"], c=d["g"], cmap="RdYlGn",
                    s=70, edgecolors="black", linewidths=0.5,
                    vmin=-g_abs, vmax=g_abs)
    lo = min(d["target"].min(), d["phi"].min())
    hi = max(d["target"].max(), d["phi"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
    for j, name in enumerate(d["names"]):
        if abs(d["g"][j]) > 0.3 * g_abs:
            ax.annotate(name.replace("cs.", ""), (d["target"][j], d["phi"][j]),
                        textcoords="offset points", xytext=(4, 3), fontsize=6.5)
    sig = "***" if d["pearson_p"] < 0.001 else "**" if d["pearson_p"] < 0.01 else \
          "*" if d["pearson_p"] < 0.05 else ""
    ax.set_title(f"t={d['t_idx']} (~{d['year']})\n"
                 f"r={d['pearson_r']:+.3f}{sig}  RMSE={d['rmse']:.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel(r"$-\tilde g_j(t)$", fontsize=9)
    if i == 0:
        ax.set_ylabel(r"$\Phi_\theta(c_j, t)$", fontsize=10)
    ax.grid(alpha=0.3)

# Bottom-left (spans first half): aggregated scatter colored by t
n_bottom = max(n_top, 3)
left_span = max(1, n_bottom // 2)
ax_agg = plt.subplot(gs[1, :left_span])
all_targets = np.array(all_targets); all_phis = np.array(all_phis)
all_t_idx = np.array(all_t_idx)
sc_agg = ax_agg.scatter(all_targets, all_phis, c=all_t_idx, cmap="viridis",
                        s=50, edgecolors="black", linewidths=0.4, alpha=0.85)
lo = min(all_targets.min(), all_phis.min()); hi = max(all_targets.max(), all_phis.max())
ax_agg.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.6,
            label=r"$\Phi = -\tilde g$ (anchor target)")
r_all, p_all = stats.pearsonr(all_phis, all_targets)
sp_all, _    = stats.spearmanr(all_phis, all_targets)
rmse_all     = float(np.sqrt(((all_phis - all_targets) ** 2).mean()))
sig_all = "***" if p_all < 0.001 else "**" if p_all < 0.01 else "*" if p_all < 0.05 else ""
ax_agg.set_title(f"Aggregated across all (j, t)  —  "
                 f"Pearson r={r_all:+.3f}{sig_all}, Spearman ρ={sp_all:+.3f}, "
                 f"RMSE={rmse_all:.3f}",
                 fontsize=11, fontweight="bold")
ax_agg.set_xlabel(r"$-\tilde g_j(t)$")
ax_agg.set_ylabel(r"$\Phi_\theta(c_j, t)$")
ax_agg.legend(fontsize=9); ax_agg.grid(alpha=0.3)
plt.colorbar(sc_agg, ax=ax_agg, label="t", fraction=0.04)

# Bottom-right: per-t metrics
ax_m = plt.subplot(gs[1, left_span:])
ts = [d["t_idx"] for d in valid]
rs = [d["pearson_r"] for d in valid]
rmses = [d["rmse"] for d in valid]
xb = np.arange(len(ts))
w = 0.35
b1 = ax_m.bar(xb - w/2, rs, width=w, color="steelblue", label="Pearson r(Φ, -g̃)")
ax_m.set_ylim(-1.05, 1.05)
ax_m.axhline(0, color="black", lw=0.5)
ax_m.set_ylabel("Pearson r", color="steelblue")
ax_m.set_xticks(xb)
ax_m.set_xticklabels([f"t={t}" for t in ts])
ax_m.grid(alpha=0.3, axis="y")
ax_m2 = ax_m.twinx()
b2 = ax_m2.bar(xb + w/2, rmses, width=w, color="coral", alpha=0.85, label="RMSE ||Φ + g̃||")
ax_m2.set_ylabel("RMSE", color="coral")
ax_m.set_title("Per-t anchor quality  (higher r = better fit, lower RMSE = tighter)",
               fontsize=11, fontweight="bold")
# 凡例まとめ
lines = [b1, b2]
labels = [b1.get_label(), b2.get_label()]
ax_m.legend(lines, labels, fontsize=9, loc="lower left")

_method = "X3-clean" if VARIANT == "clean" else "X3"
fig.suptitle(
    f"PI-SDE {_method} — Growth-anchor visual proof  |  {DOMAIN}, seed={SEED}, λ_g={LAM_G}\n"
    f"Theory claim:  $\\Phi_\\theta(c_j, t) = -\\tilde g_j(t)$   "
    f"(Gibbs density $p(c_j,t) \\propto \\exp(\\tilde g_j(t))$)",
    fontsize=13, fontweight="bold", y=1.00,
)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")
print(f"  Aggregated:  r={r_all:+.3f}, ρ={sp_all:+.3f}, RMSE={rmse_all:.3f}")
for d in valid:
    print(f"  t={d['t_idx']:>2}:  r={d['pearson_r']:+.3f}  ρ={d['spearman_r']:+.3f}  RMSE={d['rmse']:.3f}")
