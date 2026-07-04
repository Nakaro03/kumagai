"""
PI-SDE X3-clean — Topic trajectory case study figure (vis-venue ready).

注目トピックを 3 つ選び、Φ 地形上を時刻 t に沿って移動する **軌跡**を可視化する。

レイアウト:
  [A] 全 t を重ねた UMAP 散布 (年色分け)  +  注目 3 トピックの centroid 軌跡
       (矢印 + マーカー、t を進むにつれ色 → アノテーション付き)
  [B] 注目トピックごとの Φ(c_j, t) の時系列折れ線
       → 「Φ が下がる = 谷に降りる = 高成長領域へ」と読める
  [C] 注目トピックごとの実成長率 g_j(t) の時系列折れ線
       → [B] と並べて Φ と g の対応を視覚的に確認

選定: 全 active トピックから |g| が最大 / 最小 / 中央のものを 1 つずつ
       (env で `PNODE_TOPIC_IDS=3,7,15` で明示指定も可)

Usage:
  PNODE_X3_VARIANT=clean PNODE_DOMAIN_TARGET=paper PNODE_SEED=42 PNODE_LAM_G=0.5 \\
    python pnode_patent_runner/plot_pisde_x3_case_study.py
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

warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

from src.model import ForwardSDE


DOMAIN  = os.environ.get("PNODE_DOMAIN_TARGET", "paper")
SEED    = int(os.environ.get("PNODE_SEED", 42))
LAM_G   = float(os.environ.get("PNODE_LAM_G", 0.5))
D_CTX   = int(os.environ.get("PNODE_D_CTX", 32))
VARIANT = os.environ.get("PNODE_X3_VARIANT", "clean")
TOPIC_IDS_ENV = os.environ.get("PNODE_TOPIC_IDS", "")

DOMAIN_MAP = {
    "paper":              ("PNode_Paper_X1",                  "data/PNode_Paper_X1",                  2022),
    "patent_energy_top50":("PNode_Patent_Energy_X1_top50",    "data/PNode_Patent_Energy_X1_top50",    2010),
    "arxiv_construction": ("PNode_ArXiv_Construction_X1_v2",  "data/PNode_ArXiv_Construction_X1_v2",  2014),
    "jp_construction":    ("PNode_JP_Construction_X1",        "data/PNode_JP_Construction_X1",        2014),
}
DATA_NAME, DATA_DIR, YEAR_BASE = DOMAIN_MAP[DOMAIN]

if VARIANT == "baseline":
    X3_DIR = Path(f"RESULTS_X3/{DATA_NAME}/softplus-400_400-0.0-const-0.1-0.1-0.005-x3_g{LAM_G}/seed_{SEED}/alltime")
else:
    X3_DIR = Path(f"RESULTS_X3_ABLATION/{DATA_NAME}/mask/x3abl_mask_g{LAM_G}/seed_{SEED}/alltime")
DATA_PT = f"{DATA_DIR}/alltime/fate_train.pt"
OUT_PNG = X3_DIR / f"case_study_{('x3clean' if VARIANT=='clean' else 'x3')}.png"


class MinimalPredictor(nn.Module):
    def __init__(self, x_dim, d_ctx=32):
        super().__init__()
        self.embed = nn.Linear(x_dim + 2, d_ctx)
        self.attn = nn.MultiheadAttention(d_ctx, num_heads=2, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(d_ctx)
        self.out = nn.Linear(d_ctx + x_dim + 1, 1)


print(f"[X3 case study] domain={DOMAIN}  seed={SEED}  variant={VARIANT}")
print(f"  RESULTS dir: {X3_DIR}")
if not X3_DIR.exists():
    raise SystemExit(f"✗ checkpoint dir not found: {X3_DIR}")

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
print(f"  Loading: {ckpt_path}")
ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
predictor.load_state_dict(ckpt["predictor_state_dict"])
model.eval(); predictor.eval()


# ── トピック選定 ────────────────────────────────────────────────────
# 最終時点で active かつ全期間で active を維持しているトピックを候補に
g_final_raw = growth_raw[T].numpy()
all_active = np.ones(n_topics, dtype=bool)
for t in range(1, T + 1):
    all_active &= (centroids[t].abs().sum(dim=-1).numpy() > 1e-6)
candidate_ids = np.where(all_active)[0]
print(f"  candidates (active in all t): {len(candidate_ids)}")

if TOPIC_IDS_ENV:
    sel_ids = [int(x.strip()) for x in TOPIC_IDS_ENV.split(",") if x.strip()]
    print(f"  using user-specified topics: {sel_ids}")
else:
    g_cand = g_final_raw[candidate_ids]
    order = np.argsort(g_cand)   # ascending
    if len(candidate_ids) >= 3:
        sel_ids = [
            int(candidate_ids[order[-1]]),     # max growth (boom)
            int(candidate_ids[order[len(order) // 2]]),  # median
            int(candidate_ids[order[0]]),      # min growth (decline)
        ]
    else:
        sel_ids = list(candidate_ids[:3])
    print(f"  auto-selected: max-g / median / min-g")

sel_names = [str(topic_names[i]) for i in sel_ids]
print(f"  topics: {list(zip(sel_ids, sel_names))}")
print(f"  final-t growth: {[f'{g_final_raw[i]:+.2f}' for i in sel_ids]}")


# ── UMAP fit (全 obs) ──────────────────────────────────────────────
print("UMAP fitting...")
import umap
x_all = torch.cat(xp).numpy()
um = umap.UMAP(n_components=2, n_neighbors=30, metric="euclidean",
               random_state=42, transform_seed=42)
x_all_2d = um.fit_transform(x_all)
years_all = np.concatenate([np.full(v.shape[0], y[k]) for k, v in enumerate(xp)])


# ── 各 t での選定トピック centroid の 2D 投影と Φ 値 ────────────────
traj_2d = {tid: [] for tid in sel_ids}   # list of (x, y) per t
traj_phi = {tid: [] for tid in sel_ids}
traj_g   = {tid: [] for tid in sel_ids}

print("Computing per-topic trajectories...")
for t in range(0, T + 1):
    c_t = centroids[t].numpy()
    for tid in sel_ids:
        c_j = c_t[tid:tid+1]
        if c_j.sum() != 0:
            c_2d = um.transform(c_j)[0]
            xt = torch.cat([torch.tensor(c_j, dtype=torch.float32),
                            torch.full((1, 1), float(y[t]))], dim=1).to(device).requires_grad_()
            phi_v = model._func._pot(xt).squeeze(-1).item()
        else:
            c_2d = (np.nan, np.nan); phi_v = np.nan
        traj_2d[tid].append(c_2d)
        traj_phi[tid].append(phi_v)
        traj_g[tid].append(float(growth_raw[t][tid]))


# ── プロット ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 8))
gs = GridSpec(2, 3, width_ratios=[1.6, 1, 1], height_ratios=[1, 1],
              wspace=0.28, hspace=0.30)

# [A] 全観測 + 軌跡 (spans full left column)
ax_map = plt.subplot(gs[:, 0])
years_int = years_all.astype(int)
T_max = int(max(years_int))
for yt in range(T_max + 1):
    mask = years_int == yt
    ax_map.scatter(x_all_2d[mask, 0], x_all_2d[mask, 1], s=2, alpha=0.18,
                   color=plt.cm.gray(0.4 + 0.05 * yt))

palette = ["#E74C3C", "#3498DB", "#2ECC71"]   # red / blue / green
for k, tid in enumerate(sel_ids):
    pts = np.array(traj_2d[tid])
    valid = ~np.isnan(pts[:, 0])
    pts_v = pts[valid]
    ts_v = np.arange(T + 1)[valid]
    if len(pts_v) >= 2:
        for i in range(len(pts_v) - 1):
            ax_map.annotate("", xy=pts_v[i+1], xytext=pts_v[i],
                            arrowprops=dict(arrowstyle="->", color=palette[k],
                                            lw=2.2, alpha=0.9))
    # マーカー (t を色で示す)
    for i, (p, tt) in enumerate(zip(pts_v, ts_v)):
        marker_size = 80 + 40 * i
        ax_map.scatter(p[0], p[1], s=marker_size, color=palette[k],
                       edgecolors="black", linewidths=0.8, zorder=5,
                       alpha=0.6 + 0.4 * (i / max(len(pts_v) - 1, 1)))
        ax_map.annotate(f"t={tt}", p, textcoords="offset points",
                        xytext=(10, -3 - 3 * k), fontsize=7, color=palette[k])
    # 最終位置にラベル
    if len(pts_v) > 0:
        last_p = pts_v[-1]
        nm = sel_names[k].replace("cs.", "")
        g_final = traj_g[tid][-1]
        ax_map.annotate(f"{nm}\n  g={g_final:+.2f}", last_p,
                        textcoords="offset points", xytext=(15, 8),
                        fontsize=10, color=palette[k], fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="white", ec=palette[k], alpha=0.85))

ax_map.set_title(f"[A] Topic centroid trajectories through Φ landscape\n"
                 f"  Background: all observations (gray, faded)  |  "
                 f"Arrows: t=0 → t={T} per topic",
                 fontsize=11, fontweight="bold")
ax_map.set_xlabel("UMAP1"); ax_map.set_ylabel("UMAP2")
ax_map.set_xticks([]); ax_map.set_yticks([])

# [B] Φ(c_j, t) 折れ線
ax_phi = plt.subplot(gs[0, 1])
ts_x = np.arange(T + 1)
for k, tid in enumerate(sel_ids):
    phi_arr = np.array(traj_phi[tid])
    valid = ~np.isnan(phi_arr)
    ax_phi.plot(ts_x[valid], phi_arr[valid], "o-",
                color=palette[k], lw=2.2, markersize=8,
                label=sel_names[k].replace("cs.", ""))
ax_phi.set_title(r"[B] $\Phi_\theta(c_j, t)$ over time" + "\n  (low Φ = valley = high-density region)",
                 fontsize=10, fontweight="bold")
ax_phi.set_xlabel("t"); ax_phi.set_ylabel(r"$\Phi_\theta$")
ax_phi.grid(alpha=0.3); ax_phi.legend(fontsize=8, loc="best")
ax_phi.set_xticks(ts_x)

# [C] g_j(t) 折れ線
ax_g = plt.subplot(gs[1, 1])
for k, tid in enumerate(sel_ids):
    g_arr = np.array(traj_g[tid])
    ax_g.plot(ts_x, g_arr, "s-",
              color=palette[k], lw=2.2, markersize=8,
              label=sel_names[k].replace("cs.", ""))
ax_g.axhline(0, color="black", lw=0.5, alpha=0.5)
ax_g.set_title(r"[C] Actual growth rate $g_j(t)$ over time" + "\n  (positive = topic expanding, negative = shrinking)",
               fontsize=10, fontweight="bold")
ax_g.set_xlabel("t"); ax_g.set_ylabel(r"$g_j$")
ax_g.grid(alpha=0.3); ax_g.legend(fontsize=8, loc="best")
ax_g.set_xticks(ts_x)

# [D] 比較表 (Φ vs g の相関 per 選定トピック)
ax_corr = plt.subplot(gs[:, 2])
ax_corr.axis("off")
lines = ["[D] Per-topic Φ–g correspondence", "", f"{'Topic':<18} {'Final g':<10} {'Final Φ':<10} {'Φ change':<10}"]
lines.append("-" * 50)
for k, tid in enumerate(sel_ids):
    phi_arr = np.array(traj_phi[tid])
    g_arr = np.array(traj_g[tid])
    valid = ~np.isnan(phi_arr)
    if valid.sum() >= 2:
        d_phi = phi_arr[valid][-1] - phi_arr[valid][0]
    else:
        d_phi = np.nan
    nm = sel_names[k].replace("cs.", "")
    lines.append(f"{nm:<18} {g_arr[-1]:+.2f}     {phi_arr[-1]:+.2f}     {d_phi:+.2f}")

lines += ["", "Reading guide:",
          "  • If a topic's Φ DECREASES over time",
          "    while g INCREASES,",
          "    the landscape correctly captures",
          "    the growth (deepening valley).",
          "",
          "  • If Φ stays flat or rises while",
          "    g rises, the landscape misses",
          "    this trend.",
          "",
          "Note: this is a DESCRIPTIVE",
          "  visualization of historical data,",
          "  not a future prediction tool."]

for i, ln in enumerate(lines):
    weight = "bold" if i == 0 or "Reading" in ln or "Note:" in ln else "normal"
    fmly   = "monospace" if "<" in ln or "----" in ln or "+" in ln or "Topic" in ln else "sans-serif"
    ax_corr.text(0.0, 0.97 - 0.045 * i, ln, transform=ax_corr.transAxes,
                 fontsize=9, fontweight=weight, family=fmly, va="top")

_method = "X3-clean" if VARIANT == "clean" else "X3"
fig.suptitle(
    f"PI-SDE {_method} — Topic trajectory case study  |  {DOMAIN}, seed={SEED}\n"
    f"Descriptive analysis: how topics move through the learned Φ landscape over t",
    fontsize=12, fontweight="bold", y=1.00,
)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")
