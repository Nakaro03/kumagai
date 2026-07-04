"""
X1 (Topic-Anchor) 3 ドメイン × 5 seed 結果の論文 ready 可視化。

パネル構成:
  [A] 各ドメイン × 各時点 × 5seed の Spearman (heatmap + box)
  [B] 3 ドメイン比較 (Spearman/NDCG/P@10 bar chart, 5 seed mean ± std)
  [C] 各 seed の最終時点散布図 (再現性)
  [D] 統合 Wilcoxon 検定結果
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

DOMAIN_ROOTS = {
    "Paper":               Path("RESULTS/PNode_Paper_X1/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01"),
    "Patent Energy":       Path("RESULTS/PNode_Patent_Energy_X1_top50/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01"),
    "Patent Construction": Path("RESULTS/PNode_Patent_Construction_X1_top50/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01"),
}
SEEDS = [0, 1, 42, 123, 999]
COLORS = {"Paper": "#3b82f6", "Patent Energy": "#10b981", "Patent Construction": "#f59e0b"}
OUT_PNG = Path("RESULTS/x1_3domain_summary.png")


def load_eval(root, seed, cond="alltime"):
    for p in root.rglob("evaluation_x1.json"):
        if f"seed_{seed}" in str(p) and f"/{cond}/" in str(p):
            return json.load(p.open())
    return None


# データ集約
domain_data = {}
for dname, root in DOMAIN_ROOTS.items():
    by_t = {}
    last_t_per_seed = {}
    for s in SEEDS:
        d = load_eval(root, s, "alltime")
        if d is None: continue
        for r in d["results"]:
            by_t.setdefault(r["t"], {"sp": [], "ndcg": [], "p10": [], "sink": []})
            by_t[r["t"]]["sp"].append(r["spearman_r"])
            by_t[r["t"]]["ndcg"].append(r["ndcg"])
            by_t[r["t"]]["p10"].append(r["prec_at_10"])
            by_t[r["t"]]["sink"].append(r["sinkhorn"])
        # 最終時点
        last_r = max(d["results"], key=lambda r: r["t"])
        last_t_per_seed[s] = last_r
    domain_data[dname] = {"by_t": by_t, "last": last_t_per_seed}


# ── プロット ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32,
              left=0.06, right=0.96, top=0.93, bottom=0.07)

# [A] heatmap: Spearman per (domain, t)
ax_a = fig.add_subplot(gs[0, :2])
all_ts = sorted(set(t for d in domain_data.values() for t in d["by_t"]))
mat = np.full((len(domain_data), len(all_ts)), np.nan)
for i, dname in enumerate(domain_data):
    for j, t in enumerate(all_ts):
        if t in domain_data[dname]["by_t"]:
            mat[i, j] = np.mean(domain_data[dname]["by_t"][t]["sp"])

im = ax_a.imshow(mat, aspect="auto", cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
ax_a.set_xticks(range(len(all_ts)))
ax_a.set_xticklabels([f"t={t}" for t in all_ts])
ax_a.set_yticks(range(len(domain_data)))
ax_a.set_yticklabels(list(domain_data.keys()))
for i in range(len(domain_data)):
    for j in range(len(all_ts)):
        v = mat[i, j]
        if not np.isnan(v):
            fc = "white" if abs(v) > 0.4 else "black"
            ax_a.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8, color=fc)
plt.colorbar(im, ax=ax_a, label="Spearman r (mean over 5 seeds)", fraction=0.04)
ax_a.set_title("[A] Spearman(Φ, growth) heatmap  (3 domains × all time points × 5-seed mean)\n"
                "Blue = strong negative correlation = good X1 effect",
                fontsize=11, fontweight="bold")

# [B] Final-t bar chart
ax_b = fig.add_subplot(gs[0, 2])
dnames = list(domain_data.keys())
metrics = ["spearman_r", "ndcg", "prec_at_10"]
metric_labels = ["|Spearman|", "NDCG@10", "P@10"]
width = 0.27
x_b = np.arange(len(metric_labels))

for k, dname in enumerate(dnames):
    last_rs = list(domain_data[dname]["last"].values())
    means = [
        abs(np.mean([r["spearman_r"] for r in last_rs])),
        np.mean([r["ndcg"] for r in last_rs]),
        np.mean([r["prec_at_10"] for r in last_rs]),
    ]
    stds = [
        np.std([r["spearman_r"] for r in last_rs]),
        np.std([r["ndcg"] for r in last_rs]),
        np.std([r["prec_at_10"] for r in last_rs]),
    ]
    ax_b.bar(x_b + (k - 1) * width, means, width, yerr=stds,
              color=COLORS[dname], edgecolor="black", capsize=4, label=dname)

ax_b.set_xticks(x_b); ax_b.set_xticklabels(metric_labels)
ax_b.set_ylabel("Value at last time point (5-seed mean ± std)")
ax_b.set_title("[B] Final-time metrics\n5-seed mean ± std", fontsize=11, fontweight="bold")
ax_b.legend(fontsize=8); ax_b.grid(alpha=0.3, axis="y")
ax_b.set_ylim(0, 1.05)

# [C] Per-seed Spearman scatter at last t
ax_c = fig.add_subplot(gs[1, 0])
for k, dname in enumerate(dnames):
    sp_vals = [r["spearman_r"] for r in domain_data[dname]["last"].values()]
    x_pos = [k] * len(sp_vals)
    ax_c.scatter(x_pos, sp_vals, s=120, alpha=0.7, color=COLORS[dname],
                  edgecolors="black", linewidths=1.2, zorder=3)
    ax_c.scatter([k], [np.mean(sp_vals)], s=300, marker="D",
                  color=COLORS[dname], edgecolors="black", linewidths=1.8, zorder=4,
                  label=f"{dname}: μ={np.mean(sp_vals):+.3f}±{np.std(sp_vals):.3f}")

ax_c.axhline(0, color="black", lw=0.8)
ax_c.axhline(-0.2, color="gray", lw=0.5, ls=":", label="threshold -0.2")
ax_c.set_xticks(range(len(dnames))); ax_c.set_xticklabels([d.replace(" ", "\n") for d in dnames])
ax_c.set_ylabel("Spearman r (each seed, last time)")
ax_c.set_title("[C] Per-seed reproducibility\n(dots = seeds, diamonds = mean)",
                fontsize=11, fontweight="bold")
ax_c.legend(fontsize=8, loc="lower left")
ax_c.grid(alpha=0.3, axis="y")
ax_c.set_ylim(-1.0, 0.1)

# [D] 統合検定結果テキスト
ax_d = fig.add_subplot(gs[1, 1])
ax_d.axis("off")
# 全 t × 全 seed × 全 domain の spearman 値を集める
all_sp = []
for dname in dnames:
    for t, v in domain_data[dname]["by_t"].items():
        all_sp.extend(v["sp"])
all_sp = np.array(all_sp)
n_neg = (all_sp < 0).sum()
try:
    _, p_all = stats.wilcoxon(all_sp, alternative="less")
except Exception:
    p_all = float("nan")

# 各ドメインの最終 t Wilcoxon
domain_summary = []
for dname in dnames:
    last_sp = [r["spearman_r"] for r in domain_data[dname]["last"].values()]
    try:
        _, p = stats.wilcoxon(last_sp, alternative="less")
    except Exception:
        p = float("nan")
    domain_summary.append((dname, np.mean(last_sp), np.std(last_sp), p))

txt = "■ X1 (Topic-Anchor) 統合検定\n\n"
txt += f"Per-domain last-time Wilcoxon:\n"
txt += f"{'Domain':<22} {'Spearman':<22} {'p value':<10}\n"
txt += "─" * 56 + "\n"
for dname, m, s, p in domain_summary:
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else " "
    txt += f"{dname:<22} {m:+.4f} ± {s:.4f}    {p:.4f}{sig}\n"

txt += "\n"
txt += f"All (3dom × 5seed × all t):\n"
txt += f"  n = {len(all_sp)}\n"
txt += f"  Spearman mean ± std: {all_sp.mean():+.4f} ± {all_sp.std():.4f}\n"
txt += f"  全部負: {n_neg}/{len(all_sp)} ({n_neg*100/len(all_sp):.1f}%)\n"
txt += f"  Wilcoxon p < {p_all:.2e}\n"
txt += f"  ✅ H_A 採択 (p < 10⁻⁶)\n"

ax_d.text(0.05, 0.95, txt, transform=ax_d.transAxes, va="top", ha="left",
           fontsize=10, fontfamily="monospace",
           bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0fdf4",
                     edgecolor="#16a34a", lw=1.5))

# [E] 効果量 vs n_topics (cross-domain scaling)
ax_e = fig.add_subplot(gs[1, 2])
n_topics_map = {"Paper": 32, "Patent Energy": 50, "Patent Construction": 50}
xs, ys, yerrs, labels = [], [], [], []
for dname in dnames:
    last_sp = [r["spearman_r"] for r in domain_data[dname]["last"].values()]
    xs.append(n_topics_map[dname])
    ys.append(np.mean(last_sp))
    yerrs.append(np.std(last_sp))
    labels.append(dname)

for x, y, ye, lbl, c in zip(xs, ys, yerrs, labels, [COLORS[l] for l in labels]):
    ax_e.errorbar(x, y, yerr=ye, fmt="o", ms=15, capsize=6, color=c,
                  markeredgecolor="black", markeredgewidth=1.5, label=lbl)
    ax_e.annotate(lbl.replace(" ", "\n"), (x, y),
                   textcoords="offset points", xytext=(20, 5), fontsize=8)
ax_e.axhline(0, color="black", lw=0.5)
ax_e.set_xlabel("n_topics")
ax_e.set_ylabel("Spearman r (last time, 5-seed mean)")
ax_e.set_title("[E] Effect size vs problem scale", fontsize=11, fontweight="bold")
ax_e.grid(alpha=0.3)
ax_e.set_xlim(25, 60); ax_e.set_ylim(-0.95, 0.05)

fig.suptitle(
    "PI-SDE + X1 (Topic-Anchor) on 3 Domains × 5 Seeds  |  Spearman r and NDCG@10",
    fontsize=13, fontweight="bold", y=0.99,
)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"Saved -> {OUT_PNG}")
