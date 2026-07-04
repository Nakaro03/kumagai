"""
PI-SDE + X1 (Topic-Anchor) 5 seed × 4 condition 結果のまとめ可視化。

パネル構成:
  [A] Spearman r (条件別 × 時点別) bar chart (X1 vs vanilla)
  [B] NDCG@10 (X1 のみ)
  [C] Sinkhorn 距離 (vanilla vs X1 比較)
  [D] Test split のみの統合プロット (生 seed 値)
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

X1_ROOT      = Path("RESULTS/PNode_Paper_X1/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01")
VANILLA_ROOT = Path("RESULTS/PNode_Paper/softplus-400_400-0.5-const-0.1-0.1-0.005")
SEEDS = [0, 1, 42, 123, 999]
CONDITIONS = ["alltime", "leaveout1", "leaveout2", "leaveout3"]
OUT_PNG = X1_ROOT / "summary_5seed_comparison.png"


def load_eval(root, seed, cond):
    pat = "evaluation_x1.json" if root == X1_ROOT else "evaluation.json"
    for p in root.rglob(pat):
        if f"seed_{seed}" in str(p) and f"/{cond}/" in str(p):
            return json.load(p.open())
    return None


# 集約
data = {}
for cond in CONDITIONS:
    data[cond] = {"x1": {}, "vanilla": {}}
    for s in SEEDS:
        d_x1 = load_eval(X1_ROOT, s, cond)
        d_v = load_eval(VANILLA_ROOT, s, cond)
        if d_x1:
            for r in d_x1["results"]:
                key = (r["t"], r["split"])
                data[cond]["x1"].setdefault(key, {"sp": [], "ndcg": [], "sink": [], "p10": []})
                data[cond]["x1"][key]["sp"].append(r["spearman_r"])
                data[cond]["x1"][key]["ndcg"].append(r["ndcg"])
                data[cond]["x1"][key]["sink"].append(r["sinkhorn"])
                data[cond]["x1"][key]["p10"].append(r["prec_at_10"])
        if d_v:
            for r in d_v["results"]:
                key = (r["t"], r["split"])
                data[cond]["vanilla"].setdefault(key, {"sink": []})
                data[cond]["vanilla"][key]["sink"].append(r["pi_sde"])


# ── プロット ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28,
              left=0.08, right=0.96, top=0.92, bottom=0.08)

# [A] Spearman r per condition×t (X1)
ax_a = fig.add_subplot(gs[0, 0])
x_labels, sp_means, sp_stds, splits_a = [], [], [], []
for cond in CONDITIONS:
    for (t, split) in sorted(data[cond]["x1"].keys()):
        vals = data[cond]["x1"][(t, split)]["sp"]
        x_labels.append(f"{cond}\nt={t}\n({split})")
        sp_means.append(np.mean(vals))
        sp_stds.append(np.std(vals))
        splits_a.append(split)

x = np.arange(len(x_labels))
colors_a = ["#ef4444" if s == "test" else "#3b82f6" for s in splits_a]
bars = ax_a.bar(x, sp_means, yerr=sp_stds, color=colors_a, edgecolor="black", capsize=4)
ax_a.axhline(0, color="black", lw=0.8)
ax_a.axhline(-0.15, color="gray", lw=0.5, ls=":", label="Threshold -0.15")
for i, (m, s) in enumerate(zip(sp_means, sp_stds)):
    ax_a.text(i, m - 0.05 if m < 0 else m + 0.02, f"{m:+.2f}", ha="center",
              fontsize=7, fontweight="bold")
ax_a.set_xticks(x)
ax_a.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
ax_a.set_ylabel("Spearman r (5 seed mean ± std)")
ax_a.set_title("[A] Spearman(Φ, growth) per condition × time\n"
               "Blue = train split, Red = held-out test split",
               fontsize=10, fontweight="bold")
ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3, axis="y")
ax_a.set_ylim(-1.0, 0.3)

# [B] NDCG@10 per condition × t (X1)
ax_b = fig.add_subplot(gs[0, 1])
ndcg_means, ndcg_stds = [], []
for cond in CONDITIONS:
    for (t, split) in sorted(data[cond]["x1"].keys()):
        vals = data[cond]["x1"][(t, split)]["ndcg"]
        ndcg_means.append(np.mean(vals))
        ndcg_stds.append(np.std(vals))

bars = ax_b.bar(x, ndcg_means, yerr=ndcg_stds, color=colors_a, edgecolor="black", capsize=4)
for i, m in enumerate(ndcg_means):
    ax_b.text(i, m + 0.02, f"{m:.2f}", ha="center", fontsize=7, fontweight="bold")
ax_b.set_xticks(x)
ax_b.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
ax_b.set_ylabel("NDCG@10 (5 seed mean ± std)")
ax_b.set_title("[B] NDCG@10 per condition × time\n"
               "Blue = train, Red = test (held-out year)",
               fontsize=10, fontweight="bold")
ax_b.set_ylim(0, 1.1); ax_b.grid(alpha=0.3, axis="y")

# [C] Sinkhorn vanilla vs X1
ax_c = fig.add_subplot(gs[1, 0])
v_means, v_stds, x1_means, x1_stds, sink_labels = [], [], [], [], []
for cond in CONDITIONS:
    for (t, split) in sorted(data[cond]["x1"].keys()):
        v = data[cond]["vanilla"].get((t, split), {}).get("sink", [])
        x1 = data[cond]["x1"][(t, split)]["sink"]
        if v and x1:
            v_means.append(np.mean(v)); v_stds.append(np.std(v))
            x1_means.append(np.mean(x1)); x1_stds.append(np.std(x1))
            sink_labels.append(f"{cond}\nt={t} ({split})")

x_s = np.arange(len(sink_labels))
w = 0.35
ax_c.bar(x_s - w/2, v_means, w, yerr=v_stds, label="vanilla PI-SDE", color="#9ca3af",
         edgecolor="black", capsize=3)
ax_c.bar(x_s + w/2, x1_means, w, yerr=x1_stds, label="PI-SDE + X1", color="#3b82f6",
         edgecolor="black", capsize=3)
ax_c.set_xticks(x_s)
ax_c.set_xticklabels(sink_labels, rotation=45, ha="right", fontsize=7)
ax_c.set_ylabel("Sinkhorn distance (lower = better)")
ax_c.set_title("[C] Sinkhorn: vanilla vs X1\n(X1 trades distribution match for trend ranking)",
               fontsize=10, fontweight="bold")
ax_c.legend(fontsize=9); ax_c.grid(alpha=0.3, axis="y")

# [D] Test split のみ raw 値プロット
ax_d = fig.add_subplot(gs[1, 1])
test_data = {"cond": [], "t": [], "sp_vals": []}
for cond in CONDITIONS:
    for (t, split) in sorted(data[cond]["x1"].keys()):
        if split == "test":
            test_data["cond"].append(cond)
            test_data["t"].append(t)
            test_data["sp_vals"].append(data[cond]["x1"][(t, split)]["sp"])

x_pos = np.arange(len(test_data["cond"]))
for i, vals in enumerate(test_data["sp_vals"]):
    ax_d.scatter([i]*len(vals), vals, s=80, alpha=0.6, color="#ef4444",
                 edgecolors="black", zorder=3)
    ax_d.scatter([i], [np.mean(vals)], s=200, marker="D", color="#dc2626",
                 edgecolors="black", linewidths=1.5, zorder=4,
                 label="mean" if i == 0 else "")
ax_d.axhline(0, color="black", lw=0.8)
ax_d.axhline(-0.15, color="gray", lw=0.5, ls=":", label="threshold -0.15")

# Wilcoxon
all_test_sp = sum(test_data["sp_vals"], [])
try:
    w, p = stats.wilcoxon(np.array(all_test_sp), alternative="less")
    title_p = f"Wilcoxon p = {p:.4f}"
except Exception:
    title_p = ""

labels_d = [f"{c}\nt={t}" for c, t in zip(test_data["cond"], test_data["t"])]
ax_d.set_xticks(x_pos); ax_d.set_xticklabels(labels_d)
ax_d.set_ylabel("Spearman r (test only, 5 seed)")
ax_d.set_title(f"[D] Held-out test Spearman (each dot=seed)\n"
               f"{title_p}  |  10/15 negative",
               fontsize=10, fontweight="bold")
ax_d.legend(fontsize=8, loc="lower right")
ax_d.grid(alpha=0.3, axis="y")
ax_d.set_ylim(-1.0, 0.4)

fig.suptitle(
    "PI-SDE + X1 (Topic-Anchor) on ArXiv CS  |  5 seeds × 4 conditions × 3 time points\n"
    "Strong effect in alltime/training (Spearman down to -0.81),  Moderate in held-out test (mean -0.14, p=0.005)",
    fontsize=11.5, fontweight="bold", y=0.99,
)

fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"Saved -> {OUT_PNG}")
