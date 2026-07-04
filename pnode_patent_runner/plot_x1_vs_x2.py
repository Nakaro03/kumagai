"""
X1 vs X2 比較図 — 4 ドメインの leaveout 結果.
"""
from __future__ import annotations

import json, glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for cand in ["Noto Sans CJK JP", "IPAGothic", "TakaoGothic"]:
    if cand in [f.name for f in font_manager.fontManager.ttflist]:
        plt.rcParams["font.family"] = cand
        break

DOMAINS = {
    "Paper\n(arXiv CS)":            ("PNode_Paper_X1", 3, "leaveout3"),
    "Patent Energy\n(CPC Y02)":     ("PNode_Patent_Energy_X1_top50", 11, "leaveout11"),
    "arXiv Construction\n(legacy)": ("PNode_ArXiv_Construction_X1_v2", 10, "leaveout10"),
    "JP Construction\n(J-STAGE)":   ("PNode_JP_Construction_X1", 10, "leaveout10"),
}
SEEDS = [0, 1, 42, 123, 999]


def gather_x1(root, last_t, tag):
    sps, p10s = [], []
    for s in SEEDS:
        f = glob.glob(f"RESULTS/{root}/*x1_v1.0_g0.1_b0.01/seed_{s}/{tag}/evaluation_x1.json")
        if not f: continue
        d = json.load(open(f[0]))
        r = next((r for r in d["results"] if r["t"] == last_t and r.get("split") == "test"), None)
        if r is None: continue
        sps.append(r["spearman_r"]); p10s.append(r["prec_at_10"])
    return np.array(sps), np.array(p10s)


def gather_x2(root, last_t, tag):
    phi_sps, phi_p10s, g_sps, g_p10s = [], [], [], []
    for s in SEEDS:
        f = glob.glob(f"RESULTS_X2/{root}/*x2_v*/seed_{s}/{tag}/evaluation_x2.json")
        if not f: continue
        d = json.load(open(f[0]))
        r = next((r for r in d["results"] if r["t"] == last_t and r.get("split") == "test"), None)
        if r is None: continue
        phi_sps.append(r["phi_spearman_r"]); phi_p10s.append(r["phi_prec_at_10"])
        g_sps.append(r["growth_spearman_r"]); g_p10s.append(r["growth_prec_at_10"])
    return (np.array(phi_sps), np.array(phi_p10s), np.array(g_sps), np.array(g_p10s))


fig, axes = plt.subplots(1, 2, figsize=(16, 7))

method_colors = {
    "X1 PI-SDE (Φ)": "#1f4f7a",
    "X2 (Φ head)": "#4a90d9",
    "X2 (growth head) ★": "#2c7a2c",
}

# Panel 1: |Spearman|
ax = axes[0]
x_pos = np.arange(len(DOMAINS))
width = 0.27

for i, (mname, color) in enumerate(method_colors.items()):
    means, stds = [], []
    for dname, (root, last_t, tag) in DOMAINS.items():
        if "X1" in mname:
            sps, _ = gather_x1(root, last_t, tag)
            means.append(abs(np.mean(sps)) if len(sps) else np.nan)
            stds.append(np.std(sps) if len(sps) else 0)
        elif "Φ head" in mname:
            phi_sps, _, _, _ = gather_x2(root, last_t, tag)
            means.append(abs(np.mean(phi_sps)) if len(phi_sps) else np.nan)
            stds.append(np.std(phi_sps) if len(phi_sps) else 0)
        else:   # growth head
            _, _, g_sps, _ = gather_x2(root, last_t, tag)
            means.append(abs(np.mean(g_sps)) if len(g_sps) else np.nan)
            stds.append(np.std(g_sps) if len(g_sps) else 0)
    offset = (i - 1) * width
    bars = ax.bar(x_pos + offset, means, width, yerr=stds,
                  label=mname, color=color, edgecolor="black",
                  alpha=0.92, capsize=4, linewidth=0.7, error_kw=dict(lw=1))
    for b, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(b.get_x() + b.get_width()/2, m + 0.02, f"{m:.2f}",
                    ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(list(DOMAINS.keys()), fontsize=9)
ax.set_ylabel("|Spearman ρ|  (higher = better)", fontsize=11)
ax.set_title("|Spearman| 比較  (5-seed mean ± std, leaveout=last)",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 0.9)

# Panel 2: P@10
ax2 = axes[1]
for i, (mname, color) in enumerate(method_colors.items()):
    means, stds = [], []
    for dname, (root, last_t, tag) in DOMAINS.items():
        if "X1" in mname:
            _, p10 = gather_x1(root, last_t, tag)
            means.append(np.mean(p10) if len(p10) else np.nan)
            stds.append(np.std(p10) if len(p10) else 0)
        elif "Φ head" in mname:
            _, phi_p10, _, _ = gather_x2(root, last_t, tag)
            means.append(np.mean(phi_p10) if len(phi_p10) else np.nan)
            stds.append(np.std(phi_p10) if len(phi_p10) else 0)
        else:
            _, _, _, g_p10 = gather_x2(root, last_t, tag)
            means.append(np.mean(g_p10) if len(g_p10) else np.nan)
            stds.append(np.std(g_p10) if len(g_p10) else 0)
    offset = (i - 1) * width
    bars = ax2.bar(x_pos + offset, means, width, yerr=stds,
                   label=mname, color=color, edgecolor="black",
                   alpha=0.92, capsize=4, linewidth=0.7, error_kw=dict(lw=1))
    for b, m in zip(bars, means):
        if not np.isnan(m):
            ax2.text(b.get_x() + b.get_width()/2, m + 0.02, f"{m:.2f}",
                     ha="center", fontsize=8, fontweight="bold")

ax2.set_xticks(x_pos)
ax2.set_xticklabels(list(DOMAINS.keys()), fontsize=9)
ax2.set_ylabel("Precision@10  (higher = better)", fontsize=11)
ax2.set_title("P@10 比較  (5-seed mean ± std)",
              fontsize=12, fontweight="bold")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, 1.15)

plt.suptitle("X1 PI-SDE vs X2 PI-SDE (Multi-Task + Cross-Topic + MC) — true future prediction (leaveout)",
             fontsize=13, fontweight="bold", y=1.005)
plt.tight_layout()

out = Path("RESULTS/fig16_x1_vs_x2.png")
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
import shutil
shutil.copy(out, "figures/fig16_x1_vs_x2.png")
