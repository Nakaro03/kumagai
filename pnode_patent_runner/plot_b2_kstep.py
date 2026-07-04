"""
B2 k-step ahead 結果可視化 — long-horizon robustness。
"""
from __future__ import annotations

import json, glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

for cand in ["Noto Sans CJK JP", "IPAGothic", "TakaoGothic"]:
    if cand in [f.name for f in font_manager.fontManager.ttflist]:
        plt.rcParams["font.family"] = cand
        break

DOMAINS = {
    "Patent Energy\n(CPC Y02, last_t=11)": {
        "root": "RESULTS/PNode_Patent_Energy_X1_top50",
        "last_t": 11,
        "kstep_tags": {1: "leaveout11", 2: "leaveout10_11", 3: "leaveout9_10_11"},
        "color": "#1f4f7a",
    },
    "JP Construction\n(J-STAGE, last_t=10)": {
        "root": "RESULTS/PNode_JP_Construction_X1",
        "last_t": 10,
        "kstep_tags": {1: "leaveout10", 2: "leaveout9_10", 3: "leaveout8_9_10"},
        "color": "#a02050",
    },
}
SEEDS = [0, 1, 42, 123, 999]


def gather(root, tag, last_t):
    sps = []
    for s in SEEDS:
        files = list(Path(root).rglob(f"*_v1.0_g0.1_b0.01/seed_{s}/{tag}/evaluation_x1.json"))
        if not files: continue
        d = json.load(open(files[0]))
        r = next((r for r in d["results"] if r["t"] == last_t and r.get("split") == "test"), None)
        if r: sps.append(r["spearman_r"])
    return np.array(sps)


fig, ax = plt.subplots(figsize=(11, 7))

ks = [1, 2, 3]
width = 0.32
positions = np.arange(len(ks))

for i, (dname, cfg) in enumerate(DOMAINS.items()):
    means, stds, wps = [], [], []
    for k in ks:
        sps = gather(cfg["root"], cfg["kstep_tags"][k], cfg["last_t"])
        if len(sps) == 0:
            means.append(np.nan); stds.append(0); wps.append(1.0); continue
        means.append(np.mean(sps)); stds.append(np.std(sps))
        try:
            _, wp = stats.wilcoxon(sps, alternative="less")
            wps.append(wp)
        except Exception:
            wps.append(1.0)

    offset = (i - 0.5) * width
    bars = ax.bar(positions + offset, means, width, yerr=stds,
                  label=dname, color=cfg["color"], edgecolor="black", capsize=4,
                  alpha=0.92, linewidth=0.8, error_kw=dict(ecolor="black", lw=1))
    # Add Wilcoxon p annotations
    for k, m, s, wp, pos in zip(ks, means, stds, wps, positions + offset):
        sig = "***" if wp < 0.001 else ("**" if wp < 0.01 else ("*" if wp < 0.05 else ""))
        ax.text(pos, m - s - 0.04, f"{m:+.3f}\n{sig}", ha="center", va="top",
                fontsize=9, fontweight="bold", color="black")

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(positions)
ax.set_xticklabels([f"k = {k}\n({k}-step ahead)" for k in ks], fontsize=11)
ax.set_ylabel("Spearman ρ  (lower / more negative = better)", fontsize=12)
ax.set_title("B2 k-step ahead prediction — X1 PI-SDE long-horizon robustness\n"
             "(全 k で Wilcoxon p < 0.05 を維持, * = 5-seed Wilcoxon test)",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(-0.65, 0.05)

out = Path("RESULTS/fig11_b2_kstep_robustness.png")
plt.tight_layout()
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")

import shutil
shutil.copy(out, "figures/fig11_b2_kstep_robustness.png")
print("Copied to figures/")
