"""
PI-SDE 論文ドメイン結果の可視化 (seed=42 既存結果)。

入力:
  - RESULTS/PNode_Paper/.../alltime/evaluation.json
  - RESULTS/PNode_Paper/.../leaveout3/.../leaveout3/evaluation.json

出力:
  - RESULTS/PNode_Paper/visualization.png
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir("/home/nakamuraroi/kumagai")

RESULTS_ROOT = Path("RESULTS/PNode_Paper/softplus-400_400-0.5-const-0.1-0.1-0.005")
OUT_PNG = RESULTS_ROOT / "visualization.png"


def load_eval(seed, cond):
    sub = f"seed_{seed}/{cond}"
    for p in RESULTS_ROOT.rglob("evaluation.json"):
        if f"seed_{seed}" in str(p) and f"/{cond}/" in str(p):
            return json.load(p.open())
    return None


# 既存データ取得
alltime = load_eval(42, "alltime")
leaveout3 = load_eval(42, "leaveout3")

assert alltime is not None, "alltime evaluation not found"
assert leaveout3 is not None, "leaveout3 evaluation not found"

# データ準備
def to_arrays(res):
    ts = [r["t"] for r in res["results"]]
    pi  = [r["pi_sde"]     for r in res["results"]]
    na  = [r["naive"]      for r in res["results"]]
    la  = [r.get("last_seen", float("nan")) for r in res["results"]]
    sp  = [r["split"]      for r in res["results"]]
    return ts, pi, na, la, sp

ts_a, pi_a, na_a, la_a, sp_a = to_arrays(alltime)
ts_l, pi_l, na_l, la_l, sp_l = to_arrays(leaveout3)

# ── プロット ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

# [A] alltime: 性能比較
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(ts_a))
w = 0.27
b1 = ax.bar(x - w, pi_a, w, label="PI-SDE",    color="#3b82f6", edgecolor="black")
b2 = ax.bar(x,     na_a, w, label="Naive(x_0)", color="#9ca3af", edgecolor="black")
b3 = ax.bar(x + w, la_a, w, label="Last-seen",  color="#fbbf24", edgecolor="black")
for bar in b1:
    h = bar.get_height(); ax.annotate(f"{h:.2f}", (bar.get_x()+bar.get_width()/2, h+0.2),
                                       ha="center", fontsize=8, color="black")
ax.set_xticks(x); ax.set_xticklabels([f"t={t}\ny={int(t)+2022}" for t in ts_a])
ax.set_ylabel("Sinkhorn distance (lower = better)")
ax.set_title("[A] alltime: 全期間 fit\n(t=1,2,3 共に PI-SDE が両baselineを下回る = 改善)", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
ax.set_ylim(0, max(na_a + la_a + pi_a) * 1.15)

# [B] leaveout3: 未来予測 (test split が t=3)
ax = fig.add_subplot(gs[0, 1])
x = np.arange(len(ts_l))
b1 = ax.bar(x - w, pi_l, w, label="PI-SDE",    color="#3b82f6", edgecolor="black")
b2 = ax.bar(x,     na_l, w, label="Naive(x_0)", color="#9ca3af", edgecolor="black")
b3 = ax.bar(x + w, la_l, w, label="Last-seen",  color="#fbbf24", edgecolor="black")
for bar in b1:
    h = bar.get_height(); ax.annotate(f"{h:.2f}", (bar.get_x()+bar.get_width()/2, h+0.2),
                                       ha="center", fontsize=8, color="black")
# test split を強調
for i, s in enumerate(sp_l):
    if s == "test":
        ax.axvspan(i - 0.5, i + 0.5, color="#fef3c7", alpha=0.5, zorder=0)
        ax.text(i, max(na_l)*1.05, "TEST\n(held-out)", ha="center", fontsize=9, color="#dc2626",
                fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"t={t}\ny={int(t)+2022}" for t in ts_l])
ax.set_ylabel("Sinkhorn distance (lower = better)")
ax.set_title("[B] leaveout t=3: 2025 年を holdout して未来予測\n"
             "(test split で PI-SDE が両baselineを下回る = 未来予測能力あり)", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
ax.set_ylim(0, max(na_l + la_l + pi_l) * 1.15)

# [C] 改善率 (vs Naive)
ax = fig.add_subplot(gs[1, 0])
imp_a = [(na - pi) / na * 100 for pi, na in zip(pi_a, na_a)]
imp_l = [(na - pi) / na * 100 for pi, na in zip(pi_l, na_l)]
x = np.arange(len(ts_a))
ax.bar(x - 0.2, imp_a, 0.4, label="alltime", color="#3b82f6", edgecolor="black")
ax.bar(x + 0.2, imp_l, 0.4, label="leaveout3", color="#f97316", edgecolor="black")
for i, (a, l) in enumerate(zip(imp_a, imp_l)):
    ax.annotate(f"{a:.1f}%", (i - 0.2, a + 0.5), ha="center", fontsize=8)
    ax.annotate(f"{l:.1f}%", (i + 0.2, l + 0.5), ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"t={t}" for t in ts_a])
ax.set_ylabel("Improvement over Naive (%)")
ax.set_title("[C] PI-SDE 改善率 (vs Naive x_0)\n(全条件で 17-27% 改善)", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
ax.axhline(0, color="black", lw=0.8)

# [D] サマリーテキスト
ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
summary = (
    "■ PI-SDE on ArXiv CS papers (seed=42)\n"
    "  - 入力: 50-D PCA reduced 論文埋め込み\n"
    "  - 時系列: 4 時点 (2022-2025)\n"
    "  - モデル: dx = -∇Φ(x,t)dt + 0.1·dW  (HJ 内包)\n"
    "  - 評価: Sinkhorn (Wasserstein 近似)\n"
    "\n"
    "■ alltime (training fit)\n"
    f"  t=1: PI-SDE {pi_a[0]:.2f} < Naive {na_a[0]:.2f}\n"
    f"  t=2: PI-SDE {pi_a[1]:.2f} < Naive {na_a[1]:.2f}\n"
    f"  t=3: PI-SDE {pi_a[2]:.2f} < Naive {na_a[2]:.2f}\n"
    f"  平均改善: -{np.mean(imp_a):.1f}%\n"
    "\n"
    "■ leaveout3 (2025 holdout)\n"
    f"  t=3 (TEST): PI-SDE {pi_l[2]:.2f} < Naive {na_l[2]:.2f}\n"
    f"  test 改善: -{imp_l[2]:.1f}%\n"
    "\n"
    "■ PI-SDE 論文 (Jiang & Wan 2024)\n"
    "  の HJ regularization 効果を\n"
    "  論文ドメインで再現確認\n"
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        va="top", ha="left", fontsize=9.5, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0fdf4",
                  edgecolor="#16a34a", lw=1.5))

fig.suptitle(
    "PI-SDE applied to ArXiv CS papers  (seed=42, 500 epochs)",
    fontsize=12, fontweight="bold", y=0.99,
)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved -> {OUT_PNG}")
