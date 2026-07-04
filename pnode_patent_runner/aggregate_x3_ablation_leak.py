"""
X3 Ablation aggregator — Data leak hypothesis verdict.

3 つの ablation 結果と baseline (X3 本体) を 1 表で比較:
  baseline : g_n[t] 入力 (現行 X3, leak の可能性)
  mask     : g_n=0 入力 (centroid + t のみ)
  shuffle  : g_n をシャッフル
  lag      : g_n[t-1] 入力 (autoregressive)

判定ルール:
  - baseline g_pred ρ が mask/shuffle と「同程度」なら → leak の可能性低い
  - baseline g_pred ρ が mask/shuffle より「圧倒的に高い」なら → leak 確定
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path("/home/nakamuraroi/kumagai")
DOMAIN = "paper"
DATA_NAME = "PNode_Paper_X1"
SEED = 42
LAM_G = 0.5
LAST_T = 3


def load_baseline():
    p = ROOT / f"RESULTS_X3/{DATA_NAME}/softplus-400_400-0.0-const-0.1-0.1-0.005-x3_g{LAM_G}/seed_{SEED}/alltime/evaluation_x3.json"
    if not p.exists():
        return None
    return json.load(p.open())


def load_ablation(mode: str):
    p = ROOT / f"RESULTS_X3_ABLATION/{DATA_NAME}/{mode}/x3abl_{mode}_g{LAM_G}_s{SEED}/seed_{SEED}/alltime/evaluation_x3_ablation_{mode}.json"
    if not p.exists():
        return None
    return json.load(p.open())


def row_at(d, t):
    if d is None: return None
    return next((r for r in d["results"] if r["t"] == t), None)


print("=" * 110)
print("  X3 Ablation — Data Leak Hypothesis Test")
print("  baseline = g_n[t] input  |  mask = zeros  |  shuffle = permuted g_n  |  lag = g_n[t-1]")
print("=" * 110)

modes = ["baseline", "mask", "shuffle", "lag"]
data = {
    "baseline": load_baseline(),
    "mask":     load_ablation("mask"),
    "shuffle":  load_ablation("shuffle"),
    "lag":      load_ablation("lag"),
}

# t別比較
print(f"\n  {'t':<4} {'mode':<10} {'φ-Sp':<14} {'g-Sp':<14} {'g-NDCG':<10} {'g-MSE':<10}")
print("  " + "-" * 80)
for t in [1, 2, 3]:
    for m in modes:
        r = row_at(data[m], t)
        if r is None:
            print(f"  {t:<4} {m:<10} MISSING")
            continue
        sig = "*" if r["phi_spearman_p"] < 0.05 else " "
        sig_g = "*" if r["growth_spearman_p"] < 0.05 else " "
        print(f"  {t:<4} {m:<10} {r['phi_spearman_r']:+.3f}{sig:<9} "
              f"{r['growth_spearman_r']:+.3f}{sig_g:<9} "
              f"{r['growth_ndcg']:<10.3f} {r['growth_mse']:<10.4f}")
    print()

# t=LAST_T (final) で各 mode を比較
print(f"\n  [Final t={LAST_T} comparison — KEY result]")
print("  " + "=" * 80)
print(f"  {'mode':<12} {'g-Spearman':<22} {'φ-Spearman':<22} {'g-NDCG@10':<10}")
print("  " + "-" * 80)
baseline_r = row_at(data["baseline"], LAST_T)
for m in modes:
    r = row_at(data[m], LAST_T)
    if r is None:
        print(f"  {m:<12} MISSING")
        continue
    delta = ""
    if m != "baseline" and baseline_r:
        d_g = r["growth_spearman_r"] - baseline_r["growth_spearman_r"]
        delta = f"  (Δ vs baseline = {d_g:+.3f})"
    print(f"  {m:<12} {r['growth_spearman_r']:+.3f}{delta:<18} "
          f"{r['phi_spearman_r']:+.3f}                 "
          f"{r['growth_ndcg']:.3f}")

# Verdict
print("\n  [Verdict]")
print("  " + "=" * 80)
if data["baseline"] and data["mask"] and data["shuffle"]:
    base_g = row_at(data["baseline"], LAST_T)["growth_spearman_r"]
    mask_g = row_at(data["mask"], LAST_T)["growth_spearman_r"]
    shuf_g = row_at(data["shuffle"], LAST_T)["growth_spearman_r"]
    lag_g  = row_at(data["lag"], LAST_T)["growth_spearman_r"] if data["lag"] else None

    drop_mask = abs(base_g) - abs(mask_g)
    drop_shuf = abs(base_g) - abs(shuf_g)

    print(f"  baseline |ρ| = {abs(base_g):.3f}")
    print(f"  mask     |ρ| = {abs(mask_g):.3f}   (drop = {drop_mask:+.3f})")
    print(f"  shuffle  |ρ| = {abs(shuf_g):.3f}   (drop = {drop_shuf:+.3f})")
    if lag_g is not None:
        print(f"  lag      |ρ| = {abs(lag_g):.3f}   (= {abs(lag_g):.3f} vs baseline {abs(base_g):.3f})")

    if drop_mask > 0.4 and drop_shuf > 0.4:
        print("\n  ❌  LEAK CONFIRMED: removing g_n input drops |ρ| by >0.4")
        print("       → baseline X3 g_pred is mostly identity copy of input")
        print("       → the 'ρ=0.99' result is NOT real predictive capacity")
    elif drop_mask > 0.2 or drop_shuf > 0.2:
        print("\n  ⚠   PARTIAL LEAK: input g_n helps significantly but not the whole story")
        print("       → predictor uses both input g_n (leak) and centroid+attention (genuine)")
    else:
        print("\n  ✅  NO LEAK: removing g_n input has small effect")
        print("       → predictor genuinely learns from centroid + attention context")
        print("       → baseline ρ=0.99 reflects real predictive capacity")
