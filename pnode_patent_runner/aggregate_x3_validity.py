"""
X3 (Minimal EBM) — Validity aggregation across multi-seed / multi-domain / leave-one-out.

入力: 各 run の evaluation_x3.json
出力:
  - X3 multi-seed (paper × 5 seeds): mean ± std
  - X3 multi-domain (4 domains × seed=42): per-domain summary
  - X3 leave-one-out (paper, holdout t=3): generalization test
  - 全結果と X1/X2 baselines および外部 baselines を 1 表で比較
  - 「有効性が確認できたか」の機械的判定 (Spearman 有意性 + ベースライン勝率)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path("/home/nakamuraroi/kumagai")
os.chdir(ROOT)

DOMAINS = {
    "paper":               ("PNode_Paper_X1", 3),
    "patent_energy_top50": ("PNode_Patent_Energy_X1_top50", 11),
    "arxiv_construction":  ("PNode_ArXiv_Construction_X1_v2", 10),
    "jp_construction":     ("PNode_JP_Construction_X1", 10),
}
LAM_G = 0.5
X3_TAG = f"softplus-400_400-0.0-const-0.1-0.1-0.005-x3_g{LAM_G}"


def load_x3(domain: str, seed: int, leaveout: str = "alltime") -> Optional[Dict]:
    data_name, _ = DOMAINS[domain]
    p = ROOT / f"RESULTS_X3/{data_name}/{X3_TAG}/seed_{seed}/{leaveout}/evaluation_x3.json"
    if not p.exists():
        return None
    return json.load(p.open())


def load_x1_mean(domain: str, last_t: int) -> Optional[Dict]:
    """X1 5-seed mean for the final time-point."""
    data_name, _ = DOMAINS[domain]
    root = ROOT / f"RESULTS/{data_name}"
    sps, nds, p10s = [], [], []
    for s in [0, 1, 42, 123, 999]:
        for p in root.rglob("evaluation_x1.json"):
            tag = list(p.parents)[2].name if len(p.parents) >= 3 else ""
            if not tag.endswith("-x1_v1.0_g0.1_b0.01"): continue
            if f"seed_{s}" not in str(p) or "/alltime/" not in str(p): continue
            d = json.load(p.open())
            r = next((r for r in d["results"] if r["t"] == last_t), None)
            if r is None: continue
            sps.append(r["spearman_r"]); nds.append(r["ndcg"]); p10s.append(r["prec_at_10"])
            break
    if not sps:
        return None
    return {"n": len(sps), "spearman_mean": float(np.mean(sps)), "spearman_std": float(np.std(sps)),
            "ndcg_mean": float(np.mean(nds)), "p10_mean": float(np.mean(p10s))}


def load_x2(domain: str, seed: int, last_t: int) -> Optional[Dict]:
    data_name, _ = DOMAINS[domain]
    root = ROOT / f"RESULTS_X2/{data_name}"
    for p in root.rglob("evaluation_x2.json"):
        tag = list(p.parents)[2].name if len(p.parents) >= 3 else ""
        if "x2_v1.0_g0.1_b0.01_gh0.5" not in tag: continue
        if f"seed_{seed}" not in str(p) or "/alltime/" not in str(p): continue
        d = json.load(p.open())
        r = next((r for r in d["results"] if r["t"] == last_t), None)
        if r is not None:
            return r
    return None


def load_baseline_means(domain: str, methods=None) -> Dict[str, Dict]:
    """External baselines mean across 5 seeds (paper has full data)."""
    if methods is None:
        methods = ["Naive_zero", "Naive_mean", "Naive_lastg", "Linear", "ARIMA",
                   "LSTM", "Transformer", "DLinear", "PatchTST"]
    out = {}
    for m in methods:
        sps, nds, p10s = [], [], []
        for s in [0, 1, 42, 123, 999]:
            p = ROOT / f"RESULTS/baselines/{domain}/baselines_seed{s}.json"
            if not p.exists(): continue
            d = json.load(p.open())
            if m not in d: continue
            r = d[m]
            if r.get("spearman_r") == r.get("spearman_r"):  # NaN check
                sps.append(r["spearman_r"])
            nds.append(r.get("ndcg_at_10", float("nan")))
            p10s.append(r.get("prec_at_10", float("nan")))
        if sps:
            out[m] = {"n": len(sps),
                      "spearman_mean": float(np.mean(sps)),
                      "spearman_std": float(np.std(sps)),
                      "ndcg_mean": float(np.nanmean(nds)),
                      "p10_mean": float(np.nanmean(p10s))}
    return out


# ─────────────────────────────────────────────────────────────────
# 集計
# ─────────────────────────────────────────────────────────────────
print("=" * 110)
print(f"  X3 (Minimal EBM, λ_g={LAM_G}, 200 epochs) — Validity Aggregation")
print("=" * 110)

# A) Multi-seed (paper × 5 seeds)
print("\n[A] Multi-seed (paper × 5 seeds)")
print("-" * 110)
print(f"  {'seed':<8} {'t':<4} {'phi-Sp':<14} {'phi-NDCG':<10} {'g-Sp':<14} {'g-NDCG':<10} {'g-MSE':<10}")
phi_sp_t1, phi_sp_t2, phi_sp_t3 = [], [], []
g_sp_t1,  g_sp_t2,  g_sp_t3  = [], [], []
g_ndcg_t3, phi_ndcg_t3 = [], []
phi_p_t3 = []
for s in [0, 1, 42, 123, 999]:
    d = load_x3("paper", s)
    if d is None:
        print(f"  seed={s:<3}: MISSING")
        continue
    for r in d["results"]:
        sig = "*" if r["phi_spearman_p"] < 0.05 else " "
        print(f"  {s:<8} {r['t']:<4} {r['phi_spearman_r']:+.3f}{sig:<9} "
              f"{r['phi_ndcg']:<10.3f} {r['growth_spearman_r']:+.3f}     "
              f"{r['growth_ndcg']:<10.3f} {r['growth_mse']:<10.4f}")
        if r["t"] == 1: phi_sp_t1.append(r["phi_spearman_r"]); g_sp_t1.append(r["growth_spearman_r"])
        if r["t"] == 2: phi_sp_t2.append(r["phi_spearman_r"]); g_sp_t2.append(r["growth_spearman_r"])
        if r["t"] == 3:
            phi_sp_t3.append(r["phi_spearman_r"]); g_sp_t3.append(r["growth_spearman_r"])
            g_ndcg_t3.append(r["growth_ndcg"]); phi_ndcg_t3.append(r["phi_ndcg"])
            phi_p_t3.append(r["phi_spearman_p"])
    print()

def ms(a): return (f"{np.mean(a):+.3f} ± {np.std(a):.3f}" if a else "N/A", len(a))

print("  ── Mean ± SD across seeds ──")
m, n = ms(phi_sp_t1); print(f"  t=1 phi-Spearman:  {m}  (n={n})")
m, n = ms(phi_sp_t2); print(f"  t=2 phi-Spearman:  {m}  (n={n})")
m, n = ms(phi_sp_t3); print(f"  t=3 phi-Spearman:  {m}  (n={n})")
m, n = ms(g_sp_t1);  print(f"  t=1 g-Spearman:    {m}  (n={n})")
m, n = ms(g_sp_t2);  print(f"  t=2 g-Spearman:    {m}  (n={n})")
m, n = ms(g_sp_t3);  print(f"  t=3 g-Spearman:    {m}  (n={n})")
m, n = ms(g_ndcg_t3); print(f"  t=3 g-NDCG@10:     {m}  (n={n})")
m, n = ms(phi_ndcg_t3); print(f"  t=3 phi-NDCG@10:   {m}  (n={n})")
if phi_p_t3:
    n_sig = sum(1 for p in phi_p_t3 if p < 0.05)
    print(f"  t=3 phi-Spearman significant (p<0.05): {n_sig}/{len(phi_p_t3)} seeds")

# B) Multi-domain (seed=42)
print("\n\n[B] Multi-domain (seed=42)")
print("-" * 110)
print(f"  {'domain':<22} {'t':<4} {'phi-Sp':<14} {'g-Sp':<14} {'g-NDCG':<10} {'g-MSE':<10}")
domain_results = {}
for dn in DOMAINS:
    d = load_x3(dn, 42)
    if d is None:
        print(f"  {dn:<22}: MISSING")
        continue
    last_t = DOMAINS[dn][1]
    last_row = next((r for r in d["results"] if r["t"] == last_t), None)
    if last_row:
        domain_results[dn] = last_row
        for r in d["results"]:
            if r["t"] not in [1, last_t // 2, last_t]: continue
            sig = "*" if r["phi_spearman_p"] < 0.05 else " "
            print(f"  {dn:<22} {r['t']:<4} {r['phi_spearman_r']:+.3f}{sig:<9} "
                  f"{r['growth_spearman_r']:+.3f}        {r['growth_ndcg']:<10.3f} {r['growth_mse']:<10.4f}")

# C) Leave-one-out (paper, holdout t=3)
print("\n\n[C] Leave-one-out (paper, t=3 holdout — never seen during training)")
print("-" * 110)
d_lo = load_x3("paper", 42, "leaveout3")
if d_lo is not None:
    for r in d_lo["results"]:
        split = r.get("split", "?")
        sig = "*" if r["phi_spearman_p"] < 0.05 else " "
        flag = "  ⟵ TEST" if split == "test" else ""
        print(f"  t={r['t']:<4} split={split:<6} phi-Sp={r['phi_spearman_r']:+.3f}{sig}  "
              f"g-Sp={r['growth_spearman_r']:+.3f}  g-NDCG={r['growth_ndcg']:.3f}  "
              f"g-MSE={r['growth_mse']:.4f}{flag}")
else:
    print("  MISSING — leave-one-out run not yet executed")

# D) 比較表 (paper, t=3 final)
print("\n\n[D] Comparison table (paper, t=3 final timepoint)")
print("-" * 110)
print(f"  {'Method':<26} {'|Spearman|':<22} {'NDCG@10':<14} {'P@10':<10} {'n_seed':<6}")
print("  " + "-" * 100)

# External baselines
base = load_baseline_means("paper")
for m, r in base.items():
    print(f"  {m:<26} {abs(r['spearman_mean']):.3f} ± {r['spearman_std']:.3f}     "
          f"{r['ndcg_mean']:.3f}         {r['p10_mean']:.2f}       {r['n']}")

# X1
x1 = load_x1_mean("paper", 3)
if x1:
    print(f"  {'PI-SDE + X1 (Φ-rank)':<26} {abs(x1['spearman_mean']):.3f} ± {x1['spearman_std']:.3f}     "
          f"{x1['ndcg_mean']:.3f}         {x1['p10_mean']:.2f}       {x1['n']}")

# X2 (single seed=42)
x2 = load_x2("paper", 42, 3)
if x2:
    print(f"  {'PI-SDE + X2 (Φ-rank)':<26} {abs(x2['phi_spearman_r']):.3f}            "
          f"{x2['phi_ndcg']:.3f}         {x2['phi_prec_at_10']:.2f}       1")
    print(f"  {'PI-SDE + X2 (g_pred)':<26} {abs(x2['growth_spearman_r']):.3f}            "
          f"{x2['growth_ndcg']:.3f}         {x2['growth_prec_at_10']:.2f}       1")

# X3 multi-seed mean
if phi_sp_t3:
    print(f"  {'PI-SDE + X3 (Φ-rank)':<26} {abs(np.mean(phi_sp_t3)):.3f} ± {np.std(phi_sp_t3):.3f}     "
          f"{np.mean(phi_ndcg_t3):.3f}         -          {len(phi_sp_t3)}")
if g_sp_t3:
    print(f"  {'PI-SDE + X3 (g_pred)':<26} {abs(np.mean(g_sp_t3)):.3f} ± {np.std(g_sp_t3):.3f}     "
          f"{np.mean(g_ndcg_t3):.3f}         -          {len(g_sp_t3)}")

# E) Mechanical verdict
print("\n\n[E] Validity verdict (mechanical)")
print("-" * 110)
checks = []
if phi_sp_t3:
    n_seeds = len(phi_sp_t3)
    n_sig = sum(1 for p in phi_p_t3 if p < 0.05)
    pass1 = n_sig == n_seeds and n_seeds >= 3
    checks.append((f"Multi-seed Φ-rank significant @ all {n_seeds} seeds (p<0.05)",
                   pass1, f"{n_sig}/{n_seeds}"))
    pass2 = abs(np.mean(phi_sp_t3)) > 0.6
    checks.append((f"X3 |Φ-Spearman| > 0.6 (vs best baseline ~0.3)",
                   pass2, f"{abs(np.mean(phi_sp_t3)):.3f}"))
if g_sp_t3:
    pass3 = abs(np.mean(g_sp_t3)) > 0.9
    checks.append((f"X3 g_pred Spearman > 0.9 (predictor head valid)",
                   pass3, f"{abs(np.mean(g_sp_t3)):.3f}"))
if domain_results:
    n_dom = len(domain_results)
    n_dom_sig = sum(1 for r in domain_results.values() if r["phi_spearman_p"] < 0.05)
    pass4 = n_dom_sig >= 3
    checks.append((f"Multi-domain Φ-rank significant in ≥3/4 domains",
                   pass4, f"{n_dom_sig}/{n_dom}"))
if d_lo:
    test_rows = [r for r in d_lo["results"] if r.get("split") == "test"]
    if test_rows:
        pass5 = any(r["phi_spearman_p"] < 0.05 for r in test_rows)
        checks.append((f"Leave-one-out Φ-Spearman significant on holdout",
                       pass5, f"{test_rows[0]['phi_spearman_r']:+.3f} (p={test_rows[0]['phi_spearman_p']:.4f})"))

for name, ok, val in checks:
    mark = "✅" if ok else "❌"
    print(f"  {mark}  {name:<70}  →  {val}")

n_pass = sum(1 for _, ok, _ in checks if ok)
n_total = len(checks)
print(f"\n  Score: {n_pass}/{n_total} checks passed")
if n_pass == n_total:
    print("  Verdict: ✅ Effectiveness CONFIRMED across all axes")
elif n_pass >= n_total - 1:
    print("  Verdict: ⚠ Mostly confirmed (1 weak)")
else:
    print(f"  Verdict: ❌ Effectiveness NOT YET confirmed ({n_total - n_pass} failures)")
