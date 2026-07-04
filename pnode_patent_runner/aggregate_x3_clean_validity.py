"""
X3-clean (mask mode: predictor input g_n=0) — Full validity aggregation.

入力: RESULTS_X3_ABLATION/{DATA}/mask/x3abl_mask_g0.5/seed_{seed}/{alltime|leaveout3}/
  evaluation_x3_ablation_mask.json

集計:
  [A] Multi-seed × multi-domain: 4 domains × 5 seeds = 20 runs
  [B] Leave-one-out (paper, t=3 holdout): 5 seeds
  [C] X3-clean vs X3-baseline vs X1 vs external baselines (paper, final t)
  [D] 機械的判定 — 有効性が確認されたか
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np

ROOT = Path("/home/nakamuraroi/kumagai")
os.chdir(ROOT)

DOMAINS = {
    "paper":               ("PNode_Paper_X1", 3),
    "patent_energy_top50": ("PNode_Patent_Energy_X1_top50", 11),
    "arxiv_construction":  ("PNode_ArXiv_Construction_X1_v2", 10),
    "jp_construction":     ("PNode_JP_Construction_X1", 10),
}
SEEDS = [0, 1, 42, 123, 999]
LAM_G = 0.5


def load_x3_clean(domain: str, seed: int, leaveout: str = "alltime"):
    data_name, _ = DOMAINS[domain]
    p = (ROOT / "RESULTS_X3_ABLATION" / data_name / "mask" / f"x3abl_mask_g{LAM_G}"
         / f"seed_{seed}" / leaveout / "evaluation_x3_ablation_mask.json")
    if not p.exists(): return None
    return json.load(p.open())


def load_x3_baseline(domain: str, seed: int, leaveout: str = "alltime"):
    data_name, _ = DOMAINS[domain]
    tag = f"softplus-400_400-0.0-const-0.1-0.1-0.005-x3_g{LAM_G}"
    p = ROOT / "RESULTS_X3" / data_name / tag / f"seed_{seed}" / leaveout / "evaluation_x3.json"
    if not p.exists(): return None
    return json.load(p.open())


def load_x1_mean(domain: str, last_t: int):
    data_name, _ = DOMAINS[domain]
    root = ROOT / f"RESULTS/{data_name}"
    sps, nds, p10s = [], [], []
    for s in SEEDS:
        for p in root.rglob("evaluation_x1.json"):
            tag = list(p.parents)[2].name if len(p.parents) >= 3 else ""
            if not tag.endswith("-x1_v1.0_g0.1_b0.01"): continue
            if f"seed_{s}" not in str(p) or "/alltime/" not in str(p): continue
            d = json.load(p.open())
            r = next((r for r in d["results"] if r["t"] == last_t), None)
            if r is None: continue
            sps.append(r["spearman_r"]); nds.append(r["ndcg"]); p10s.append(r["prec_at_10"])
            break
    if not sps: return None
    return {"n": len(sps), "spearman_mean": float(np.mean(sps)),
            "spearman_std": float(np.std(sps)),
            "ndcg_mean": float(np.mean(nds)), "p10_mean": float(np.mean(p10s))}


def load_baseline_means(domain: str):
    methods = ["Naive_zero", "Naive_mean", "Naive_lastg", "Linear", "ARIMA",
               "LSTM", "Transformer", "DLinear", "PatchTST"]
    out = {}
    for m in methods:
        sps, nds, p10s = [], [], []
        for s in SEEDS:
            p = ROOT / f"RESULTS/baselines/{domain}/baselines_seed{s}.json"
            if not p.exists(): continue
            d = json.load(p.open())
            if m not in d: continue
            r = d[m]
            if r.get("spearman_r") == r.get("spearman_r"):
                sps.append(r["spearman_r"])
            nds.append(r.get("ndcg_at_10", float("nan")))
            p10s.append(r.get("prec_at_10", float("nan")))
        if sps:
            out[m] = {"n": len(sps), "spearman_mean": float(np.mean(sps)),
                      "spearman_std": float(np.std(sps)),
                      "ndcg_mean": float(np.nanmean(nds)),
                      "p10_mean": float(np.nanmean(p10s))}
    return out


print("=" * 110)
print(f"  X3-clean (mask mode, λ_g={LAM_G}, 200 epochs) — Full Validity Aggregation")
print("=" * 110)

# ── [A] Multi-seed × multi-domain ──────────────────────────────────
print("\n[A] Multi-seed × multi-domain (alltime)")
print("-" * 110)
print(f"  {'domain':<22} {'phi_sp_mean':<22} {'g_sp_mean':<22} {'g_ndcg_mean':<14} {'n_sig/n':<10}")
print("  " + "-" * 100)

domain_summary = {}
for dn in DOMAINS:
    last_t = DOMAINS[dn][1]
    phi_sps, g_sps, g_ndcgs, phi_ps = [], [], [], []
    for s in SEEDS:
        d = load_x3_clean(dn, s)
        if d is None: continue
        r = next((r for r in d["results"] if r["t"] == last_t), None)
        if r is None: continue
        phi_sps.append(r["phi_spearman_r"])
        phi_ps.append(r["phi_spearman_p"])
        g_sps.append(r["growth_spearman_r"])
        g_ndcgs.append(r["growth_ndcg"])
    if not phi_sps:
        print(f"  {dn:<22}  MISSING")
        continue
    n_sig = sum(1 for p in phi_ps if p < 0.05)
    domain_summary[dn] = {
        "phi_mean": float(np.mean(phi_sps)), "phi_std": float(np.std(phi_sps)),
        "g_mean":  float(np.mean(g_sps)),  "g_std":  float(np.std(g_sps)),
        "g_ndcg_mean": float(np.mean(g_ndcgs)),
        "n": len(phi_sps), "n_sig": n_sig,
    }
    print(f"  {dn:<22}  {np.mean(phi_sps):+.3f} ± {np.std(phi_sps):.3f}      "
          f"{np.mean(g_sps):+.3f} ± {np.std(g_sps):.3f}      "
          f"{np.mean(g_ndcgs):.3f}          {n_sig}/{len(phi_ps)}")

# ── [B] Leave-one-out (paper t=3 holdout) ──────────────────────────
print("\n\n[B] Leave-one-out generalization (paper, t=3 holdout, 5 seeds)")
print("-" * 110)
print(f"  {'seed':<8} {'split':<8} {'t':<4} {'phi_sp':<14} {'g_sp':<14} {'g_ndcg':<10} {'g_mse':<10}")
print("  " + "-" * 80)
test_phi_sps, test_g_sps, test_g_ndcgs, test_g_mses = [], [], [], []
test_phi_ps = []
for s in SEEDS:
    d = load_x3_clean("paper", s, "leaveout3")
    if d is None:
        print(f"  seed={s:<3}  MISSING")
        continue
    for r in d["results"]:
        if r.get("split") == "test":
            sig = "*" if r["phi_spearman_p"] < 0.05 else " "
            sig_g = "*" if r["growth_spearman_p"] < 0.05 else " "
            print(f"  {s:<8} {'TEST':<8} {r['t']:<4} {r['phi_spearman_r']:+.3f}{sig:<9} "
                  f"{r['growth_spearman_r']:+.3f}{sig_g:<9} "
                  f"{r['growth_ndcg']:<10.3f} {r['growth_mse']:<10.4f}")
            test_phi_sps.append(r["phi_spearman_r"])
            test_phi_ps.append(r["phi_spearman_p"])
            test_g_sps.append(r["growth_spearman_r"])
            test_g_ndcgs.append(r["growth_ndcg"])
            test_g_mses.append(r["growth_mse"])

if test_phi_sps:
    print(f"\n  ── Test-set mean ± SD ──")
    print(f"  phi_spearman   = {np.mean(test_phi_sps):+.3f} ± {np.std(test_phi_sps):.3f}")
    print(f"  g_spearman     = {np.mean(test_g_sps):+.3f} ± {np.std(test_g_sps):.3f}")
    print(f"  g_ndcg@10      = {np.mean(test_g_ndcgs):.3f} ± {np.std(test_g_ndcgs):.3f}")
    print(f"  g_mse          = {np.mean(test_g_mses):.4f} ± {np.std(test_g_mses):.4f}")
    n_sig = sum(1 for p in test_phi_ps if p < 0.05)
    print(f"  phi-Spearman significant (p<0.05): {n_sig}/{len(test_phi_ps)} seeds")

# ── [C] Method comparison (paper, t=3) ─────────────────────────────
print("\n\n[C] Comparison table (paper, t=3 final timepoint)")
print("-" * 110)
print(f"  {'Method':<32} {'|Spearman|':<22} {'NDCG@10':<14} {'P@10':<10} {'n_seed':<6}")
print("  " + "-" * 90)

base = load_baseline_means("paper")
for m, r in base.items():
    print(f"  {m:<32} {abs(r['spearman_mean']):.3f} ± {r['spearman_std']:.3f}     "
          f"{r['ndcg_mean']:.3f}         {r['p10_mean']:.2f}       {r['n']}")

x1 = load_x1_mean("paper", 3)
if x1:
    print(f"  {'PI-SDE X1 (Φ-rank)':<32} {abs(x1['spearman_mean']):.3f} ± {x1['spearman_std']:.3f}     "
          f"{x1['ndcg_mean']:.3f}         {x1['p10_mean']:.2f}       {x1['n']}")

# X3 baseline (5 seeds)
x3b_phi, x3b_g, x3b_phi_ndcg, x3b_g_ndcg = [], [], [], []
for s in SEEDS:
    d = load_x3_baseline("paper", s)
    if d is None: continue
    r = next((rr for rr in d["results"] if rr["t"] == 3), None)
    if r:
        x3b_phi.append(r["phi_spearman_r"]); x3b_g.append(r["growth_spearman_r"])
        x3b_phi_ndcg.append(r["phi_ndcg"]); x3b_g_ndcg.append(r["growth_ndcg"])
if x3b_phi:
    print(f"  {'PI-SDE X3 baseline (Φ-rank)':<32} {abs(np.mean(x3b_phi)):.3f} ± {np.std(x3b_phi):.3f}     "
          f"{np.mean(x3b_phi_ndcg):.3f}         -          {len(x3b_phi)}")
    print(f"  {'PI-SDE X3 baseline (g_pred)':<32} {abs(np.mean(x3b_g)):.3f} ± {np.std(x3b_g):.3f}     "
          f"{np.mean(x3b_g_ndcg):.3f}         -          {len(x3b_g)}")

# X3-clean
if "paper" in domain_summary:
    ds = domain_summary["paper"]
    print(f"  {'PI-SDE X3-clean (Φ-rank)':<32} {abs(ds['phi_mean']):.3f} ± {ds['phi_std']:.3f}     "
          f"-             -          {ds['n']}")
    print(f"  {'PI-SDE X3-clean (g_pred)':<32} {abs(ds['g_mean']):.3f} ± {ds['g_std']:.3f}     "
          f"{ds['g_ndcg_mean']:.3f}         -          {ds['n']}")

# ── [D] Validity verdict ───────────────────────────────────────────
print("\n\n[D] Validity verdict (mechanical)")
print("-" * 110)
checks = []

# (1) Multi-domain Φ-rank significant
for dn, ds in domain_summary.items():
    pass_sig = ds["n_sig"] == ds["n"] and ds["n"] >= 3
    checks.append((f"{dn:<22} Φ-rank significant in all {ds['n']} seeds",
                   pass_sig, f"{ds['n_sig']}/{ds['n']}"))

# (2) X3-clean g_pred outperforms best external baseline
if "paper" in domain_summary and base:
    best_baseline = max(abs(r["spearman_mean"]) for r in base.values())
    x3c_g = abs(domain_summary["paper"]["g_mean"])
    checks.append((f"X3-clean g_pred > 2× best external baseline (paper)",
                   x3c_g > 2 * best_baseline, f"{x3c_g:.3f} vs {best_baseline:.3f}"))

# (3) X3-clean Φ-rank competitive with X1
if x1 and "paper" in domain_summary:
    x1_phi = abs(x1["spearman_mean"])
    x3c_phi = abs(domain_summary["paper"]["phi_mean"])
    checks.append((f"X3-clean Φ-rank ≥ 0.8× X1 (paper)",
                   x3c_phi >= 0.8 * x1_phi, f"{x3c_phi:.3f} vs {x1_phi:.3f}"))

# (4) Leave-one-out Φ-rank significant on majority of seeds
if test_phi_sps:
    n_sig_lo = sum(1 for p in test_phi_ps if p < 0.05)
    checks.append((f"Leave-one-out Φ-Spearman significant in ≥3/5 seeds",
                   n_sig_lo >= 3, f"{n_sig_lo}/{len(test_phi_ps)}"))

# (5) Leave-one-out g-rank holds
if test_g_sps:
    g_mean_lo = abs(np.mean(test_g_sps))
    checks.append((f"Leave-one-out g_pred mean |ρ| > 0.5 (real prediction)",
                   g_mean_lo > 0.5, f"{g_mean_lo:.3f}"))

n_pass = sum(1 for _, ok, _ in checks if ok)
n_total = len(checks)
print(f"\n  Total: {n_pass}/{n_total} checks passed\n")
for name, ok, val in checks:
    mark = "✅" if ok else "❌"
    print(f"  {mark}  {name:<70}  →  {val}")

print()
if n_pass == n_total:
    print(f"  Verdict: ✅✅✅  EFFECTIVENESS FULLY CONFIRMED ({n_pass}/{n_total})")
elif n_pass >= n_total - 1:
    print(f"  Verdict: ✅  Mostly confirmed ({n_pass}/{n_total}, 1 weak)")
elif n_pass >= n_total * 0.7:
    print(f"  Verdict: ⚠   Partially confirmed ({n_pass}/{n_total})")
else:
    print(f"  Verdict: ❌  NOT confirmed ({n_pass}/{n_total} only)")
