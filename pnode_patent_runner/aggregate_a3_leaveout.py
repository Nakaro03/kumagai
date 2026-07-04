"""
A3 集約: X1 leaveout (5 seed × 3 domain) vs A2 baselines (no-leak future prediction)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

DOMAINS = {
    "paper":               {"label": "Paper",            "root": "PNode_Paper_X1",                 "last_t": 3},
    "patent_energy_top50": {"label": "Patent Energy",    "root": "PNode_Patent_Energy_X1_top50",   "last_t": 11},
    "arxiv_construction":  {"label": "arXiv Construction","root": "PNode_ArXiv_Construction_X1_v2","last_t": 10},
    "jp_construction":     {"label": "JP Construction",   "root": "PNode_JP_Construction_X1",       "last_t": 10},
}
SEEDS = [0, 1, 42, 123, 999]
BASELINE_METHODS = ["Naive_lastg", "Linear", "ARIMA", "LSTM", "Transformer"]


def gather_baseline(domain, method, metric):
    vals = []
    for s in SEEDS:
        p = Path(f"RESULTS/baselines/{domain}/baselines_seed{s}.json")
        if not p.exists(): continue
        d = json.load(p.open())
        if method in d and metric in d[method] and d[method][metric] == d[method][metric]:
            vals.append(d[method][metric])
    return vals


def gather_x1_leaveout(domain):
    cfg = DOMAINS[domain]
    last_t = cfg["last_t"]
    out_root = Path(f"RESULTS/{cfg['root']}")
    sp_test, nd_test, p10_test = [], [], []
    sp_p = []
    for s in SEEDS:
        for p in out_root.rglob("evaluation_x1.json"):
            tag_dir = list(p.parents)[2].name if len(p.parents) >= 3 else ""
            if not tag_dir.endswith("-x1_v1.0_g0.1_b0.01"): continue
            if f"seed_{s}" not in str(p): continue
            if f"/leaveout{last_t}/" not in str(p): continue
            d = json.load(p.open())
            r = next((r for r in d["results"] if r["t"] == last_t and r.get("split") == "test"), None)
            if r is None: continue
            sp_test.append(r["spearman_r"])
            sp_p.append(r["spearman_p"])
            nd_test.append(r["ndcg"])
            p10_test.append(r["prec_at_10"])
            break
    return sp_test, sp_p, nd_test, p10_test


def main():
    print("=" * 110)
    print("  A3 Comparison: X1 PI-SDE leaveout (true future) vs Baselines (also no-leak)")
    print("=" * 110)

    for domain, cfg in DOMAINS.items():
        print(f"\n{'='*110}")
        print(f"  {cfg['label']} - leaveout t={cfg['last_t']} (held-out)")
        print(f"{'='*110}")
        print(f"  {'Method':<14} {'Spearman':<26} {'NDCG@10':<14} {'P@10':<10} {'n_seed':<6}")
        print("  " + "-" * 80)

        # Baselines
        for m in BASELINE_METHODS:
            sps = gather_baseline(domain, m, "spearman_r")
            sp_ps = gather_baseline(domain, m, "spearman_p")
            nds = gather_baseline(domain, m, "ndcg_at_10")
            p10s = gather_baseline(domain, m, "prec_at_10")
            if not sps: continue
            n_sig = sum(1 for p in sp_ps if p < 0.05)
            sp_s = f"{np.mean(sps):+.3f}±{np.std(sps):.3f} ({n_sig}/5*)"
            nd_s = f"{np.mean(nds):.3f}±{np.std(nds):.3f}" if nds else "-"
            ps_s = f"{np.mean(p10s):.2f}±{np.std(p10s):.2f}" if p10s else "-"
            print(f"  {m:<14} {sp_s:<26} {nd_s:<14} {ps_s:<10} {len(sps)}/5")

        # X1 leaveout
        sp_t, sp_p_t, nd_t, p10_t = gather_x1_leaveout(domain)
        if sp_t:
            n_sig = sum(1 for p in sp_p_t if p < 0.05)
            sp_s = f"{np.mean(sp_t):+.3f}±{np.std(sp_t):.3f} ({n_sig}/5*)"
            nd_s = f"{np.mean(nd_t):.3f}±{np.std(nd_t):.3f}"
            ps_s = f"{np.mean(p10_t):.2f}±{np.std(p10_t):.2f}"
            try:
                _, wp = stats.wilcoxon(sp_t, alternative="less")
            except Exception:
                wp = float("nan")
            print(f"  {'X1 PI-SDE':<14} {sp_s:<26} {nd_s:<14} {ps_s:<10} {len(sp_t)}/5  (Wilcoxon p={wp:.4f})")
        else:
            print(f"  X1 PI-SDE  (not yet)")


if __name__ == "__main__":
    main()
