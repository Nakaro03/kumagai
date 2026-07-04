"""
A2 集約: baselines (ARIMA / LSTM / Transformer / Linear / Naive) vs PI-SDE+X1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DOMAINS = ["paper", "patent_energy_top50", "arxiv_construction"]
DOMAIN_LABELS = {
    "paper": "Paper (ArXiv CS)",
    "patent_energy_top50": "Patent Energy",
    "arxiv_construction": "arXiv Construction",
}
SEEDS = [0, 1, 42, 123, 999]
METHODS = ["Naive_zero", "Naive_mean", "Naive_lastg", "Linear", "ARIMA", "LSTM", "Transformer"]


def gather(domain: str, method: str, metric: str):
    vals = []
    for s in SEEDS:
        p = Path(f"RESULTS/baselines/{domain}/baselines_seed{s}.json")
        if not p.exists(): continue
        d = json.load(p.open())
        if method in d and metric in d[method] and d[method][metric] == d[method][metric]:
            vals.append(d[method][metric])
    return vals


def load_x1(domain: str):
    """X1 (full) の 5 seed 結果"""
    # 設定 D (full X1) の結果から alltime last_t を取得
    last_t = {"paper": 3, "patent_energy_top50": 11, "arxiv_construction": 10}[domain]
    root_map = {
        "paper": "PNode_Paper_X1",
        "patent_energy_top50": "PNode_Patent_Energy_X1_top50",
        "arxiv_construction": "PNode_ArXiv_Construction_X1_v2",
    }
    out_root = Path(f"RESULTS/{root_map[domain]}")
    sp, nd, p10 = [], [], []
    for s in SEEDS:
        for p in out_root.rglob("evaluation_x1.json"):
            tag_dir = list(p.parents)[2].name if len(p.parents) >= 3 else ""
            if not tag_dir.endswith("-x1_v1.0_g0.1_b0.01"): continue
            if f"seed_{s}" not in str(p) or "/alltime/" not in str(p): continue
            d = json.load(p.open())
            r = next((r for r in d["results"] if r["t"] == last_t), None)
            if r is None: continue
            sp.append(r["spearman_r"])
            nd.append(r["ndcg"])
            p10.append(r["prec_at_10"])
            break
    return sp, nd, p10


def main():
    print("=" * 110)
    print("  A2 Baseline Comparison: Time-series methods vs PI-SDE + X1")
    print("=" * 110)

    for domain in DOMAINS:
        print(f"\n{'='*110}")
        print(f"  {DOMAIN_LABELS[domain]}")
        print(f"{'='*110}")
        print(f"  {'Method':<14} {'MSE':<18} {'MAE':<14} {'Spearman':<22} {'NDCG@10':<14} {'P@10':<10}")
        print("  " + "-" * 95)

        for method in METHODS:
            mses = gather(domain, method, "mse")
            maes = gather(domain, method, "mae")
            sps  = gather(domain, method, "spearman_r")
            sp_ps = gather(domain, method, "spearman_p")
            nds  = gather(domain, method, "ndcg_at_10")
            ps   = gather(domain, method, "prec_at_10")
            if not sps:
                continue
            n_sig = sum(1 for p in sp_ps if p < 0.05)
            mse_s = f"{np.mean(mses):.3f}±{np.std(mses):.2f}" if mses else "-"
            mae_s = f"{np.mean(maes):.3f}±{np.std(maes):.2f}" if maes else "-"
            sp_s  = f"{np.mean(sps):+.3f}±{np.std(sps):.3f} ({n_sig}/5*)"
            nd_s  = f"{np.mean(nds):.3f}±{np.std(nds):.3f}" if nds else "-"
            ps_s  = f"{np.mean(ps):.2f}±{np.std(ps):.2f}" if ps else "-"
            print(f"  {method:<14} {mse_s:<18} {mae_s:<14} {sp_s:<22} {nd_s:<14} {ps_s:<10}")

        # PI-SDE + X1 (full)
        x1_sp, x1_nd, x1_p10 = load_x1(domain)
        if x1_sp:
            sp_s = f"{np.mean(x1_sp):+.3f}±{np.std(x1_sp):.3f}"
            nd_s = f"{np.mean(x1_nd):.3f}±{np.std(x1_nd):.3f}"
            ps_s = f"{np.mean(x1_p10):.2f}±{np.std(x1_p10):.2f}"
            print(f"  {'X1 PI-SDE':<14} {'(no MSE)':<18} {'-':<14} {sp_s:<22} {nd_s:<14} {ps_s:<10}")

    print("\n注:")
    print("  - 全 baseline は input=counts[0..T-2] で count[T-1] を予測 (no data leak)")
    print("  - X1 PI-SDE は alltime fit (Sinkhorn matching on all years)")
    print("  - 公平比較には leaveout が必要 (A3)")


if __name__ == "__main__":
    main()
