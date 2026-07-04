"""
B1 λ_X1 sensitivity sweep 集約: {0.1, 0.5, 1.0, 2.0, 5.0} × 3 domain × 5 seed
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DOMAINS = {
    "Paper":               ("PNode_Paper_X1", 3),
    "Patent Energy":       ("PNode_Patent_Energy_X1_top50", 11),
    "arXiv Construction":  ("PNode_ArXiv_Construction_X1_v2", 10),
}
SEEDS = [0, 1, 42, 123, 999]
LAMBDAS = [0.1, 0.5, 1.0, 2.0, 5.0]


def load_results(root_name: str, seed: int, lam_x1: float):
    """tag suffix:
      λ_X1=1.0 → "-x1_v1.0_g0.1_b0.01"
      λ_X1≠1.0 → "-x1_v1.0_g0.1_b0.01_lx{L}"
    """
    if lam_x1 == 1.0:
        suffix = "-x1_v1.0_g0.1_b0.01"
    else:
        suffix = f"-x1_v1.0_g0.1_b0.01_lx{lam_x1}"

    out_root = Path(f"RESULTS/{root_name}")
    for p in out_root.rglob("evaluation_x1.json"):
        parents = list(p.parents)
        if len(parents) < 3: continue
        tag = parents[2].name
        if not tag.endswith(suffix): continue
        # exclude α-tagged variants (which add _a{...} after _lx)
        if "_a" in tag.split(suffix)[-1]: continue
        if f"seed_{seed}" not in str(p): continue
        if "/alltime/" not in str(p): continue
        return json.load(p.open())
    return None


def main():
    print("=" * 110)
    print("  B1 λ_X1 Sensitivity Sweep  (3 domain × 5 seed × 5 λ values)")
    print("=" * 110)

    for dname, (root_name, last_t) in DOMAINS.items():
        print(f"\n{'='*110}")
        print(f"  {dname}  (last_t={last_t})")
        print(f"{'='*110}")
        print(f"  {'λ_X1':<8} {'Spearman':<22} {'NDCG@10':<14} {'P@10':<12} {'n_seed':<6}")
        print("  " + "-" * 80)

        for L in LAMBDAS:
            sp, nd, p10 = [], [], []
            for s in SEEDS:
                d = load_results(root_name, s, L)
                if d is None: continue
                r = next((r for r in d["results"] if r["t"] == last_t), None)
                if r is None: continue
                sp.append(r["spearman_r"])
                nd.append(r["ndcg"])
                p10.append(r["prec_at_10"])
            if not sp:
                print(f"  {L:<8} (no data)")
                continue
            sp_s = f"{np.mean(sp):+.4f}±{np.std(sp):.4f}"
            nd_s = f"{np.mean(nd):.3f}±{np.std(nd):.3f}"
            ps_s = f"{np.mean(p10):.2f}±{np.std(p10):.2f}"
            print(f"  {L:<8} {sp_s:<22} {nd_s:<14} {ps_s:<12} {len(sp)}/5")

    print("\n" + "=" * 110)
    print("  解釈:")
    print("=" * 110)
    print("""
  λ_X1 が小さい (0.1):  X1 寄与弱 → vanilla PI-SDE に近い
  λ_X1 が中程度 (1.0):  default (A1 で best)
  λ_X1 が大きい (5.0):  X1 過剰 → Sinkhorn matching 妨害の可能性

  期待されるパターン:
    Spearman は λ_X1 ↑ で改善 (or 飽和) するが、5.0 で破綻するか?
    NDCG/P@10 も同様。robustness 確認が目的。
""")


if __name__ == "__main__":
    main()
