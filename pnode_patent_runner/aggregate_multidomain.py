"""
マルチドメイン × 5 seed の結果集約。

入力: pnode_patent_runner/outputs/multidomain_trend/trend_<domain>_a_y2010-2021_seed<N>.json

出力:
  - aggregated_multidomain.json
  - aggregated_multidomain_table.txt   (paper-ready table)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

OUTDIR = Path("pnode_patent_runner/outputs/multidomain_trend")
DOMAINS = ["energy", "agrifood", "construction", "pharma", "semiconductor", "computing"]
SEEDS   = [0, 1, 42, 123, 999]
YEAR_RANGE = (2010, 2021)
TAG     = "_a"


def load_runs(domain: str, key: str = "pnode_pc"):
    out = []
    for s in SEEDS:
        f = OUTDIR / f"trend_{domain}{TAG}_y{YEAR_RANGE[0]}-{YEAR_RANGE[1]}_seed{s}.json"
        if not f.exists():
            continue
        d = json.load(f.open())
        for r in d["results"]:
            if r["key"] == key:
                r["seed"] = s
                out.append(r)
    return out


def stat(vals):
    arr = np.array([v for v in vals if v == v])
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.std()), len(arr)


def main():
    print("=" * 90)
    print(f"  Multi-domain trend benchmark集約 (PC-PNODE A+BEF, n_seeds=5, years {YEAR_RANGE[0]}-{YEAR_RANGE[1]})")
    print("=" * 90)
    print(f"\n  {'Domain':<14} {'n_topics':<10} {'Spearman r':<22} {'Wilcoxon':<10} {'sig 0.05':<9} {'NDCG@10':<14}  {'Link-AUC':<14}")
    print("  " + "-" * 96)

    summary = {}
    for d in DOMAINS:
        runs = load_runs(d)
        if not runs:
            print(f"  {d:<14} (no data)")
            continue

        n_topics = runs[0].get("n_topics", "?")
        sp_vals = [r["spearman_r"] for r in runs if r["spearman_r"] == r["spearman_r"]]
        sp_p    = [r["spearman_p"] for r in runs if r["spearman_p"] == r["spearman_p"]]
        link_v  = [r["link_auc"]   for r in runs]
        ndcg_v  = [r["ndcg_at_10"] for r in runs]

        sp_mean, sp_std, _ = stat(sp_vals)
        link_mean, link_std, _ = stat(link_v)
        ndcg_mean, ndcg_std, _ = stat(ndcg_v)
        n_sig = sum(1 for p in sp_p if p < 0.05)

        try:
            w, wp = stats.wilcoxon(sp_vals, alternative="less")
        except Exception:
            wp = float("nan")

        sp_str = f"{sp_mean:+.3f} ± {sp_std:.3f}"
        link_str = f"{link_mean:.3f} ± {link_std:.3f}"
        ndcg_str = f"{ndcg_mean:.3f} ± {ndcg_std:.3f}"

        print(f"  {d:<14} {n_topics:<10} {sp_str:<22} p={wp:.4f}  {n_sig}/5      {ndcg_str:<14}  {link_str:<14}")

        summary[d] = {
            "n_topics": n_topics,
            "spearman": {"mean": sp_mean, "std": sp_std, "values": sp_vals},
            "wilcoxon_p": wp, "n_sig": n_sig,
            "link_auc": {"mean": link_mean, "std": link_std},
            "ndcg":     {"mean": ndcg_mean, "std": ndcg_std},
        }

    # ── 全ドメイン統合検定 ────────────────────────────────────────────────
    all_sp = []
    for d in DOMAINS:
        runs = load_runs(d)
        all_sp.extend([r["spearman_r"] for r in runs if r["spearman_r"] == r["spearman_r"]])

    print("\n" + "=" * 90)
    print(f"  全ドメイン統合: PC-PNODE Spearman r (n = {len(all_sp)})")
    print("=" * 90)
    if len(all_sp) >= 5:
        m, s = np.mean(all_sp), np.std(all_sp)
        try:
            w, p = stats.wilcoxon(all_sp, alternative="less")
        except Exception:
            p = float("nan")
        n_neg = sum(1 for v in all_sp if v < 0)
        print(f"  mean ± std: {m:+.4f} ± {s:.4f}")
        print(f"  全体 Wilcoxon p = {p:.6f}  (one-sided, alternative=less)")
        print(f"  負の符号: {n_neg}/{len(all_sp)} ({n_neg*100/len(all_sp):.1f}%)")
        decision = "✅ H_A 採択" if p < 0.05 else "❌ H_A 棄却"
        print(f"  {decision}")

    # 保存
    out = {
        "per_domain": summary,
        "all_seeds_all_domains": {
            "n": len(all_sp), "mean": float(np.mean(all_sp)) if all_sp else float("nan"),
            "std": float(np.std(all_sp)) if all_sp else float("nan"),
            "values": all_sp,
        },
    }
    with open(OUTDIR / "aggregated_multidomain.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUTDIR / 'aggregated_multidomain.json'}")


if __name__ == "__main__":
    main()
