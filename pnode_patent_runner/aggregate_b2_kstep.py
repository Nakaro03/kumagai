"""
B2 k-step ahead 集約: Patent + JP × (k=1, 2, 3) × 5 seed.

k=1 は既存の leaveout (last t のみ除外)
k=2 は leaveout (T-2, T-1)
k=3 は leaveout (T-3, T-2, T-1)

評価: 最終時点 (T-1) の Spearman を 5 seed で集約 + Wilcoxon.
"""
from __future__ import annotations

import json, glob
from pathlib import Path

import numpy as np
from scipy import stats

DOMAINS = {
    "Patent Energy": {
        "root": "RESULTS/PNode_Patent_Energy_X1_top50",
        "TRAIN_T": list(range(1, 12)),    # t=1..11, last=11
        "kstep_tags": {1: "leaveout11", 2: "leaveout10_11", 3: "leaveout9_10_11"},
    },
    "JP Construction": {
        "root": "RESULTS/PNode_JP_Construction_X1",
        "TRAIN_T": list(range(1, 11)),    # t=1..10, last=10
        "kstep_tags": {1: "leaveout10", 2: "leaveout9_10", 3: "leaveout8_9_10"},
    },
}
SEEDS = [0, 1, 42, 123, 999]


def gather(root, tag, last_t):
    """Get Spearman at test split for last_t across seeds."""
    sps, p10s, ndcgs = [], [], []
    for s in SEEDS:
        files = list(Path(root).rglob(f"*_v1.0_g0.1_b0.01/seed_{s}/{tag}/evaluation_x1.json"))
        if not files: continue
        d = json.load(open(files[0]))
        r = next((r for r in d["results"] if r["t"] == last_t and r.get("split") == "test"), None)
        if r is None: continue
        sps.append(r["spearman_r"])
        p10s.append(r["prec_at_10"])
        ndcgs.append(r["ndcg"])
    return np.array(sps), np.array(p10s), np.array(ndcgs)


def main():
    print("=" * 88)
    print("  B2 k-step ahead long-horizon prediction (X1 PI-SDE)")
    print("=" * 88)

    for dname, cfg in DOMAINS.items():
        print(f"\n{'='*88}")
        print(f"  {dname}  (TRAIN_T: t={cfg['TRAIN_T']})")
        print(f"{'='*88}")
        print(f"  {'k':<4} {'leaveout tag':<22} {'eval t':<8} {'Spearman':<22} {'NDCG@10':<12} {'P@10':<8} {'Wilcoxon p':<10}")
        print("  " + "-" * 86)

        last_t = cfg["TRAIN_T"][-1]
        for k, tag in cfg["kstep_tags"].items():
            sps, p10s, ndcgs = gather(cfg["root"], tag, last_t)
            if len(sps) == 0:
                print(f"  k={k:<3} {tag:<22} t={last_t:<5}  (no data)")
                continue
            sp_str = f"{np.mean(sps):+.4f}±{np.std(sps):.4f}"
            nd_str = f"{np.mean(ndcgs):.3f}±{np.std(ndcgs):.3f}"
            p10_str = f"{np.mean(p10s):.2f}±{np.std(p10s):.2f}"
            try:
                _, wp = stats.wilcoxon(sps, alternative="less")
                wp_str = f"{wp:.4f}{'*' if wp < 0.05 else ''}"
            except Exception:
                wp_str = "n/a"
            print(f"  k={k:<3} {tag:<22} t={last_t:<5}  {sp_str:<22} {nd_str:<12} {p10_str:<8} {wp_str}")

    print(f"\n{'='*88}")
    print("  解釈:")
    print(f"{'='*88}")
    print("""
  k=1: 1-step ahead = 通常の leaveout (last year 除外, last year 予測)
  k=2: 2-step ahead = 直近 2 年除外, 直近年予測 (より難しい)
  k=3: 3-step ahead = 直近 3 年除外, 直近年予測 (更に難しい)

  期待:
    k ↑ で性能 (|Spearman|) 低下するはず
    全 k で Wilcoxon p < 0.05 を維持できれば、long-horizon robustness 主張可能
""")


if __name__ == "__main__":
    main()
