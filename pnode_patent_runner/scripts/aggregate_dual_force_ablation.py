#!/usr/bin/env python3
"""
Dual-Force v2 の5設定アブレーション集計:
  raw / learnable / zscore / learnable_renorm / zscore_renorm
を construction / agrifood それぞれで3 seed平均し、
Gate S (GEM full, gem_skill.json) と Gate L 実績 (TAP-NODE+burst) に対して比較する。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from scipy import stats

DF_DIR = Path("pnode_patent_runner/outputs/dual_force_patent")
TAP_DIR = Path("pnode_patent_runner/outputs/tap_node_patent")
GEM_JSON = Path("pnode_patent_runner/outputs/predictability_map/gem_skill.json")

CONFIGS = ["raw", "learnable", "zscore", "learnable_renorm", "zscore_renorm"]
SEEDS = [42, 7, 123]
DOMAINS = ["construction", "agrifood"]


def load_auc(path: Path, key: str) -> float:
    d = json.load(open(path, encoding="utf-8"))
    return d[key]


def main() -> None:
    gem = json.load(open(GEM_JSON, encoding="utf-8"))

    tap_aucs = {}
    for dom in DOMAINS:
        files = sorted(TAP_DIR.glob(f"tap_node_burst_{dom}_seed*.json"))
        aucs = [load_auc(f, "tap_node_final_val_auc") for f in files]
        tap_aucs[dom] = aucs

    for dom in DOMAINS:
        gem_full = gem["results"][dom]["variants"]["gem_full"]["auc_mean"]
        relatedness = gem["results"][dom]["ceilings"]["relatedness"]
        tap_mean = statistics.mean(tap_aucs[dom]) if tap_aucs[dom] else float("nan")
        tap_sd = statistics.pstdev(tap_aucs[dom]) if len(tap_aucs[dom]) > 1 else 0.0

        print(f"\n==== {dom} ====")
        print(f"  relatedness (training-free): {relatedness:.4f}")
        print(f"  Gate S (GEM full):           {gem_full:.4f}")
        print(f"  Gate L winner (TAP-NODE+burst, n={len(tap_aucs[dom])}): {tap_mean:.4f} +/- {tap_sd:.4f}")
        print(f"  {'config':<20} {'auc_mean':>10} {'auc_sd':>8} {'vs GEM':>10} {'vs TAP+burst':>14} {'p(vs GEM)':>10}")

        for cfg in CONFIGS:
            aucs = []
            for seed in SEEDS:
                f = DF_DIR / f"dual_force_{cfg}_{dom}_seed{seed}.json"
                if f.is_file():
                    aucs.append(load_auc(f, "dual_force_final_val_auc"))
            if not aucs:
                print(f"  {cfg:<20} (no data)")
                continue
            mean_auc = statistics.mean(aucs)
            sd_auc = statistics.pstdev(aucs) if len(aucs) > 1 else 0.0
            diff_gem = mean_auc - gem_full
            diff_tap = mean_auc - tap_mean
            if len(aucs) > 1:
                _, pval = stats.ttest_1samp(aucs, gem_full)
            else:
                pval = float("nan")
            print(
                f"  {cfg:<20} {mean_auc:>10.4f} {sd_auc:>8.4f} {diff_gem:>+10.4f} {diff_tap:>+14.4f} {pval:>10.4f}"
            )


if __name__ == "__main__":
    main()
