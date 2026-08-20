#!/usr/bin/env python3
"""Gate L 集計: TAP-NODE(+burst) の 10 seed holdout AUC を Gate S（gem_skill.json）と比較する。"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from scipy import stats

OUT_DIR = Path("pnode_patent_runner/outputs/tap_node_patent")
GEM_JSON = Path("pnode_patent_runner/outputs/predictability_map/gem_skill.json")


def main() -> None:
    gem = json.load(open(GEM_JSON, encoding="utf-8"))
    for dom in ("construction", "agrifood"):
        files = sorted(OUT_DIR.glob(f"tap_node_burst_{dom}_seed*.json"))
        aucs = []
        cs = []
        for f in files:
            d = json.load(open(f, encoding="utf-8"))
            aucs.append(d["tap_node_final_val_auc"])
            cs.append(d["learned_scalars"]["c"])
        if not aucs:
            print(f"{dom}: no results found in {OUT_DIR}")
            continue
        mean_auc = statistics.mean(aucs)
        sd_auc = statistics.pstdev(aucs)
        gem_full = gem["results"][dom]["variants"]["gem_full"]["auc_mean"]
        relatedness = gem["results"][dom]["ceilings"]["relatedness"]
        ceil_max = gem["results"][dom]["ceiling_max"]
        # 対応なし t 検定（TAP-NODE 10 seed の分散のみ既知。Gate S は決定的特徴+ロジスティック回帰10 seedの平均のみ保存）
        tstat, pval = stats.ttest_1samp(aucs, gem_full)
        print(f"==== {dom} (n={len(aucs)} seeds) ====")
        print(f"  TAP-NODE+burst holdout AUC: {mean_auc:.4f} ± {sd_auc:.4f}  (range {min(aucs):.4f}-{max(aucs):.4f})")
        print(f"  learned c (burst interaction): mean={statistics.mean(cs):.4f}, sd={statistics.pstdev(cs):.4f}")
        print(f"  Gate S (GEM full, rel+mom+mom*burst+seen): {gem_full:.4f}")
        print(f"  relatedness (training-free): {relatedness:.4f}")
        print(f"  ceiling_max (popularity/seen/rel/seen+pop best): {ceil_max:.4f}")
        print(f"  TAP-NODE+burst vs GEM full: diff={mean_auc - gem_full:+.4f}, one-sample t-test p={pval:.4f}")
        verdict = "GATE L PASS" if (mean_auc > gem_full and pval < 0.05) else "GATE L NOT PASSED"
        print(f"  => {verdict}")
        print()


if __name__ == "__main__":
    main()
