"""
A1 Ablation 集約: L_X1 の各成分 (val, grad, basin) の寄与を測定。

4 設定 × 3 ドメイン × 5 seed:
  A (vanilla):       v=0,   g=0,   b=0
  B (val only):      v=1.0, g=0,   b=0
  C (val + grad):    v=1.0, g=0.1, b=0
  D (full X1):       v=1.0, g=0.1, b=0.01
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

DOMAINS = {
    "Paper":               ("PNode_Paper_X1", 3),
    "Patent Energy":       ("PNode_Patent_Energy_X1_top50", 11),
    "arXiv Construction":  ("PNode_ArXiv_Construction_X1_v2", 10),
}
SEEDS = [0, 1, 42, 123, 999]

SETTINGS = [
    ("A (vanilla)",  "0.0",  "0.0",  "0.0"),
    ("B (val)",      "1.0",  "0.0",  "0.0"),
    ("C (val+grad)", "1.0",  "0.1",  "0.0"),
    ("D (full X1)",  "1.0",  "0.1",  "0.01"),
]


def load_results(dataset_name, seed, lam_v, lam_g, lam_b):
    # ディレクトリ名は -x1_v{V}_g{G}_b{B} で終わる (substring 一致を避ける)
    out_root = Path(f"RESULTS/{dataset_name}")
    suffix = f"-x1_v{lam_v}_g{lam_g}_b{lam_b}"
    for p in out_root.rglob("evaluation_x1.json"):
        path_str = str(p)
        # parent name は seed_*  alltime のように 2 階層上が ablation tag
        parent_parts = list(p.parents)
        # 通常: .../tag_dir/seed_X/alltime/evaluation_x1.json
        if len(parent_parts) < 3: continue
        tag_dir = parent_parts[2].name   # 3 上 = tag ディレクトリ
        if not tag_dir.endswith(suffix): continue
        if f"seed_{seed}" not in path_str: continue
        if "/alltime/" not in path_str: continue
        return json.load(p.open())
    return None


def main():
    print("=" * 110)
    print("  A1 Ablation: L_X1 各成分の寄与")
    print("=" * 110)

    for dname, (root_name, last_t) in DOMAINS.items():
        print(f"\n{'='*110}")
        print(f"  {dname}  (last_t={last_t})")
        print(f"{'='*110}")
        print(f"  {'Setting':<14} {'Spearman(Φ)':<22} {'NDCG@10':<14} {'P@10':<12} {'n_seed':<6} {'Wilcoxon p':<10}")
        print("  " + "-" * 95)

        for label, lv, lg, lb in SETTINGS:
            sp_vals, sp_p, ndcg, p10 = [], [], [], []
            for s in SEEDS:
                d = load_results(root_name, s, lv, lg, lb)
                if d is None:
                    continue
                r_last = next((r for r in d["results"] if r["t"] == last_t), None)
                if r_last is None:
                    continue
                sp_vals.append(r_last["spearman_r"])
                sp_p.append(r_last["spearman_p"])
                ndcg.append(r_last["ndcg"])
                p10.append(r_last["prec_at_10"])

            if not sp_vals:
                print(f"  {label:<14} (no data)")
                continue

            n_sig = sum(1 for p in sp_p if p < 0.05)
            try:
                _, p_wilc = stats.wilcoxon(sp_vals, alternative="less")
            except Exception:
                p_wilc = float("nan")

            sp_str = f"{np.mean(sp_vals):+.4f}±{np.std(sp_vals):.4f}"
            ndcg_str = f"{np.mean(ndcg):.3f}±{np.std(ndcg):.3f}"
            p10_str = f"{np.mean(p10):.2f}±{np.std(p10):.2f}"
            print(f"  {label:<14} {sp_str:<22} {ndcg_str:<14} {p10_str:<12} {len(sp_vals)}/5    "
                  f"{p_wilc:.4f}{'*' if p_wilc < 0.05 else ''}")

    print("\n" + "=" * 110)
    print("  解釈:")
    print("=" * 110)
    print("""
  期待される結果:
    A (vanilla):  Spearman ≈ 0  (X1 効果なし → 順位無相関)
    B (val):      強い負相関   (主成分)
    C (val+grad): やや改善     (臨界点制約で安定化)
    D (full):    最良          (谷形状制約で完成)

  もし D > C > B >> A なら → 3 成分全て効果あり (best)
  もし B ≈ D なら          → grad/basin は冗長 (simplify 可)
""")


if __name__ == "__main__":
    main()
