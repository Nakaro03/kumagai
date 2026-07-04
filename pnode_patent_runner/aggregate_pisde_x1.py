"""
PI-SDE + X1 の 5 seed × 4 condition 結果集約。
vanilla PI-SDE との比較表を生成。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

X1_ROOT      = Path("RESULTS/PNode_Paper_X1/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01")
VANILLA_ROOT = Path("RESULTS/PNode_Paper/softplus-400_400-0.5-const-0.1-0.1-0.005")
SEEDS = [0, 1, 42, 123, 999]
CONDITIONS = ["alltime", "leaveout1", "leaveout2", "leaveout3"]


def load_eval(root, seed, cond, fname="evaluation.json"):
    pat = "evaluation_x1.json" if root == X1_ROOT else fname
    for p in root.rglob(pat):
        if f"seed_{seed}" in str(p) and f"/{cond}/" in str(p):
            return json.load(p.open())
    return None


def main():
    print("=" * 95)
    print(f"  PI-SDE + X1 (Topic-Anchor) vs Vanilla PI-SDE  |  5 seed × 4 condition")
    print("=" * 95)

    summary_rows = []
    for cond in CONDITIONS:
        print(f"\n[ {cond} ]")
        print(f"  {'t':<4} {'split':<8} | {'Sinkhorn':<22} | {'Spearman r':<22} | {'NDCG@10':<14} | {'P@10':<10}")
        print("  " + "-" * 90)
        # vanilla 集計
        van_results = {}
        for s in SEEDS:
            d = load_eval(VANILLA_ROOT, s, cond)
            if d:
                for r in d["results"]:
                    van_results.setdefault((r["t"], r["split"]), {"pi_sde": [], "sp": [], "ndcg": [], "p10": []})
                    van_results[(r["t"], r["split"])]["pi_sde"].append(r["pi_sde"])
        # X1 集計
        x1_results = {}
        for s in SEEDS:
            d = load_eval(X1_ROOT, s, cond)
            if d:
                for r in d["results"]:
                    key = (r["t"], r["split"])
                    x1_results.setdefault(key, {"sink": [], "sp": [], "sp_p": [], "ndcg": [], "p10": []})
                    x1_results[key]["sink"].append(r["sinkhorn"])
                    x1_results[key]["sp"].append(r["spearman_r"])
                    x1_results[key]["sp_p"].append(r["spearman_p"])
                    x1_results[key]["ndcg"].append(r["ndcg"])
                    x1_results[key]["p10"].append(r["prec_at_10"])

        if not x1_results and not van_results:
            print("  (no data yet)")
            continue

        for (t, split) in sorted(set(list(x1_results.keys()) + list(van_results.keys()))):
            van = van_results.get((t, split), {})
            x1  = x1_results.get((t, split), {})
            # vanilla
            van_sink_str = f"{np.mean(van['pi_sde']):.3f}±{np.std(van['pi_sde']):.3f}" if van.get('pi_sde') else "  -  "
            # X1
            if x1.get('sink'):
                x1_sink_str = f"{np.mean(x1['sink']):.3f}±{np.std(x1['sink']):.3f}"
                x1_sp = np.array(x1['sp'])
                x1_sp_str = f"{x1_sp.mean():+.3f}±{x1_sp.std():.3f}"
                n_sig = sum(1 for p in x1['sp_p'] if p < 0.05)
                x1_sp_str += f" ({n_sig}/5*)"
                x1_ndcg = np.mean(x1['ndcg'])
                x1_ndcg_std = np.std(x1['ndcg'])
                x1_ndcg_str = f"{x1_ndcg:.3f}±{x1_ndcg_std:.3f}"
                x1_p10 = np.mean(x1['p10'])
                x1_p10_str = f"{x1_p10:.2f}"
            else:
                x1_sink_str, x1_sp_str, x1_ndcg_str, x1_p10_str = " - ", " - ", " - ", " - "

            # 表示
            split_mark = "★test" if split == "test" else "train"
            print(f"  {t:<4} {split_mark:<8} |")
            print(f"     vanilla     | Sink={van_sink_str}")
            print(f"     X1 (新規)  | Sink={x1_sink_str}  Sp={x1_sp_str}  NDCG={x1_ndcg_str}  P@10={x1_p10_str}")

            summary_rows.append({
                "cond": cond, "t": t, "split": split,
                "vanilla_sink": np.mean(van['pi_sde']) if van.get('pi_sde') else None,
                "x1_sink": np.mean(x1['sink']) if x1.get('sink') else None,
                "x1_spearman": np.mean(x1['sp']) if x1.get('sp') else None,
                "x1_spearman_std": np.std(x1['sp']) if x1.get('sp') else None,
                "x1_ndcg": np.mean(x1['ndcg']) if x1.get('ndcg') else None,
                "x1_p10": np.mean(x1['p10']) if x1.get('p10') else None,
            })

    # ── 統合検定: leaveout test を全部集めて統計的有意性 ──────────
    print("\n" + "=" * 95)
    print("  X1 統合検定 (leaveout test): Wilcoxon  Spearman r < 0")
    print("=" * 95)
    all_sp = []; all_p = []
    for cond in ["leaveout1", "leaveout2", "leaveout3"]:
        for s in SEEDS:
            d = load_eval(X1_ROOT, s, cond)
            if d:
                for r in d["results"]:
                    if r["split"] == "test":
                        all_sp.append(r["spearman_r"])
                        all_p.append(r["spearman_p"])
    if all_sp:
        arr = np.array(all_sp)
        n_neg = (arr < 0).sum()
        n_sig = sum(1 for p in all_p if p < 0.05)
        try:
            w, p = stats.wilcoxon(arr, alternative="less")
        except Exception:
            p = float("nan")
        print(f"  n={len(arr)}, mean ± std: {arr.mean():+.4f} ± {arr.std():.4f}")
        print(f"  全部負方向: {n_neg}/{len(arr)}")
        print(f"  p<0.05 個別有意: {n_sig}/{len(arr)}")
        print(f"  Wilcoxon p = {p:.6f}")
        print(f"  {'✅ X1 → H_A 採択 (Spearman < 0 有意)' if p < 0.05 else '❌'}")

    # JSON 保存
    out = Path("RESULTS/PNode_Paper_X1/aggregated_x1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": summary_rows, "all_leaveout_test_spearman": all_sp}, out.open("w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
