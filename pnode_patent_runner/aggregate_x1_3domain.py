"""
3 ドメイン (Paper / Patent Energy / Patent Construction) × 5 seed の X1 結果集約。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

DOMAIN_ROOTS = {
    "Paper":               Path("RESULTS/PNode_Paper_X1/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01"),
    "Patent Energy":       Path("RESULTS/PNode_Patent_Energy_X1_top50/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01"),
    "Patent Construction": Path("RESULTS/PNode_Patent_Construction_X1_top50/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01"),
}
SEEDS = [0, 1, 42, 123, 999]


def load_eval(root, seed, cond="alltime"):
    for p in root.rglob("evaluation_x1.json"):
        if f"seed_{seed}" in str(p) and f"/{cond}/" in str(p):
            return json.load(p.open())
    return None


def main():
    print("=" * 100)
    print("  X1 (Topic-Anchor) 3 domain × 5 seed × alltime — 最終時点 (t=last) サマリ")
    print("=" * 100)

    summary = {}
    for dname, root in DOMAIN_ROOTS.items():
        print(f"\n[ {dname} ]")
        last_t_per_seed = {}    # {seed: result at last t}
        n_topics = None
        for s in SEEDS:
            d = load_eval(root, s, "alltime")
            if d is None:
                print(f"  seed {s}: (no data)")
                continue
            n_topics = max(d["results"], key=lambda r: r["n_active"])["n_active"]
            # 最終 t の結果のみ取り出す
            last_r = max(d["results"], key=lambda r: r["t"])
            last_t_per_seed[s] = last_r

        if not last_t_per_seed:
            continue

        last_t = next(iter(last_t_per_seed.values()))["t"]
        sp = [r["spearman_r"] for r in last_t_per_seed.values()]
        sp_p = [r["spearman_p"] for r in last_t_per_seed.values()]
        ndcg = [r["ndcg"] for r in last_t_per_seed.values()]
        p10 = [r["prec_at_10"] for r in last_t_per_seed.values()]
        sink = [r["sinkhorn"] for r in last_t_per_seed.values()]

        n_sig = sum(1 for p in sp_p if p < 0.05)
        n_neg = sum(1 for r in sp if r < 0)
        try:
            w, p = stats.wilcoxon(sp, alternative="less")
        except Exception:
            p = float("nan")

        print(f"  n_topics={n_topics}  t=last={last_t}  n_seed={len(sp)}")
        print(f"  Spearman r  : {np.mean(sp):+.4f} ± {np.std(sp):.4f}  ({n_neg}/{len(sp)} neg, {n_sig}/{len(sp)} p<0.05)")
        print(f"  NDCG@10     : {np.mean(ndcg):.4f} ± {np.std(ndcg):.4f}")
        print(f"  Precision@10: {np.mean(p10):.2f} ± {np.std(p10):.2f}")
        print(f"  Sinkhorn    : {np.mean(sink):.4f} ± {np.std(sink):.4f}")
        print(f"  Wilcoxon p (Sp<0): {p:.4f}  {'✅' if p<0.05 else '❌'}")

        summary[dname] = {
            "n_topics": n_topics, "last_t": last_t, "n_seed": len(sp),
            "spearman": {"mean": float(np.mean(sp)), "std": float(np.std(sp)),
                          "values": sp, "n_neg": n_neg, "n_sig": n_sig},
            "ndcg":     {"mean": float(np.mean(ndcg)), "std": float(np.std(ndcg)), "values": ndcg},
            "p10":      {"mean": float(np.mean(p10)),  "std": float(np.std(p10)),  "values": p10},
            "sinkhorn": {"mean": float(np.mean(sink)), "std": float(np.std(sink)), "values": sink},
            "wilcoxon_p": float(p),
        }

    # 全 t 集約: 各 domain × seed × t を統合
    print("\n" + "=" * 100)
    print("  全 t × 全 seed × 全 domain 統合検定")
    print("=" * 100)
    all_sp_train, all_sp_test = [], []
    for dname, root in DOMAIN_ROOTS.items():
        for s in SEEDS:
            d = load_eval(root, s, "alltime")
            if d is None: continue
            for r in d["results"]:
                arr = all_sp_test if r.get("split") == "test" else all_sp_train
                arr.append(r["spearman_r"])
    if all_sp_train:
        m, std = np.mean(all_sp_train), np.std(all_sp_train)
        try:
            _, p = stats.wilcoxon(all_sp_train, alternative="less")
        except:
            p = float("nan")
        n_neg = sum(1 for v in all_sp_train if v < 0)
        print(f"  Train全体 (n={len(all_sp_train)}): Spearman {m:+.4f} ± {std:.4f}, {n_neg} negative, Wilcoxon p={p:.6f}")

    # 保存
    out = Path("RESULTS/aggregated_x1_3domain.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, out.open("w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
