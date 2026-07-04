"""
X1 + Growth Head 3 domain × 5 seed × {alltime, leaveout最終年} の集約。

集計対象:
  - Paper (ArXiv CS): leaveout t=3
  - Patent Energy top-50: (alltime のみ、leaveout 未実施)
  - arXiv Construction v2: leaveout t=10

出力指標:
  - alltime: R², MSE, MAE, Spearman(ĝ, g), Spearman(Φ, g)
  - leaveout test: 同上 (真の未来予測精度)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

DOMAINS = {
    "Paper (ArXiv CS)": {
        "root": Path("RESULTS/PNode_Paper_X1_GROWTH/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01_GROWTH"),
        "last_t": 3,
        "leaveout_t": 3,
    },
    "Patent Energy": {
        "root": Path("RESULTS/PNode_Patent_Energy_X1_top50_GROWTH/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01_GROWTH"),
        "last_t": 11,
        "leaveout_t": 11,
    },
    "arXiv Construction": {
        "root": Path("RESULTS/PNode_ArXiv_Construction_X1_v2_GROWTH/softplus-400_400-0.5-const-0.1-0.1-0.005-x1_v1.0_g0.1_b0.01_GROWTH"),
        "last_t": 10,
        "leaveout_t": 10,
    },
}
SEEDS = [0, 1, 42, 123, 999]


def load_results(root, seed, cond):
    p = root / f"seed_{seed}" / cond / "evaluation_growth.json"
    if not p.exists():
        return None
    return json.load(p.open())


def aggregate(results_list, t_filter=None, split_filter=None):
    if not results_list:
        return None
    rs = []
    for d in results_list:
        for r in d["results"]:
            if t_filter is not None and r["t"] != t_filter:
                continue
            if split_filter is not None and r["split"] != split_filter:
                continue
            rs.append(r)
    if not rs:
        return None
    return {
        "r2":          (np.mean([r["r2"] for r in rs]),       np.std([r["r2"] for r in rs])),
        "mse":         (np.mean([r["mse"] for r in rs]),      np.std([r["mse"] for r in rs])),
        "mae":         (np.mean([r["mae"] for r in rs]),      np.std([r["mae"] for r in rs])),
        "sp_g":        (np.mean([r["spearman_growth"] for r in rs]), np.std([r["spearman_growth"] for r in rs])),
        "sp_phi":      (np.mean([r["spearman_phi"] for r in rs]),    np.std([r["spearman_phi"] for r in rs])),
        "n":           len(rs),
    }


def main():
    print("=" * 100)
    print("  X1 + Growth Head  3 Domain × 5 Seed  Comparison")
    print("=" * 100)

    summary = {}
    for dname, cfg in DOMAINS.items():
        print(f"\n{'='*100}")
        print(f"  {dname}")
        print(f"{'='*100}")

        # alltime - 最終時点
        alltime_results = [load_results(cfg["root"], s, "alltime") for s in SEEDS]
        alltime_results = [r for r in alltime_results if r]
        print(f"\n  [alltime t={cfg['last_t']}] ({len(alltime_results)}/5 seeds)")
        ag_at = aggregate(alltime_results, t_filter=cfg["last_t"])
        if ag_at:
            print(f"    R²:    {ag_at['r2'][0]:+.4f} ± {ag_at['r2'][1]:.4f}")
            print(f"    MSE:   {ag_at['mse'][0]:.4f} ± {ag_at['mse'][1]:.4f}")
            print(f"    MAE:   {ag_at['mae'][0]:.4f} ± {ag_at['mae'][1]:.4f}")
            print(f"    Sp(ĝ): {ag_at['sp_g'][0]:+.4f} ± {ag_at['sp_g'][1]:.4f}")
            print(f"    Sp(Φ): {ag_at['sp_phi'][0]:+.4f} ± {ag_at['sp_phi'][1]:.4f}")

        # leaveout - test split (真の未来予測)
        leaveout_results = [load_results(cfg["root"], s, f"leaveout{cfg['leaveout_t']}") for s in SEEDS]
        leaveout_results = [r for r in leaveout_results if r]
        print(f"\n  [leaveout{cfg['leaveout_t']} test split] ({len(leaveout_results)}/5 seeds)")
        ag_lo = aggregate(leaveout_results, t_filter=cfg["leaveout_t"], split_filter="test")
        if ag_lo:
            print(f"    R²:    {ag_lo['r2'][0]:+.4f} ± {ag_lo['r2'][1]:.4f}")
            print(f"    MSE:   {ag_lo['mse'][0]:.4f} ± {ag_lo['mse'][1]:.4f}")
            print(f"    MAE:   {ag_lo['mae'][0]:.4f} ± {ag_lo['mae'][1]:.4f}")
            print(f"    Sp(ĝ): {ag_lo['sp_g'][0]:+.4f} ± {ag_lo['sp_g'][1]:.4f}")
            print(f"    Sp(Φ): {ag_lo['sp_phi'][0]:+.4f} ± {ag_lo['sp_phi'][1]:.4f}")
            # 統計検定
            try:
                sp_g_vals = [r["spearman_growth"] for d in leaveout_results for r in d["results"]
                              if r["t"] == cfg["leaveout_t"] and r["split"] == "test"]
                sp_phi_vals = [r["spearman_phi"] for d in leaveout_results for r in d["results"]
                                if r["t"] == cfg["leaveout_t"] and r["split"] == "test"]
                _, p_sp_g = stats.wilcoxon(sp_g_vals, alternative="greater")
                _, p_sp_phi = stats.wilcoxon(sp_phi_vals, alternative="less")
                print(f"    Wilcoxon Sp(ĝ) > 0: p = {p_sp_g:.4f}")
                print(f"    Wilcoxon Sp(Φ) < 0: p = {p_sp_phi:.4f}")
            except Exception as e:
                print(f"    Wilcoxon failed: {e}")
        else:
            print("    (no leaveout data)")

        summary[dname] = {"alltime": ag_at, "leaveout": ag_lo}

    # 3 ドメイン比較表
    print("\n\n" + "=" * 100)
    print("  3 ドメイン比較 (mean ± std)")
    print("=" * 100)
    print(f"\n{'Domain':<22} {'split':<10} {'R²':<22} {'MSE':<18} {'Sp(ĝ)':<22} {'Sp(Φ)':<22}")
    print("-" * 116)
    for dname, ag in summary.items():
        for split, key in [("alltime", "alltime"), ("test", "leaveout")]:
            data = ag[key]
            if not data: continue
            print(f"{dname:<22} {split:<10} "
                  f"{data['r2'][0]:+.3f}±{data['r2'][1]:.3f}      "
                  f"{data['mse'][0]:.3f}±{data['mse'][1]:.3f}    "
                  f"{data['sp_g'][0]:+.3f}±{data['sp_g'][1]:.3f}      "
                  f"{data['sp_phi'][0]:+.3f}±{data['sp_phi'][1]:.3f}")

    # JSON 保存
    out = Path("RESULTS/aggregated_x1_growth_3domain.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({k: {
        "alltime": {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in v["alltime"].items()} if v.get("alltime") else None,
        "leaveout": {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in v["leaveout"].items()} if v.get("leaveout") else None,
    } for k, v in summary.items()}, out.open("w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
