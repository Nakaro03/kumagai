"""
PI-SDE 論文ドメイン 5 seed × 4 condition の結果集約。

入力: RESULTS/PNode_Paper/.../seed_{S}/{alltime|leaveout_K}/evaluation.json
出力: 集約テーブル + プロット
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_ROOT = Path("RESULTS/PNode_Paper/softplus-400_400-0.5-const-0.1-0.1-0.005")
SEEDS = [0, 1, 42, 123, 999]
CONDITIONS = ["alltime", "leaveout1", "leaveout2", "leaveout3"]


def load_eval(seed, cond):
    sub = f"seed_{seed}/{cond}"
    # nested directory bug fallback
    for path_try in [
        RESULTS_ROOT / sub / "evaluation.json",
        RESULTS_ROOT / sub / RESULTS_ROOT.name.replace("/", "_") / sub / "evaluation.json",
    ]:
        if path_try.exists():
            return json.load(path_try.open())
    # broad search
    for p in RESULTS_ROOT.rglob("evaluation.json"):
        if f"seed_{seed}" in str(p) and f"/{cond}/" in str(p):
            return json.load(p.open())
    return None


def main():
    print("=" * 90)
    print("  PI-SDE Paper Domain  5 seed × 4 condition Sinkhorn distance summary")
    print("=" * 90)

    all_data = {}
    for cond in CONDITIONS:
        per_t = {}
        for seed in SEEDS:
            d = load_eval(seed, cond)
            if d is None:
                continue
            for r in d["results"]:
                t = r["t"]
                key = (t, r["split"])
                per_t.setdefault(key, {"pi_sde": [], "naive": [], "last_seen": []})
                per_t[key]["pi_sde"].append(r["pi_sde"])
                per_t[key]["naive"].append(r["naive"])
                if r.get("last_seen") == r.get("last_seen"):  # not nan
                    per_t[key]["last_seen"].append(r["last_seen"])
        all_data[cond] = per_t

    # 表示
    for cond in CONDITIONS:
        print(f"\n[ {cond} ]")
        per_t = all_data[cond]
        if not per_t:
            print("  (no data)")
            continue
        print(f"  {'t':<4} {'split':<8} {'PI-SDE':<22} {'Naive':<22} {'Last-seen':<22} {'n_seed':<7}")
        print("  " + "-" * 90)
        for (t, split), v in sorted(per_t.items()):
            ps = np.array(v["pi_sde"]); na = np.array(v["naive"]); ls = np.array(v["last_seen"]) if v["last_seen"] else np.array([])
            ps_str = f"{ps.mean():.3f} ± {ps.std():.3f}"
            na_str = f"{na.mean():.3f} ± {na.std():.3f}"
            ls_str = f"{ls.mean():.3f} ± {ls.std():.3f}" if ls.size > 0 else "N/A"
            try:
                _, p = stats.wilcoxon(ps - na, alternative="less")
                sig = "*" if p < 0.05 else ""
            except Exception:
                p, sig = float("nan"), ""
            print(f"  {t:<4} {split:<8} {ps_str:<22} {na_str:<22} {ls_str:<22} {ps.size:<7}  vs-Naive p={p:.3f}{sig}")

    # 全 leaveout の test split を集めて統合検定
    print("\n" + "=" * 90)
    print("  Leaveout test 集約 (全 leaveout_{1,2,3} の test split)")
    print("=" * 90)
    test_pi  = []
    test_na  = []
    test_ls  = []
    for cond in ["leaveout1", "leaveout2", "leaveout3"]:
        per_t = all_data.get(cond, {})
        for (t, split), v in per_t.items():
            if split == "test":
                test_pi.extend(v["pi_sde"])
                test_na.extend(v["naive"])
                test_ls.extend(v["last_seen"])
    if test_pi:
        ps = np.array(test_pi); na = np.array(test_na); ls = np.array(test_ls) if test_ls else np.array([])
        print(f"  PI-SDE :   {ps.mean():.3f} ± {ps.std():.3f}   (n={ps.size})")
        print(f"  Naive  :   {na.mean():.3f} ± {na.std():.3f}")
        if ls.size > 0:
            print(f"  Last sn:   {ls.mean():.3f} ± {ls.std():.3f}")
        try:
            _, pn = stats.wilcoxon(ps - na, alternative="less")
            print(f"  Wilcoxon PI-SDE < Naive    : p = {pn:.4f}")
            if ls.size > 0:
                _, pl = stats.wilcoxon(ps - ls, alternative="less")
                print(f"  Wilcoxon PI-SDE < Last-seen: p = {pl:.4f}")
        except Exception as e:
            print(f"  検定失敗: {e}")

    # 保存 + プロット
    out_summary = Path("RESULTS/PNode_Paper/aggregated_summary.json")
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {c: {f"{t}_{s}": {k: [float(x) for x in vv] for k, vv in v.items()}
              for (t, s), v in per_t.items()}
          for c, per_t in all_data.items()},
        out_summary.open("w"), indent=2,
    )
    print(f"\nSaved -> {out_summary}")

    # プロット: leaveout_3 (= 未来予測) を強調
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, cond in zip(axes, CONDITIONS):
        per_t = all_data.get(cond, {})
        if not per_t:
            ax.set_title(f"{cond} (no data)"); continue
        ts = sorted([t for (t, _) in per_t.keys()])
        means_p, stds_p = [], []; means_n, stds_n = [], []; means_l, stds_l = [], []
        splits = []
        for t in ts:
            keys = [(t, s) for (tt, s) in per_t.keys() if tt == t]
            if not keys: continue
            t_split = keys[0][1]
            v = per_t[keys[0]]
            means_p.append(np.mean(v["pi_sde"])); stds_p.append(np.std(v["pi_sde"]))
            means_n.append(np.mean(v["naive"]));  stds_n.append(np.std(v["naive"]))
            if v["last_seen"]:
                means_l.append(np.mean(v["last_seen"])); stds_l.append(np.std(v["last_seen"]))
            else:
                means_l.append(float("nan")); stds_l.append(float("nan"))
            splits.append(t_split)
        x = np.arange(len(ts))
        w = 0.27
        ax.bar(x - w, means_p, w, yerr=stds_p, label="PI-SDE",  color="#3b82f6", capsize=3)
        ax.bar(x,     means_n, w, yerr=stds_n, label="Naive",   color="#9ca3af", capsize=3)
        ax.bar(x + w, means_l, w, yerr=stds_l, label="Last-seen",color="#fbbf24", capsize=3)
        # test split highlight
        for i, sp in enumerate(splits):
            if sp == "test":
                ax.axvspan(i - 0.5, i + 0.5, color="#fef3c7", alpha=0.4, zorder=0)
        ax.set_xticks(x); ax.set_xticklabels(ts)
        ax.set_title(f"{cond}", fontsize=10)
        ax.set_xlabel("t"); ax.set_ylabel("Sinkhorn (lower = better)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"PI-SDE on ArXiv CS Papers  |  {len(SEEDS)} seeds  |  yellow shade = test split",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    plot_path = Path("RESULTS/PNode_Paper/aggregated_plot.png")
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    print(f"Saved -> {plot_path}")


if __name__ == "__main__":
    main()
