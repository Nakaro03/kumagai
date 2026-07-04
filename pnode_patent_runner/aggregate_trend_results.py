"""
全実験結果を集約して論文用テーブルとプロットを生成。

入力:
  - trend_benchmark_seed{0,1,42,123,999}.json     (PC-PNODE 5 seed)
  - timeseries_baselines_seed{0,1,42,123,999}.json (時系列ベースライン)

出力:
  - aggregated_table.txt            (Markdown 表)
  - aggregated_plot.png             (mean ± std バープロット)
  - aggregated_results.json         (raw)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUTDIR = Path("pnode_patent_runner/outputs/trend_benchmark")
SEEDS  = [0, 1, 42, 123, 999]


def load_pnode_results(tag: str = ""):
    """PC-PNODE 系の結果を集約。tag='' (元) or '_bef' (B+E+F 改善版)"""
    by_method = {"static": [], "neural_ode": [], "pnode": [], "pnode_pc": []}
    for s in SEEDS:
        f = OUTDIR / f"trend_benchmark{tag}_seed{s}.json"
        if not f.exists():
            print(f"  warning: {f} 未存在 (skip)")
            continue
        d = json.load(f.open())
        for r in d["results"]:
            by_method[r["key"]].append({"seed": s, **r})
    return by_method


def load_timeseries_results():
    """時系列ベースライン (naive, arima, lstm, transformer)"""
    by_method = {"naive": [], "arima": [], "lstm": [], "transformer": []}
    for s in SEEDS:
        f = OUTDIR / f"timeseries_baselines_seed{s}.json"
        if not f.exists():
            print(f"  warning: {f} 未存在 (skip)")
            continue
        d = json.load(f.open())
        for r in d["results"]:
            if "error" in r: continue
            by_method[r["method"]].append({"seed": s, **r})
    return by_method


def stats_summary(values, name=""):
    arr = np.array([v for v in values if v == v])
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.std()), len(arr)


def main():
    pnode_res = load_pnode_results()
    ts_res    = load_timeseries_results()

    print("\n" + "=" * 80)
    print("  PC-PNODE & 時系列ベースライン  集約結果 (mean ± std over seeds)")
    print("=" * 80)

    # ── PC-PNODE 系 ──────────────────────────────────────────────────────────
    print("\n[グラフベース手法]")
    print(f"  {'手法':<12} {'Link-AUC':>14} {'Entry-AUC':>14} {'Spearman(Φ,g)':>17} {'NDCG@10':>14}")
    print("  " + "-" * 73)
    summary_pnode = {}
    for key, runs in pnode_res.items():
        if not runs:
            continue
        link  = stats_summary([r.get("link_auc") for r in runs])
        entry = stats_summary([r.get("entry_auc") for r in runs])
        sp    = stats_summary([r.get("spearman_r") for r in runs])
        ndcg  = stats_summary([r.get("ndcg_at_10") for r in runs])
        sp_p_vals = [r.get("spearman_p") for r in runs if r.get("spearman_p") == r.get("spearman_p")]
        sig_count = sum(1 for p in sp_p_vals if p < 0.05)

        spr_str = f"{sp[0]:+.3f}±{sp[1]:.3f}" if sp[2] > 0 else "    N/A    "
        if sig_count > 0:
            spr_str += f" ({sig_count}/{len(sp_p_vals)}*)"
        print(f"  {key:<12} {link[0]:>7.4f}±{link[1]:.3f}  {entry[0]:>7.4f}±{entry[1]:.3f}  "
              f"{spr_str:>17} {ndcg[0]:>7.3f}±{ndcg[1]:.3f}")
        summary_pnode[key] = {
            "link_auc":   {"mean": link[0],  "std": link[1],  "n": link[2]},
            "entry_auc":  {"mean": entry[0], "std": entry[1], "n": entry[2]},
            "spearman_r": {"mean": sp[0],    "std": sp[1],    "n": sp[2], "n_sig": sig_count, "n_total": len(sp_p_vals)},
            "ndcg":       {"mean": ndcg[0],  "std": ndcg[1],  "n": ndcg[2]},
        }

    # ── 時系列ベースライン ────────────────────────────────────────────────────
    print("\n[時系列ベースライン (グラフ構造を使わない)]")
    print(f"  {'手法':<12} {'MSE':>14} {'MAE':>14} {'DirAcc':>14} {'NDCG@10':>14}")
    print("  " + "-" * 70)
    summary_ts = {}
    for key, runs in ts_res.items():
        if not runs:
            continue
        mse  = stats_summary([r.get("mse") for r in runs])
        mae  = stats_summary([r.get("mae") for r in runs])
        da   = stats_summary([r.get("dir_acc") for r in runs])
        ndcg = stats_summary([r.get("ndcg_at_10") for r in runs])
        print(f"  {key:<12} {mse[0]:>7.3f}±{mse[1]:.3f}  {mae[0]:>7.3f}±{mae[1]:.3f}  "
              f"{da[0]:>7.3f}±{da[1]:.3f}  {ndcg[0]:>7.3f}±{ndcg[1]:.3f}")
        summary_ts[key] = {
            "mse":  {"mean": mse[0],  "std": mse[1],  "n": mse[2]},
            "mae":  {"mean": mae[0],  "std": mae[1],  "n": mae[2]},
            "dir_acc":  {"mean": da[0],   "std": da[1],   "n": da[2]},
            "ndcg": {"mean": ndcg[0], "std": ndcg[1], "n": ndcg[2]},
        }

    # ── H_A 仮説検定: PC-PNODE の Spearman r が 0 より有意に小さいか ─────────
    print("\n" + "=" * 80)
    print("  H_A 仮説検定: Spearman(Φ, g) < 0  [Wilcoxon 符号順位検定]")
    print("=" * 80)
    sp_vals = [r.get("spearman_r") for r in pnode_res.get("pnode_pc", [])
               if r.get("spearman_r") == r.get("spearman_r")]
    if len(sp_vals) >= 3:
        try:
            stat, p = stats.wilcoxon(sp_vals, alternative="less")
            print(f"  PC-PNODE Spearman 値 (n={len(sp_vals)}): {[f'{v:+.3f}' for v in sp_vals]}")
            print(f"  Wilcoxon W={stat:.2f}, p={p:.4f}")
            decision = "✅ H_A 採択 (Spearman < 0 が有意)" if p < 0.05 else "❌ H_A 棄却 (有意でない)"
            print(f"  {decision}")
        except Exception as e:
            print(f"  検定失敗: {e}")
    else:
        print(f"  サンプル不足 (n={len(sp_vals)})")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    out = {
        "pnode_summary": summary_pnode,
        "timeseries_summary": summary_ts,
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
    }
    with open(OUTDIR / "aggregated_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUTDIR / 'aggregated_results.json'}")

    # ── プロット ──────────────────────────────────────────────────────────────
    methods_g  = ["static", "neural_ode", "pnode", "pnode_pc"]
    methods_ts = ["naive", "arima", "lstm", "transformer"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # [A] NDCG@10 比較
    ax = axes[0]
    labels, means, stds, colors = [], [], [], []
    for m in methods_g:
        if m in summary_pnode and summary_pnode[m]["ndcg"]["n"] > 0:
            labels.append(m + " (graph)")
            means.append(summary_pnode[m]["ndcg"]["mean"])
            stds.append(summary_pnode[m]["ndcg"]["std"])
            colors.append("#3b82f6" if m == "pnode_pc" else "#cbd5e1")
    for m in methods_ts:
        if m in summary_ts and summary_ts[m]["ndcg"]["n"] > 0:
            labels.append(m + " (TS)")
            means.append(summary_ts[m]["ndcg"]["mean"])
            stds.append(summary_ts[m]["ndcg"]["std"])
            colors.append("#fbbf24")
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="black", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("NDCG@10")
    ax.set_title("NDCG@10 (higher = better)\nmean ± std over seeds")
    ax.grid(alpha=0.3, axis="y")

    # [B] Spearman r (PC-PNODE only, others N/A)
    ax = axes[1]
    pcp = summary_pnode.get("pnode_pc", {}).get("spearman_r", {})
    if pcp.get("n", 0) > 0:
        ax.bar([0], [pcp["mean"]], yerr=[pcp["std"]], color="#3b82f6",
               edgecolor="black", capsize=6, width=0.5)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.axhline(-0.15, color="red", lw=0.6, ls=":", label="threshold -0.15")
        ax.set_xticks([0]); ax.set_xticklabels(["PC-PNODE"])
        ax.set_ylabel("Spearman r")
        ax.set_title(f"Spearman(Φ, g) over {pcp['n']} seeds\n"
                     f"mean={pcp['mean']:+.3f} ± {pcp['std']:.3f}\n"
                     f"{pcp.get('n_sig', 0)}/{pcp.get('n_total', 0)} seeds significant (p<0.05)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    # [C] 時系列ベースラインの MSE 比較
    ax = axes[2]
    labels, means, stds = [], [], []
    for m in methods_ts:
        if m in summary_ts and summary_ts[m]["mse"]["n"] > 0:
            labels.append(m); means.append(summary_ts[m]["mse"]["mean"])
            stds.append(summary_ts[m]["mse"]["std"])
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color="#fbbf24", edgecolor="black", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("MSE")
    ax.set_title("Time-series baselines: MSE (lower = better)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"PC-PNODE vs Baselines on ArXiv CS  |  {len(SEEDS)} seeds × 4 graph methods × 4 TS methods",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTDIR / "aggregated_plot.png", dpi=140, bbox_inches="tight")
    print(f"Saved -> {OUTDIR / 'aggregated_plot.png'}")


if __name__ == "__main__":
    main()
