"""
H1 検証: 保存場 (P-NODE) vs 自由場 (Neural ODE) の多ステップ AUC 分析

入力 : run_h1_conservative_vs_free.sh が生成した JSON ファイル群
出力 :
  h1_auc_horizon.png/pdf  — AUC@k 折れ線グラフ（ドメイン別）
  h1_degradation.png/pdf  — Δ AUC(k) = AUC(k) − AUC(1) の劣化曲線
  h1_results_table.md     — 平均 ± 標準偏差の表（論文向け）
  h1_significance.csv     — Wilcoxon signed-rank 検定結果
"""

from __future__ import annotations

import argparse
import glob
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ── 設定 ──────────────────────────────────────────────────────────────────────

METHODS_ORDER = ["static", "rnn", "neural_ode", "pnode"]
METHOD_LABELS = {
    "static":     "Static",
    "rnn":        "RNN",
    "neural_ode": "Neural ODE",
    "pnode":      "P-NODE (ours)",
}
COLORS = {
    "static":     "#888888",
    "rnn":        "#f5a623",
    "neural_ode": "#4363d8",
    "pnode":      "#e6194b",
}
MARKERS = {"static": "s", "rnn": "^", "neural_ode": "D", "pnode": "o"}
LINESTYLES = {"static": "--", "rnn": "-.", "neural_ode": ":", "pnode": "-"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})


# ── データ読み込み ──────────────────────────────────────────────────────────────

def load_results(out_dir: Path) -> pd.DataFrame:
    """
    JSON → DataFrame with columns:
      domain, seed, method, k, auc, ap
    """
    records = []
    pattern = str(out_dir / "h1_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"JSON が見つかりません: {pattern}")

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)

        fname = Path(fpath).stem            # h1_energy_seed42
        parts = fname.split("_")
        # domain は seed の前まで
        seed_idx = next(i for i, p in enumerate(parts) if p.startswith("seed"))
        domain = "_".join(parts[1:seed_idx])
        seed   = int(parts[seed_idx].replace("seed", ""))

        for result in data.get("results", []):
            method = result.get("key", "")
            if method not in METHODS_ORDER:
                continue

            hg = (result.get("final_metrics_by_horizon_gap")
                  or result.get("horizon_gap_results")
                  or {})

            for k_str, metrics in hg.items():
                k = int(k_str)
                auc = metrics.get("auc", float("nan"))
                ap  = metrics.get("ap",  float("nan"))
                records.append(dict(
                    domain=domain, seed=seed, method=method,
                    k=k, auc=auc, ap=ap,
                ))

    if not records:
        raise ValueError(
            "horizon_gap_results が空です。"
            " --eval-horizon-gaps を付けて再実行してください。"
        )

    return pd.DataFrame(records)


# ── 統計処理 ─────────────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """domain × method × k ごとに mean/std を集計。"""
    agg = (
        df.groupby(["domain", "method", "k"])["auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "auc_mean", "std": "auc_std", "count": "n"})
    )
    agg_ap = (
        df.groupby(["domain", "method", "k"])["ap"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "ap_mean", "std": "ap_std"})
    )
    return agg.merge(agg_ap, on=["domain", "method", "k"])


def compute_degradation(df: pd.DataFrame) -> pd.DataFrame:
    """Δ AUC(k) = AUC(k) − AUC(1) を各 seed で計算。"""
    base = df[df["k"] == 1][["domain", "seed", "method", "auc"]].rename(
        columns={"auc": "auc_k1"}
    )
    merged = df.merge(base, on=["domain", "seed", "method"])
    merged["delta_auc"] = merged["auc"] - merged["auc_k1"]
    return merged


def wilcoxon_pairwise(df: pd.DataFrame, method_a: str, method_b: str) -> pd.DataFrame:
    """
    domain × k ごとに method_a vs method_b の Wilcoxon signed-rank test。
    """
    rows = []
    for (domain, k), grp in df.groupby(["domain", "k"]):
        a = grp[grp["method"] == method_a]["auc"].values
        b = grp[grp["method"] == method_b]["auc"].values
        if len(a) < 3 or len(b) < 3 or len(a) != len(b):
            rows.append(dict(domain=domain, k=k, stat=float("nan"), p=float("nan"),
                             a_mean=float("nan"), b_mean=float("nan"), delta=float("nan")))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stat, p = wilcoxon(a, b, alternative="two-sided")
            except ValueError:
                stat, p = float("nan"), float("nan")
        rows.append(dict(
            domain=domain, k=k,
            stat=stat, p=p,
            a_mean=a.mean(), b_mean=b.mean(),
            delta=b.mean() - a.mean(),   # P-NODE − Neural ODE
        ))
    return pd.DataFrame(rows)


# ── 可視化 ────────────────────────────────────────────────────────────────────

def plot_auc_horizon(stats: pd.DataFrame, domains: list[str], out_dir: Path):
    """ドメイン別に AUC@k の折れ線グラフを描く。"""
    n = len(domains)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), sharey=False)
    if n == 1:
        axes = [axes]

    ks = sorted(stats["k"].unique())

    for ax, domain in zip(axes, domains):
        sub = stats[stats["domain"] == domain]
        for method in METHODS_ORDER:
            ms = sub[sub["method"] == method].sort_values("k")
            if ms.empty:
                continue
            label = METHOD_LABELS.get(method, method)
            lw = 2.0 if method == "pnode" else 1.4
            ax.errorbar(
                ms["k"], ms["auc_mean"],
                yerr=ms["auc_std"],
                label=label,
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle=LINESTYLES[method],
                linewidth=lw,
                markersize=6,
                capsize=3,
            )
        ax.set_xticks(ks)
        ax.set_xlabel("Prediction horizon $k$ (years)")
        ax.set_ylabel("AUC-ROC")
        ax.set_title(f"Domain: {domain}")
        ax.legend(loc="lower left")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    fig.suptitle(
        "H1: Multi-step AUC@k — Conservative (P-NODE) vs Free-field (Neural ODE)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = out_dir / f"h1_auc_horizon.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


def plot_degradation(ddf: pd.DataFrame, domains: list[str], out_dir: Path):
    """Δ AUC(k) = AUC(k) − AUC(1) の劣化曲線。k=1 は 0 固定。"""
    agg = (
        ddf.groupby(["domain", "method", "k"])["delta_auc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    n = len(domains)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]

    ks = sorted(ddf["k"].unique())

    for ax, domain in zip(axes, domains):
        sub = agg[agg["domain"] == domain]
        for method in METHODS_ORDER:
            ms = sub[sub["method"] == method].sort_values("k")
            if ms.empty:
                continue
            lw = 2.0 if method == "pnode" else 1.4
            ax.errorbar(
                ms["k"], ms["mean"],
                yerr=ms["std"],
                label=METHOD_LABELS.get(method, method),
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle=LINESTYLES[method],
                linewidth=lw,
                markersize=6,
                capsize=3,
            )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(ks)
        ax.set_xlabel("Prediction horizon $k$ (years)")
        ax.set_ylabel("ΔAUC = AUC($k$) − AUC($k$=1)")
        ax.set_title(f"Domain: {domain}")
        ax.legend(loc="lower left")
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(
        "H1: AUC Degradation vs Horizon — Smaller drop = better multi-step stability",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = out_dir / f"h1_degradation.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ── テキスト出力 ──────────────────────────────────────────────────────────────

def save_markdown_table(stats: pd.DataFrame, sig: pd.DataFrame, out_dir: Path):
    lines = ["# H1: AUC@k — Mean ± Std (n seeds)\n"]
    for domain in sorted(stats["domain"].unique()):
        lines.append(f"\n## Domain: {domain}\n")
        sub   = stats[stats["domain"] == domain]
        ks    = sorted(sub["k"].unique())
        # header
        header = "| Method | " + " | ".join(f"AUC k={k}" for k in ks) + " |"
        sep    = "|--------|" + "|".join(["--------"] * len(ks)) + "|"
        lines += [header, sep]
        for method in METHODS_ORDER:
            row = sub[sub["method"] == method].sort_values("k")
            if row.empty:
                continue
            cells = []
            for _, r in row.iterrows():
                s = f"{r['auc_mean']:.4f} ±{r['auc_std']:.4f}"
                # 有意差マーク (* p<0.05)
                sv = sig[
                    (sig["domain"] == domain) & (sig["k"] == r["k"])
                ]
                if not sv.empty and sv.iloc[0]["p"] < 0.05:
                    s += " *"
                cells.append(s)
            label = METHOD_LABELS.get(method, method)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("\n`*` p < 0.05 (Wilcoxon signed-rank, P-NODE vs Neural ODE)\n")

    path = out_dir / "h1_results_table.md"
    path.write_text("\n".join(lines))
    print(f"Saved → {path}")


def save_significance_csv(sig: pd.DataFrame, out_dir: Path):
    path = out_dir / "h1_significance.csv"
    sig.to_csv(path, index=False, float_format="%.4f")
    print(f"Saved → {path}")


def print_summary(sig: pd.DataFrame):
    print("\n" + "=" * 60)
    print(" H1 仮説検定結果: P-NODE vs Neural ODE")
    print("=" * 60)
    print(f"{'Domain':15s} {'k':>3}  {'NeuralODE':>10}  {'P-NODE':>10}  {'Δ(+好)':>8}  {'p':>7}  判定")
    print("-" * 60)
    for _, row in sig.sort_values(["domain", "k"]).iterrows():
        verdict = "✓ H1支持" if (row["delta"] > 0 and row["p"] < 0.05) else \
                  "△ 差なし" if row["p"] >= 0.05 else \
                  "✗ H1棄却"
        print(f"{row['domain']:15s} {int(row['k']):>3}  "
              f"{row['a_mean']:10.4f}  {row['b_mean']:10.4f}  "
              f"{row['delta']:+8.4f}  {row['p']:7.4f}  {verdict}")
    print("=" * 60)
    print()


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="pnode_patent_runner/outputs/h1_conservative_vs_free")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise SystemExit(f"出力ディレクトリが見つかりません: {out_dir}")

    print(f"Loading JSON from {out_dir} ...")
    df = load_results(out_dir)
    print(f"  {len(df)} records  "
          f"domains={sorted(df['domain'].unique())}  "
          f"seeds={sorted(df['seed'].unique())}  "
          f"horizons={sorted(df['k'].unique())}")

    domains = sorted(df["domain"].unique())
    stats   = compute_stats(df)
    ddf     = compute_degradation(df)
    sig     = wilcoxon_pairwise(df, method_a="neural_ode", method_b="pnode")

    plot_auc_horizon(stats, domains, out_dir)
    plot_degradation(ddf, domains, out_dir)
    save_markdown_table(stats, sig, out_dir)
    save_significance_csv(sig, out_dir)
    print_summary(sig)


if __name__ == "__main__":
    main()
