"""
Aggregate baseline_all.py results across 3 domains and N seeds into a
publication-ready comparison table.

Output: stdout (markdown + ASCII) and CSV file.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

import numpy as np

ROOT = Path("/home/nakamuraroi/kumagai")
DOMAINS = [
    ("patent_energy",       "PNode_Patent_Energy_X1_top50"),
    ("arxiv_construction",  "PNode_ArXiv_Construction_X1_v2"),
    ("jp_construction",     "PNode_JP_Construction_X1"),
]
METHODS = ["persistence", "mean", "linear", "arima", "ets", "chronos", "moirai", "timesfm"]
METHOD_LABEL = {
    "persistence": "Persistence",
    "mean":        "Mean",
    "linear":      "Linear OLS",
    "arima":       "ARIMA",
    "ets":         "ETS",
    "chronos":     "Chronos-Bolt-S",
    "moirai":      "Moirai-2-R-S",
    "timesfm":     "TimesFM-2.0-500m",
}


def load_domain(data_name: str):
    f = ROOT / f"RESULTS_TSFM_BASELINE/{data_name}/all_methods/split70/evaluation_all.json"
    if not f.exists():
        return None
    return json.load(f.open())


def collect_metric(domain_data, method: str, metric: str = "spearman_r"):
    """Return list of per-seed values (mean across test_t).  If deterministic, returns single-elem list."""
    if method not in domain_data["results"]:
        return None
    runs = domain_data["results"][method]["runs"]
    out = []
    for r in runs:
        vals = [t[metric] for t in r["per_t"]]
        out.append(mean(vals))
    return out


def fmt(values, k=3):
    """Format mean ± std for the multi-seed list, or just the value if 1 elem."""
    if values is None or len(values) == 0:
        return "—"
    if len(values) == 1:
        return f"{values[0]:+.{k}f}"
    return f"{mean(values):+.{k}f}±{stdev(values):.{k}f}"


def main():
    # Collect mean Spearman per domain x method (averaged across test_t, then over seeds)
    rho_table = defaultdict(dict)        # rho_table[method][domain] = (mean, std)
    ndcg_table = defaultdict(dict)
    mse_table = defaultdict(dict)
    sig_count_table = defaultdict(dict)  # how many seed × test_t are p<0.05

    domain_status = {}
    for short, data_name in DOMAINS:
        d = load_domain(data_name)
        if d is None:
            print(f"  [WARN] missing: {data_name}")
            domain_status[short] = "missing"
            continue
        domain_status[short] = f"seeds={d['seeds']}, test_t={d['test_t']}"
        for m in METHODS:
            rho_seeds  = collect_metric(d, m, "spearman_r")
            ndcg_seeds = collect_metric(d, m, "ndcg")
            mse_seeds  = collect_metric(d, m, "mse_norm")
            rho_table[m][short]  = rho_seeds
            ndcg_table[m][short] = ndcg_seeds
            mse_table[m][short]  = mse_seeds
            # Count significant negative + significant positive across seeds and test_t
            sig_pos = sig_neg = total = 0
            for r in d["results"][m]["runs"]:
                for tres in r["per_t"]:
                    total += 1
                    if tres["spearman_p"] < 0.05:
                        if tres["spearman_r"] > 0:
                            sig_pos += 1
                        else:
                            sig_neg += 1
            sig_count_table[m][short] = (sig_pos, sig_neg, total)

    print("=" * 88)
    print("  STATUS")
    print("=" * 88)
    for short, _ in DOMAINS:
        print(f"  {short:<20} {domain_status[short]}")
    print()

    # ── Markdown table 1: Spearman ρ ─────────────────────────────────
    print("=" * 88)
    print("  TABLE 1: Mean Spearman ρ (across test_t, then ± across seeds)")
    print("=" * 88)
    header = ["Method"] + [s for s, _ in DOMAINS] + ["3-domain avg"]
    widths = [18, 18, 18, 18, 14]
    print("| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for m in METHODS:
        row = [METHOD_LABEL[m]]
        avg_means = []
        for short, _ in DOMAINS:
            seeds = rho_table[m].get(short)
            row.append(fmt(seeds, k=3))
            if seeds:
                avg_means.append(mean(seeds))
        if avg_means:
            row.append(f"{mean(avg_means):+.3f}")
        else:
            row.append("—")
        print("| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |")
    print()

    # ── Significance count ───────────────────────────────────────────
    print("=" * 88)
    print("  TABLE 2: Significant (p<0.05) test_t count across seeds × test_t  [pos / neg / total]")
    print("=" * 88)
    print("| " + " | ".join(h.ljust(w) for h, w in zip(header[:-1], widths[:-1])) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths[:-1]) + "|")
    for m in METHODS:
        row = [METHOD_LABEL[m]]
        for short, _ in DOMAINS:
            cnt = sig_count_table[m].get(short)
            if cnt is None:
                row.append("—")
            else:
                row.append(f"{cnt[0]} / {cnt[1]} / {cnt[2]}")
        print("| " + " | ".join(v.ljust(w) for v, w in zip(row, widths[:-1])) + " |")
    print()

    # ── NDCG ─────────────────────────────────────────────────────────
    print("=" * 88)
    print("  TABLE 3: Mean NDCG@10 (across test_t, then ± across seeds)")
    print("=" * 88)
    print("| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for m in METHODS:
        row = [METHOD_LABEL[m]]
        avg_means = []
        for short, _ in DOMAINS:
            seeds = ndcg_table[m].get(short)
            row.append(fmt(seeds, k=3))
            if seeds:
                avg_means.append(mean(seeds))
        row.append(f"{mean(avg_means):.3f}" if avg_means else "—")
        print("| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |")
    print()

    # ── CSV ──────────────────────────────────────────────────────────
    csv_path = ROOT / "RESULTS_TSFM_BASELINE/comparison_table.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as f:
        f.write("method,metric,domain,seed,mean_value\n")
        for m in METHODS:
            for short, data_name in DOMAINS:
                d = load_domain(data_name)
                if d is None or m not in d["results"]:
                    continue
                for r in d["results"][m]["runs"]:
                    for metric in ["spearman_r", "ndcg", "mse_norm", "mae_norm"]:
                        v = mean(t[metric] for t in r["per_t"])
                        f.write(f"{m},{metric},{short},{r['seed']},{v}\n")
    print(f"Saved CSV -> {csv_path}")


if __name__ == "__main__":
    main()
