"""aggregate_x5_full.py — aggregate full X5 sweep (3 domain × 5 seed × LOTO).

Reads RESULTS_X5/{DATA}/{ABL?}/seed_{S}/{loto|alltime}/evaluation.json and
writes a markdown table + CSV summary.

Usage:
  python pnode_patent_runner/aggregate_x5_full.py                       # full sweep (no ABL prefix)
  python pnode_patent_runner/aggregate_x5_full.py --ablation A0          # ablation A0 only
  python pnode_patent_runner/aggregate_x5_full.py --out RESULTS_X5/SUMMARY.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

ROOT = Path("/home/nakamuraroi/kumagai")

DOMAINS = [
    ("patent_energy",      "PNode_Patent_Energy_X1_top50"),
    ("arxiv_construction", "PNode_ArXiv_Construction_X1_v2"),
    ("jp_construction",    "PNode_JP_Construction_X1"),
]

METRICS = ["w1_marginal", "mmd_rbf", "hits_at_10", "mrr", "ap", "ndcg_at_10", "spearman"]
HIGHER_IS_BETTER = {"hits_at_10", "mrr", "ap", "ndcg_at_10", "spearman", "spearman_r"}


def find_eval(data_name: str, seed: int, *, ablation: Optional[str] = None,
              mode: str = "loto") -> Optional[Path]:
    base = ROOT / "RESULTS_X5" / data_name
    if ablation:
        base = base / ablation
    f = base / f"seed_{seed}" / mode / "evaluation.json"
    return f if f.exists() else None


def load_eval(p: Path) -> dict:
    with p.open() as f:
        return json.load(f)


def collect(*, seeds, ablation=None, mode="loto"):
    """Return dict: rows[domain_short][metric] = list-of-per-seed-means."""
    rows: dict[str, dict[str, list[float]]] = {d[0]: {m: [] for m in METRICS} for d in DOMAINS}
    summary_files: list[str] = []
    for short, data_name in DOMAINS:
        for s in seeds:
            p = find_eval(data_name, s, ablation=ablation, mode=mode)
            if p is None:
                continue
            summary_files.append(str(p))
            data = load_eval(p)
            agg = data.get("__mean__", {})
            for m in METRICS:
                if m in agg and agg[m] == agg[m]:  # NaN check
                    rows[short][m].append(float(agg[m]))
    return rows, summary_files


def fmt(values: list[float], k: int = 3) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:+.{k}f}"
    return f"{mean(values):+.{k}f}±{stdev(values):.{k}f}"


def render_markdown(rows, ablation, seeds, files) -> str:
    title = "X5 Full Sweep" if ablation is None else f"X5 Ablation = {ablation}"
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Seeds: `{seeds}` — {len(files)} evaluation files found")
    lines.append("")
    header = ["Metric"] + [d[0] for d in DOMAINS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for m in METRICS:
        cells = [m] + [fmt(rows[d[0]][m]) for d in DOMAINS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Higher is better: " + ", ".join(sorted(HIGHER_IS_BETTER)))
    lines.append("Lower is better:  w1_marginal, mmd_rbf")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default=None,
                    help="Ablation tag (A0..A6). Omit for full-sweep no-ABL path.")
    ap.add_argument("--seeds", default="42,0,1,123,999",
                    help="Comma-separated seed list")
    ap.add_argument("--mode", default="loto", choices=["loto", "alltime"])
    ap.add_argument("--out", default=None, help="Write markdown to this file")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    rows, files = collect(seeds=seeds, ablation=args.ablation, mode=args.mode)
    md = render_markdown(rows, args.ablation, seeds, files)
    print(md)
    if args.out:
        Path(args.out).write_text(md)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
