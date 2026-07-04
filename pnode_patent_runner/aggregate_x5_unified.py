"""aggregate_x5_unified.py — single comparison table for paper Table 1.

Combines:
  * X5 full sweep                (RESULTS_X5/{DATA}/seed_S/loto/...)
  * X5 ablations  A0..A6         (RESULTS_X5/{DATA}/A?/seed_S/loto/...)
  * PRESCIENT-style baseline     (RESULTS_X5/{DATA}/PRESCIENT/seed_S/alltime/...)
  * MIOFlow-style baseline       (RESULTS_X5/{DATA}/MIOFLOW/seed_S/alltime/...)
  * TSFM/naive baselines         (RESULTS_TSFM_BASELINE/{DATA}/all_methods/split70/...)

Output: a single markdown matrix where rows are methods and columns are
(domain × primary metric).

Usage:
  python pnode_patent_runner/aggregate_x5_unified.py
  python pnode_patent_runner/aggregate_x5_unified.py --out RESULTS_X5/TABLE1.md
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

METRICS = ["w1_marginal", "hits_at_10", "ndcg_at_10", "mrr"]

# (label, source, lookup-spec)
#   source:
#     "x5"           -> RESULTS_X5/{data}/seed_S/loto/evaluation.json
#     "x5_ablation"  -> RESULTS_X5/{data}/{ABL}/seed_S/loto/evaluation.json
#     "x5_baseline"  -> RESULTS_X5/{data}/{TAG}/seed_S/alltime/evaluation.json
#     "tsfm"         -> RESULTS_TSFM_BASELINE/{data}/all_methods/split70/evaluation_all.json
METHODS = [
    ("X5 (full A0)",           "x5",          None),
    ("A1 no LOTO",             "x5_ablation", "A1"),
    ("A2 no L_phys",           "x5_ablation", "A2"),
    ("A3 no L_geom",           "x5_ablation", "A3"),
    ("A4 no L_smooth",         "x5_ablation", "A4"),
    ("A5 no Fourier",          "x5_ablation", "A5"),
    ("A6 LOTO-only",           "x5_ablation", "A6"),
    ("PRESCIENT (reimpl)",     "x5_baseline", "PRESCIENT"),
    ("MIOFlow (reimpl)",       "x5_baseline", "MIOFLOW"),
    ("Persistence",            "tsfm",        "persistence"),
    ("Chronos-Bolt-S",         "tsfm",        "chronos"),
    ("Moirai-2-R-S",           "tsfm",        "moirai"),
]


def _load_x5_eval(data_name: str, seed: int, *, ablation: Optional[str] = None,
                  mode: str = "loto") -> Optional[dict]:
    p = ROOT / "RESULTS_X5" / data_name
    if ablation:
        p = p / ablation
    p = p / f"seed_{seed}" / mode / "evaluation.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _load_tsfm(data_name: str) -> Optional[dict]:
    p = ROOT / "RESULTS_TSFM_BASELINE" / data_name / "all_methods" / "split70" / "evaluation_all.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def collect_x5(data_name: str, seeds: list[int], *, ablation=None, mode="loto") -> dict[str, list[float]]:
    out = {m: [] for m in METRICS}
    for s in seeds:
        d = _load_x5_eval(data_name, s, ablation=ablation, mode=mode)
        if d is None:
            continue
        agg = d.get("__mean__", {})
        for m in METRICS:
            v = agg.get(m)
            if v is not None and v == v:
                out[m].append(float(v))
    return out


def collect_tsfm(data_name: str, method: str) -> dict[str, list[float]]:
    """Pull per-test-t scores for the given tsfm/naive method, average across t per seed.

    The TSFM eval JSON only has ndcg / spearman; W1/MMD/Hits/MRR are unavailable.
    We map ndcg -> ndcg_at_10. For Hits@10 we approximate from prec_at_10 if present.
    """
    out = {m: [] for m in METRICS}
    d = _load_tsfm(data_name)
    if d is None:
        return out
    if method not in d.get("results", {}):
        return out
    for run in d["results"][method]["runs"]:
        per_t = run["per_t"]
        ndcg = [t.get("ndcg") for t in per_t if t.get("ndcg") is not None]
        if ndcg:
            out["ndcg_at_10"].append(mean(ndcg))
        # Hits@10 ≈ prec_at_10 from baseline_all metrics (top-10 from raw growth)
        prec = [t.get("prec_at_10") for t in per_t if t.get("prec_at_10") is not None]
        if prec:
            out["hits_at_10"].append(mean(prec))
        # No W1/MRR available for TSFM in current eval JSON
    return out


def fmt(values: list[float], decimals: int = 3) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.{decimals}f}"
    return f"{mean(values):.{decimals}f}±{stdev(values):.{decimals}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,0,1,123,999")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    rows: list[str] = []
    rows.append("# X5 Unified Comparison Table")
    rows.append("")
    rows.append(f"Seeds: `{seeds}`  |  W1↓, others↑")
    rows.append("")

    header = ["Method"]
    for short, _ in DOMAINS:
        for m in METRICS:
            arrow = "↓" if m == "w1_marginal" else "↑"
            header.append(f"{short}:{m}{arrow}")
    rows.append("| " + " | ".join(header) + " |")
    rows.append("|" + "|".join("---" for _ in header) + "|")

    for label, src, spec in METHODS:
        cells = [label]
        for short, data_name in DOMAINS:
            if src == "x5":
                stats_ = collect_x5(data_name, seeds, ablation=None, mode="loto")
            elif src == "x5_ablation":
                stats_ = collect_x5(data_name, seeds, ablation=spec, mode="loto")
            elif src == "x5_baseline":
                stats_ = collect_x5(data_name, seeds, ablation=spec, mode="alltime")
            elif src == "tsfm":
                stats_ = collect_tsfm(data_name, spec)
            else:
                stats_ = {m: [] for m in METRICS}
            for m in METRICS:
                cells.append(fmt(stats_[m]))
        rows.append("| " + " | ".join(cells) + " |")

    md = "\n".join(rows)
    print(md)
    if args.out:
        Path(args.out).write_text(md)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
