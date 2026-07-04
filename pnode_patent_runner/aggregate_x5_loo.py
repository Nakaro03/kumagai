"""aggregate_x5_loo.py — verdict on whether X5 has TRUE predictive ability.

Reads RESULTS_X5_LOO/{DATA}/h{T}/{ABL}/seed_S/evaluation.json and produces a
comparison table. Critically, this is the table that decides whether X5 is a
real predictive method or just X3-clean re-implemented.

Expected outcome interpretation:
  * If "full" Spearman > 0 across seeds on all 3 domains → X5 has predictive
    ability → NeurIPS/TMLR/KDD pivot keeps method-paper claim.
  * If "full" Spearman ≤ 0 (negative, like X3-clean's -0.35) → X5 has NO
    predictive ability → X5 must be presented as descriptive only or framed as
    a negative-result study.
  * If "full" >> "no_anchor" and "no_anchor" ≈ "prescient" → confirms anchor
    is the only mechanism that matters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

ROOT = Path("/home/nakamuraroi/kumagai")

DOMAINS = [
    ("patent_energy",      "PNode_Patent_Energy_X1_top50", 5),
    ("arxiv_construction", "PNode_ArXiv_Construction_X1_v2", 5),
    ("jp_construction",    "PNode_JP_Construction_X1", 5),
]

ABLATIONS = [
    ("full",      "X5 full (anchor + Fourier + geom + smooth)"),
    ("no_anchor", "no Φ-anchor"),
    ("prescient", "PRESCIENT-equivalent (no anchor, no Fourier, no geom, no smooth)"),
]

METRICS = ["w1_marginal", "hits_at_10", "ndcg_at_10", "mrr", "ap", "spearman"]


def load_eval(data_name: str, holdout: int, abl: str, seed: int) -> Optional[dict]:
    f = ROOT / "RESULTS_X5_LOO" / data_name / f"h{holdout}" / abl / f"seed_{seed}" / "evaluation.json"
    if not f.exists():
        return None
    with f.open() as fh:
        return json.load(fh)


def collect(data_name: str, holdout: int, abl: str, seeds: list[int]) -> dict[str, list[float]]:
    vals = {m: [] for m in METRICS}
    for s in seeds:
        d = load_eval(data_name, holdout, abl, s)
        if d is None:
            continue
        agg = d.get("__mean__", {})
        for m in METRICS:
            v = agg.get(m)
            if v is not None and v == v:
                vals[m].append(float(v))
    return vals


def fmt(vs: list[float]) -> str:
    if not vs:
        return "—"
    if len(vs) == 1:
        return f"{vs[0]:+.3f}"
    return f"{mean(vs):+.3f}±{stdev(vs):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,0,1,123,999")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    lines: list[str] = []
    lines.append("# X5 True Leave-One-Out Verdict")
    lines.append("")
    lines.append(f"Seeds: `{seeds}`  |  W1↓, others↑")
    lines.append("")
    lines.append("Critical comparison: X3-clean's prior leave-one-out experiment reported")
    lines.append("Spearman ρ = −0.35 (negative correlation, no predictive ability).")
    lines.append("X5 must show Spearman > 0 here to claim predictive recovery.")
    lines.append("")

    for short, data_name, holdout in DOMAINS:
        lines.append(f"## Domain: **{short}** (holdout t={holdout})")
        lines.append("")
        header = ["Ablation"] + METRICS
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for abl_id, abl_descr in ABLATIONS:
            vals = collect(data_name, holdout, abl_id, seeds)
            row = [f"{abl_id} ({abl_descr})"] + [fmt(vals[m]) for m in METRICS]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    md = "\n".join(lines)
    print(md)
    if args.out:
        Path(args.out).write_text(md)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
