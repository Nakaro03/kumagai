"""aggregate_x5_ablations.py — assemble the A0..A6 ablation table.

Output: a single markdown matrix where rows are ablations and columns are
(domain × metric).
"""
from __future__ import annotations

import argparse
from statistics import mean, stdev
from pathlib import Path
import json

ROOT = Path("/home/nakamuraroi/kumagai")

ABLATIONS = [
    ("A0", "X5 full (LOTO + 4-term)"),
    ("A1", "no LOTO"),
    ("A2", "no L_phys"),
    ("A3", "no L_geom"),
    ("A4", "no L_smooth"),
    ("A5", "no Fourier (scalar t)"),
    ("A6", "LOTO only (phys=geom=smooth=0)"),
]

DOMAINS = [
    ("patent_energy",      "PNode_Patent_Energy_X1_top50"),
    ("arxiv_construction", "PNode_ArXiv_Construction_X1_v2"),
    ("jp_construction",    "PNode_JP_Construction_X1"),
]

KEY_METRICS = ["w1_marginal", "hits_at_10", "ndcg_at_10", "mrr"]


def load_one(data_name: str, abl: str, seed: int, mode: str = "loto") -> dict | None:
    f = ROOT / "RESULTS_X5" / data_name / abl / f"seed_{seed}" / mode / "evaluation.json"
    if not f.exists():
        return None
    with f.open() as fh:
        return json.load(fh)


def fmt(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:+.3f}"
    return f"{mean(vals):+.3f}±{stdev(vals):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,0,1,123,999")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    lines = ["# X5 Ablation Matrix (A0..A6)", "",
             f"Seeds: `{seeds}`", ""]
    header = ["Ablation"]
    for short, _ in DOMAINS:
        for m in KEY_METRICS:
            header.append(f"{short}:{m}")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")

    for abl, descr in ABLATIONS:
        cells = [f"{abl} ({descr})"]
        for short, data_name in DOMAINS:
            for m in KEY_METRICS:
                vals = []
                for s in seeds:
                    data = load_one(data_name, abl, s)
                    if data and m in data.get("__mean__", {}):
                        v = data["__mean__"][m]
                        if v == v:  # not NaN
                            vals.append(float(v))
                cells.append(fmt(vals))
        lines.append("| " + " | ".join(cells) + " |")

    md = "\n".join(lines)
    print(md)
    if args.out:
        Path(args.out).write_text(md)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
