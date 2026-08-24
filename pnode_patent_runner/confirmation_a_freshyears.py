#!/usr/bin/env python3
"""Confirmation A: fresh-year analysis using the existing inventor/CPC panels.

This is a non-gating companion analysis.  It intentionally reuses Gate 0's
existing panel loaders and regression implementation, but does not make a
pass/fail decision about the occupancy-share hypothesis.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from pnode_patent_runner.gate0_regime_detectability import (
    _build_observations,
    _fit_interaction,
    _mass_table,
    load_author_topic_pairs,
    load_domain_pairs,
)

PATENT_DOMAINS = (
    "construction",
    "energy",
    "computing",
    "pharma",
    "semiconductor",
    "agrifood",
)
PATENT_TRANSITIONS = [
    (2020, 2021, 2022),
    (2021, 2022, 2023),
    (2022, 2023, 2024),
]
AUTHOR_TOPIC_TRANSITIONS = [(t - 1, t, t + 1) for t in range(2022, 2025)]
PURPOSE = "占有率仮説の合否判定には使わない参考情報（確認A: fresh years の非gating別分析）"
LABEL = "[CONFIRMATION A — 非gating/参考情報のみ]"


def _analyze_pairs(pairs, transitions) -> Dict:
    mass = _mass_table(pairs)
    observations = _build_observations(mass, transitions)
    return _fit_interaction(observations)


def run_analysis(cpc_level: str) -> Dict[str, Dict]:
    """Run the fixed fresh-year protocol for all six patent domains plus author/topic."""
    results: Dict[str, Dict] = {}
    for domain in PATENT_DOMAINS:
        pairs = load_domain_pairs(domain, cpc_level, year_min=2020, year_max=2024)
        fit = _analyze_pairs(pairs, PATENT_TRANSITIONS)
        results[domain] = fit
        print(
            f"[{domain}] n={fit.get('n')} n_burst={fit.get('n_burst')} "
            f"coef(mom×burst)={fit.get('coef_mom_burst')} "
            f"p_hc1={fit.get('p_mom_burst_hc1')} "
            f"p_cluster={fit.get('p_mom_burst_cluster')}"
        )

    author_topic_pairs = load_author_topic_pairs(2020, 2025)
    fit = _analyze_pairs(author_topic_pairs, AUTHOR_TOPIC_TRANSITIONS)
    results["author_topic"] = fit
    print(
        f"[author_topic] n={fit.get('n')} n_burst={fit.get('n_burst')} "
        f"coef(mom×burst)={fit.get('coef_mom_burst')} "
        f"p_hc1={fit.get('p_mom_burst_hc1')} "
        f"p_cluster={fit.get('p_mom_burst_cluster')}"
    )
    return results


def build_report(results: Dict[str, Dict]) -> Dict:
    """Build a deliberately non-gating report; no pass/fail fields belong here."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "non_gating": True,
        "purpose": PURPOSE,
        "protocol": {
            "patent_year_range": [2020, 2024],
            "patent_transitions": [list(t) for t in PATENT_TRANSITIONS],
            "author_topic_year_range": [2020, 2025],
            "author_topic_transitions": [list(t) for t in AUTHOR_TOPIC_TRANSITIONS],
            "regression": "next_mom ~ mom + burst + mom:burst (existing Gate 0 implementation)",
        },
        "results": results,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Confirmation A fresh-year analysis (non-gating reference information only)"
    )
    parser.add_argument(
        "--cpc-level", choices=("maingroup", "subclass"), default="maingroup"
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print(LABEL)
    results = run_analysis(args.cpc_level)
    report = build_report(results)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, ensure_ascii=False)
    print(f"{LABEL} Wrote: {args.output_json}")
    return 0


if __name__ == "__main__":
    main()
