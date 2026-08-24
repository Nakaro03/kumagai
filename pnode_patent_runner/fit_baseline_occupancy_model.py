#!/usr/bin/env python3
"""Fit and freeze the Confirmation B baseline on exploration data only.

The fit reuses the same guarded observations as the occupancy models, so all
transition centers are at most 2015 and all responses are observed by 2016.
The five fitted coefficients are serialized in the seven-coefficient format
used by :func:`fit_occupancy_model.predict_next_mom`, with both occupancy
terms fixed to exactly zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

if __package__:
    from pnode_patent_runner.evaluate_occupancy_model import fit_baseline_model
    from pnode_patent_runner.fit_occupancy_model import (
        DEFAULT_OCCUPANCY_FEATURES_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_TARGET_PANEL_DIR,
        MAX_TRANSITION_YEAR,
        build_fit_dataset,
        save_coefficients,
        validate_transition_year_max,
    )
    from pnode_patent_runner.slice_occupancy_panel_by_domain import (
        DEFAULT_DOMAINS,
        validate_domains,
    )
else:  # Support ``python pnode_patent_runner/<script>.py``.
    from evaluate_occupancy_model import fit_baseline_model  # type: ignore[no-redef]
    from fit_occupancy_model import (  # type: ignore[no-redef]
        DEFAULT_OCCUPANCY_FEATURES_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_TARGET_PANEL_DIR,
        MAX_TRANSITION_YEAR,
        build_fit_dataset,
        save_coefficients,
        validate_transition_year_max,
    )
    from slice_occupancy_panel_by_domain import (  # type: ignore[no-redef]
        DEFAULT_DOMAINS,
        validate_domains,
    )


def fit_and_freeze_baseline(
    domains: Sequence[str] = DEFAULT_DOMAINS,
    target_panel_dir: Path = DEFAULT_TARGET_PANEL_DIR,
    occupancy_features_dir: Path = DEFAULT_OCCUPANCY_FEATURES_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    transition_year_max: int = MAX_TRANSITION_YEAR,
) -> Dict[str, Dict[str, float]]:
    """Fit and save the baseline for each domain using only guarded rows."""
    domains = validate_domains(domains)
    transition_year_max = validate_transition_year_max(transition_year_max)
    results: Dict[str, Dict[str, float]] = {}
    for domain in domains:
        observations = build_fit_dataset(
            domain,
            target_panel_dir=target_panel_dir,
            occupancy_features_dir=occupancy_features_dir,
            transition_year_max=transition_year_max,
        )
        fitted = fit_baseline_model(observations)
        frozen = {
            **fitted,
            "occ_centered": 0.0,
            "occ_centered_burst": 0.0,
        }
        path = Path(output_dir) / (
            f"fitted_coefficients_{domain}_baseline.json"
        )
        save_coefficients(frozen, path)
        results[domain] = frozen
        print(
            f"[{domain} baseline] n={len(observations)} "
            f"coefficients={json.dumps(frozen)}"
        )
    return results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument(
        "--target-panel-dir", type=Path, default=DEFAULT_TARGET_PANEL_DIR
    )
    parser.add_argument(
        "--occupancy-features-dir",
        type=Path,
        default=DEFAULT_OCCUPANCY_FEATURES_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--transition-year-max", type=int, default=MAX_TRANSITION_YEAR
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_transition_year_max(args.transition_year_max)
    fit_and_freeze_baseline(
        domains=args.domains,
        target_panel_dir=args.target_panel_dir,
        occupancy_features_dir=args.occupancy_features_dir,
        output_dir=args.output_dir,
        transition_year_max=args.transition_year_max,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
