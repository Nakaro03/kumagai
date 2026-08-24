#!/usr/bin/env python3
"""Build and run the preregistered nine-cell Confirmation B evaluation.

This module is an explicit post-exploration path.  Feature preparation requires
centering constants recovered from the frozen exploration-period feature
files.  Model coefficients are loaded only; no model is fitted here.

The repository's real-data defaults are provided for the user's later run.
Tests for this module use synthetic data exclusively.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

if __package__:
    from pnode_patent_runner.build_occupancy_panel import (
        DEFAULT_OUTPUT_DIR as DEFAULT_PANEL_DIR,
    )
    from pnode_patent_runner.compute_occupancy_features import (
        OUTPUT_COLUMNS,
        compute_occupancy_features_confirmation_b,
        recover_frozen_centers,
    )
    from pnode_patent_runner.evaluate_occupancy_model import (
        HOLM_MODEL_NAMES,
        PREREGISTERED_CELLS,
        PREREGISTERED_DOMAINS,
        PREREGISTERED_TRANSITION_YEARS,
        PairedCellErrors,
        evaluate_occupancy_model_gate,
    )
    from pnode_patent_runner.fit_occupancy_model import (
        load_coefficients,
        predict_next_mom,
    )
    from pnode_patent_runner.phase0b_baseline_reverification import (
        BURST_PERCENTILE,
        _mass_table,
    )
    from pnode_patent_runner.slice_occupancy_panel_by_domain import (
        TARGET_COLUMNS,
        slice_occupancy_panel_by_domain_confirmation_b,
    )
else:  # Support ``python pnode_patent_runner/<script>.py``.
    from build_occupancy_panel import DEFAULT_OUTPUT_DIR as DEFAULT_PANEL_DIR
    from compute_occupancy_features import OUTPUT_COLUMNS  # type: ignore[no-redef]
    from compute_occupancy_features import (  # type: ignore[no-redef]
        compute_occupancy_features_confirmation_b,
        recover_frozen_centers,
    )
    from evaluate_occupancy_model import (  # type: ignore[no-redef]
        HOLM_MODEL_NAMES,
        PREREGISTERED_CELLS,
        PREREGISTERED_DOMAINS,
        PREREGISTERED_TRANSITION_YEARS,
        PairedCellErrors,
        evaluate_occupancy_model_gate,
    )
    from fit_occupancy_model import (  # type: ignore[no-redef]
        load_coefficients,
        predict_next_mom,
    )
    from phase0b_baseline_reverification import (  # type: ignore[no-redef]
        BURST_PERCENTILE,
        _mass_table,
    )
    from slice_occupancy_panel_by_domain import (  # type: ignore[no-redef]
        TARGET_COLUMNS,
        slice_occupancy_panel_by_domain_confirmation_b,
    )


CONFIRMATION_B_MAX_YEAR = 2019
DEFAULT_CONFIRMATION_DIR = DEFAULT_PANEL_DIR / "confirmation_b"
DEFAULT_BY_DOMAIN_DIR = DEFAULT_CONFIRMATION_DIR / "by_domain"
DEFAULT_FEATURES_DIR = DEFAULT_CONFIRMATION_DIR / "occupancy_features"
DEFAULT_FROZEN_FEATURES_DIR = DEFAULT_PANEL_DIR / "occupancy_features"
DEFAULT_COEFFICIENT_DIR = DEFAULT_PANEL_DIR / "fitted_coefficients"
DEFAULT_RESULT_PATH = DEFAULT_CONFIRMATION_DIR / "confirmation_b_result.json"


def prepare_confirmation_b_data(
    *,
    target_panel_path: Path = DEFAULT_PANEL_DIR / "target_panel.tsv",
    firm_edges_path: Path = DEFAULT_PANEL_DIR / "firm_edges.tsv",
    frozen_features_dir: Path = DEFAULT_FROZEN_FEATURES_DIR,
    by_domain_dir: Path = DEFAULT_BY_DOMAIN_DIR,
    output_features_dir: Path = DEFAULT_FEATURES_DIR,
) -> Dict[str, Dict[str, float]]:
    """Slice through 2019 and compute features with recovered frozen centers."""
    centers = {
        domain: recover_frozen_centers(
            Path(frozen_features_dir) / f"occupancy_features_{domain}.tsv"
        )
        for domain in PREREGISTERED_DOMAINS
    }
    slice_occupancy_panel_by_domain_confirmation_b(
        target_panel_path=Path(target_panel_path),
        firm_edges_path=Path(firm_edges_path),
        domains=PREREGISTERED_DOMAINS,
        max_reporting_year=CONFIRMATION_B_MAX_YEAR,
        output_dir=Path(by_domain_dir),
    )
    compute_occupancy_features_confirmation_b(
        override_centers=centers,
        domains=PREREGISTERED_DOMAINS,
        full_firm_edges_path=Path(firm_edges_path),
        by_domain_dir=Path(by_domain_dir),
        output_dir=Path(output_features_dir),
        max_reporting_year=CONFIRMATION_B_MAX_YEAR,
    )
    return centers


def _load_confirmation_target_panel(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", dtype="string")
    missing = set(TARGET_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame.loc[:, list(TARGET_COLUMNS)].copy()
    years = pd.to_numeric(frame["filing_year"], errors="coerce")
    mass = pd.to_numeric(frame["target_mass"], errors="coerce")
    if years.isna().any() or ((years % 1) != 0).any():
        raise ValueError(f"{path} contains a missing or non-integer filing_year")
    if (years > CONFIRMATION_B_MAX_YEAR).any():
        raise ValueError(
            f"Confirmation B target panel must end by {CONFIRMATION_B_MAX_YEAR}"
        )
    if mass.isna().any() or not np.isfinite(mass).all() or (mass < 0).any():
        raise ValueError(f"{path} contains invalid target_mass")
    if frame["maingroup"].isna().any() or frame["maingroup"].str.strip().eq("").any():
        raise ValueError(f"{path} contains a missing or empty maingroup")
    frame["filing_year"] = years.astype("int64")
    frame["target_mass"] = mass.astype(float)
    frame["maingroup"] = frame["maingroup"].str.strip()
    if frame.duplicated(["filing_year", "maingroup"]).any():
        raise ValueError(f"{path} must be unique by (filing_year, maingroup)")
    return frame


def _load_confirmation_features(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", dtype="string")
    missing = set(OUTPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame.loc[:, list(OUTPUT_COLUMNS)].copy()
    years = pd.to_numeric(frame["filing_year"], errors="coerce")
    if years.isna().any() or ((years % 1) != 0).any():
        raise ValueError(f"{path} contains a missing or non-integer filing_year")
    if (years > CONFIRMATION_B_MAX_YEAR).any():
        raise ValueError(
            f"Confirmation B feature panel must end by {CONFIRMATION_B_MAX_YEAR}"
        )
    universe = frame["in_topic_universe"].str.strip().str.lower()
    if (~universe.isin(("true", "false"))).any():
        raise ValueError(f"{path} contains a non-boolean in_topic_universe")
    if frame["maingroup"].isna().any() or frame["maingroup"].str.strip().eq("").any():
        raise ValueError(f"{path} contains a missing or empty maingroup")
    frame["filing_year"] = years.astype("int64")
    frame["in_topic_universe"] = universe.eq("true")
    frame["maingroup"] = frame["maingroup"].str.strip()
    for column in ("occ_a_centered", "occ_b_centered"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        eligible = frame["in_topic_universe"]
        values = frame.loc[eligible, column]
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{path} contains invalid {column} in topic universe")
    if frame.duplicated(["filing_year", "maingroup"]).any():
        raise ValueError(f"{path} must be unique by (filing_year, maingroup)")
    return frame


def _coefficient_paths(coefficient_dir: Path, domain: str) -> Dict[str, Path]:
    directory = Path(coefficient_dir)
    return {
        model: directory / f"fitted_coefficients_{domain}_{model}.json"
        for model in ("baseline", *HOLM_MODEL_NAMES)
    }


def _build_prediction_features(mass: pd.DataFrame) -> pd.DataFrame:
    """Apply Phase 0-b's leak-free momentum and burst logic without outcomes."""
    rows = []
    for year in PREREGISTERED_TRANSITION_YEARS:
        previous_year = year - 1
        if previous_year not in mass.index or year not in mass.index:
            continue
        previous_mass = mass.loc[previous_year]
        current_mass = mass.loc[year]
        topics = current_mass.index
        momentum = np.log1p(current_mass.reindex(topics, fill_value=0.0)) - np.log1p(
            previous_mass.reindex(topics, fill_value=0.0)
        )
        positive_momentum = momentum[momentum > 0]
        threshold = (
            np.percentile(positive_momentum, BURST_PERCENTILE)
            if len(positive_momentum)
            else np.inf
        )
        rows.append(
            pd.DataFrame(
                {
                    "cat": topics,
                    "t": year,
                    "mom": momentum.to_numpy(dtype=float),
                    "burst": (momentum >= threshold).to_numpy(dtype=float),
                    "log1p_M": np.log1p(current_mass).to_numpy(dtype=float),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=("cat", "t", "mom", "burst", "log1p_M"))
    return pd.concat(rows, ignore_index=True)


def build_confirmation_b_observations(
    domain: str,
    *,
    by_domain_dir: Path = DEFAULT_BY_DOMAIN_DIR,
    occupancy_features_dir: Path = DEFAULT_FEATURES_DIR,
    coefficient_dir: Path = DEFAULT_COEFFICIENT_DIR,
) -> pd.DataFrame:
    """Build topic-aligned predictions and absolute errors for one domain."""
    if domain not in PREREGISTERED_DOMAINS:
        raise ValueError(f"domain must be one of {PREREGISTERED_DOMAINS}")
    panel = _load_confirmation_target_panel(
        Path(by_domain_dir) / f"target_panel_{domain}.tsv"
    )
    mass = _mass_table(panel)
    observations = _build_prediction_features(mass)
    if observations.empty:
        raise ValueError(f"{domain} has no Confirmation B observations")
    features = _load_confirmation_features(
        Path(occupancy_features_dir) / f"occupancy_features_{domain}.tsv"
    ).rename(columns={"filing_year": "t", "maingroup": "cat"})
    observations = observations.merge(
        features.loc[
            :, [
                "t",
                "cat",
                "occ_a_centered",
                "occ_b_centered",
                "in_topic_universe",
            ]
        ],
        on=["t", "cat"],
        how="inner",
        validate="one_to_one",
    )
    observations = observations.loc[
        observations["in_topic_universe"].astype(bool)
    ].copy()

    paths = _coefficient_paths(coefficient_dir, domain)
    coefficients = {name: load_coefficients(path) for name, path in paths.items()}
    feature_by_model: Mapping[str, Optional[str]] = {
        "baseline": None,
        "a": "occ_a_centered",
        "b": "occ_b_centered",
    }
    for model, occupancy_column in feature_by_model.items():
        predictions = []
        for row in observations.itertuples(index=False):
            occupancy_value = (
                0.0 if occupancy_column is None else float(getattr(row, occupancy_column))
            )
            predictions.append(
                predict_next_mom(
                    coefficients[model],
                    mom=float(row.mom),
                    log1p_M=float(row.log1p_M),
                    burst=float(row.burst),
                    occ_centered=occupancy_value,
                )
            )
        observations[f"{model}_prediction"] = predictions

    # Outcome masses are accessed only after every model prediction has been
    # constructed.  No value from t + 1 can enter momentum, burst, centering,
    # or any prediction feature.
    actual_next_momentum = []
    for row in observations.itertuples(index=False):
        outcome_year = int(row.t) + 1
        if outcome_year not in mass.index:
            raise ValueError(f"Missing outcome year {outcome_year} for {domain}")
        current_mass = float(mass.at[int(row.t), row.cat])
        outcome_mass = float(mass.at[outcome_year, row.cat])
        actual_next_momentum.append(
            math.log1p(outcome_mass) - math.log1p(current_mass)
        )
    observations["next_mom"] = actual_next_momentum
    for model in feature_by_model:
        observations[f"{model}_error"] = (
            observations["next_mom"] - observations[f"{model}_prediction"]
        ).abs()

    expected_years = set(PREREGISTERED_TRANSITION_YEARS)
    actual_years = set(observations["t"].astype(int))
    if actual_years != expected_years:
        raise ValueError(
            f"{domain} Confirmation B cells differ; "
            f"missing={sorted(expected_years - actual_years)}, "
            f"extra={sorted(actual_years - expected_years)}"
        )
    return observations.sort_values(["t", "cat"]).reset_index(drop=True)


def build_confirmation_b_cell_errors(
    *,
    by_domain_dir: Path = DEFAULT_BY_DOMAIN_DIR,
    occupancy_features_dir: Path = DEFAULT_FEATURES_DIR,
    coefficient_dir: Path = DEFAULT_COEFFICIENT_DIR,
) -> Dict[str, Dict[tuple[str, int], PairedCellErrors]]:
    """Construct the exact 3-domain x 3-transition paired-error family."""
    model_cells: Dict[str, Dict[tuple[str, int], PairedCellErrors]] = {
        model: {} for model in HOLM_MODEL_NAMES
    }
    for domain in PREREGISTERED_DOMAINS:
        observations = build_confirmation_b_observations(
            domain,
            by_domain_dir=by_domain_dir,
            occupancy_features_dir=occupancy_features_dir,
            coefficient_dir=coefficient_dir,
        )
        for year in PREREGISTERED_TRANSITION_YEARS:
            cell_rows = observations.loc[observations["t"] == year]
            if cell_rows.empty:
                raise ValueError(f"Missing Confirmation B cell {(domain, year)}")
            baseline_errors = cell_rows["baseline_error"].to_numpy(dtype=float)
            topic_ids = tuple(cell_rows["cat"].astype(str))
            for model in HOLM_MODEL_NAMES:
                model_cells[model][(domain, year)] = PairedCellErrors(
                    baseline_errors=baseline_errors,
                    augmented_errors=cell_rows[f"{model}_error"].to_numpy(
                        dtype=float
                    ),
                    topic_ids=topic_ids,
                )
    if any(set(cells) != PREREGISTERED_CELLS for cells in model_cells.values()):
        raise ValueError("Internal error: Confirmation B did not build all nine cells")
    return model_cells


def run_confirmation_b(
    *,
    by_domain_dir: Path = DEFAULT_BY_DOMAIN_DIR,
    occupancy_features_dir: Path = DEFAULT_FEATURES_DIR,
    coefficient_dir: Path = DEFAULT_COEFFICIENT_DIR,
    n_bootstraps: int = 9999,
    random_seed: int = 8128,
) -> Dict[str, object]:
    """Build nine cells and pass them unchanged to the preregistered gate."""
    cell_errors = build_confirmation_b_cell_errors(
        by_domain_dir=by_domain_dir,
        occupancy_features_dir=occupancy_features_dir,
        coefficient_dir=coefficient_dir,
    )
    hhi_paths = {
        domain: _coefficient_paths(coefficient_dir, domain)["b"]
        for domain in PREREGISTERED_DOMAINS
    }
    return evaluate_occupancy_model_gate(
        cell_errors,
        hhi_paths,
        n_bootstraps=n_bootstraps,
        random_seed=random_seed,
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            (f"{key[0]}:{key[1]}" if isinstance(key, tuple) else str(key)): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-panel-path", type=Path, default=DEFAULT_PANEL_DIR / "target_panel.tsv"
    )
    parser.add_argument(
        "--firm-edges-path", type=Path, default=DEFAULT_PANEL_DIR / "firm_edges.tsv"
    )
    parser.add_argument(
        "--frozen-features-dir", type=Path, default=DEFAULT_FROZEN_FEATURES_DIR
    )
    parser.add_argument("--by-domain-dir", type=Path, default=DEFAULT_BY_DOMAIN_DIR)
    parser.add_argument(
        "--occupancy-features-dir", type=Path, default=DEFAULT_FEATURES_DIR
    )
    parser.add_argument(
        "--coefficient-dir", type=Path, default=DEFAULT_COEFFICIENT_DIR
    )
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--n-bootstraps", type=int, default=9999)
    parser.add_argument("--random-seed", type=int, default=8128)
    parser.add_argument(
        "--skip-data-preparation",
        action="store_true",
        help="Use already-prepared by-domain and occupancy-feature inputs",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.skip_data_preparation:
        prepare_confirmation_b_data(
            target_panel_path=args.target_panel_path,
            firm_edges_path=args.firm_edges_path,
            frozen_features_dir=args.frozen_features_dir,
            by_domain_dir=args.by_domain_dir,
            output_features_dir=args.occupancy_features_dir,
        )
    result = run_confirmation_b(
        by_domain_dir=args.by_domain_dir,
        occupancy_features_dir=args.occupancy_features_dir,
        coefficient_dir=args.coefficient_dir,
        n_bootstraps=args.n_bootstraps,
        random_seed=args.random_seed,
    )
    payload = _jsonable(result)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.result_json.with_suffix(args.result_json.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(args.result_json)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    # Real-data execution is intentionally left to the user.
    raise SystemExit(main())
