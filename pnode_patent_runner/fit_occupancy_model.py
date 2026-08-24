#!/usr/bin/env python3
"""Fit and freeze the issue #8 section 12.7 occupancy regressions.

Only transition centers through 2015 are admissible: every fitted target is
therefore observed by 2016.  Prediction is a separate scalar-only linear
combination over already-frozen coefficients; this module provides no path for
the prediction function to fit or compare against observed outcomes.
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
    from pnode_patent_runner.phase0b_baseline_reverification import (
        MAX_REPORTING_YEAR,
        _build_observations,
        _mass_table,
        load_target_panel,
    )
    from pnode_patent_runner.slice_occupancy_panel_by_domain import (
        DEFAULT_DOMAINS,
        validate_domains,
    )
else:  # Support ``python pnode_patent_runner/<script>.py``.
    from phase0b_baseline_reverification import (  # type: ignore[no-redef]
        MAX_REPORTING_YEAR,
        _build_observations,
        _mass_table,
        load_target_panel,
    )
    from slice_occupancy_panel_by_domain import (  # type: ignore[no-redef]
        DEFAULT_DOMAINS,
        validate_domains,
    )


DEFAULT_TARGET_PANEL_DIR = Path("data/processed/occupancy_panel/by_domain")
DEFAULT_OCCUPANCY_FEATURES_DIR = Path(
    "data/processed/occupancy_panel/occupancy_features"
)
DEFAULT_OUTPUT_DIR = Path("data/processed/occupancy_panel/fitted_coefficients")
MAX_TRANSITION_YEAR = MAX_REPORTING_YEAR - 1

MODEL_NAMES = ("a", "b")
COEFFICIENT_NAMES = (
    "intercept",
    "mom",
    "log1p_M",
    "burst",
    "mom_burst",
    "occ_centered",
    "occ_centered_burst",
)
OCCUPANCY_FEATURE_COLUMNS = (
    "filing_year",
    "maingroup",
    "occ_a",
    "occ_a_centered",
    "occ_b",
    "occ_b_centered",
    "coverage",
    "n_j",
    "in_topic_universe",
)
FIT_DATASET_COLUMNS = (
    "cat",
    "t",
    "mom",
    "burst",
    "next_mom",
    "log1p_M",
    "coverage",
    "occ_a_centered",
    "occ_b_centered",
    "in_topic_universe",
)


def validate_transition_year_max(transition_year_max: int) -> int:
    """Reject any fit window whose response would enter the holdout."""
    if transition_year_max > MAX_TRANSITION_YEAR:
        raise ValueError(
            "Holdout guard: transition_year_max must be <= "
            f"{MAX_TRANSITION_YEAR} so t + 1 <= {MAX_REPORTING_YEAR}; "
            f"got {transition_year_max}"
        )
    return transition_year_max


def _load_occupancy_features(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", dtype="string")
    missing = set(OCCUPANCY_FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame.loc[:, list(OCCUPANCY_FEATURE_COLUMNS)].copy()

    years = pd.to_numeric(frame["filing_year"], errors="coerce")
    if years.isna().any() or ((years % 1) != 0).any():
        raise ValueError(f"{path} contains a missing or non-integer filing_year")
    frame["filing_year"] = years.astype("int64")
    if (frame["filing_year"] > MAX_REPORTING_YEAR).any():
        raise ValueError(
            f"Holdout guard: {path} contains filing_year beyond "
            f"{MAX_REPORTING_YEAR}"
        )
    if frame["maingroup"].isna().any() or frame["maingroup"].str.strip().eq("").any():
        raise ValueError(f"{path} contains a missing or empty maingroup")
    frame["maingroup"] = frame["maingroup"].str.strip()
    if frame.duplicated(["filing_year", "maingroup"]).any():
        raise ValueError(
            f"{path} must be unique by (filing_year, maingroup)"
        )

    universe_text = frame["in_topic_universe"].str.strip().str.lower()
    invalid_universe = ~universe_text.isin(("true", "false"))
    if invalid_universe.any():
        raise ValueError(f"{path} contains a non-boolean in_topic_universe")
    frame["in_topic_universe"] = universe_text.map(
        {"true": True, "false": False}
    ).astype(bool)

    for column in ("coverage", "occ_a_centered", "occ_b_centered"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        reportable = frame["in_topic_universe"]
        if frame.loc[reportable, column].isna().any() or not np.isfinite(
            frame.loc[reportable, column]
        ).all():
            raise ValueError(
                f"{path} contains a missing or non-finite {column} in the "
                "topic universe"
            )
    return frame


def build_fit_dataset(
    domain: str,
    target_panel_dir: Path = DEFAULT_TARGET_PANEL_DIR,
    occupancy_features_dir: Path = DEFAULT_OCCUPANCY_FEATURES_DIR,
    transition_year_max: int = MAX_TRANSITION_YEAR,
) -> pd.DataFrame:
    """Build one domain's guarded, topic-universe-only fit observations."""
    validate_domains((domain,))
    transition_year_max = validate_transition_year_max(transition_year_max)
    target_path = Path(target_panel_dir) / f"target_panel_{domain}.tsv"
    features_path = (
        Path(occupancy_features_dir) / f"occupancy_features_{domain}.tsv"
    )

    panel = load_target_panel(target_path)
    mass = _mass_table(panel)
    if mass.empty:
        raise ValueError(f"{target_path} contains no rows")
    first_transition_year = int(mass.index.min()) + 1
    transitions = [
        (t - 1, t, t + 1)
        for t in range(first_transition_year, transition_year_max + 1)
    ]
    observations = _build_observations(mass, transitions)

    mass_at_t = (
        mass.rename_axis(index="t", columns="cat")
        .stack(dropna=False)
        .rename("target_mass")
        .reset_index()
    )
    observations = observations.merge(
        mass_at_t,
        on=["cat", "t"],
        how="left",
        validate="one_to_one",
    )
    if observations["target_mass"].isna().any():
        raise ValueError("Internal error: target mass is missing for an observation")
    observations["log1p_M"] = np.log1p(observations.pop("target_mass"))

    features = _load_occupancy_features(features_path).rename(
        columns={"filing_year": "t", "maingroup": "cat"}
    )
    observations = observations.merge(
        features.loc[
            :,
            [
                "t",
                "cat",
                "occ_a_centered",
                "occ_b_centered",
                "coverage",
                "in_topic_universe",
            ],
        ],
        on=["t", "cat"],
        # A category can be active at t-1 or t+1 while having zero mass and no
        # organization edge at t.  _build_observations retains that row, but
        # the occupancy feature panel has no (t, cat) candidate to mark as in
        # universe.  An inner merge correctly makes it ineligible for fitting.
        how="inner",
        validate="one_to_one",
    )
    observations = observations.loc[
        observations["in_topic_universe"].astype(bool),
        list(FIT_DATASET_COLUMNS),
    ]
    return observations.sort_values(["t", "cat"]).reset_index(drop=True)


def _validate_model_name(model: str) -> str:
    if model not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}; got {model!r}")
    return model


def validate_fit_observations(
    observations: pd.DataFrame,
    numeric_columns: Sequence[str],
) -> pd.DataFrame:
    """Validate a fit matrix and enforce the shared pre-holdout guard.

    Every occupancy-model variant, including the non-gating coverage
    sensitivity models, must pass through this guard before OLS is run.
    """
    required = {"t", *numeric_columns}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Fit observations are missing columns: {sorted(missing)}")
    if observations.empty:
        raise ValueError("Cannot fit an empty observation dataset")

    transition_years = pd.to_numeric(observations["t"], errors="coerce")
    if transition_years.isna().any() or not np.equal(
        transition_years, np.floor(transition_years)
    ).all():
        raise ValueError("t must contain only integer transition years")
    if (transition_years > MAX_TRANSITION_YEAR).any():
        bad_year = int(transition_years[transition_years > MAX_TRANSITION_YEAR].min())
        raise ValueError(
            "Holdout guard: fit accepts only transition years t <= "
            f"{MAX_TRANSITION_YEAR}; got t={bad_year}"
        )

    numeric = observations.loc[:, list(numeric_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any() or not np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all():
        raise ValueError("Fit observations contain missing or non-finite values")
    return numeric


def fit_model(observations: pd.DataFrame, model: str) -> Dict[str, float]:
    """Fit one OLS model, refusing every transition center after 2015."""
    model = _validate_model_name(model)
    occ_column = f"occ_{model}_centered"
    numeric_columns = ["mom", "log1p_M", "burst", "next_mom", occ_column]
    numeric = validate_fit_observations(observations, numeric_columns)

    mom = numeric["mom"].to_numpy(dtype=float)
    log1p_mass = numeric["log1p_M"].to_numpy(dtype=float)
    burst = numeric["burst"].to_numpy(dtype=float)
    occ = numeric[occ_column].to_numpy(dtype=float)
    design = np.column_stack(
        (
            np.ones(len(numeric)),
            mom,
            log1p_mass,
            burst,
            mom * burst,
            occ,
            occ * burst,
        )
    )
    response = numeric["next_mom"].to_numpy(dtype=float)
    estimates, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if rank < len(COEFFICIENT_NAMES):
        raise ValueError(
            "OLS design matrix is rank deficient: "
            f"rank={rank}, required={len(COEFFICIENT_NAMES)}"
        )
    return {
        name: float(value) for name, value in zip(COEFFICIENT_NAMES, estimates)
    }


def _validated_coefficients(
    coefficients: Mapping[str, float],
) -> Dict[str, float]:
    missing = set(COEFFICIENT_NAMES) - set(coefficients)
    extra = set(coefficients) - set(COEFFICIENT_NAMES)
    if missing or extra:
        raise ValueError(
            f"Coefficient keys differ; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    validated = {name: float(coefficients[name]) for name in COEFFICIENT_NAMES}
    if not all(math.isfinite(value) for value in validated.values()):
        raise ValueError("Coefficients must all be finite")
    return validated


def save_coefficients(coefficients: Mapping[str, float], path: Path) -> Path:
    """Serialize one frozen coefficient vector as JSON."""
    frozen = _validated_coefficients(coefficients)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(frozen, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)
    return path


def load_coefficients(path: Path) -> Dict[str, float]:
    """Load and validate one frozen JSON coefficient vector."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Coefficient JSON must contain an object")
    return _validated_coefficients(payload)


def predict_next_mom(
    coefficients: Mapping[str, float],
    *,
    mom: float,
    log1p_M: float,
    burst: float,
    occ_centered: float,
) -> float:
    """Predict by scalar linear combination only; never fit or score a model."""
    frozen = _validated_coefficients(coefficients)
    inputs = (float(mom), float(log1p_M), float(burst), float(occ_centered))
    if not all(math.isfinite(value) for value in inputs):
        raise ValueError("Prediction features must all be finite scalars")
    mom_value, mass_value, burst_value, occ_value = inputs
    return float(
        frozen["intercept"]
        + frozen["mom"] * mom_value
        + frozen["log1p_M"] * mass_value
        + frozen["burst"] * burst_value
        + frozen["mom_burst"] * mom_value * burst_value
        + frozen["occ_centered"] * occ_value
        + frozen["occ_centered_burst"] * occ_value * burst_value
    )


def fit_and_freeze(
    domains: Sequence[str] = DEFAULT_DOMAINS,
    target_panel_dir: Path = DEFAULT_TARGET_PANEL_DIR,
    occupancy_features_dir: Path = DEFAULT_OCCUPANCY_FEATURES_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    transition_year_max: int = MAX_TRANSITION_YEAR,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Fit and save models A/B for every requested domain."""
    domains = validate_domains(domains)
    transition_year_max = validate_transition_year_max(transition_year_max)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for domain in domains:
        observations = build_fit_dataset(
            domain,
            target_panel_dir=target_panel_dir,
            occupancy_features_dir=occupancy_features_dir,
            transition_year_max=transition_year_max,
        )
        results[domain] = {}
        for model in MODEL_NAMES:
            coefficients = fit_model(observations, model)
            path = Path(output_dir) / (
                f"fitted_coefficients_{domain}_{model}.json"
            )
            save_coefficients(coefficients, path)
            results[domain][model] = coefficients
            print(
                f"[{domain} model {model.upper()}] "
                f"n={len(observations)} coefficients={json.dumps(coefficients)}"
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
    # Validate before any input path is opened.
    validate_transition_year_max(args.transition_year_max)
    fit_and_freeze(
        domains=args.domains,
        target_panel_dir=args.target_panel_dir,
        occupancy_features_dir=args.occupancy_features_dir,
        output_dir=args.output_dir,
        transition_year_max=args.transition_year_max,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
