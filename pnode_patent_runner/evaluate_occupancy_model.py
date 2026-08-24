"""Issue #8 section 12.8 occupancy-model fits and preregistered gate logic.

The evaluation layer accepts already-computed, topic-level paired errors.  It
does not load target panels, construct holdout outcomes, or run Confirmation B.
This keeps the gate implementation independently testable with synthetic data.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pnode_patent_runner.fit_occupancy_model import (
    MODEL_NAMES,
    load_coefficients,
    validate_fit_observations,
)
from pnode_patent_runner.slice_occupancy_panel_by_domain import DEFAULT_DOMAINS


# These gate values and the exact 3 x 3 cell set were preregistered in issue
# #8.  They are deliberately module constants and cannot be changed at runtime.
SESOI_THRESHOLD = 0.05
ZERO_DELTA_TOLERANCE = 1e-9
PREREGISTERED_DOMAINS = tuple(DEFAULT_DOMAINS)
PREREGISTERED_TRANSITION_YEARS = (2016, 2017, 2018)
PREREGISTERED_CELLS = frozenset(
    (domain, year)
    for domain in PREREGISTERED_DOMAINS
    for year in PREREGISTERED_TRANSITION_YEARS
)
PREREGISTERED_CELL_COUNT = 9
HOLM_FAMILY_SIZE = 2
# Only the two preregistered occupancy models belong to the Holm family;
# baseline_cov and augmented_cov are descriptive sensitivity models.
HOLM_MODEL_NAMES = tuple(MODEL_NAMES)

BASELINE_COEFFICIENT_NAMES = (
    "intercept",
    "mom",
    "log1p_M",
    "burst",
    "mom_burst",
)
BASELINE_COVERAGE_COEFFICIENT_NAMES = (
    *BASELINE_COEFFICIENT_NAMES,
    "coverage",
)
AUGMENTED_COVERAGE_COEFFICIENT_NAMES = (
    *BASELINE_COVERAGE_COEFFICIENT_NAMES,
    "occ_centered",
    "occ_centered_burst",
)

Cell = Tuple[str, int]


@dataclass(frozen=True)
class PairedCellErrors:
    """Topic-aligned errors for one domain-year cell.

    ``topic_ids`` should be supplied when a topic occurs in multiple year
    cells so the auxiliary bootstrap can retain the entire topic cluster.
    When omitted, each paired row is treated as a cell-local topic cluster.
    """

    baseline_errors: Sequence[float]
    augmented_errors: Sequence[float]
    topic_ids: Optional[Sequence[Hashable]] = None


def _fit_ols(
    numeric: pd.DataFrame,
    coefficient_names: Sequence[str],
    design_columns: Sequence[np.ndarray],
) -> Dict[str, float]:
    design = np.column_stack((np.ones(len(numeric)), *design_columns))
    response = numeric["next_mom"].to_numpy(dtype=float)
    estimates, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if rank < len(coefficient_names):
        raise ValueError(
            "OLS design matrix is rank deficient: "
            f"rank={rank}, required={len(coefficient_names)}"
        )
    return {
        name: float(value) for name, value in zip(coefficient_names, estimates)
    }


def fit_baseline_model(observations: pd.DataFrame) -> Dict[str, float]:
    """Fit the five-coefficient, no-occupancy baseline with the shared guard."""
    columns = ("mom", "log1p_M", "burst", "next_mom")
    numeric = validate_fit_observations(observations, columns)
    mom = numeric["mom"].to_numpy(dtype=float)
    burst = numeric["burst"].to_numpy(dtype=float)
    return _fit_ols(
        numeric,
        BASELINE_COEFFICIENT_NAMES,
        (
            mom,
            numeric["log1p_M"].to_numpy(dtype=float),
            burst,
            mom * burst,
        ),
    )


def fit_baseline_coverage_model(observations: pd.DataFrame) -> Dict[str, float]:
    """Fit baseline + coverage for descriptive, non-gating sensitivity only.

    This model is excluded from the preregistered main Holm family (m=2).
    """
    columns = ("mom", "log1p_M", "burst", "coverage", "next_mom")
    numeric = validate_fit_observations(observations, columns)
    mom = numeric["mom"].to_numpy(dtype=float)
    burst = numeric["burst"].to_numpy(dtype=float)
    return _fit_ols(
        numeric,
        BASELINE_COVERAGE_COEFFICIENT_NAMES,
        (
            mom,
            numeric["log1p_M"].to_numpy(dtype=float),
            burst,
            mom * burst,
            numeric["coverage"].to_numpy(dtype=float),
        ),
    )


def fit_augmented_coverage_model(
    observations: pd.DataFrame, model: str
) -> Dict[str, float]:
    """Fit coverage + occupancy model k for non-gating sensitivity only.

    Neither model A nor model B from this coverage-adjusted fit is included in
    the preregistered main Holm family (m=2).
    """
    if model not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}; got {model!r}")
    occ_column = f"occ_{model}_centered"
    columns = (
        "mom",
        "log1p_M",
        "burst",
        "coverage",
        occ_column,
        "next_mom",
    )
    numeric = validate_fit_observations(observations, columns)
    mom = numeric["mom"].to_numpy(dtype=float)
    burst = numeric["burst"].to_numpy(dtype=float)
    occ = numeric[occ_column].to_numpy(dtype=float)
    return _fit_ols(
        numeric,
        AUGMENTED_COVERAGE_COEFFICIENT_NAMES,
        (
            mom,
            numeric["log1p_M"].to_numpy(dtype=float),
            burst,
            mom * burst,
            numeric["coverage"].to_numpy(dtype=float),
            occ,
            occ * burst,
        ),
    )


def _coerce_cell_errors(
    cell: Cell, paired: PairedCellErrors
) -> tuple[np.ndarray, np.ndarray, tuple[Hashable, ...]]:
    baseline = np.asarray(paired.baseline_errors, dtype=float)
    augmented = np.asarray(paired.augmented_errors, dtype=float)
    if baseline.ndim != 1 or augmented.ndim != 1:
        raise ValueError(f"{cell}: error arrays must be one-dimensional")
    if len(baseline) == 0 or len(baseline) != len(augmented):
        raise ValueError(f"{cell}: paired error arrays must have equal nonzero length")
    if not np.isfinite(baseline).all() or not np.isfinite(augmented).all():
        raise ValueError(f"{cell}: errors must all be finite")

    if paired.topic_ids is None:
        topic_ids: tuple[Hashable, ...] = tuple((cell, i) for i in range(len(baseline)))
    else:
        topic_ids = tuple(paired.topic_ids)
        if len(topic_ids) != len(baseline):
            raise ValueError(f"{cell}: topic_ids must align with paired errors")
        for topic_id in topic_ids:
            try:
                hash(topic_id)
            except TypeError as error:
                raise ValueError(f"{cell}: topic_ids must be hashable") from error
        if len(set(topic_ids)) != len(topic_ids):
            raise ValueError(f"{cell}: topic_ids must be unique within a cell")
    return baseline, augmented, topic_ids


def _validate_cell_mapping(
    cell_errors: Mapping[Cell, PairedCellErrors],
) -> Dict[Cell, tuple[np.ndarray, np.ndarray, tuple[Hashable, ...]]]:
    extras = set(cell_errors) - PREREGISTERED_CELLS
    if extras:
        raise ValueError(f"Unexpected cells outside preregistration: {sorted(extras)}")
    return {
        cell: _coerce_cell_errors(cell, paired)
        for cell, paired in cell_errors.items()
    }


def compute_error_statistics(
    cell_errors: Mapping[Cell, PairedCellErrors],
) -> Dict[str, object]:
    """Compute cell deltas and the preregistered T_k, B, and R_k statistics.

    Inputs are abstract paired errors and have no dependency on a panel or an
    outcome data source.  If a preregistered cell is absent, aggregate
    statistics are undefined (``None``), allowing the gate to fail closed.
    """
    validated = _validate_cell_mapping(cell_errors)
    deltas: Dict[Cell, float] = {}
    baseline_cell_mae: Dict[Cell, float] = {}
    for cell, (baseline, augmented, _) in validated.items():
        deltas[cell] = float(np.median(np.abs(baseline) - np.abs(augmented)))
        baseline_cell_mae[cell] = float(np.median(np.abs(baseline)))

    missing_cells = tuple(sorted(PREREGISTERED_CELLS - set(validated)))
    if missing_cells:
        mean_delta = None
        baseline_mae = None
        relative_effect = None
    else:
        mean_delta = float(sum(deltas.values()) / PREREGISTERED_CELL_COUNT)
        baseline_mae = float(
            sum(baseline_cell_mae.values()) / PREREGISTERED_CELL_COUNT
        )
        relative_effect = (
            float(mean_delta / baseline_mae) if baseline_mae > 0.0 else None
        )
    return {
        "cell_deltas": deltas,
        "T_k": mean_delta,
        "B": baseline_mae,
        "R_k": relative_effect,
        "missing_cells": missing_cells,
    }


def nine_cell_sign_consistency(cell_deltas: Mapping[Cell, float]) -> bool:
    """Require all nine preregistered deltas to be strictly positive.

    Per issue #8, values with absolute magnitude below 1e-9 are treated as
    zero, and any missing or additional cell fails the immutable gate.
    """
    if set(cell_deltas) != PREREGISTERED_CELLS:
        return False
    for value in cell_deltas.values():
        delta = float(value)
        if not math.isfinite(delta):
            return False
        if abs(delta) < ZERO_DELTA_TOLERANCE or delta <= 0.0:
            return False
    return True


def meets_sesoi(relative_effect: Optional[float]) -> bool:
    """Apply the immutable issue #8 SESOI threshold (R_k >= 0.05)."""
    if relative_effect is None:
        return False
    value = float(relative_effect)
    return math.isfinite(value) and value >= SESOI_THRESHOLD


def check_hhi_frozen_coefficient(path: Path) -> Dict[str, object]:
    """Check whether a frozen model-B ``occ_centered`` coefficient is negative."""
    coefficient = load_coefficients(Path(path))["occ_centered"]
    return {
        "occ_centered": coefficient,
        "is_negative": bool(coefficient < 0.0),
    }


def check_hhi_frozen_coefficients(
    coefficient_paths: Mapping[str, Path],
) -> Dict[str, object]:
    """Check the three preregistered domains' exploration-only HHI fits."""
    if set(coefficient_paths) != set(PREREGISTERED_DOMAINS):
        raise ValueError(
            "HHI coefficient paths must match preregistered domains: "
            f"{PREREGISTERED_DOMAINS}"
        )
    by_domain = {
        domain: check_hhi_frozen_coefficient(coefficient_paths[domain])
        for domain in PREREGISTERED_DOMAINS
    }
    return {
        "by_domain": by_domain,
        "all_negative": all(
            bool(result["is_negative"]) for result in by_domain.values()
        ),
    }


def holm_adjust_two_pvalues(raw_p_values: Mapping[str, float]) -> Dict[str, float]:
    """Apply Holm's step-down adjustment to the fixed main family m=2."""
    if len(raw_p_values) != HOLM_FAMILY_SIZE:
        raise ValueError(f"Holm family must contain exactly m={HOLM_FAMILY_SIZE}")
    validated = {name: float(value) for name, value in raw_p_values.items()}
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in validated.values()
    ):
        raise ValueError("Raw p-values must be finite and in [0, 1]")

    ordered = sorted(validated, key=lambda name: validated[name])
    first, second = ordered
    first_adjusted = min(1.0, 2.0 * validated[first])
    second_adjusted = min(1.0, max(first_adjusted, validated[second]))
    return {first: first_adjusted, second: second_adjusted}


def _cluster_bootstrap_sample(
    validated: Mapping[Cell, tuple[np.ndarray, np.ndarray, tuple[Hashable, ...]]],
    rng: np.random.Generator,
) -> Dict[Cell, PairedCellErrors]:
    sampled: Dict[Cell, PairedCellErrors] = {}
    for domain in PREREGISTERED_DOMAINS:
        domain_cells = {
            cell: values for cell, values in validated.items() if cell[0] == domain
        }
        clusters = tuple(
            dict.fromkeys(
                topic_id
                for _, _, topic_ids in domain_cells.values()
                for topic_id in topic_ids
            )
        )
        draws = rng.integers(0, len(clusters), size=len(clusters))
        multiplicities = Counter(clusters[index] for index in draws)
        for cell, (baseline, augmented, topic_ids) in domain_cells.items():
            repeats = np.asarray(
                [multiplicities[topic_id] for topic_id in topic_ids], dtype=int
            )
            sampled[cell] = PairedCellErrors(
                np.repeat(baseline, repeats),
                np.repeat(augmented, repeats),
            )
    return sampled


def _one_sided_topic_cluster_bootstrap_pvalue(
    cell_errors: Mapping[Cell, PairedCellErrors],
    n_bootstraps: int,
    rng: np.random.Generator,
) -> float:
    if isinstance(n_bootstraps, bool) or int(n_bootstraps) != n_bootstraps:
        raise ValueError("n_bootstraps must be a positive integer")
    n_bootstraps = int(n_bootstraps)
    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be a positive integer")
    validated = _validate_cell_mapping(cell_errors)
    if set(validated) != PREREGISTERED_CELLS:
        raise ValueError("Auxiliary bootstrap requires all nine cells")

    nonpositive = 0
    completed = 0
    attempts = 0
    max_attempts = 20 * n_bootstraps
    while completed < n_bootstraps and attempts < max_attempts:
        attempts += 1
        resampled = _cluster_bootstrap_sample(validated, rng)
        try:
            statistic = compute_error_statistics(resampled)["T_k"]
        except ValueError:
            # A sparse synthetic/source layout can leave a cell empty in one
            # cluster draw.  Such a draw is undefined and is redrawn.
            continue
        if statistic is None:
            continue
        nonpositive += float(statistic) <= 0.0
        completed += 1
    if completed < n_bootstraps:
        raise ValueError("Too few valid topic-cluster bootstrap replicates")
    return float((nonpositive + 1) / (n_bootstraps + 1))


def compute_auxiliary_holm_pvalues(
    model_cell_errors: Mapping[str, Mapping[Cell, PairedCellErrors]],
    *,
    n_bootstraps: int = 9999,
    random_seed: int = 8128,
) -> Dict[str, Dict[str, float]]:
    """Return one-sided cluster-bootstrap p-values as auxiliary information.

    The direction is fixed to augmented MAE < baseline MAE.  Models are not
    re-estimated.  These p-values are explicitly excluded from the main pass/
    fail decision; the coverage sensitivity fits are also excluded from this
    fixed Holm family (m=2).
    """
    if set(model_cell_errors) != set(HOLM_MODEL_NAMES):
        raise ValueError(
            f"Auxiliary Holm family must be exactly {HOLM_MODEL_NAMES}"
        )
    seed_sequence = np.random.SeedSequence(random_seed)
    raw = {
        model: _one_sided_topic_cluster_bootstrap_pvalue(
            model_cell_errors[model],
            n_bootstraps,
            np.random.default_rng(child_seed),
        )
        for model, child_seed in zip(
            HOLM_MODEL_NAMES, seed_sequence.spawn(HOLM_FAMILY_SIZE)
        )
    }
    return {
        "auxiliary_raw_one_sided_pvalues": raw,
        "auxiliary_holm_adjusted_pvalues": holm_adjust_two_pvalues(raw),
    }


def evaluate_occupancy_model_gate(
    model_cell_errors: Mapping[str, Mapping[Cell, PairedCellErrors]],
    hhi_coefficient_paths: Mapping[str, Path],
    *,
    n_bootstraps: int = 9999,
    random_seed: int = 8128,
) -> Dict[str, object]:
    """Evaluate the immutable main gate and separately return auxiliary p-values.

    Models A and B are evaluated independently.  Model A's ``full_passed`` is
    its effect-size gate alone; model B additionally requires the frozen HHI
    coefficient-sign check.  ``positive_result_detected`` uses preregistered
    OR semantics and is true when either model's ``full_passed`` is true.  No
    outcome source is read here; callers provide paired errors directly.
    Holm-adjusted p-values are returned under ``auxiliary_inference`` and never
    affect ``main_decision``.
    """
    if set(model_cell_errors) != set(HOLM_MODEL_NAMES):
        raise ValueError(f"Main model set must be exactly {HOLM_MODEL_NAMES}")

    effect_size_gates: Dict[str, Dict[str, object]] = {}
    for model in HOLM_MODEL_NAMES:
        statistics = compute_error_statistics(model_cell_errors[model])
        sign_passed = nine_cell_sign_consistency(statistics["cell_deltas"])
        sesoi_passed = meets_sesoi(statistics["R_k"])
        effect_size_gates[model] = {
            "statistics": statistics,
            "nine_cell_sign_passed": sign_passed,
            "sesoi_passed": sesoi_passed,
            "passed": sign_passed and sesoi_passed,
        }

    hhi_sign_check = check_hhi_frozen_coefficients(hhi_coefficient_paths)
    model_a_full_passed = bool(effect_size_gates["a"]["passed"])
    model_b_full_passed = bool(effect_size_gates["b"]["passed"]) and bool(
        hhi_sign_check["all_negative"]
    )
    effect_size_gates["a"].update(
        {
            "hhi_sign_required": False,
            "hhi_sign_passed": None,
            "full_passed": model_a_full_passed,
        }
    )
    effect_size_gates["b"].update(
        {
            "hhi_sign_required": True,
            "hhi_sign_passed": bool(hhi_sign_check["all_negative"]),
            "full_passed": model_b_full_passed,
        }
    )
    positive_result_detected = model_a_full_passed or model_b_full_passed
    any_missing_cells = any(
        bool(gate["statistics"]["missing_cells"])
        for gate in effect_size_gates.values()
    )
    if any_missing_cells:
        # A missing cell fails that model's effect-size gate.  Regardless of
        # whether the other model yields a positive result under OR semantics,
        # a complete m=2 Holm family cannot be formed.
        auxiliary_inference: Dict[str, object] = {
            "auxiliary_raw_one_sided_pvalues": {
                model: None for model in HOLM_MODEL_NAMES
            },
            "auxiliary_holm_adjusted_pvalues": {
                model: None for model in HOLM_MODEL_NAMES
            },
            "auxiliary_unavailable_reason": "one or more preregistered cells are missing",
        }
    else:
        auxiliary_inference = compute_auxiliary_holm_pvalues(
            model_cell_errors,
            n_bootstraps=n_bootstraps,
            random_seed=random_seed,
        )
    return {
        "main_decision": {
            "positive_result_detected": positive_result_detected,
            "effect_size_gates": effect_size_gates,
            "hhi_sign_check": hhi_sign_check,
        },
        "auxiliary_inference": auxiliary_inference,
    }
