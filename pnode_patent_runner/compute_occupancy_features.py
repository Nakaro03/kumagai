"""Compute topic-year organization-occupancy features.

This is the GitHub issue #8, section 12.6 feature-construction layer only.
It does not fit a model or inspect any holdout-period aggregate.  Organization
specialization always uses the full, domain-independent firm portfolio as its
denominator, even when the feature numerator comes from a domain slice.

Rows outside the topic universe are retained with ``in_topic_universe=False``
and missing occupancy features.  This makes zero-organization-mass exclusions
auditable.  Centering constants and printed summaries use reportable rows only;
the constants are then applied to all years in the output files.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

if __package__:
    from pnode_patent_runner.build_occupancy_panel import (
        DEFAULT_OUTPUT_DIR as DEFAULT_PANEL_DIR,
        MAX_REPORTING_YEAR,
        MIN_FILING_YEAR,
    )
    from pnode_patent_runner.slice_occupancy_panel_by_domain import (
        DEFAULT_DOMAINS,
        validate_domains,
    )
else:  # Support ``python pnode_patent_runner/<script>.py``.
    from build_occupancy_panel import (  # type: ignore[no-redef]
        DEFAULT_OUTPUT_DIR as DEFAULT_PANEL_DIR,
        MAX_REPORTING_YEAR,
        MIN_FILING_YEAR,
    )
    from slice_occupancy_panel_by_domain import (  # type: ignore[no-redef]
        DEFAULT_DOMAINS,
        validate_domains,
    )


DEFAULT_FULL_FIRM_EDGES_PATH = DEFAULT_PANEL_DIR / "firm_edges.tsv"
DEFAULT_BY_DOMAIN_DIR = DEFAULT_PANEL_DIR / "by_domain"
DEFAULT_OUTPUT_DIR = DEFAULT_PANEL_DIR / "occupancy_features"
DEFAULT_CHUNK_SIZE = 500_000
DEFAULT_TOLERANCE = 1e-9

TARGET_COLUMNS = ("filing_year", "maingroup", "target_mass")
EDGE_COLUMNS = ("filing_year", "assignee_id", "maingroup", "edge_weight")
KEY_COLUMNS = ("filing_year", "assignee_id")
OUTPUT_COLUMNS = (
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


@dataclass(frozen=True)
class OccupancyFeaturesResult:
    output_paths: Dict[str, Path]
    centering_constants: Dict[str, Dict[str, float]]
    summaries: pd.DataFrame


def validate_max_reporting_year(year: int) -> int:
    """Reject any request that would derive or print a holdout aggregate."""
    if year > MAX_REPORTING_YEAR:
        raise ValueError(
            "Holdout guard: max_reporting_year must be <= "
            f"{MAX_REPORTING_YEAR}; got {year}"
        )
    if year < MIN_FILING_YEAR:
        raise ValueError(
            f"max_reporting_year must be >= {MIN_FILING_YEAR}; got {year}"
        )
    return year


def _validate_frame(
    frame: pd.DataFrame,
    path: Path,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame.loc[:, list(required_columns)].copy()

    years = pd.to_numeric(frame["filing_year"], errors="raise")
    if years.isna().any() or ((years % 1) != 0).any():
        raise ValueError(f"{path} contains a missing or non-integer filing_year")
    frame["filing_year"] = years.astype("int64")

    for column in ("maingroup", "assignee_id"):
        if column not in frame:
            continue
        if frame[column].isna().any() or frame[column].str.strip().eq("").any():
            raise ValueError(f"{path} contains a missing or empty {column}")
        frame[column] = frame[column].str.strip()

    mass_column = "target_mass" if "target_mass" in frame else "edge_weight"
    frame[mass_column] = pd.to_numeric(frame[mass_column], errors="raise")
    if frame[mass_column].isna().any() or not frame[mass_column].map(
        math.isfinite
    ).all():
        raise ValueError(f"{path} contains a missing or non-finite {mass_column}")
    if (frame[mass_column] < 0).any():
        raise ValueError(f"{path} contains a negative {mass_column}")
    return frame


def _read_panel(path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", dtype="string")
    return _validate_frame(frame, path, required_columns)


def _aggregate_target(target: pd.DataFrame) -> pd.DataFrame:
    return (
        target.groupby(["filing_year", "maingroup"], as_index=False, sort=True)[
            "target_mass"
        ]
        .sum()
        .reset_index(drop=True)
    )


def _aggregate_edges(edges: pd.DataFrame) -> pd.DataFrame:
    return (
        edges.groupby(
            ["filing_year", "assignee_id", "maingroup"],
            as_index=False,
            sort=True,
        )["edge_weight"]
        .sum()
        .reset_index(drop=True)
    )


def _full_portfolio_totals(
    path: Path,
    relevant_keys: pd.DataFrame,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    """Sum all-maingroup portfolios for firm-years used in domain edges."""
    path = Path(path)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}")
    if not path.is_file():
        raise FileNotFoundError(path)

    key_index = pd.MultiIndex.from_frame(
        relevant_keys.loc[:, list(KEY_COLUMNS)].drop_duplicates()
    )
    partials: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        usecols=list(EDGE_COLUMNS),
        chunksize=chunk_size,
    )
    for chunk in reader:
        chunk = _validate_frame(chunk, path, EDGE_COLUMNS)
        if key_index.empty:
            continue
        chunk_keys = pd.MultiIndex.from_frame(chunk.loc[:, list(KEY_COLUMNS)])
        selected = chunk.loc[chunk_keys.isin(key_index), list(EDGE_COLUMNS)]
        if selected.empty:
            continue
        partials.append(
            selected.groupby(list(KEY_COLUMNS), as_index=False)["edge_weight"].sum()
        )

    if not partials:
        return pd.DataFrame(columns=(*KEY_COLUMNS, "portfolio_weight"))
    totals = pd.concat(partials, ignore_index=True).groupby(
        list(KEY_COLUMNS), as_index=False
    )["edge_weight"].sum()
    return totals.rename(columns={"edge_weight": "portfolio_weight"})


def compute_edge_contributions(
    domain_edges: pd.DataFrame,
    portfolio_totals: pd.DataFrame,
    tolerance: float = DEFAULT_TOLERANCE,
) -> pd.DataFrame:
    """Return positive edge-level q and specialization values.

    ``portfolio_totals`` must have been calculated from the full, unsliced
    firm-edge table.  Keeping this operation separate makes that invariant
    directly testable.
    """
    positive = domain_edges.loc[domain_edges["edge_weight"] > 0].copy()
    if positive.empty:
        return positive.assign(
            organization_mass=pd.Series(dtype="float64"),
            portfolio_weight=pd.Series(dtype="float64"),
            q=pd.Series(dtype="float64"),
            spec=pd.Series(dtype="float64"),
        )

    organization_mass = positive.groupby(
        ["filing_year", "maingroup"]
    )["edge_weight"].transform("sum")
    positive["organization_mass"] = organization_mass
    positive = positive.merge(
        portfolio_totals,
        on=list(KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    missing = positive["portfolio_weight"].isna() | (
        positive["portfolio_weight"] <= 0
    )
    if missing.any():
        keys = positive.loc[missing, list(KEY_COLUMNS)].drop_duplicates().to_dict(
            "records"
        )
        raise ValueError(
            "Positive domain edges have no positive full-portfolio denominator: "
            f"{keys[:10]}"
        )
    inconsistent = positive["edge_weight"] > positive["portfolio_weight"] + tolerance
    if inconsistent.any():
        rows = positive.loc[
            inconsistent, [*KEY_COLUMNS, "maingroup", "edge_weight", "portfolio_weight"]
        ].to_dict("records")
        raise ValueError(
            "A domain edge exceeds its full-portfolio denominator: " f"{rows[:10]}"
        )

    positive["q"] = positive["edge_weight"] / positive["organization_mass"]
    positive["spec"] = positive["edge_weight"] / positive["portfolio_weight"]
    return positive


def _compute_domain_features(
    target: pd.DataFrame,
    domain_edges: pd.DataFrame,
    portfolio_totals: pd.DataFrame,
    max_reporting_year: int,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, float]]:
    contributions = compute_edge_contributions(
        domain_edges, portfolio_totals, tolerance=tolerance
    )
    if contributions.empty:
        edge_stats = pd.DataFrame(
            columns=(
                "filing_year",
                "maingroup",
                "organization_mass",
                "occ_a",
                "HHI",
                "n_j",
            )
        )
    else:
        contributions["q_spec"] = contributions["q"] * contributions["spec"]
        contributions["q_squared"] = contributions["q"] ** 2
        edge_stats = contributions.groupby(
            ["filing_year", "maingroup"], as_index=False, sort=True
        ).agg(
            organization_mass=("edge_weight", "sum"),
            occ_a=("q_spec", "sum"),
            HHI=("q_squared", "sum"),
            n_j=("assignee_id", "nunique"),
        )

    candidates = _aggregate_target(target).merge(
        edge_stats,
        on=["filing_year", "maingroup"],
        how="outer",
        validate="one_to_one",
    )
    candidates["target_mass"] = candidates["target_mass"].fillna(0.0)
    candidates["organization_mass"] = candidates["organization_mass"].fillna(0.0)
    candidates["n_j"] = candidates["n_j"].fillna(0).astype("int64")
    candidates["coverage"] = candidates["organization_mass"] / candidates[
        "target_mass"
    ].where(candidates["target_mass"] > 0)
    candidates["in_topic_universe"] = (candidates["target_mass"] > 0) & (
        candidates["organization_mass"] > 0
    )

    covered = candidates["target_mass"] > 0
    invalid_coverage = covered & (
        (candidates["coverage"] < -tolerance)
        | (candidates["coverage"] > 1.0 + tolerance)
    )
    if invalid_coverage.any():
        bad = candidates.loc[
            invalid_coverage,
            ["filing_year", "maingroup", "target_mass", "organization_mass", "coverage"],
        ].to_dict("records")
        raise ValueError(f"Topic-year coverage is outside [0, 1]: {bad[:10]}")

    candidates["occ_b"] = math.nan
    singleton = candidates["n_j"] == 1
    multiple = candidates["n_j"] > 1
    candidates.loc[singleton, "occ_b"] = 1.0
    reciprocal_n = 1.0 / candidates.loc[multiple, "n_j"]
    candidates.loc[multiple, "occ_b"] = (
        candidates.loc[multiple, "HHI"] - reciprocal_n
    ) / (1.0 - reciprocal_n)

    outside = ~candidates["in_topic_universe"]
    candidates.loc[outside, ["occ_a", "occ_b"]] = math.nan
    reportable = candidates["in_topic_universe"] & (
        candidates["filing_year"] <= max_reporting_year
    )
    if not reportable.any():
        raise ValueError(
            "No topic-universe rows are available at or before "
            f"max_reporting_year={max_reporting_year}; centering is undefined"
        )
    centers = {
        "occ_a": float(candidates.loc[reportable, "occ_a"].mean()),
        "occ_b": float(candidates.loc[reportable, "occ_b"].mean()),
    }
    candidates["occ_a_centered"] = candidates["occ_a"] - centers["occ_a"]
    candidates["occ_b_centered"] = candidates["occ_b"] - centers["occ_b"]

    reportable_targets = covered & (candidates["filing_year"] <= max_reporting_year)
    n_target_rows = int(reportable_targets.sum())
    n_zero = int((reportable_targets & (candidates["n_j"] == 0)).sum())
    summary = {
        "topic_year_count": int(reportable.sum()),
        "occ_a_mean": centers["occ_a"],
        "occ_b_mean": centers["occ_b"],
        "n_j_zero_excluded_count": n_zero,
        "n_j_zero_excluded_fraction": (
            n_zero / n_target_rows if n_target_rows else math.nan
        ),
    }
    features = candidates.loc[:, list(OUTPUT_COLUMNS)].sort_values(
        ["filing_year", "maingroup"]
    )
    return features.reset_index(drop=True), contributions, centers, summary


def compute_occupancy_features(
    domains: Sequence[str] = DEFAULT_DOMAINS,
    full_firm_edges_path: Path = DEFAULT_FULL_FIRM_EDGES_PATH,
    by_domain_dir: Path = DEFAULT_BY_DOMAIN_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_reporting_year: int = MAX_REPORTING_YEAR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    tolerance: float = DEFAULT_TOLERANCE,
) -> OccupancyFeaturesResult:
    """Compute and write occupancy features for each requested domain."""
    max_reporting_year = validate_max_reporting_year(max_reporting_year)
    domains = validate_domains(domains)
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be finite and non-negative; got {tolerance}")

    by_domain_dir = Path(by_domain_dir)
    domain_inputs: Dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    all_key_frames: list[pd.DataFrame] = []
    for domain in domains:
        target_path = by_domain_dir / f"target_panel_{domain}.tsv"
        edge_path = by_domain_dir / f"firm_edges_{domain}.tsv"
        target = _aggregate_target(_read_panel(target_path, TARGET_COLUMNS))
        edges = _aggregate_edges(_read_panel(edge_path, EDGE_COLUMNS))
        domain_inputs[domain] = (target, edges)
        all_key_frames.append(edges.loc[:, list(KEY_COLUMNS)])

    relevant_keys = pd.concat(all_key_frames, ignore_index=True).drop_duplicates()
    portfolio_totals = _full_portfolio_totals(
        Path(full_firm_edges_path), relevant_keys, chunk_size=chunk_size
    )

    output_frames: Dict[str, pd.DataFrame] = {}
    centers_by_domain: Dict[str, Dict[str, float]] = {}
    summary_rows: list[Dict[str, object]] = []
    for domain in domains:
        target, edges = domain_inputs[domain]
        features, _, centers, summary = _compute_domain_features(
            target,
            edges,
            portfolio_totals,
            max_reporting_year=max_reporting_year,
            tolerance=tolerance,
        )
        output_frames[domain] = features
        centers_by_domain[domain] = centers
        summary_rows.append({"domain": domain, **summary})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: Dict[str, Path] = {}
    for domain in domains:
        path = output_dir / f"occupancy_features_{domain}.tsv"
        temporary = path.with_suffix(path.suffix + ".tmp")
        output_frames[domain].to_csv(temporary, sep="\t", index=False)
        temporary.replace(path)
        output_paths[domain] = path

    summaries = pd.DataFrame(summary_rows)
    print(
        "Occupancy feature summary "
        f"(guarded: filing_year <= {max_reporting_year}):",
        flush=True,
    )
    print(summaries.to_string(index=False), flush=True)
    return OccupancyFeaturesResult(output_paths, centers_by_domain, summaries)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument(
        "--full-firm-edges-path", type=Path, default=DEFAULT_FULL_FIRM_EDGES_PATH
    )
    parser.add_argument("--by-domain-dir", type=Path, default=DEFAULT_BY_DOMAIN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-reporting-year", type=int, default=MAX_REPORTING_YEAR
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    compute_occupancy_features(
        domains=args.domains,
        full_firm_edges_path=args.full_firm_edges_path,
        by_domain_dir=args.by_domain_dir,
        output_dir=args.output_dir,
        max_reporting_year=args.max_reporting_year,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
