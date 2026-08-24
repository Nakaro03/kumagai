"""Slice the occupancy panel into CPC-prefix domains and report coverage.

This is the Phase 0-c (GitHub issue #8, section 12.5) data-slicing layer.
It does not compute occupancy features, fit models, or report any aggregate
from the holdout period.

The default comparison uses construction, agrifood, and computing.  Energy
(Y02) remains available as an explicit domain, but is not a default because
Y02 is a cross-cutting tag normally recorded as ``additional``: only 2 of
869,956 observed Y02 records were ``inventional``, making it effectively empty
in the inventional-only occupancy panel.
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
    from pnode_patent_runner.extract_domain_bipartite import (
        DOMAINS,
        domain_prefixes_match,
    )
else:  # Support ``python pnode_patent_runner/<script>.py``.
    from build_occupancy_panel import (  # type: ignore[no-redef]
        DEFAULT_OUTPUT_DIR as DEFAULT_PANEL_DIR,
        MAX_REPORTING_YEAR,
        MIN_FILING_YEAR,
    )
    from extract_domain_bipartite import (  # type: ignore[no-redef]
        DOMAINS,
        domain_prefixes_match,
    )


DEFAULT_TARGET_PANEL_PATH = DEFAULT_PANEL_DIR / "target_panel.tsv"
DEFAULT_FIRM_EDGES_PATH = DEFAULT_PANEL_DIR / "firm_edges.tsv"
DEFAULT_OUTPUT_DIR = DEFAULT_PANEL_DIR / "by_domain"
# Y02 energy is almost entirely an ``additional`` cross-cutting tag (only 2 of
# 869,956 observed rows were ``inventional``), so it is unusable as a default
# domain for this inventional-only panel.  Keep it in DOMAINS for explicit use.
DEFAULT_DOMAINS = ("construction", "agrifood", "computing")
DEFAULT_TOLERANCE = 1e-6

TARGET_COLUMNS = ("filing_year", "maingroup", "target_mass")
EDGE_COLUMNS = ("filing_year", "assignee_id", "maingroup", "edge_weight")


@dataclass(frozen=True)
class SliceResult:
    target_paths: Dict[str, Path]
    firm_edges_paths: Dict[str, Path]
    summary_path: Path
    summary: pd.DataFrame


def validate_max_reporting_year(year: int) -> int:
    """Reject reporting requests that enter the project holdout period."""
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


def validate_domains(domains: Sequence[str]) -> tuple[str, ...]:
    """Validate domain names while retaining the requested output order."""
    requested = tuple(domains)
    if not requested:
        raise ValueError("At least one domain must be specified")
    unknown = sorted(set(requested) - set(DOMAINS))
    if unknown:
        raise ValueError(
            f"Unknown domain(s): {unknown}; available domains: {sorted(DOMAINS)}"
        )
    duplicates = sorted({domain for domain in requested if requested.count(domain) > 1})
    if duplicates:
        raise ValueError(f"Duplicate domain(s): {duplicates}")
    return requested


def _read_panel(path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", dtype="string")
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame.loc[:, list(required_columns)].copy()

    years = pd.to_numeric(frame["filing_year"], errors="raise")
    if years.isna().any() or ((years % 1) != 0).any():
        raise ValueError(f"{path} contains a missing or non-integer filing_year")
    frame["filing_year"] = years.astype("int64")

    mass_column = "target_mass" if "target_mass" in frame else "edge_weight"
    frame[mass_column] = pd.to_numeric(frame[mass_column], errors="raise")
    if (
        frame[mass_column].isna().any()
        or not frame[mass_column].map(math.isfinite).all()
    ):
        raise ValueError(f"{path} contains a missing or non-finite {mass_column}")
    if frame["maingroup"].isna().any():
        raise ValueError(f"{path} contains a missing maingroup")
    return frame


def _audit_domain_summary(
    domain: str,
    yearly: pd.DataFrame,
    target: pd.DataFrame,
    edges: pd.DataFrame,
    tolerance: float,
) -> None:
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be finite and non-negative; got {tolerance}")
    if (target["target_mass"] < 0).any():
        raise ValueError(f"[{domain}] target_mass is negative")
    if (edges["edge_weight"] < 0).any():
        raise ValueError(f"[{domain}] organization mass (edge_weight) is negative")
    if (yearly["target_mass"] < 0).any():
        raise ValueError(f"[{domain}] yearly target_mass is negative")
    if (yearly["organization_mass"] < 0).any():
        raise ValueError(f"[{domain}] yearly organization_mass is negative")
    coverage = yearly["coverage"]
    if coverage.isna().any() or not coverage.map(math.isfinite).all():
        raise ValueError(f"[{domain}] coverage is missing or non-finite")
    if ((coverage < -tolerance) | (coverage > 1.0 + tolerance)).any():
        bad = yearly.loc[
            (coverage < -tolerance) | (coverage > 1.0 + tolerance),
            ["filing_year", "coverage"],
        ].to_dict("records")
        raise ValueError(
            f"[{domain}] coverage is outside [0, 1] with tolerance {tolerance}: {bad}"
        )


def _yearly_summary(target: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    target_yearly = target.groupby("filing_year", as_index=False).agg(
        maingroup_count=("maingroup", "nunique"),
        target_mass=("target_mass", "sum"),
    )
    edge_yearly = edges.groupby("filing_year", as_index=False).agg(
        organization_mass=("edge_weight", "sum")
    )
    yearly = target_yearly.merge(edge_yearly, on="filing_year", how="outer")
    yearly["maingroup_count"] = yearly["maingroup_count"].fillna(0).astype("int64")
    yearly[["target_mass", "organization_mass"]] = yearly[
        ["target_mass", "organization_mass"]
    ].fillna(0.0)
    yearly["coverage"] = yearly["organization_mass"] / yearly["target_mass"]
    both_zero = (yearly["target_mass"] == 0) & (yearly["organization_mass"] == 0)
    yearly.loc[both_zero, "coverage"] = 0.0
    return yearly.sort_values("filing_year").reset_index(drop=True)


def slice_occupancy_panel_by_domain(
    target_panel_path: Path = DEFAULT_TARGET_PANEL_PATH,
    firm_edges_path: Path = DEFAULT_FIRM_EDGES_PATH,
    domains: Sequence[str] = DEFAULT_DOMAINS,
    max_reporting_year: int = MAX_REPORTING_YEAR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tolerance: float = DEFAULT_TOLERANCE,
) -> SliceResult:
    """Write guarded domain slices and a horizontally arranged yearly summary."""
    max_reporting_year = validate_max_reporting_year(max_reporting_year)
    domains = validate_domains(domains)
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be finite and non-negative; got {tolerance}")

    target_panel = _read_panel(Path(target_panel_path), TARGET_COLUMNS)
    firm_edges = _read_panel(Path(firm_edges_path), EDGE_COLUMNS)

    # The reporting guard is applied before domain slicing, aggregation, or output.
    target_panel = target_panel[target_panel["filing_year"] <= max_reporting_year]
    firm_edges = firm_edges[firm_edges["filing_year"] <= max_reporting_year]

    domain_frames: Dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    domain_yearly: Dict[str, pd.DataFrame] = {}
    all_years: set[int] = set()
    for domain in domains:
        prefixes = DOMAINS[domain]
        target = target_panel[
            domain_prefixes_match(target_panel["maingroup"], prefixes)
        ].copy()
        edges = firm_edges[
            domain_prefixes_match(firm_edges["maingroup"], prefixes)
        ].copy()
        target = target.sort_values(["filing_year", "maingroup"]).reset_index(drop=True)
        edges = edges.sort_values(
            ["filing_year", "assignee_id", "maingroup"]
        ).reset_index(drop=True)
        yearly = _yearly_summary(target, edges)
        _audit_domain_summary(domain, yearly, target, edges, tolerance)
        domain_frames[domain] = (target, edges)
        domain_yearly[domain] = yearly
        all_years.update(yearly["filing_year"].astype(int).tolist())

    summary = pd.DataFrame({"filing_year": sorted(all_years)})
    for domain in domains:
        renamed = domain_yearly[domain].rename(
            columns={
                column: f"{domain}_{column}"
                for column in (
                    "maingroup_count",
                    "target_mass",
                    "organization_mass",
                    "coverage",
                )
            }
        )
        summary = summary.merge(renamed, on="filing_year", how="left")
        count_column = f"{domain}_maingroup_count"
        mass_columns = [
            f"{domain}_target_mass",
            f"{domain}_organization_mass",
        ]
        coverage_column = f"{domain}_coverage"
        summary[count_column] = summary[count_column].fillna(0).astype("int64")
        summary[mass_columns] = summary[mass_columns].fillna(0.0)
        summary[coverage_column] = summary[coverage_column].fillna(0.0)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_paths: Dict[str, Path] = {}
    edge_paths: Dict[str, Path] = {}
    for domain in domains:
        target_path = output_dir / f"target_panel_{domain}.tsv"
        edge_path = output_dir / f"firm_edges_{domain}.tsv"
        target, edges = domain_frames[domain]
        target.to_csv(target_path, sep="\t", index=False)
        edges.to_csv(edge_path, sep="\t", index=False)
        target_paths[domain] = target_path
        edge_paths[domain] = edge_path

    summary_path = output_dir / "domain_coverage_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(
        f"Domain coverage summary (guarded: filing_year <= {max_reporting_year}):",
        flush=True,
    )
    print(summary.to_string(index=False), flush=True)
    return SliceResult(target_paths, edge_paths, summary_path, summary)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-panel-path", type=Path, default=DEFAULT_TARGET_PANEL_PATH
    )
    parser.add_argument("--firm-edges-path", type=Path, default=DEFAULT_FIRM_EDGES_PATH)
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument(
        "--max-reporting-year", type=int, default=MAX_REPORTING_YEAR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    slice_occupancy_panel_by_domain(
        target_panel_path=args.target_panel_path,
        firm_edges_path=args.firm_edges_path,
        domains=args.domains,
        max_reporting_year=args.max_reporting_year,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
