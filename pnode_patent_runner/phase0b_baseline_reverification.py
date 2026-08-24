#!/usr/bin/env python3
"""Reverify the momentum x burst baseline on the occupancy target panel.

This module implements GitHub issue #8, section 12.3 (Phase 0-b) only.  It
uses the domain-independent ``target_panel.tsv`` and never treats statistical
significance as a gate for later occupancy analysis.

The observation construction and interaction fit are mathematically identical
to ``gate0_regime_detectability._build_observations`` and
``gate0_regime_detectability._fit_interaction`` respectively.  They are kept
local so that the older Gate 0 program and its decision semantics remain
unchanged.

Example::

    python -m pnode_patent_runner.phase0b_baseline_reverification \
        --panel-path data/processed/occupancy_panel/target_panel.tsv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

BURST_PERCENTILE = 80.0
HOLDOUT_START_YEAR = 2017
# Keep this hard limit aligned with build_occupancy_panel.MAX_REPORTING_YEAR.
MAX_REPORTING_YEAR = HOLDOUT_START_YEAR - 1
DEFAULT_TRANSITION_YEAR_MIN = 2011
REQUIRED_PANEL_COLUMNS = ("filing_year", "maingroup", "target_mass")


def validate_analysis_years(year_min: int, max_reporting_year: int) -> None:
    """Validate the public-entrypoint years against the 2017+ holdout.

    ``year_min`` is the first transition center year ``t``.  Therefore both
    ``t - 1`` and ``t + 1`` must be legal, and at least one transition must fit
    within ``max_reporting_year``.
    """
    if max_reporting_year > MAX_REPORTING_YEAR:
        raise ValueError(
            "Holdout guard: max_reporting_year must be <= "
            f"{MAX_REPORTING_YEAR}; got {max_reporting_year}"
        )
    if year_min + 1 > max_reporting_year:
        raise ValueError(
            "Holdout guard: year_min is a transition year t and requires "
            f"t + 1 <= {max_reporting_year}; got year_min={year_min}"
        )


def load_target_panel(
    panel_path: Path, max_reporting_year: int = MAX_REPORTING_YEAR
) -> pd.DataFrame:
    """Load and validate ``target_panel.tsv`` without admitting holdout rows."""
    if max_reporting_year > MAX_REPORTING_YEAR:
        raise ValueError(
            "Holdout guard: max_reporting_year must be <= "
            f"{MAX_REPORTING_YEAR}; got {max_reporting_year}"
        )

    panel = pd.read_csv(panel_path, sep="\t")
    missing = set(REQUIRED_PANEL_COLUMNS) - set(panel.columns)
    if missing:
        raise ValueError(f"{panel_path} is missing required columns: {sorted(missing)}")
    panel = panel.loc[:, list(REQUIRED_PANEL_COLUMNS)].copy()

    numeric_years = pd.to_numeric(panel["filing_year"], errors="coerce")
    if numeric_years.isna().any() or not np.equal(
        numeric_years, np.floor(numeric_years)
    ).all():
        raise ValueError("filing_year must contain only integer years")
    panel["filing_year"] = numeric_years.astype(int)

    holdout_rows = panel["filing_year"] > max_reporting_year
    if holdout_rows.any():
        first_holdout_year = int(panel.loc[holdout_rows, "filing_year"].min())
        raise ValueError(
            "Holdout guard: target_panel contains filing_year "
            f"{first_holdout_year}, beyond max_reporting_year={max_reporting_year}"
        )

    panel["target_mass"] = pd.to_numeric(panel["target_mass"], errors="coerce")
    if not np.isfinite(panel["target_mass"]).all():
        raise ValueError("target_mass must contain only finite numeric values")
    if (panel["target_mass"] < 0).any():
        raise ValueError("target_mass must be non-negative")
    if panel["maingroup"].isna().any() or (
        panel["maingroup"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("maingroup must contain only non-empty values")
    if panel.duplicated(["filing_year", "maingroup"]).any():
        raise ValueError("target_panel must be unique by (filing_year, maingroup)")

    panel["maingroup"] = panel["maingroup"].astype(str)
    return panel


def _mass_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Return M_j(t), indexed by filing year with maingroups as columns."""
    return panel.pivot(
        index="filing_year", columns="maingroup", values="target_mass"
    ).fillna(0.0)


def _build_observations(
    mass: pd.DataFrame, transitions: List[Tuple[int, int, int]]
) -> pd.DataFrame:
    """Build observations exactly as Gate 0's ``_build_observations`` does.

    For every transition ``(t-1, t, t+1)``, burst is based only on that
    transition year's positive momentum distribution.  Rows whose three-year
    mass is zero are discarded.
    """
    rows = []
    for (tm1, t, t1) in transitions:
        if tm1 not in mass.index or t not in mass.index or t1 not in mass.index:
            continue
        M_tm1 = mass.loc[tm1]
        M_t = mass.loc[t]
        M_t1 = mass.loc[t1]
        cats = M_t.index
        mom_t = np.log1p(M_t.reindex(cats, fill_value=0)) - np.log1p(
            M_tm1.reindex(cats, fill_value=0)
        )
        next_mom = np.log1p(M_t1.reindex(cats, fill_value=0)) - np.log1p(
            M_t.reindex(cats, fill_value=0)
        )
        pos_mom = mom_t[mom_t > 0]
        thr = (
            np.percentile(pos_mom, BURST_PERCENTILE)
            if len(pos_mom)
            else np.inf
        )
        burst = (mom_t >= thr).astype(float)
        keep = (
            M_tm1.reindex(cats, fill_value=0)
            + M_t.reindex(cats, fill_value=0)
            + M_t1.reindex(cats, fill_value=0)
        ) > 0
        df = pd.DataFrame(
            {
                "cat": cats,
                "t": t,
                "mom": mom_t.values,
                "burst": burst.values,
                "next_mom": next_mom.values,
            }
        )
        rows.append(df[keep.values])
    if not rows:
        return pd.DataFrame(columns=["cat", "t", "mom", "burst", "next_mom"])
    return pd.concat(rows, ignore_index=True)


def _fit_interaction(obs: pd.DataFrame) -> Dict:
    """Fit Gate 0's identical OLS/HC1/category-cluster specification."""
    if len(obs) < 20 or obs["burst"].sum() < 5:
        return {
            "n": int(len(obs)),
            "n_burst": int(obs["burst"].sum()),
            "status": "insufficient_data",
        }
    X = pd.DataFrame(
        {
            "mom": obs["mom"],
            "burst": obs["burst"],
            "mom_burst": obs["mom"] * obs["burst"],
        }
    )
    X = sm.add_constant(X)
    y = obs["next_mom"]
    model = sm.OLS(y, X).fit(cov_type="HC1")
    model_cluster = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": obs["cat"].to_numpy()}
    )
    p_hc1 = float(model.pvalues["mom_burst"])
    p_cluster = float(model_cluster.pvalues["mom_burst"])
    p_conservative = max(p_hc1, p_cluster)
    return {
        "n": int(len(obs)),
        "n_burst": int(obs["burst"].sum()),
        "n_categories": int(obs["cat"].nunique()),
        "coef_mom_burst": float(model.params["mom_burst"]),
        "se_mom_burst_hc1": float(model.bse["mom_burst"]),
        "p_mom_burst_hc1": p_hc1,
        "p_mom_burst_cluster": p_cluster,
        "p_mom_burst": p_conservative,
        "coef_mom": float(model.params["mom"]),
        "p_mom": float(model.pvalues["mom"]),
        "r2": float(model.rsquared),
        "status": "ok",
    }


def run_baseline_reverification(
    panel_path: Path,
    year_min: int = DEFAULT_TRANSITION_YEAR_MIN,
    max_reporting_year: int = MAX_REPORTING_YEAR,
) -> Dict:
    """Run Phase 0-b for all maingroups, returning statistics without a gate."""
    validate_analysis_years(year_min, max_reporting_year)
    panel = load_target_panel(panel_path, max_reporting_year=max_reporting_year)
    mass = _mass_table(panel)
    transitions = [
        (t - 1, t, t + 1) for t in range(year_min, max_reporting_year)
    ]
    obs = _build_observations(mass, transitions)
    return _fit_interaction(obs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 0-b: reverify momentum x burst on the pre-holdout "
            "occupancy target panel"
        )
    )
    parser.add_argument(
        "--panel-path", type=Path, required=True, help="Path to target_panel.tsv"
    )
    parser.add_argument(
        "--year-min",
        type=int,
        default=DEFAULT_TRANSITION_YEAR_MIN,
        help="Minimum transition center year t (default: %(default)s)",
    )
    parser.add_argument(
        "--max-reporting-year",
        type=int,
        default=MAX_REPORTING_YEAR,
        help="Maximum year allowed in data and transitions (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_baseline_reverification(
        panel_path=args.panel_path,
        year_min=args.year_min,
        max_reporting_year=args.max_reporting_year,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
