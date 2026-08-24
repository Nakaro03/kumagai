"""Tests for the Phase 0-b occupancy-panel baseline reverification."""
from __future__ import annotations

import csv
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from pnode_patent_runner.build_occupancy_panel import (
    MAX_REPORTING_YEAR as BUILDER_MAX_REPORTING_YEAR,
)
from pnode_patent_runner.gate0_regime_detectability import (
    _build_observations as gate0_build_observations,
)
from pnode_patent_runner.gate0_regime_detectability import (
    _fit_interaction as gate0_fit_interaction,
)
from pnode_patent_runner.phase0b_baseline_reverification import (
    _build_observations,
    _mass_table,
    MAX_REPORTING_YEAR,
    load_target_panel,
    main,
    run_baseline_reverification,
)


YEARS = (2012, 2013, 2014, 2015, 2016)
CATEGORIES = tuple(f"G00X{i}" for i in range(10))

# Each row is the known log1p-mass increment from the previous year.  All ten
# categories have positive momentum, so the 80th-percentile rule selects the
# two largest entries in each transition year (six burst observations total).
KNOWN_MOMENTA = {
    2013: (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20),
    2014: (0.15, 0.04, 0.12, 0.09, 0.20, 0.03, 0.08, 0.18, 0.06, 0.11),
    2015: (0.05, 0.16, 0.07, 0.20, 0.02, 0.14, 0.09, 0.04, 0.18, 0.12),
    2016: (0.10, 0.02, 0.19, 0.06, 0.17, 0.08, 0.15, 0.03, 0.13, 0.11),
}


def _write_synthetic_panel(path: Path, include_holdout: bool = False) -> None:
    log_mass = np.array([3.0 + 0.05 * i for i in range(len(CATEGORIES))])
    rows = []
    for year in YEARS:
        if year > YEARS[0]:
            log_mass = log_mass + np.array(KNOWN_MOMENTA[year])
        for category, value in zip(CATEGORIES, np.expm1(log_mass)):
            rows.append((year, category, value))
    if include_holdout:
        rows.append((2017, CATEGORIES[0], 1.0))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("filing_year", "maingroup", "target_mass"))
        writer.writerows(rows)


class Phase0bBaselineReverificationTest(unittest.TestCase):
    def test_observations_and_fit_match_gate0_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            panel_path = Path(directory) / "target_panel.tsv"
            _write_synthetic_panel(panel_path)

            panel = load_target_panel(panel_path)
            mass = _mass_table(panel)
            transitions = [
                (2012, 2013, 2014),
                (2013, 2014, 2015),
                (2014, 2015, 2016),
            ]
            actual_obs = _build_observations(mass, transitions)
            gate0_obs = gate0_build_observations(mass, transitions)
            pd.testing.assert_frame_equal(actual_obs, gate0_obs)

            first_category = actual_obs[
                (actual_obs["cat"] == CATEGORIES[0]) & (actual_obs["t"] == 2013)
            ].iloc[0]
            self.assertAlmostEqual(first_category["mom"], KNOWN_MOMENTA[2013][0])
            self.assertAlmostEqual(
                first_category["next_mom"], KNOWN_MOMENTA[2014][0]
            )
            self.assertEqual(first_category["burst"], 0.0)
            self.assertEqual(int(actual_obs["burst"].sum()), 6)

            expected = gate0_fit_interaction(actual_obs)
            actual = run_baseline_reverification(
                panel_path, year_min=2013, max_reporting_year=2016
            )
            self.assertEqual(actual["status"], "ok")
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                if isinstance(actual[key], float):
                    self.assertTrue(
                        math.isclose(
                            actual[key],
                            expected[key],
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        ),
                        msg=f"{key}: {actual[key]} != {expected[key]}",
                    )
                else:
                    self.assertEqual(actual[key], expected[key])

    def test_nonsignificant_result_is_reported_with_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            panel_path = Path(directory) / "target_panel.tsv"
            _write_synthetic_panel(panel_path)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = main(
                    [
                        "--panel-path",
                        str(panel_path),
                        "--year-min",
                        "2013",
                    ]
                )

            result = json.loads(stdout.getvalue())
            self.assertGreater(result["p_mom_burst"], 0.05)
            self.assertEqual(return_code, 0)
            self.assertNotIn("gate0_pass", result)

    def test_panel_with_2017_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            panel_path = Path(directory) / "target_panel.tsv"
            _write_synthetic_panel(panel_path, include_holdout=True)
            with self.assertRaisesRegex(ValueError, "Holdout guard"):
                run_baseline_reverification(
                    panel_path, year_min=2013, max_reporting_year=2016
                )

    def test_transition_year_min_requiring_2017_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            panel_path = Path(directory) / "target_panel.tsv"
            _write_synthetic_panel(panel_path)
            with self.assertRaisesRegex(ValueError, r"t \+ 1 <= 2016"):
                run_baseline_reverification(
                    panel_path, year_min=2016, max_reporting_year=2016
                )

    def test_max_reporting_year_cannot_enter_holdout(self) -> None:
        self.assertEqual(MAX_REPORTING_YEAR, BUILDER_MAX_REPORTING_YEAR)
        with tempfile.TemporaryDirectory() as directory:
            panel_path = Path(directory) / "target_panel.tsv"
            _write_synthetic_panel(panel_path)
            with self.assertRaisesRegex(ValueError, "Holdout guard"):
                run_baseline_reverification(
                    panel_path, year_min=2013, max_reporting_year=2017
                )


if __name__ == "__main__":
    unittest.main()
