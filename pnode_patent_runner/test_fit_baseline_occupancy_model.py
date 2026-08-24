"""Synthetic-only tests for freezing the Confirmation B baseline."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pnode_patent_runner import fit_baseline_occupancy_model as baseline_fit
from pnode_patent_runner.fit_occupancy_model import load_coefficients


class FitBaselineOccupancyModelTest(unittest.TestCase):
    def test_fit_freezes_existing_prediction_format_with_zero_occupancy_terms(self) -> None:
        rng = np.random.default_rng(1210)
        rows = 50
        observations = pd.DataFrame(
            {
                "t": np.full(rows, 2015),
                "mom": rng.normal(size=rows),
                "log1p_M": rng.uniform(0.0, 4.0, size=rows),
                "burst": np.tile((0.0, 1.0), rows // 2),
            }
        )
        expected = {
            "intercept": 0.4,
            "mom": -0.2,
            "log1p_M": 0.1,
            "burst": 0.7,
            "mom_burst": 0.3,
        }
        observations["next_mom"] = (
            expected["intercept"]
            + expected["mom"] * observations["mom"]
            + expected["log1p_M"] * observations["log1p_M"]
            + expected["burst"] * observations["burst"]
            + expected["mom_burst"]
            * observations["mom"]
            * observations["burst"]
        )

        with tempfile.TemporaryDirectory() as directory, patch.object(
            baseline_fit, "build_fit_dataset", return_value=observations
        ):
            output_dir = Path(directory)
            result = baseline_fit.fit_and_freeze_baseline(
                domains=("construction",), output_dir=output_dir
            )
            loaded = load_coefficients(
                output_dir / "fitted_coefficients_construction_baseline.json"
            )

        for name, value in expected.items():
            self.assertAlmostEqual(result["construction"][name], value, places=12)
        self.assertEqual(loaded, result["construction"])
        self.assertEqual(loaded["occ_centered"], 0.0)
        self.assertEqual(loaded["occ_centered_burst"], 0.0)

    def test_cli_rejects_post_exploration_fit_before_reading_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Holdout guard"):
            baseline_fit.main(
                [
                    "--domains",
                    "construction",
                    "--target-panel-dir",
                    "does-not-exist",
                    "--occupancy-features-dir",
                    "does-not-exist",
                    "--transition-year-max",
                    "2016",
                ]
            )


if __name__ == "__main__":
    unittest.main()
