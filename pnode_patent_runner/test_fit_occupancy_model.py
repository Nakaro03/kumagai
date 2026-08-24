"""Synthetic-only tests for issue #8 section 12.7 fit freezing."""
from __future__ import annotations

import inspect
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pnode_patent_runner import fit_occupancy_model as occupancy_fit
from pnode_patent_runner import phase0b_baseline_reverification as baseline


class FitOccupancyModelTest(unittest.TestCase):
    def _dataset_fixture(self, root: Path) -> tuple[Path, Path]:
        target_dir = root / "by_domain"
        feature_dir = root / "occupancy_features"
        target_dir.mkdir()
        feature_dir.mkdir()

        target_rows = []
        masses = {
            "X00A1": (3.0, 7.0, 15.0, 31.0, 63.0),
            "Y00B2": (8.0, 4.0, 10.0, 5.0, 12.0),
        }
        for year_index, year in enumerate(range(2012, 2017)):
            for cat, values in masses.items():
                target_rows.append((year, cat, values[year_index]))
        pd.DataFrame(
            target_rows,
            columns=("filing_year", "maingroup", "target_mass"),
        ).to_csv(
            target_dir / "target_panel_construction.tsv", sep="\t", index=False
        )

        feature_rows = []
        for t in (2013, 2014, 2015):
            for cat_index, cat in enumerate(masses):
                in_universe = not (t == 2014 and cat == "Y00B2")
                occ_a_centered = 0.1 * (t - 2012) + cat_index
                occ_b_centered = -0.2 * (t - 2012) - cat_index
                if not in_universe:
                    occ_a_centered = math.nan
                    occ_b_centered = math.nan
                feature_rows.append(
                    (
                        t,
                        cat,
                        0.5,
                        occ_a_centered,
                        0.25,
                        occ_b_centered,
                        0.8,
                        2,
                        in_universe,
                    )
                )
        pd.DataFrame(
            feature_rows,
            columns=occupancy_fit.OCCUPANCY_FEATURE_COLUMNS,
        ).to_csv(
            feature_dir / "occupancy_features_construction.tsv",
            sep="\t",
            index=False,
        )
        return target_dir, feature_dir

    @staticmethod
    def _known_linear_observations() -> tuple[pd.DataFrame, dict[str, float]]:
        rng = np.random.default_rng(8127)
        n_rows = 40
        mom = rng.normal(size=n_rows)
        log1p_mass = rng.uniform(0.0, 5.0, size=n_rows)
        burst = np.tile((0.0, 1.0), n_rows // 2)
        occ = rng.normal(size=n_rows)
        coefficients = {
            "intercept": 0.75,
            "mom": -0.4,
            "log1p_M": 0.12,
            "burst": 1.1,
            "mom_burst": 0.35,
            "occ_centered": -0.8,
            "occ_centered_burst": 0.6,
        }
        next_mom = (
            coefficients["intercept"]
            + coefficients["mom"] * mom
            + coefficients["log1p_M"] * log1p_mass
            + coefficients["burst"] * burst
            + coefficients["mom_burst"] * mom * burst
            + coefficients["occ_centered"] * occ
            + coefficients["occ_centered_burst"] * occ * burst
        )
        observations = pd.DataFrame(
            {
                "t": np.full(n_rows, 2015),
                "mom": mom,
                "log1p_M": log1p_mass,
                "burst": burst,
                "occ_a_centered": occ,
                "next_mom": next_mom,
            }
        )
        return observations, coefficients

    def test_dataset_reuses_baseline_builders_and_merges_features(self) -> None:
        self.assertIs(occupancy_fit.load_target_panel, baseline.load_target_panel)
        self.assertIs(occupancy_fit._mass_table, baseline._mass_table)
        self.assertIs(occupancy_fit._build_observations, baseline._build_observations)

        with tempfile.TemporaryDirectory() as directory:
            target_dir, feature_dir = self._dataset_fixture(Path(directory))
            observations = occupancy_fit.build_fit_dataset(
                "construction",
                target_panel_dir=target_dir,
                occupancy_features_dir=feature_dir,
            )

        row = observations.set_index(["t", "cat"]).loc[(2013, "X00A1")]
        self.assertAlmostEqual(row["mom"], math.log(2.0))
        self.assertAlmostEqual(row["next_mom"], math.log(2.0))
        self.assertEqual(row["burst"], 1.0)
        self.assertAlmostEqual(row["log1p_M"], math.log(8.0))
        self.assertAlmostEqual(row["occ_a_centered"], 0.1)
        self.assertAlmostEqual(row["occ_b_centered"], -0.2)
        self.assertTrue(bool(row["in_topic_universe"]))
        self.assertEqual(tuple(observations.columns), occupancy_fit.FIT_DATASET_COLUMNS)

    def test_outside_topic_universe_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target_dir, feature_dir = self._dataset_fixture(Path(directory))
            observations = occupancy_fit.build_fit_dataset(
                "construction",
                target_panel_dir=target_dir,
                occupancy_features_dir=feature_dir,
            )

        keys = set(zip(observations["t"], observations["cat"]))
        self.assertNotIn((2014, "Y00B2"), keys)
        self.assertEqual(len(observations), 5)
        self.assertTrue(observations["in_topic_universe"].all())

    def test_observation_without_a_feature_key_is_not_fit_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target_dir, feature_dir = self._dataset_fixture(Path(directory))
            path = feature_dir / "occupancy_features_construction.tsv"
            features = pd.read_csv(path, sep="\t")
            features = features.loc[
                ~(
                    (features["filing_year"] == 2015)
                    & (features["maingroup"] == "Y00B2")
                )
            ]
            features.to_csv(path, sep="\t", index=False)
            observations = occupancy_fit.build_fit_dataset(
                "construction",
                target_panel_dir=target_dir,
                occupancy_features_dir=feature_dir,
            )

        keys = set(zip(observations["t"], observations["cat"]))
        self.assertNotIn((2015, "Y00B2"), keys)
        self.assertEqual(len(observations), 4)

    def test_fit_rejects_transition_center_2016(self) -> None:
        observations, _ = self._known_linear_observations()
        observations.loc[0, "t"] = 2016
        with self.assertRaisesRegex(ValueError, r"t <= 2015"):
            occupancy_fit.fit_model(observations, "a")

    def test_coefficients_json_round_trip(self) -> None:
        _, coefficients = self._known_linear_observations()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fitted_coefficients_construction_a.json"
            returned_path = occupancy_fit.save_coefficients(coefficients, path)
            loaded = occupancy_fit.load_coefficients(path)
        self.assertEqual(returned_path, path)
        self.assertEqual(loaded, coefficients)

    def test_fit_recovers_known_relation_and_prediction_matches_by_hand(self) -> None:
        observations, expected_coefficients = self._known_linear_observations()
        fitted = occupancy_fit.fit_model(observations, "a")
        for name, expected in expected_coefficients.items():
            self.assertAlmostEqual(fitted[name], expected, places=12)

        features = {
            "mom": 0.5,
            "log1p_M": 2.0,
            "burst": 1.0,
            "occ_centered": -0.25,
        }
        expected_prediction = (
            0.75
            - 0.4 * 0.5
            + 0.12 * 2.0
            + 1.1
            + 0.35 * 0.5
            - 0.8 * -0.25
            + 0.6 * -0.25
        )
        actual = occupancy_fit.predict_next_mom(fitted, **features)
        self.assertAlmostEqual(actual, expected_prediction, places=12)

    def test_prediction_signature_is_scalar_only_and_contains_no_fit(self) -> None:
        signature = inspect.signature(occupancy_fit.predict_next_mom)
        self.assertEqual(
            tuple(signature.parameters),
            ("coefficients", "mom", "log1p_M", "burst", "occ_centered"),
        )
        for name in ("mom", "log1p_M", "burst", "occ_centered"):
            self.assertIs(
                signature.parameters[name].kind,
                inspect.Parameter.KEYWORD_ONLY,
            )
        source = inspect.getsource(occupancy_fit.predict_next_mom)
        self.assertNotIn(".fit(", source)
        self.assertNotIn("lstsq", source)
        self.assertNotIn("DataFrame", source)

    def test_cli_transition_max_guard_runs_before_input_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "Holdout guard"):
            occupancy_fit.main(
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
