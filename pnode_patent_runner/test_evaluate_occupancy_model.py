"""Synthetic tests for issue #8 section 12.8 model and gate logic.

The only repository data read here are the explicitly permitted, exploration-
period frozen model-B coefficient JSON files used by the HHI sign test.
"""
from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pnode_patent_runner import evaluate_occupancy_model as evaluation
from pnode_patent_runner.fit_occupancy_model import (
    COEFFICIENT_NAMES,
    load_coefficients,
    save_coefficients,
)


class EvaluateOccupancyModelTest(unittest.TestCase):
    @staticmethod
    def _fit_observations() -> pd.DataFrame:
        rng = np.random.default_rng(1208)
        n_rows = 80
        return pd.DataFrame(
            {
                "t": np.full(n_rows, 2015),
                "mom": rng.normal(size=n_rows),
                "log1p_M": rng.uniform(0.0, 5.0, size=n_rows),
                "burst": np.tile((0.0, 1.0), n_rows // 2),
                "coverage": rng.uniform(0.2, 0.95, size=n_rows),
                "occ_a_centered": rng.normal(size=n_rows),
                "occ_b_centered": rng.normal(size=n_rows),
                "next_mom": np.zeros(n_rows),
            }
        )

    @staticmethod
    def _synthetic_cells(delta: float = 1.0) -> dict:
        cells = {}
        for domain in evaluation.PREREGISTERED_DOMAINS:
            for year in evaluation.PREREGISTERED_TRANSITION_YEARS:
                cells[(domain, year)] = evaluation.PairedCellErrors(
                    baseline_errors=(10.0, -10.0, 10.0),
                    augmented_errors=(10.0 - delta, delta - 10.0, 10.0 - delta),
                    topic_ids=("topic-1", "topic-2", "topic-3"),
                )
        return cells

    @staticmethod
    def _write_negative_hhi_coefficients(root: Path) -> dict[str, Path]:
        paths = {}
        for domain in evaluation.PREREGISTERED_DOMAINS:
            path = root / f"fitted_coefficients_{domain}_b.json"
            save_coefficients(
                {
                    name: (-0.25 if name == "occ_centered" else 0.0)
                    for name in COEFFICIENT_NAMES
                },
                path,
            )
            paths[domain] = path
        return paths

    def test_baseline_fit_recovers_known_coefficients(self) -> None:
        observations = self._fit_observations()
        expected = {
            "intercept": 0.8,
            "mom": -0.3,
            "log1p_M": 0.15,
            "burst": 0.7,
            "mom_burst": -0.2,
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
        actual = evaluation.fit_baseline_model(observations)
        for name, value in expected.items():
            self.assertAlmostEqual(actual[name], value, places=12)

    def test_coverage_sensitivity_fits_recover_known_coefficients(self) -> None:
        observations = self._fit_observations()
        baseline_expected = {
            "intercept": -0.4,
            "mom": 0.2,
            "log1p_M": -0.1,
            "burst": 0.9,
            "mom_burst": 0.35,
            "coverage": 1.25,
        }
        observations["next_mom"] = (
            baseline_expected["intercept"]
            + baseline_expected["mom"] * observations["mom"]
            + baseline_expected["log1p_M"] * observations["log1p_M"]
            + baseline_expected["burst"] * observations["burst"]
            + baseline_expected["mom_burst"]
            * observations["mom"]
            * observations["burst"]
            + baseline_expected["coverage"] * observations["coverage"]
        )
        actual = evaluation.fit_baseline_coverage_model(observations)
        for name, value in baseline_expected.items():
            self.assertAlmostEqual(actual[name], value, places=12)

        augmented_expected = {
            **baseline_expected,
            "occ_centered": -0.65,
            "occ_centered_burst": 0.45,
        }
        observations["next_mom"] += (
            augmented_expected["occ_centered"] * observations["occ_a_centered"]
            + augmented_expected["occ_centered_burst"]
            * observations["occ_a_centered"]
            * observations["burst"]
        )
        actual = evaluation.fit_augmented_coverage_model(observations, "a")
        for name, value in augmented_expected.items():
            self.assertAlmostEqual(actual[name], value, places=12)

    def test_all_new_fits_reject_transition_center_after_2015(self) -> None:
        observations = self._fit_observations()
        observations.loc[0, "t"] = 2016
        fit_calls = (
            lambda: evaluation.fit_baseline_model(observations),
            lambda: evaluation.fit_baseline_coverage_model(observations),
            lambda: evaluation.fit_augmented_coverage_model(observations, "a"),
            lambda: evaluation.fit_augmented_coverage_model(observations, "b"),
        )
        for fit_call in fit_calls:
            with self.subTest(fit_call=fit_call), self.assertRaisesRegex(
                ValueError, r"t <= 2015"
            ):
                fit_call()

    def test_coverage_fits_are_documented_as_non_gating_and_outside_holm(self) -> None:
        for function in (
            evaluation.fit_baseline_coverage_model,
            evaluation.fit_augmented_coverage_model,
        ):
            docstring = inspect.getdoc(function)
            self.assertIn("non-gating", docstring)
            self.assertIn("Holm family (m=2)", docstring)

    def test_delta_T_B_R_match_hand_calculation(self) -> None:
        cells = {}
        for delta, cell in enumerate(sorted(evaluation.PREREGISTERED_CELLS), start=1):
            cells[cell] = evaluation.PairedCellErrors(
                baseline_errors=(10.0, -10.0, 10.0),
                augmented_errors=(10.0 - delta, delta - 10.0, 10.0 - delta),
            )
        statistics = evaluation.compute_error_statistics(cells)
        expected_deltas = {
            cell: float(delta)
            for delta, cell in enumerate(
                sorted(evaluation.PREREGISTERED_CELLS), start=1
            )
        }
        self.assertEqual(statistics["cell_deltas"], expected_deltas)
        self.assertAlmostEqual(statistics["T_k"], 5.0)
        self.assertAlmostEqual(statistics["B"], 10.0)
        self.assertAlmostEqual(statistics["R_k"], 0.5)
        self.assertEqual(statistics["missing_cells"], ())

    def test_nine_cell_sign_gate_passes_only_complete_strictly_positive_set(self) -> None:
        deltas = {cell: 0.2 for cell in evaluation.PREREGISTERED_CELLS}
        self.assertTrue(evaluation.nine_cell_sign_consistency(deltas))

        first_cell = next(iter(evaluation.PREREGISTERED_CELLS))
        for nonpositive in (0.0, -0.1):
            changed = {**deltas, first_cell: nonpositive}
            self.assertFalse(evaluation.nine_cell_sign_consistency(changed))

        missing = dict(deltas)
        missing.pop(first_cell)
        self.assertFalse(evaluation.nine_cell_sign_consistency(missing))

    def test_nine_cell_zero_tolerance_boundary(self) -> None:
        first_cell = next(iter(evaluation.PREREGISTERED_CELLS))
        deltas = {cell: 0.2 for cell in evaluation.PREREGISTERED_CELLS}
        deltas[first_cell] = 0.5e-9
        self.assertFalse(evaluation.nine_cell_sign_consistency(deltas))
        deltas[first_cell] = 1e-9
        self.assertTrue(evaluation.nine_cell_sign_consistency(deltas))

    def test_preregistered_gate_values_cannot_be_runtime_arguments(self) -> None:
        self.assertEqual(evaluation.SESOI_THRESHOLD, 0.05)
        self.assertEqual(evaluation.ZERO_DELTA_TOLERANCE, 1e-9)
        self.assertEqual(len(evaluation.PREREGISTERED_CELLS), 9)
        self.assertTrue(evaluation.meets_sesoi(0.05))
        self.assertFalse(evaluation.meets_sesoi(np.nextafter(0.05, 0.0)))
        self.assertNotIn("threshold", inspect.signature(evaluation.meets_sesoi).parameters)
        sign_parameters = inspect.signature(
            evaluation.nine_cell_sign_consistency
        ).parameters
        self.assertEqual(tuple(sign_parameters), ("cell_deltas",))

    def test_model_b_only_full_pass_detects_positive_result(self) -> None:
        model_cells = {
            "a": self._synthetic_cells(delta=0.4),
            "b": self._synthetic_cells(delta=1.0),
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_negative_hhi_coefficients(Path(directory))
            result = evaluation.evaluate_occupancy_model_gate(
                model_cells,
                paths,
                n_bootstraps=19,
                random_seed=7,
            )
        main = result["main_decision"]
        self.assertTrue(main["effect_size_gates"]["a"]["nine_cell_sign_passed"])
        self.assertFalse(main["effect_size_gates"]["a"]["sesoi_passed"])
        self.assertFalse(main["effect_size_gates"]["a"]["passed"])
        self.assertFalse(main["effect_size_gates"]["a"]["full_passed"])
        self.assertTrue(main["effect_size_gates"]["b"]["full_passed"])
        self.assertTrue(main["positive_result_detected"])

    def test_model_a_only_full_pass_detects_positive_result(self) -> None:
        model_cells = {
            "a": self._synthetic_cells(delta=1.0),
            "b": self._synthetic_cells(delta=0.4),
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_negative_hhi_coefficients(Path(directory))
            result = evaluation.evaluate_occupancy_model_gate(
                model_cells,
                paths,
                n_bootstraps=19,
                random_seed=17,
            )
        main = result["main_decision"]
        self.assertTrue(main["effect_size_gates"]["a"]["full_passed"])
        self.assertFalse(main["effect_size_gates"]["b"]["passed"])
        self.assertFalse(main["effect_size_gates"]["b"]["full_passed"])
        self.assertTrue(main["positive_result_detected"])

    def test_missing_a_cell_fails_a_but_b_can_still_detect_positive_result(self) -> None:
        model_cells = {
            "a": self._synthetic_cells(delta=1.0),
            "b": self._synthetic_cells(delta=1.0),
        }
        model_cells["a"].pop(next(iter(evaluation.PREREGISTERED_CELLS)))
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_negative_hhi_coefficients(Path(directory))
            result = evaluation.evaluate_occupancy_model_gate(
                model_cells,
                paths,
                n_bootstraps=19,
                random_seed=8,
            )
        self.assertTrue(result["main_decision"]["positive_result_detected"])
        self.assertFalse(
            result["main_decision"]["effect_size_gates"]["a"][
                "nine_cell_sign_passed"
            ]
        )
        self.assertFalse(
            result["main_decision"]["effect_size_gates"]["a"]["full_passed"]
        )
        self.assertTrue(
            result["main_decision"]["effect_size_gates"]["b"]["full_passed"]
        )
        self.assertEqual(
            result["auxiliary_inference"][
                "auxiliary_holm_adjusted_pvalues"
            ],
            {"a": None, "b": None},
        )

    def test_nonnegative_hhi_fails_b_but_a_can_detect_positive_result(self) -> None:
        model_cells = {
            "a": self._synthetic_cells(delta=1.0),
            "b": self._synthetic_cells(delta=1.0),
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_negative_hhi_coefficients(Path(directory))
            save_coefficients(
                {
                    name: (0.0 if name == "occ_centered" else 0.1)
                    for name in COEFFICIENT_NAMES
                },
                paths["construction"],
            )
            result = evaluation.evaluate_occupancy_model_gate(
                model_cells,
                paths,
                n_bootstraps=9,
                random_seed=9,
            )
        self.assertTrue(
            all(
                gate["passed"]
                for gate in result["main_decision"]["effect_size_gates"].values()
            )
        )
        self.assertFalse(result["main_decision"]["hhi_sign_check"]["all_negative"])
        self.assertTrue(
            result["main_decision"]["effect_size_gates"]["a"]["full_passed"]
        )
        self.assertFalse(
            result["main_decision"]["effect_size_gates"]["b"]["full_passed"]
        )
        self.assertTrue(result["main_decision"]["positive_result_detected"])

    def test_no_positive_result_when_a_fails_and_b_hhi_sign_fails(self) -> None:
        model_cells = {
            "a": self._synthetic_cells(delta=0.4),
            "b": self._synthetic_cells(delta=1.0),
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_negative_hhi_coefficients(Path(directory))
            save_coefficients(
                {
                    name: (0.0 if name == "occ_centered" else 0.1)
                    for name in COEFFICIENT_NAMES
                },
                paths["construction"],
            )
            result = evaluation.evaluate_occupancy_model_gate(
                model_cells,
                paths,
                n_bootstraps=9,
                random_seed=19,
            )
        main = result["main_decision"]
        self.assertFalse(main["effect_size_gates"]["a"]["full_passed"])
        self.assertTrue(main["effect_size_gates"]["b"]["passed"])
        self.assertFalse(main["effect_size_gates"]["b"]["full_passed"])
        self.assertFalse(main["positive_result_detected"])

    def test_hhi_sign_check_reads_three_frozen_exploration_json_files(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        coefficient_dir = (
            repository_root
            / "data"
            / "processed"
            / "occupancy_panel"
            / "fitted_coefficients"
        )
        paths = {
            domain: coefficient_dir / f"fitted_coefficients_{domain}_b.json"
            for domain in evaluation.PREREGISTERED_DOMAINS
        }
        result = evaluation.check_hhi_frozen_coefficients(paths)
        for domain, path in paths.items():
            expected_coefficient = load_coefficients(path)["occ_centered"]
            actual = result["by_domain"][domain]
            self.assertEqual(actual["occ_centered"], expected_coefficient)
            self.assertEqual(actual["is_negative"], expected_coefficient < 0.0)
        self.assertEqual(
            result["all_negative"],
            all(item["is_negative"] for item in result["by_domain"].values()),
        )

    def test_holm_adjustment_and_one_sided_bootstrap(self) -> None:
        adjusted = evaluation.holm_adjust_two_pvalues({"a": 0.01, "b": 0.04})
        self.assertEqual(adjusted, {"a": 0.02, "b": 0.04})

        auxiliary = evaluation.compute_auxiliary_holm_pvalues(
            {
                "a": self._synthetic_cells(delta=1.0),
                "b": self._synthetic_cells(delta=2.0),
            },
            n_bootstraps=19,
            random_seed=91,
        )
        self.assertEqual(
            auxiliary["auxiliary_raw_one_sided_pvalues"],
            {"a": 0.05, "b": 0.05},
        )
        self.assertEqual(
            auxiliary["auxiliary_holm_adjusted_pvalues"],
            {"a": 0.1, "b": 0.1},
        )

    def test_integrated_result_separates_main_decision_from_auxiliary_pvalues(self) -> None:
        model_cells = {
            "a": self._synthetic_cells(delta=1.0),
            "b": self._synthetic_cells(delta=2.0),
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_negative_hhi_coefficients(Path(directory))
            result = evaluation.evaluate_occupancy_model_gate(
                model_cells,
                paths,
                n_bootstraps=19,
                random_seed=12,
            )
        self.assertEqual(set(result), {"main_decision", "auxiliary_inference"})
        main = result["main_decision"]
        self.assertEqual(
            set(main),
            {"positive_result_detected", "effect_size_gates", "hhi_sign_check"},
        )
        self.assertTrue(main["positive_result_detected"])
        self.assertTrue(
            all(
                gate["full_passed"]
                for gate in main["effect_size_gates"].values()
            )
        )
        self.assertFalse(main["effect_size_gates"]["a"]["hhi_sign_required"])
        self.assertIsNone(main["effect_size_gates"]["a"]["hhi_sign_passed"])
        self.assertTrue(main["effect_size_gates"]["b"]["hhi_sign_required"])
        self.assertTrue(main["effect_size_gates"]["b"]["hhi_sign_passed"])
        self.assertIn(
            "OR semantics",
            inspect.getdoc(evaluation.evaluate_occupancy_model_gate),
        )
        self.assertNotIn("pvalue", repr(main).lower())
        self.assertTrue(
            all(
                key.startswith("auxiliary_")
                for key in result["auxiliary_inference"]
            )
        )


if __name__ == "__main__":
    unittest.main()
