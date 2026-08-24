"""Synthetic-only tests for the Confirmation B execution mechanism."""
from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from pnode_patent_runner import run_confirmation_b as confirmation
from pnode_patent_runner.compute_occupancy_features import OUTPUT_COLUMNS
from pnode_patent_runner.evaluate_occupancy_model import (
    PREREGISTERED_CELLS,
    PREREGISTERED_DOMAINS,
)
from pnode_patent_runner.fit_occupancy_model import COEFFICIENT_NAMES, save_coefficients
from pnode_patent_runner.slice_occupancy_panel_by_domain import (
    EDGE_COLUMNS,
    TARGET_COLUMNS,
)


class RunConfirmationBTest(unittest.TestCase):
    TOPICS = {
        "construction": ("E04B1", "E04B2"),
        "agrifood": ("A01B1", "A01B2"),
        "computing": ("G06F1", "G06F2"),
    }

    @staticmethod
    def _coefficients(**changes: float) -> dict[str, float]:
        coefficients = {name: 0.0 for name in COEFFICIENT_NAMES}
        coefficients.update(changes)
        return coefficients

    def _prepared_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        by_domain = root / "by_domain"
        features_dir = root / "features"
        coefficients_dir = root / "coefficients"
        by_domain.mkdir()
        features_dir.mkdir()
        coefficients_dir.mkdir()

        for domain, topics in self.TOPICS.items():
            target_rows = []
            for year in range(2015, 2020):
                step = year - 2014
                target_rows.extend(
                    [
                        (year, topics[0], math.expm1(float(step))),
                        (year, topics[1], math.expm1(0.5 * step)),
                    ]
                )
            pd.DataFrame(target_rows, columns=TARGET_COLUMNS).to_csv(
                by_domain / f"target_panel_{domain}.tsv", sep="\t", index=False
            )

            feature_rows = []
            for year in (2016, 2017, 2018):
                for topic in topics:
                    feature_rows.append(
                        (year, topic, 0.2, 0.0, 0.4, 0.0, 1.0, 1, True)
                    )
            pd.DataFrame(feature_rows, columns=OUTPUT_COLUMNS).to_csv(
                features_dir / f"occupancy_features_{domain}.tsv",
                sep="\t",
                index=False,
            )

            save_coefficients(
                self._coefficients(),
                coefficients_dir / f"fitted_coefficients_{domain}_baseline.json",
            )
            save_coefficients(
                self._coefficients(mom=1.0),
                coefficients_dir / f"fitted_coefficients_{domain}_a.json",
            )
            save_coefficients(
                self._coefficients(mom=0.5, occ_centered=-0.1),
                coefficients_dir / f"fitted_coefficients_{domain}_b.json",
            )
        return by_domain, features_dir, coefficients_dir

    def test_nine_cells_predictions_and_errors_match_known_relation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            by_domain, features_dir, coefficients_dir = self._prepared_fixture(
                Path(directory)
            )
            observations = confirmation.build_confirmation_b_observations(
                "construction",
                by_domain_dir=by_domain,
                occupancy_features_dir=features_dir,
                coefficient_dir=coefficients_dir,
            )
            for year in (2016, 2017, 2018):
                rows = observations[observations["t"] == year].set_index("cat")
                high = rows.loc["E04B1"]
                low = rows.loc["E04B2"]
                self.assertAlmostEqual(high["mom"], 1.0)
                self.assertAlmostEqual(high["next_mom"], 1.0)
                self.assertEqual(high["burst"], 1.0)
                self.assertAlmostEqual(low["mom"], 0.5)
                self.assertAlmostEqual(low["next_mom"], 0.5)
                self.assertEqual(low["burst"], 0.0)
                self.assertAlmostEqual(high["baseline_prediction"], 0.0)
                self.assertAlmostEqual(high["a_prediction"], 1.0)
                self.assertAlmostEqual(high["b_prediction"], 0.5)
                self.assertAlmostEqual(high["baseline_error"], 1.0)
                self.assertAlmostEqual(high["a_error"], 0.0)
                self.assertAlmostEqual(high["b_error"], 0.5)

            cells = confirmation.build_confirmation_b_cell_errors(
                by_domain_dir=by_domain,
                occupancy_features_dir=features_dir,
                coefficient_dir=coefficients_dir,
            )
            self.assertEqual(set(cells), {"a", "b"})
            self.assertTrue(
                all(set(model_cells) == PREREGISTERED_CELLS for model_cells in cells.values())
            )

    def test_prepare_data_recovers_and_applies_synthetic_frozen_centers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_path = root / "target_panel.tsv"
            edge_path = root / "firm_edges.tsv"
            frozen_dir = root / "frozen"
            frozen_dir.mkdir()
            target_rows = []
            edge_rows = []
            expected_centers = {}
            for index, (domain, topics) in enumerate(self.TOPICS.items(), start=1):
                center_a = 0.1 * index
                center_b = 0.2 * index
                expected_centers[domain] = {"occ_a": center_a, "occ_b": center_b}
                frozen_rows = []
                for topic_index, topic in enumerate(topics):
                    raw_a = center_a + 0.01 * (topic_index + 1)
                    raw_b = center_b + 0.02 * (topic_index + 1)
                    frozen_rows.append(
                        (
                            2016,
                            topic,
                            raw_a,
                            raw_a - center_a,
                            raw_b,
                            raw_b - center_b,
                            1.0,
                            1,
                            True,
                        )
                    )
                pd.DataFrame(frozen_rows, columns=OUTPUT_COLUMNS).to_csv(
                    frozen_dir / f"occupancy_features_{domain}.tsv",
                    sep="\t",
                    index=False,
                )
                for year in range(2015, 2020):
                    for topic_index, topic in enumerate(topics):
                        mass = float(year - 2013 + topic_index)
                        target_rows.append((year, topic, mass))
                        edge_rows.append(
                            (year, f"{domain}-{topic_index}", topic, mass)
                        )
            pd.DataFrame(target_rows, columns=TARGET_COLUMNS).to_csv(
                target_path, sep="\t", index=False
            )
            pd.DataFrame(edge_rows, columns=EDGE_COLUMNS).to_csv(
                edge_path, sep="\t", index=False
            )

            centers = confirmation.prepare_confirmation_b_data(
                target_panel_path=target_path,
                firm_edges_path=edge_path,
                frozen_features_dir=frozen_dir,
                by_domain_dir=root / "confirmation_by_domain",
                output_features_dir=root / "confirmation_features",
            )
            for domain in PREREGISTERED_DOMAINS:
                self.assertAlmostEqual(
                    centers[domain]["occ_a"], expected_centers[domain]["occ_a"]
                )
                self.assertAlmostEqual(
                    centers[domain]["occ_b"], expected_centers[domain]["occ_b"]
                )
                generated = pd.read_csv(
                    root / "confirmation_features" / f"occupancy_features_{domain}.tsv",
                    sep="\t",
                )
                self.assertIn(2019, generated["filing_year"].tolist())
                reportable = generated[generated["in_topic_universe"]]
                for name in ("occ_a", "occ_b"):
                    differences = reportable[name] - reportable[f"{name}_centered"]
                    self.assertTrue(
                        all(
                            math.isclose(value, centers[domain][name], abs_tol=1e-12)
                            for value in differences
                        )
                    )

    def test_main_runs_end_to_end_on_synthetic_prepared_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            by_domain, features_dir, coefficients_dir = self._prepared_fixture(root)
            result_path = root / "result.json"
            with redirect_stdout(io.StringIO()):
                status = confirmation.main(
                    [
                        "--skip-data-preparation",
                        "--by-domain-dir",
                        str(by_domain),
                        "--occupancy-features-dir",
                        str(features_dir),
                        "--coefficient-dir",
                        str(coefficients_dir),
                        "--result-json",
                        str(result_path),
                        "--n-bootstraps",
                        "19",
                        "--random-seed",
                        "7",
                    ]
                )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(status, 0)
            self.assertTrue(payload["main_decision"]["positive_result_detected"])
            self.assertEqual(
                len(
                    payload["main_decision"]["effect_size_gates"]["a"][
                        "statistics"
                    ]["cell_deltas"]
                ),
                9,
            )


if __name__ == "__main__":
    unittest.main()
