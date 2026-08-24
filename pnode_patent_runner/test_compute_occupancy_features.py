"""Synthetic tests for issue #8 section 12.6 occupancy features."""
from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pnode_patent_runner import compute_occupancy_features as occupancy
from pnode_patent_runner.build_occupancy_panel import MAX_REPORTING_YEAR


class OccupancyFeaturesTest(unittest.TestCase):
    def _fixture(self, root: Path, include_holdout: bool = True) -> tuple[Path, Path]:
        full_path = root / "firm_edges.tsv"
        by_domain = root / "by_domain"
        by_domain.mkdir()

        full_rows = [
            # u1 spans agrifood and construction.  Its full 2016 portfolio is
            # 8, not the construction-only value 2.
            (2016, "u1", "A01B1", 6.0),
            (2016, "u1", "E04B1", 2.0),
            (2016, "u2", "E04B1", 2.0),
            (2016, "u3", "E04B2", 3.0),
            (2016, "u4", "E04B4", 1.0),
        ]
        construction_edges = [
            (2016, "u1", "E04B1", 2.0),
            (2016, "u2", "E04B1", 2.0),
            (2016, "u3", "E04B2", 3.0),
            # No corresponding target row: M=0, W>0 is outside the universe.
            (2016, "u4", "E04B4", 1.0),
        ]
        agrifood_edges = [(2016, "u1", "A01B1", 6.0)]
        construction_targets = [
            (2016, "E04B1", 5.0),
            (2016, "E04B2", 4.0),
            # No firm edge: M>0, W=0, n_j=0.
            (2016, "E04B3", 2.0),
        ]
        agrifood_targets = [(2016, "A01B1", 6.0)]

        if include_holdout:
            full_rows.extend(
                [
                    (2017, "u1", "A01B1", 8.0),
                    (2017, "u1", "E04B1", 8.0),
                    (2017, "u2", "E04B1", 2.0),
                ]
            )
            construction_edges.extend(
                [
                    (2017, "u1", "E04B1", 8.0),
                    (2017, "u2", "E04B1", 2.0),
                ]
            )
            agrifood_edges.append((2017, "u1", "A01B1", 8.0))
            construction_targets.append((2017, "E04B1", 10.0))
            agrifood_targets.append((2017, "A01B1", 8.0))

        pd.DataFrame(full_rows, columns=occupancy.EDGE_COLUMNS).to_csv(
            full_path, sep="\t", index=False
        )
        for domain, targets, edges in (
            ("construction", construction_targets, construction_edges),
            ("agrifood", agrifood_targets, agrifood_edges),
        ):
            pd.DataFrame(targets, columns=occupancy.TARGET_COLUMNS).to_csv(
                by_domain / f"target_panel_{domain}.tsv", sep="\t", index=False
            )
            pd.DataFrame(edges, columns=occupancy.EDGE_COLUMNS).to_csv(
                by_domain / f"firm_edges_{domain}.tsv", sep="\t", index=False
            )
        return full_path, by_domain

    def _run(
        self,
        root: Path,
        include_holdout: bool = True,
        domains: tuple[str, ...] = ("construction", "agrifood"),
    ) -> tuple[occupancy.OccupancyFeaturesResult, str]:
        full_path, by_domain = self._fixture(root, include_holdout=include_holdout)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = occupancy.compute_occupancy_features(
                domains=domains,
                full_firm_edges_path=full_path,
                by_domain_dir=by_domain,
                output_dir=root / "features",
                chunk_size=2,
            )
        return result, stdout.getvalue()

    def test_q_spec_occ_a_hhi_and_occ_b_match_hand_calculation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_path, by_domain = self._fixture(root)
            edges = occupancy._aggregate_edges(
                occupancy._read_panel(
                    by_domain / "firm_edges_construction.tsv", occupancy.EDGE_COLUMNS
                )
            )
            totals = occupancy._full_portfolio_totals(
                full_path, edges.loc[:, list(occupancy.KEY_COLUMNS)], chunk_size=2
            )
            contributions = occupancy.compute_edge_contributions(edges, totals)

            topic = contributions[
                (contributions["filing_year"] == 2016)
                & (contributions["maingroup"] == "E04B1")
            ].set_index("assignee_id")
            self.assertAlmostEqual(topic.loc["u1", "q"], 0.5)
            self.assertAlmostEqual(topic.loc["u2", "q"], 0.5)
            # Crucial cross-domain check: spec(u1)=2/(2+6), not 2/2.
            self.assertAlmostEqual(topic.loc["u1", "portfolio_weight"], 8.0)
            self.assertAlmostEqual(topic.loc["u1", "spec"], 0.25)
            self.assertAlmostEqual(topic.loc["u2", "spec"], 1.0)
            self.assertAlmostEqual((topic["q"] * topic["spec"]).sum(), 0.625)
            hhi = float((topic["q"] ** 2).sum())
            self.assertAlmostEqual(hhi, 0.5)
            normalized_hhi = (hhi - 1.0 / 2.0) / (1.0 - 1.0 / 2.0)
            self.assertAlmostEqual(normalized_hhi, 0.0)

            result, _ = self._run(Path(tempfile.mkdtemp(dir=root)))
            features = pd.read_csv(
                result.output_paths["construction"], sep="\t"
            ).set_index(["filing_year", "maingroup"])
            self.assertAlmostEqual(features.loc[(2016, "E04B1"), "occ_a"], 0.625)
            self.assertAlmostEqual(features.loc[(2016, "E04B1"), "occ_b"], 0.0)
            self.assertAlmostEqual(features.loc[(2016, "E04B1"), "coverage"], 0.8)
            self.assertEqual(features.loc[(2016, "E04B1"), "n_j"], 2)

    def test_topic_universe_zero_firm_and_singleton_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self._run(Path(directory))
            features = pd.read_csv(
                result.output_paths["construction"], sep="\t"
            ).set_index(["filing_year", "maingroup"])

            singleton = features.loc[(2016, "E04B2")]
            self.assertEqual(singleton["n_j"], 1)
            self.assertAlmostEqual(singleton["occ_b"], 1.0)
            self.assertTrue(bool(singleton["in_topic_universe"]))

            zero_firm = features.loc[(2016, "E04B3")]
            self.assertEqual(zero_firm["n_j"], 0)
            self.assertFalse(bool(zero_firm["in_topic_universe"]))
            self.assertTrue(pd.isna(zero_firm["occ_a"]))
            self.assertTrue(pd.isna(zero_firm["occ_b"]))

            zero_target = features.loc[(2016, "E04B4")]
            self.assertEqual(zero_target["n_j"], 1)
            self.assertFalse(bool(zero_target["in_topic_universe"]))
            self.assertTrue(pd.isna(zero_target["occ_a"]))
            self.assertTrue(pd.isna(zero_target["coverage"]))

            summary = result.summaries.set_index("domain").loc["construction"]
            self.assertEqual(summary["n_j_zero_excluded_count"], 1)
            self.assertAlmostEqual(summary["n_j_zero_excluded_fraction"], 1.0 / 3.0)

    def test_centering_uses_only_rows_through_2016(self) -> None:
        with tempfile.TemporaryDirectory() as with_holdout_directory:
            with_holdout, _ = self._run(Path(with_holdout_directory))
            with tempfile.TemporaryDirectory() as without_holdout_directory:
                without_holdout, _ = self._run(
                    Path(without_holdout_directory), include_holdout=False
                )

                expected = {"occ_a": 0.8125, "occ_b": 0.5}
                for feature, value in expected.items():
                    self.assertAlmostEqual(
                        with_holdout.centering_constants["construction"][feature],
                        value,
                    )
                    self.assertAlmostEqual(
                        with_holdout.centering_constants["construction"][feature],
                        without_holdout.centering_constants["construction"][feature],
                    )

            frame = pd.read_csv(
                with_holdout.output_paths["construction"], sep="\t"
            ).set_index(["filing_year", "maingroup"])
            holdout = frame.loc[(2017, "E04B1")]
            # 2017: occ_a=.8*.5 + .2*1=.6; HHI=.68; normalized HHI=.36.
            self.assertAlmostEqual(holdout["occ_a"], 0.6)
            self.assertAlmostEqual(holdout["occ_a_centered"], 0.6 - 0.8125)
            self.assertAlmostEqual(holdout["occ_b"], 0.36)
            self.assertAlmostEqual(holdout["occ_b_centered"], 0.36 - 0.5)

    def test_override_centers_are_used_verbatim_without_computing_a_mean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_path, by_domain = self._fixture(root, include_holdout=False)
            target = occupancy._aggregate_target(
                occupancy._read_panel(
                    by_domain / "target_panel_construction.tsv",
                    occupancy.TARGET_COLUMNS,
                )
            )
            edges = occupancy._aggregate_edges(
                occupancy._read_panel(
                    by_domain / "firm_edges_construction.tsv",
                    occupancy.EDGE_COLUMNS,
                )
            )
            totals = occupancy._full_portfolio_totals(
                full_path, edges.loc[:, list(occupancy.KEY_COLUMNS)]
            )
            frozen = {"occ_a": 10.0, "occ_b": -4.0}
            with patch.object(
                pd.Series,
                "mean",
                side_effect=AssertionError("center mean must be skipped"),
            ):
                features, _, centers, _ = occupancy._compute_domain_features(
                    target,
                    edges,
                    totals,
                    max_reporting_year=2019,
                    override_centers=frozen,
                )

            self.assertEqual(centers, frozen)
            reportable = features[features["in_topic_universe"]].copy()
            self.assertTrue(
                (
                    reportable["occ_a"] - reportable["occ_a_centered"]
                ).eq(frozen["occ_a"]).all()
            )
            self.assertTrue(
                (
                    reportable["occ_b"] - reportable["occ_b_centered"]
                ).eq(frozen["occ_b"]).all()
            )

    def test_recover_frozen_centers_uses_exploration_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self._run(Path(directory))
            recovered = occupancy.recover_frozen_centers(
                result.output_paths["construction"]
            )
            self.assertEqual(
                recovered,
                result.centering_constants["construction"],
            )

    def test_holdout_guard_and_stdout_do_not_report_holdout(self) -> None:
        with self.assertRaisesRegex(ValueError, "Holdout guard"):
            occupancy.compute_occupancy_features(
                domains=("construction",),
                full_firm_edges_path=Path("does-not-exist"),
                by_domain_dir=Path("does-not-exist"),
                max_reporting_year=MAX_REPORTING_YEAR + 1,
            )

        with tempfile.TemporaryDirectory() as directory:
            result, stdout = self._run(Path(directory))
            self.assertIn("2016", stdout)
            self.assertNotIn("2017", stdout)
            rows = pd.read_csv(result.output_paths["construction"], sep="\t")
            self.assertIn(2017, rows["filing_year"].tolist())
            self.assertEqual(tuple(rows.columns), occupancy.OUTPUT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
