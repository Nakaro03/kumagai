"""Synthetic tests for the Phase 0-c domain slicer."""
from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from pnode_patent_runner import extract_domain_bipartite as extract
from pnode_patent_runner import slice_occupancy_panel_by_domain as slicer
from pnode_patent_runner.build_occupancy_panel import MAX_REPORTING_YEAR


class OccupancyDomainSlicerTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        target_path = root / "target_panel.tsv"
        edge_path = root / "firm_edges.tsv"
        pd.DataFrame(
            [
                (2015, "E04B1", 4.0),
                (2015, "E02D3", 2.0),
                (2015, "Y02E1", 8.0),
                (2015, "G06F3", 10.0),
                (2015, "A01B1", 7.0),
                (2016, "E21B1", 5.0),
                (2016, "Y02T1", 4.0),
                (2016, "G06N3", 6.0),
                (2017, "E04C1", 100.0),
                (2017, "Y02P1", 100.0),
                (2017, "G06Q1", 100.0),
            ],
            columns=slicer.TARGET_COLUMNS,
        ).to_csv(target_path, sep="\t", index=False)
        pd.DataFrame(
            [
                (2015, "c1", "E04B1", 2.0),
                (2015, "c2", "E02D3", 1.0),
                (2015, "e1", "Y02E1", 2.0),
                (2015, "g1", "G06F3", 9.0),
                (2015, "x1", "A01B1", 7.0),
                (2016, "c1", "E21B1", 5.0),
                (2016, "e1", "Y02T1", 1.0),
                (2016, "g1", "G06N3", 3.0),
                (2017, "c1", "E04C1", 100.0),
                (2017, "e1", "Y02P1", 100.0),
                (2017, "g1", "G06Q1", 100.0),
            ],
            columns=slicer.EDGE_COLUMNS,
        ).to_csv(edge_path, sep="\t", index=False)
        return target_path, edge_path

    def _run(self, root: Path) -> slicer.SliceResult:
        target_path, edge_path = self._fixture(root)
        with redirect_stdout(io.StringIO()):
            return slicer.slice_occupancy_panel_by_domain(
                target_panel_path=target_path,
                firm_edges_path=edge_path,
                output_dir=root / "by_domain",
            )

    def test_real_domain_dictionary_and_prefix_function_are_imported(self) -> None:
        self.assertIs(slicer.DOMAINS, extract.DOMAINS)
        self.assertIs(slicer.domain_prefixes_match, extract.domain_prefixes_match)
        groups = pd.Series(["E04B1", "Y02E1", "G06F3", "A01B1"])
        self.assertEqual(
            slicer.domain_prefixes_match(
                groups, slicer.DOMAINS["construction"]
            ).tolist(),
            [True, False, False, False],
        )

    def test_slices_and_concrete_coverage_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(Path(directory))

            construction = pd.read_csv(
                result.target_paths["construction"], sep="\t"
            )
            agrifood = pd.read_csv(result.target_paths["agrifood"], sep="\t")
            computing = pd.read_csv(result.target_paths["computing"], sep="\t")
            construction_edges = pd.read_csv(
                result.firm_edges_paths["construction"], sep="\t"
            )
            agrifood_edges = pd.read_csv(
                result.firm_edges_paths["agrifood"], sep="\t"
            )
            computing_edges = pd.read_csv(
                result.firm_edges_paths["computing"], sep="\t"
            )
            self.assertEqual(
                construction["maingroup"].tolist(), ["E02D3", "E04B1", "E21B1"]
            )
            self.assertEqual(agrifood["maingroup"].tolist(), ["A01B1"])
            self.assertEqual(computing["maingroup"].tolist(), ["G06F3", "G06N3"])
            for frame in (construction, agrifood, computing):
                self.assertFalse(frame["maingroup"].str.startswith("Y02").any())
            self.assertEqual(
                set(construction_edges["maingroup"]), {"E02D3", "E04B1", "E21B1"}
            )
            self.assertEqual(set(agrifood_edges["maingroup"]), {"A01B1"})
            self.assertEqual(set(computing_edges["maingroup"]), {"G06F3", "G06N3"})

            summary = pd.read_csv(result.summary_path, sep="\t").set_index(
                "filing_year"
            )
            self.assertEqual(summary.index.tolist(), [2015, 2016])
            self.assertEqual(summary.loc[2015, "construction_maingroup_count"], 2)
            self.assertAlmostEqual(summary.loc[2015, "construction_target_mass"], 6.0)
            self.assertAlmostEqual(
                summary.loc[2015, "construction_organization_mass"], 3.0
            )
            self.assertAlmostEqual(summary.loc[2015, "construction_coverage"], 0.5)
            self.assertEqual(summary.loc[2015, "agrifood_maingroup_count"], 1)
            self.assertAlmostEqual(summary.loc[2015, "agrifood_target_mass"], 7.0)
            self.assertAlmostEqual(
                summary.loc[2015, "agrifood_organization_mass"], 7.0
            )
            self.assertAlmostEqual(summary.loc[2015, "agrifood_coverage"], 1.0)
            self.assertAlmostEqual(summary.loc[2015, "computing_coverage"], 9.0 / 10.0)
            self.assertAlmostEqual(summary.loc[2016, "construction_coverage"], 1.0)
            self.assertEqual(summary.loc[2016, "agrifood_maingroup_count"], 0)
            self.assertAlmostEqual(summary.loc[2016, "agrifood_target_mass"], 0.0)
            self.assertAlmostEqual(
                summary.loc[2016, "agrifood_organization_mass"], 0.0
            )
            self.assertAlmostEqual(summary.loc[2016, "agrifood_coverage"], 0.0)
            self.assertAlmostEqual(summary.loc[2016, "computing_coverage"], 3.0 / 6.0)

    def test_holdout_rows_are_absent_from_every_output_and_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_path, edge_path = self._fixture(root)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = slicer.slice_occupancy_panel_by_domain(
                    target_panel_path=target_path,
                    firm_edges_path=edge_path,
                    output_dir=root / "by_domain",
                )

            self.assertNotIn("2017", stdout.getvalue())
            self.assertNotIn(2017, result.summary["filing_year"].tolist())
            output_paths = (
                *result.target_paths.values(),
                *result.firm_edges_paths.values(),
            )
            for path in output_paths:
                rows = pd.read_csv(path, sep="\t")
                self.assertTrue((rows["filing_year"] <= MAX_REPORTING_YEAR).all())

    def test_holdout_guard_rejects_year_above_2016_before_reading(self) -> None:
        with self.assertRaisesRegex(ValueError, "Holdout guard"):
            slicer.slice_occupancy_panel_by_domain(
                target_panel_path=Path("does-not-exist"),
                firm_edges_path=Path("does-not-exist"),
                max_reporting_year=MAX_REPORTING_YEAR + 1,
            )

    def test_unknown_domain_is_rejected_before_reading(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown domain"):
            slicer.slice_occupancy_panel_by_domain(
                target_panel_path=Path("does-not-exist"),
                firm_edges_path=Path("does-not-exist"),
                domains=("construction", "unknown"),
            )

    def test_coverage_and_nonnegative_mass_audits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_path = root / "target.tsv"
            edge_path = root / "edges.tsv"
            pd.DataFrame([(2016, "G06F3", 1.0)], columns=slicer.TARGET_COLUMNS).to_csv(
                target_path, sep="\t", index=False
            )
            pd.DataFrame(
                [(2016, "u1", "G06F3", 1.000002)], columns=slicer.EDGE_COLUMNS
            ).to_csv(edge_path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "outside"):
                slicer.slice_occupancy_panel_by_domain(
                    target_panel_path=target_path,
                    firm_edges_path=edge_path,
                    domains=("computing",),
                    output_dir=root / "coverage-failure",
                )
            self.assertFalse((root / "coverage-failure").exists())

            pd.DataFrame(
                [(2016, "u1", "G06F3", -0.1)], columns=slicer.EDGE_COLUMNS
            ).to_csv(edge_path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "negative"):
                slicer.slice_occupancy_panel_by_domain(
                    target_panel_path=target_path,
                    firm_edges_path=edge_path,
                    domains=("computing",),
                    output_dir=root / "mass-failure",
                )


if __name__ == "__main__":
    unittest.main()
