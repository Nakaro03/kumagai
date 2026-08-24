"""Small invariant tests for build_occupancy_panel.py.

These tests use only unittest from the standard library, and are also
collectable by pytest when pytest is available.
"""
from __future__ import annotations

import csv
import io
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from pnode_patent_runner.build_occupancy_panel import (
    BuildConfig,
    MAX_REPORTING_YEAR,
    build_occupancy_panel,
    validate_reporting_year,
)


CPC_HEADER = (
    "patent_id",
    "cpc_sequence",
    "cpc_section",
    "cpc_class",
    "cpc_subclass",
    "cpc_group",
    "cpc_type",
)
APPLICATION_HEADER = (
    "application_id",
    "patent_id",
    "patent_application_type",
    "filing_date",
    "series_code",
    "rule_47_flag",
)
ASSIGNEE_HEADER = (
    "patent_id",
    "assignee_sequence",
    "assignee_id",
    "disambig_assignee_individual_name_first",
    "disambig_assignee_individual_name_last",
    "disambig_assignee_organization",
    "assignee_type",
    "location_id",
)


def _write_tsv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


class OccupancyPanelTest(unittest.TestCase):
    def _fixture(self, root: Path) -> BuildConfig:
        cpc = root / "g_cpc_current.tsv"
        application = root / "g_application.tsv"
        assignee = root / "g_assignee_disambiguated.tsv"
        output = root / "output"

        _write_tsv(
            cpc,
            CPC_HEADER,
            (
                ("p1", 0, "A", "A01", "A01B", "A01B1/00", "inventional"),
                # Same maingroup: this must not inflate K_p.
                ("p1", 1, "A", "A01", "A01B", "A01B1/02", "inventional"),
                ("p1", 2, "G", "G06", "G06F", "G06F3/00", "inventional"),
                ("p1", 3, "G", "G06", "G06F", "G06F3/00", "inventional"),
                ("p2", 0, "A", "A01", "A01B", "A01B1/00", "inventional"),
                ("p2", 1, "Y", "Y02", "Y02E", "Y02E1/00", "additional"),
                # Holdout row: retained in full panels but never reported.
                ("p3", 0, "H", "H01", "H01L", "H01L1/00", "inventional"),
                # Invalid filing year: excluded from year-indexed panels.
                ("p4", 0, "C", "C12", "C12N", "C12N1/00", "inventional"),
            ),
        )
        _write_tsv(
            application,
            APPLICATION_HEADER,
            (
                ("a1", "p1", "utility", "2016-01-03", "", 0),
                ("a2", "p2", "utility", "2016-02-03", "", 0),
                ("a3", "p3", "utility", "2017-03-03", "", 0),
                ("a4", "p4", "utility", "1074-01-01", "", 0),
            ),
        )
        _write_tsv(
            assignee,
            ASSIGNEE_HEADER,
            (
                ("p1", 0, "u1", "", "", "Firm One", 2, ""),
                ("p1", 1, "u2", "", "", "Firm Two", 2, ""),
                ("p1", 2, "u1", "", "", "Firm One", 2, ""),
                # Individual-only p2 must contribute to M but not W.
                ("p2", 0, "person", "Pat", "Ent", "", 1, ""),
                ("p3", 0, "u3", "", "", "Holdout Firm", 2, ""),
                ("p4", 0, "u4", "", "", "Invalid Date Firm", 2, ""),
            ),
        )
        return BuildConfig(
            cpc_path=cpc,
            application_path=application,
            assignee_path=assignee,
            output_dir=output,
            database_path=output / "occupancy_panel.sqlite3",
            batch_size=2,
            sqlite_cache_mb=8,
            diagnostic_cpc_groups=True,
            patent_ids_file=self._patent_ids(root),
        )

    @staticmethod
    def _patent_ids(root: Path) -> Path:
        path = root / "patent_ids.txt"
        path.write_text("p1\np2\np3\np4\n", encoding="utf-8")
        return path

    def test_fractional_mass_invariants_and_holdout_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = build_occupancy_panel(config)

            self.assertTrue(result.audit.passed)
            # K diagnostics are guarded too: p3 (2017) and invalid-date p4
            # are not included in anything printed or logged.
            self.assertEqual(result.k_distribution.counts_by_k, {1: 1, 2: 1})
            self.assertAlmostEqual(
                result.k_distribution.fraction_with_maingroup_collapse, 0.5
            )

            conn = sqlite3.connect(str(result.database_path))
            try:
                p1_edge_mass = conn.execute(
                    """
                    SELECT SUM(1.0 / (a.a_p * k.k_p))
                    FROM patent_maingroups AS m
                    JOIN patent_k AS k USING (patent_id)
                    JOIN patent_a AS a USING (patent_id)
                    JOIN patent_assignees AS u USING (patent_id)
                    WHERE m.patent_id = 'p1'
                    """
                ).fetchone()[0]
                p1_topic_mass = conn.execute(
                    """
                    SELECT SUM(1.0 / k.k_p)
                    FROM patent_maingroups AS m
                    JOIN patent_k AS k USING (patent_id)
                    WHERE m.patent_id = 'p1'
                    """
                ).fetchone()[0]
                coverage = dict(
                    conn.execute(
                        """
                        SELECT maingroup, coverage
                        FROM topic_year_coverage WHERE filing_year = 2016
                        """
                    )
                )
                holdout_count = conn.execute(
                    "SELECT COUNT(*) FROM target_panel WHERE filing_year = 2017"
                ).fetchone()[0]
            finally:
                conn.close()

            self.assertAlmostEqual(p1_edge_mass, 1.0)
            self.assertAlmostEqual(p1_topic_mass, 1.0)
            self.assertAlmostEqual(coverage["A01B1"], 1.0 / 3.0)
            self.assertAlmostEqual(coverage["G06F3"], 1.0)
            self.assertEqual(holdout_count, 1)

            coverage_text = result.coverage_report_path.read_text(encoding="utf-8")
            gap_text = result.mass_gap_report_path.read_text(encoding="utf-8")
            self.assertNotIn("2017", coverage_text)
            self.assertNotIn("2017", gap_text)
            self.assertNotIn("filing_year=2017", stdout.getvalue())

    def test_reporting_guard_rejects_holdout(self) -> None:
        self.assertEqual(validate_reporting_year(2016), MAX_REPORTING_YEAR)
        with self.assertRaisesRegex(ValueError, "Holdout guard"):
            validate_reporting_year(2017)


if __name__ == "__main__":
    unittest.main()
