from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import pandas as pd

from pnode_patent_runner import confirmation_a_freshyears as confirmation
from pnode_patent_runner import extract_domain_bipartite as extract
from pnode_patent_runner.gate0_regime_detectability import (
    _build_observations,
    _fit_interaction,
    _mass_table,
    load_domain_pairs,
)


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


class ConfirmationAFreshYearsTest(unittest.TestCase):
    def test_patent_transitions_are_the_three_fixed_fresh_year_transitions(self):
        self.assertEqual(
            confirmation.PATENT_TRANSITIONS,
            [(2020, 2021, 2022), (2021, 2022, 2023), (2022, 2023, 2024)],
        )

    def test_written_json_is_explicitly_non_gating_and_has_no_gate0_pass(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "confirmation_a.json"
            stdout = io.StringIO()
            with mock.patch.object(
                confirmation, "run_analysis", return_value={"construction": {"status": "ok"}}
            ), redirect_stdout(stdout):
                result = confirmation.main(["--output-json", str(output_path)])

            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result, 0)
            self.assertIs(report["non_gating"], True)
            self.assertIn("占有率仮説の合否判定には使わない参考情報", report["purpose"])
            self.assertFalse(_contains_key(report, "gate0_pass"))
            self.assertTrue(stdout.getvalue().startswith(confirmation.LABEL))

    def test_existing_loader_observation_builder_and_fit_work_on_synthetic_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            rows = []
            for year in range(2020, 2025):
                step = year - 2020
                for category_index in range(40):
                    base = 8 + category_index % 7
                    count = (
                        base
                        + step * (1 + category_index % 5)
                        + (step * step if category_index % 4 == 0 else 0)
                        + (step + 1) * (category_index % 3)
                    )
                    for row_index in range(count):
                        rows.append(
                            {
                                "ts": f"{year}-01-01",
                                "u": f"u{category_index}_{year}_{row_index}",
                                "i": f"G{category_index:02d}/00",
                            }
                        )
            pd.DataFrame(rows).to_csv(processed / "bipartite_computing.csv", index=False)

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                pairs = load_domain_pairs("computing", "maingroup", 2020, 2024)
            finally:
                os.chdir(previous_cwd)

            observations = _build_observations(
                _mass_table(pairs), confirmation.PATENT_TRANSITIONS
            )
            fit = _fit_interaction(observations)
            self.assertEqual(fit["status"], "ok")
            self.assertEqual(fit["n"], 120)
            self.assertGreaterEqual(fit["n_burst"], 5)
            self.assertIn("coef_mom_burst", fit)


class ExtractDomainBipartiteCompatibilityTest(unittest.TestCase):
    def _write_bulk_fixture(self, base: Path) -> None:
        pd.DataFrame(
            [
                {"patent_id": "p2021", "patent_date": "2021-06-01"},
                {"patent_id": "p2022", "patent_date": "2022-06-01"},
            ]
        ).to_csv(base / "g_patent.tsv", sep="\t", index=False)
        pd.DataFrame(
            [
                {"patent_id": "p2021", "inventor_id": "u1"},
                {"patent_id": "p2022", "inventor_id": "u2"},
            ]
        ).to_csv(base / "g_inventor_disambiguated.tsv", sep="\t", index=False)
        pd.DataFrame(
            [
                {"patent_id": "p2021", "cpc_group": "G06F1/00", "cpc_type": "inventional"},
                {"patent_id": "p2022", "cpc_group": "G06N3/00", "cpc_type": "additional"},
            ]
        ).to_csv(base / "g_cpc_current.tsv", sep="\t", index=False)

    def test_year_end_default_matches_existing_2021_behavior(self):
        self.assertEqual(extract.parse_args(["computing"]).year_end, extract.YEAR_END)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "bulk"
            output = root / "processed"
            base.mkdir()
            output.mkdir()
            self._write_bulk_fixture(base)

            with mock.patch.object(extract, "BASE", str(base)), mock.patch.object(
                extract, "OUT_DIR", str(output)
            ), redirect_stdout(io.StringIO()):
                extract.main(["computing"])
                default_bytes = (output / "bipartite_computing.csv").read_bytes()
                extract.main(["computing", "--year-end", "2021"])
                explicit_bytes = (output / "bipartite_computing.csv").read_bytes()

            self.assertEqual(default_bytes, explicit_bytes)
            default_rows = pd.read_csv(io.BytesIO(default_bytes))
            self.assertEqual(default_rows["ts"].tolist(), ["2021-06-01"])

    def test_year_end_2024_includes_post_2021_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "bulk"
            output = root / "processed"
            base.mkdir()
            output.mkdir()
            self._write_bulk_fixture(base)

            with mock.patch.object(extract, "BASE", str(base)), mock.patch.object(
                extract, "OUT_DIR", str(output)
            ), redirect_stdout(io.StringIO()):
                extract.main(["computing", "--year-end", "2024"])

            rows = pd.read_csv(output / "bipartite_computing.csv")
            self.assertEqual(rows["ts"].tolist(), ["2021-06-01", "2022-06-01"])


if __name__ == "__main__":
    unittest.main()
