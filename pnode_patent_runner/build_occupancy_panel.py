"""Build the organization x CPC-maingroup occupancy panel.

This is the data-construction and audit layer for GitHub issue #8, sections
12.1 and 12.2 only.  It deliberately does not select domains, fit models, or
inspect holdout-period aggregate outcomes.

The large PatentsView TSVs are streamed into a disk-backed SQLite database.
In particular, no list of pandas chunks is retained in memory.  The database
also provides the required patent-level audit trail:

* patent_maingroups(patent_id, maingroup), unique by construction
* patent_assignees(patent_id, assignee_id), unique by construction
* patent_k and patent_a, containing K_p and A_p

The final full-period products are target_panel.tsv and firm_edges.tsv.
Coverage and organization-mass-gap reports are hard-limited to filing years
through 2016 so that running this builder cannot print or log holdout-period
aggregates.

Example (small smoke sample; the final patent may be truncated at the row
boundary and is therefore only suitable for diagnostics)::

    python pnode_patent_runner/build_occupancy_panel.py \
      --output-dir /tmp/occupancy-smoke \
      --max-cpc-rows 300000 --diagnostic-cpc-groups

For a patent-complete sample, put one patent_id per line in a file and use
--patent-ids-file.  The CPC file is then fully scanned, but only those patents
are retained.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


ROOT = Path("/home/nakamuraroi/kumagai")
DEFAULT_BULK_DIR = ROOT / "notebooks/work/dataset/PatentsViewBulkData"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/occupancy_panel"

MIN_FILING_YEAR = 1900
MAX_FILING_YEAR = 2025

# Project-level holdout rule.  Do not make this configurable upward.
HOLDOUT_START_YEAR = 2017
MAX_REPORTING_YEAR = HOLDOUT_START_YEAR - 1
DEFAULT_BATCH_SIZE = 50_000
DEFAULT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class BuildConfig:
    cpc_path: Path
    application_path: Path
    assignee_path: Path
    output_dir: Path
    database_path: Path
    batch_size: int = DEFAULT_BATCH_SIZE
    sqlite_cache_mb: int = 256
    report_through_year: int = MAX_REPORTING_YEAR
    tolerance: float = DEFAULT_TOLERANCE
    skip_cpc_rows: int = 0
    max_cpc_rows: Optional[int] = None
    patent_ids_file: Optional[Path] = None
    diagnostic_cpc_groups: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class AuditReport:
    tolerance: float
    patent_maingroup_unique: bool
    patent_assignee_unique: bool
    application_year_unambiguous: bool
    patent_topic_mass_conserved: bool
    max_patent_topic_mass_error: float
    firm_edge_mass_conserved: bool
    max_firm_edge_mass_error: float
    mass_gap_matches_unassigned_patent_mass: bool
    max_mass_gap_error: float
    coverage_in_unit_interval: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.patent_maingroup_unique,
                self.patent_assignee_unique,
                self.application_year_unambiguous,
                self.patent_topic_mass_conserved,
                self.firm_edge_mass_conserved,
                self.mass_gap_matches_unassigned_patent_mass,
                self.coverage_in_unit_interval,
            )
        )


@dataclass(frozen=True)
class KDistribution:
    patent_count: int
    mean_k: float
    fraction_k_gt_one: float
    counts_by_k: Dict[int, int]
    mean_inventional_rows: Optional[float] = None
    mean_unique_cpc_groups: Optional[float] = None
    fraction_with_maingroup_collapse: Optional[float] = None


@dataclass(frozen=True)
class BuildResult:
    audit: AuditReport
    k_distribution: KDistribution
    database_path: Path
    target_path: Path
    firm_edges_path: Path
    coverage_report_path: Path
    mass_gap_report_path: Path
    audit_report_path: Path


def validate_reporting_year(year: int) -> int:
    """Reject any attempt to report aggregate outcomes from the holdout."""
    if year > MAX_REPORTING_YEAR:
        raise ValueError(
            "Holdout guard: report_through_year must be <= "
            f"{MAX_REPORTING_YEAR}; got {year}"
        )
    if year < MIN_FILING_YEAR:
        raise ValueError(
            f"report_through_year must be >= {MIN_FILING_YEAR}; got {year}"
        )
    return year


def coarsen_to_maingroup(cpc_group: str) -> str:
    """Match techtrend_common.coarsen(): take the part before the first '/'."""
    return str(cpc_group).split("/", 1)[0].strip()


def _batched(rows: Iterable[Tuple[str, ...]], size: int) -> Iterator[List[Tuple[str, ...]]]:
    batch: List[Tuple[str, ...]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _dict_rows(path: Path, required: Sequence[str]) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quotechar='"')
        fields = set(reader.fieldnames or [])
        missing = set(required) - fields
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            yield row


def _valid_year(filing_date: str) -> Optional[int]:
    value = (filing_date or "").strip()
    prefix = value[:4]
    if len(prefix) != 4 or not prefix.isdigit():
        return None
    year = int(prefix)
    if not MIN_FILING_YEAR <= year <= MAX_FILING_YEAR:
        return None
    return year


def _connect_database(path: Path, cache_mb: int) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute(f"PRAGMA cache_size={-1024 * cache_mb}")
    conn.execute("PRAGMA foreign_keys=OFF")
    return conn


def _create_schema(conn: sqlite3.Connection, diagnostics: bool) -> None:
    conn.executescript(
        """
        CREATE TABLE patent_maingroups (
            patent_id TEXT NOT NULL,
            maingroup TEXT NOT NULL,
            PRIMARY KEY (patent_id, maingroup)
        ) WITHOUT ROWID;

        CREATE TABLE patent_year_candidates (
            patent_id TEXT NOT NULL,
            filing_year INTEGER NOT NULL,
            PRIMARY KEY (patent_id, filing_year)
        ) WITHOUT ROWID;

        CREATE TABLE patent_assignees (
            patent_id TEXT NOT NULL,
            assignee_id TEXT NOT NULL,
            PRIMARY KEY (patent_id, assignee_id)
        ) WITHOUT ROWID;

        CREATE TEMP TABLE application_batch (
            patent_id TEXT NOT NULL,
            filing_year INTEGER NOT NULL,
            PRIMARY KEY (patent_id, filing_year)
        ) WITHOUT ROWID;

        CREATE TEMP TABLE assignee_batch (
            patent_id TEXT NOT NULL,
            assignee_id TEXT NOT NULL,
            PRIMARY KEY (patent_id, assignee_id)
        ) WITHOUT ROWID;
        """
    )
    if diagnostics:
        conn.executescript(
            """
            CREATE TABLE diagnostic_cpc_rows (patent_id TEXT NOT NULL);
            CREATE TABLE diagnostic_cpc_groups (
                patent_id TEXT NOT NULL,
                cpc_group TEXT NOT NULL,
                PRIMARY KEY (patent_id, cpc_group)
            ) WITHOUT ROWID;
            """
        )
    conn.commit()


def _load_requested_patents(conn: sqlite3.Connection, path: Path, batch_size: int) -> None:
    conn.execute(
        "CREATE TABLE requested_patents (patent_id TEXT PRIMARY KEY) WITHOUT ROWID"
    )

    def ids() -> Iterator[Tuple[str]]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                patent_id = line.rstrip("\r\n").split("\t", 1)[0].strip().strip('"')
                if patent_id and patent_id.lower() != "patent_id":
                    yield (patent_id,)

    for batch in _batched(ids(), batch_size):
        with conn:
            conn.executemany(
                "INSERT OR IGNORE INTO requested_patents VALUES (?)", batch
            )


def _ingest_cpc(conn: sqlite3.Connection, config: BuildConfig) -> None:
    use_requested = config.patent_ids_file is not None
    if use_requested:
        _load_requested_patents(conn, config.patent_ids_file, config.batch_size)
        conn.execute(
            """
            CREATE TEMP TABLE cpc_batch (
                patent_id TEXT NOT NULL,
                maingroup TEXT NOT NULL,
                cpc_group TEXT NOT NULL
            )
            """
        )

    def cpc_rows() -> Iterator[Tuple[str, str, str]]:
        for row_number, row in enumerate(
            _dict_rows(config.cpc_path, ("patent_id", "cpc_group", "cpc_type")),
            start=1,
        ):
            if row_number <= config.skip_cpc_rows:
                continue
            if (
                config.max_cpc_rows is not None
                and row_number > config.skip_cpc_rows + config.max_cpc_rows
            ):
                break
            if (row.get("cpc_type") or "").strip().lower() != "inventional":
                continue
            patent_id = (row.get("patent_id") or "").strip()
            cpc_group = (row.get("cpc_group") or "").strip()
            maingroup = coarsen_to_maingroup(cpc_group)
            if patent_id and cpc_group and maingroup:
                yield patent_id, maingroup, cpc_group

    for batch in _batched(cpc_rows(), config.batch_size):
        with conn:
            if use_requested:
                conn.executemany("INSERT INTO cpc_batch VALUES (?, ?, ?)", batch)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO patent_maingroups
                    SELECT b.patent_id, b.maingroup
                    FROM cpc_batch AS b
                    JOIN requested_patents AS r USING (patent_id)
                    """
                )
                if config.diagnostic_cpc_groups:
                    conn.execute(
                        """
                        INSERT INTO diagnostic_cpc_rows
                        SELECT b.patent_id FROM cpc_batch AS b
                        JOIN requested_patents AS r USING (patent_id)
                        """
                    )
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO diagnostic_cpc_groups
                        SELECT b.patent_id, b.cpc_group FROM cpc_batch AS b
                        JOIN requested_patents AS r USING (patent_id)
                        """
                    )
                conn.execute("DELETE FROM cpc_batch")
            else:
                conn.executemany(
                    "INSERT OR IGNORE INTO patent_maingroups VALUES (?, ?)",
                    ((patent_id, maingroup) for patent_id, maingroup, _ in batch),
                )
                if config.diagnostic_cpc_groups:
                    conn.executemany(
                        "INSERT INTO diagnostic_cpc_rows VALUES (?)",
                        ((patent_id,) for patent_id, _, _ in batch),
                    )
                    conn.executemany(
                        "INSERT OR IGNORE INTO diagnostic_cpc_groups VALUES (?, ?)",
                        ((patent_id, cpc_group) for patent_id, _, cpc_group in batch),
                    )


def _ingest_applications(conn: sqlite3.Connection, config: BuildConfig) -> None:
    def application_rows() -> Iterator[Tuple[str, int]]:
        for row in _dict_rows(
            config.application_path, ("patent_id", "filing_date")
        ):
            patent_id = (row.get("patent_id") or "").strip()
            year = _valid_year(row.get("filing_date") or "")
            if patent_id and year is not None:
                yield patent_id, year

    for batch in _batched(application_rows(), config.batch_size):
        with conn:
            conn.executemany(
                "INSERT OR IGNORE INTO application_batch VALUES (?, ?)", batch
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO patent_year_candidates
                SELECT b.patent_id, b.filing_year
                FROM application_batch AS b
                WHERE EXISTS (
                    SELECT 1 FROM patent_maingroups AS p
                    WHERE p.patent_id = b.patent_id
                )
                """
            )
            conn.execute("DELETE FROM application_batch")


def _ingest_assignees(conn: sqlite3.Connection, config: BuildConfig) -> None:
    def assignee_rows() -> Iterator[Tuple[str, str]]:
        for row in _dict_rows(
            config.assignee_path,
            ("patent_id", "assignee_id", "disambig_assignee_organization"),
        ):
            patent_id = (row.get("patent_id") or "").strip()
            assignee_id = (row.get("assignee_id") or "").strip()
            organization = (row.get("disambig_assignee_organization") or "").strip()
            # A_p is the number of organization-assignee IDs, so both fields
            # must identify an organization and a usable ID.
            if patent_id and assignee_id and organization:
                yield patent_id, assignee_id

    for batch in _batched(assignee_rows(), config.batch_size):
        with conn:
            conn.executemany(
                "INSERT OR IGNORE INTO assignee_batch VALUES (?, ?)", batch
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO patent_assignees
                SELECT b.patent_id, b.assignee_id
                FROM assignee_batch AS b
                WHERE EXISTS (
                    SELECT 1 FROM patent_maingroups AS p
                    WHERE p.patent_id = b.patent_id
                )
                """
            )
            conn.execute("DELETE FROM assignee_batch")


def _materialize_panel(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE patent_year AS
        SELECT patent_id, MIN(filing_year) AS filing_year
        FROM patent_year_candidates
        GROUP BY patent_id;
        CREATE UNIQUE INDEX patent_year_pk ON patent_year(patent_id);

        CREATE TABLE patent_k AS
        SELECT patent_id, COUNT(*) AS k_p
        FROM patent_maingroups
        GROUP BY patent_id;
        CREATE UNIQUE INDEX patent_k_pk ON patent_k(patent_id);

        CREATE TABLE patent_a AS
        SELECT patent_id, COUNT(*) AS a_p
        FROM patent_assignees
        GROUP BY patent_id;
        CREATE UNIQUE INDEX patent_a_pk ON patent_a(patent_id);

        CREATE TABLE target_panel AS
        SELECT y.filing_year,
               m.maingroup,
               SUM(1.0 / k.k_p) AS target_mass
        FROM patent_maingroups AS m
        JOIN patent_k AS k USING (patent_id)
        JOIN patent_year AS y USING (patent_id)
        GROUP BY y.filing_year, m.maingroup;
        CREATE UNIQUE INDEX target_panel_pk
            ON target_panel(filing_year, maingroup);

        CREATE TABLE firm_edges AS
        SELECT y.filing_year,
               a.assignee_id,
               m.maingroup,
               SUM(1.0 / (pa.a_p * k.k_p)) AS edge_weight
        FROM patent_maingroups AS m
        JOIN patent_k AS k USING (patent_id)
        JOIN patent_a AS pa USING (patent_id)
        JOIN patent_assignees AS a USING (patent_id)
        JOIN patent_year AS y USING (patent_id)
        GROUP BY y.filing_year, a.assignee_id, m.maingroup;
        CREATE UNIQUE INDEX firm_edges_pk
            ON firm_edges(filing_year, assignee_id, maingroup);

        CREATE TABLE topic_year_coverage AS
        WITH organization_mass AS (
            SELECT filing_year, maingroup, SUM(edge_weight) AS organization_mass
            FROM firm_edges
            GROUP BY filing_year, maingroup
        )
        SELECT t.filing_year,
               t.maingroup,
               t.target_mass,
               COALESCE(o.organization_mass, 0.0) AS organization_mass,
               t.target_mass - COALESCE(o.organization_mass, 0.0) AS mass_gap,
               CASE WHEN t.target_mass > 0.0
                    THEN COALESCE(o.organization_mass, 0.0) / t.target_mass
                    ELSE NULL END AS coverage
        FROM target_panel AS t
        LEFT JOIN organization_mass AS o
          USING (filing_year, maingroup);
        CREATE UNIQUE INDEX topic_year_coverage_pk
            ON topic_year_coverage(filing_year, maingroup);

        CREATE TABLE unassigned_topic_mass AS
        SELECT y.filing_year,
               m.maingroup,
               SUM(1.0 / k.k_p) AS unassigned_mass
        FROM patent_maingroups AS m
        JOIN patent_k AS k USING (patent_id)
        JOIN patent_year AS y USING (patent_id)
        LEFT JOIN patent_a AS a USING (patent_id)
        WHERE a.patent_id IS NULL
        GROUP BY y.filing_year, m.maingroup;
        CREATE UNIQUE INDEX unassigned_topic_mass_pk
            ON unassigned_topic_mass(filing_year, maingroup);
        """
    )
    conn.commit()
    conn.execute("ANALYZE")


def _scalar(conn: sqlite3.Connection, sql: str, parameters: Sequence[object] = ()) -> float:
    value = conn.execute(sql, parameters).fetchone()[0]
    return float(value or 0.0)


def audit_panel(
    conn: sqlite3.Connection, tolerance: float = DEFAULT_TOLERANCE
) -> AuditReport:
    """Run patent-level conservation and uniqueness audits.

    The report contains only algebraic invariant results, never year-specific
    coverage/topic/exclusion aggregates.
    """
    maingroup_duplicates = _scalar(
        conn,
        """
        SELECT COUNT(*) FROM (
            SELECT patent_id, maingroup
            FROM patent_maingroups
            GROUP BY patent_id, maingroup
            HAVING COUNT(*) <> 1
        )
        """,
    )
    assignee_duplicates = _scalar(
        conn,
        """
        SELECT COUNT(*) FROM (
            SELECT patent_id, assignee_id
            FROM patent_assignees
            GROUP BY patent_id, assignee_id
            HAVING COUNT(*) <> 1
        )
        """,
    )
    ambiguous_years = _scalar(
        conn,
        """
        SELECT COUNT(*) FROM (
            SELECT patent_id
            FROM patent_year_candidates
            GROUP BY patent_id
            HAVING COUNT(*) <> 1
        )
        """,
    )
    topic_error = _scalar(
        conn,
        """
        SELECT COALESCE(MAX(ABS(topic_mass - 1.0)), 0.0)
        FROM (
            SELECT m.patent_id, SUM(1.0 / k.k_p) AS topic_mass
            FROM patent_maingroups AS m
            JOIN patent_k AS k USING (patent_id)
            GROUP BY m.patent_id
        )
        """,
    )
    edge_error = _scalar(
        conn,
        """
        SELECT COALESCE(MAX(ABS(edge_mass - 1.0)), 0.0)
        FROM (
            SELECT m.patent_id,
                   SUM(1.0 / (a.a_p * k.k_p)) AS edge_mass
            FROM patent_maingroups AS m
            JOIN patent_k AS k USING (patent_id)
            JOIN patent_a AS a USING (patent_id)
            JOIN patent_assignees AS u USING (patent_id)
            GROUP BY m.patent_id
        )
        """,
    )
    mass_gap_error = _scalar(
        conn,
        """
        SELECT COALESCE(MAX(ABS(c.mass_gap - COALESCE(u.unassigned_mass, 0.0))), 0.0)
        FROM topic_year_coverage AS c
        LEFT JOIN unassigned_topic_mass AS u
          USING (filing_year, maingroup)
        """,
    )
    coverage_violations = _scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM topic_year_coverage
        WHERE coverage < -? OR coverage > 1.0 + ?
        """,
        (tolerance, tolerance),
    )

    return AuditReport(
        tolerance=tolerance,
        patent_maingroup_unique=maingroup_duplicates == 0,
        patent_assignee_unique=assignee_duplicates == 0,
        application_year_unambiguous=ambiguous_years == 0,
        patent_topic_mass_conserved=topic_error <= tolerance,
        max_patent_topic_mass_error=topic_error,
        firm_edge_mass_conserved=edge_error <= tolerance,
        max_firm_edge_mass_error=edge_error,
        mass_gap_matches_unassigned_patent_mass=mass_gap_error <= tolerance,
        max_mass_gap_error=mass_gap_error,
        coverage_in_unit_interval=coverage_violations == 0,
    )


def k_distribution(
    conn: sqlite3.Connection,
    diagnostics: bool = False,
    report_through_year: int = MAX_REPORTING_YEAR,
) -> KDistribution:
    """Return K diagnostics for the reportable pre-holdout period only."""
    report_through_year = validate_reporting_year(report_through_year)
    rows = conn.execute(
        """
        SELECT k.k_p, COUNT(*)
        FROM patent_k AS k
        JOIN patent_year AS y USING (patent_id)
        WHERE y.filing_year <= ?
        GROUP BY k.k_p ORDER BY k.k_p
        """,
        (report_through_year,),
    ).fetchall()
    counts = {int(k): int(n) for k, n in rows}
    patent_count = sum(counts.values())
    mean_k = (
        sum(k * count for k, count in counts.items()) / patent_count
        if patent_count
        else math.nan
    )
    fraction_multi = (
        sum(count for k, count in counts.items() if k > 1) / patent_count
        if patent_count
        else math.nan
    )

    mean_rows = None
    mean_groups = None
    collapse_fraction = None
    if diagnostics and patent_count:
        mean_rows = _scalar(
            conn,
            """
            SELECT AVG(row_count) FROM (
                SELECT d.patent_id, COUNT(*) AS row_count
                FROM diagnostic_cpc_rows AS d
                JOIN patent_year AS y USING (patent_id)
                WHERE y.filing_year <= ?
                GROUP BY d.patent_id
            )
            """,
            (report_through_year,),
        )
        mean_groups = _scalar(
            conn,
            """
            SELECT AVG(group_count) FROM (
                SELECT d.patent_id, COUNT(*) AS group_count
                FROM diagnostic_cpc_groups AS d
                JOIN patent_year AS y USING (patent_id)
                WHERE y.filing_year <= ?
                GROUP BY d.patent_id
            )
            """,
            (report_through_year,),
        )
        collapse_fraction = _scalar(
            conn,
            """
            WITH group_counts AS (
                SELECT patent_id, COUNT(*) AS group_count
                FROM diagnostic_cpc_groups GROUP BY patent_id
            )
            SELECT AVG(CASE WHEN g.group_count > k.k_p THEN 1.0 ELSE 0.0 END)
            FROM group_counts AS g
            JOIN patent_k AS k USING (patent_id)
            JOIN patent_year AS y USING (patent_id)
            WHERE y.filing_year <= ?
            """,
            (report_through_year,),
        )

    return KDistribution(
        patent_count=patent_count,
        mean_k=mean_k,
        fraction_k_gt_one=fraction_multi,
        counts_by_k=counts,
        mean_inventional_rows=mean_rows,
        mean_unique_cpc_groups=mean_groups,
        fraction_with_maingroup_collapse=collapse_fraction,
    )


def _export_query(
    conn: sqlite3.Connection,
    path: Path,
    header: Sequence[str],
    sql: str,
    parameters: Sequence[object] = (),
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        cursor = conn.execute(sql, parameters)
        while True:
            rows = cursor.fetchmany(10_000)
            if not rows:
                break
            writer.writerows(rows)
    temporary.replace(path)


def _audit_json(report: AuditReport) -> Dict[str, object]:
    result = asdict(report)
    result["passed"] = report.passed
    return result


def _write_outputs(
    conn: sqlite3.Connection,
    config: BuildConfig,
    report: AuditReport,
) -> Tuple[Path, Path, Path, Path, Path]:
    # Repeat the guard at the reporting boundary as defense in depth.
    report_year = validate_reporting_year(config.report_through_year)
    target_path = config.output_dir / "target_panel.tsv"
    edges_path = config.output_dir / "firm_edges.tsv"
    coverage_path = config.output_dir / f"coverage_through_{report_year}.tsv"
    mass_gap_path = (
        config.output_dir / f"organization_mass_gap_through_{report_year}.tsv"
    )
    audit_path = config.output_dir / "audit_report.json"

    _export_query(
        conn,
        target_path,
        ("filing_year", "maingroup", "target_mass"),
        """
        SELECT filing_year, maingroup, target_mass
        FROM target_panel ORDER BY filing_year, maingroup
        """,
    )
    _export_query(
        conn,
        edges_path,
        ("filing_year", "assignee_id", "maingroup", "edge_weight"),
        """
        SELECT filing_year, assignee_id, maingroup, edge_weight
        FROM firm_edges ORDER BY filing_year, assignee_id, maingroup
        """,
    )
    _export_query(
        conn,
        coverage_path,
        (
            "filing_year",
            "maingroup",
            "target_mass",
            "organization_mass",
            "coverage",
        ),
        """
        SELECT filing_year, maingroup, target_mass, organization_mass, coverage
        FROM topic_year_coverage
        WHERE filing_year <= ?
        ORDER BY filing_year, maingroup
        """,
        (report_year,),
    )
    _export_query(
        conn,
        mass_gap_path,
        ("filing_year", "maingroup", "target_mass", "organization_mass", "mass_gap"),
        """
        SELECT filing_year, maingroup, target_mass, organization_mass, mass_gap
        FROM topic_year_coverage
        WHERE filing_year <= ?
        ORDER BY filing_year, maingroup
        """,
        (report_year,),
    )
    audit_path.write_text(
        json.dumps(_audit_json(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target_path, edges_path, coverage_path, mass_gap_path, audit_path


def _print_safe_coverage(conn: sqlite3.Connection, report_year: int) -> None:
    """Print only explicitly pre-holdout yearly coverage aggregates."""
    report_year = validate_reporting_year(report_year)
    rows = conn.execute(
        """
        SELECT filing_year,
               SUM(organization_mass) / SUM(target_mass) AS yearly_coverage
        FROM topic_year_coverage
        WHERE filing_year <= ?
        GROUP BY filing_year
        ORDER BY filing_year
        """,
        (report_year,),
    )
    print(f"Coverage report (guarded: filing_year <= {report_year}):", flush=True)
    for year, coverage in rows:
        print(f"  filing_year={year} coverage={coverage:.6f}", flush=True)


def build_occupancy_panel(config: BuildConfig) -> BuildResult:
    """Build full-period panels and guarded audit reports."""
    # The holdout guard is the first data-policy check at the public entrypoint.
    validate_reporting_year(config.report_through_year)
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.skip_cpc_rows < 0:
        raise ValueError("skip_cpc_rows must be non-negative")
    if config.max_cpc_rows is not None and config.max_cpc_rows <= 0:
        raise ValueError("max_cpc_rows must be positive")
    if config.max_cpc_rows is not None and config.patent_ids_file is not None:
        raise ValueError("Use only one of max_cpc_rows and patent_ids_file")
    if config.skip_cpc_rows and config.patent_ids_file is not None:
        raise ValueError("skip_cpc_rows cannot be combined with patent_ids_file")
    if config.diagnostic_cpc_groups and not (
        config.max_cpc_rows is not None or config.patent_ids_file is not None
    ):
        raise ValueError(
            "diagnostic_cpc_groups is sample-only; also set max_cpc_rows "
            "or patent_ids_file"
        )
    for path in (config.cpc_path, config.application_path, config.assignee_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if config.patent_ids_file is not None and not config.patent_ids_file.is_file():
        raise FileNotFoundError(config.patent_ids_file)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.database_path.parent.mkdir(parents=True, exist_ok=True)
    database_files = (
        config.database_path,
        Path(str(config.database_path) + "-wal"),
        Path(str(config.database_path) + "-shm"),
    )
    existing_database_files = [path for path in database_files if path.exists()]
    if existing_database_files:
        if not config.overwrite:
            raise FileExistsError(
                f"Database already exists: {config.database_path}; use --overwrite"
            )
        for path in existing_database_files:
            path.unlink()

    print(
        f"Holdout guard active: aggregate reporting ends at "
        f"{config.report_through_year}.",
        flush=True,
    )
    conn = _connect_database(config.database_path, config.sqlite_cache_mb)
    try:
        _create_schema(conn, config.diagnostic_cpc_groups)
        print("Streaming inventional CPC records...", flush=True)
        _ingest_cpc(conn, config)
        if _scalar(conn, "SELECT COUNT(*) FROM patent_maingroups") == 0:
            raise ValueError("No inventional patent-maingroup records were retained")
        print("Streaming filing years (valid range 1900-2025)...", flush=True)
        _ingest_applications(conn, config)
        print("Streaming organization assignees...", flush=True)
        _ingest_assignees(conn, config)
        print("Materializing target and firm-edge panels on disk...", flush=True)
        _materialize_panel(conn)
        print("Running conservation and uniqueness audits...", flush=True)
        report = audit_panel(conn, config.tolerance)
        distribution = k_distribution(
            conn, config.diagnostic_cpc_groups, config.report_through_year
        )
        output_paths = _write_outputs(conn, config, report)
        _print_safe_coverage(conn, config.report_through_year)
        print(
            "K_p: "
            f"patents={distribution.patent_count} "
            f"mean={distribution.mean_k:.6f} "
            f"fraction(K_p>1)={distribution.fraction_k_gt_one:.6f} "
            f"counts={distribution.counts_by_k}",
            flush=True,
        )
        if distribution.mean_inventional_rows is not None:
            print(
                "CPC sample diagnostic: "
                f"mean_inventional_rows={distribution.mean_inventional_rows:.6f} "
                f"mean_unique_cpc_groups={distribution.mean_unique_cpc_groups:.6f} "
                "fraction_with_maingroup_collapse="
                f"{distribution.fraction_with_maingroup_collapse:.6f}",
                flush=True,
            )
        print(f"Audit passed={report.passed}", flush=True)
        if not report.passed:
            raise RuntimeError(f"Occupancy-panel audit failed: {_audit_json(report)}")
        return BuildResult(report, distribution, config.database_path, *output_paths)
    finally:
        conn.close()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-dir", type=Path, default=DEFAULT_BULK_DIR)
    parser.add_argument("--cpc-path", type=Path)
    parser.add_argument("--application-path", type=Path)
    parser.add_argument("--assignee-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--database-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sqlite-cache-mb", type=int, default=256)
    parser.add_argument(
        "--report-through-year", type=int, default=MAX_REPORTING_YEAR
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    sample = parser.add_mutually_exclusive_group()
    sample.add_argument("--max-cpc-rows", type=int)
    sample.add_argument("--patent-ids-file", type=Path)
    parser.add_argument(
        "--skip-cpc-rows",
        type=int,
        default=0,
        help="sample mode: skip this many CPC data rows before max-cpc-rows",
    )
    parser.add_argument("--diagnostic-cpc-groups", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    bulk = args.bulk_dir
    database_path = args.database_path or args.output_dir / "occupancy_panel.sqlite3"
    config = BuildConfig(
        cpc_path=args.cpc_path or bulk / "g_cpc_current.tsv",
        application_path=args.application_path or bulk / "g_application.tsv",
        assignee_path=args.assignee_path or bulk / "g_assignee_disambiguated.tsv",
        output_dir=args.output_dir,
        database_path=database_path,
        batch_size=args.batch_size,
        sqlite_cache_mb=args.sqlite_cache_mb,
        report_through_year=args.report_through_year,
        tolerance=args.tolerance,
        skip_cpc_rows=args.skip_cpc_rows,
        max_cpc_rows=args.max_cpc_rows,
        patent_ids_file=args.patent_ids_file,
        diagnostic_cpc_groups=args.diagnostic_cpc_groups,
        overwrite=args.overwrite,
    )
    try:
        build_occupancy_panel(config)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
