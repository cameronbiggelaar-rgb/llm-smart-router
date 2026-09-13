"""B26.3 — the rollup must not average v1 and v2 quality together.

v1 scored every model at 0.0-0.087 on real payloads; v2 spans 0.0-1.0. Averaging
across methods would blend two different scales into a number that describes
neither — exactly the class of artifact this fix exists to remove.

So: ``daily_findings.quality_avg`` counts ONLY v2 rows, and ``quality_method``
is what distinguishes them. v1 rows stay in ``router_logs`` untouched.
"""
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import quality  # noqa: E402
import rollup  # noqa: E402

DAY = "2026-09-13"

_LOGS_SCHEMA = """
CREATE TABLE router_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_id TEXT,
    model_used TEXT,
    provider TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    latency_seconds REAL,
    estimated_cost_usd REAL DEFAULT 0,
    success INTEGER DEFAULT 1,
    escalated INTEGER DEFAULT 0,
    compression_level INTEGER DEFAULT 0,
    request_id TEXT,
    workload_type TEXT,
    cost_usd REAL DEFAULT 0,
    cost_unknown INTEGER DEFAULT 0,
    quality_score REAL,
    quality_method TEXT NOT NULL DEFAULT '',
    experiment TEXT,
    experiment_arm TEXT,
    is_shadow INTEGER DEFAULT 0,
    error_type TEXT
);
"""


def _row(conn, *, model, method, score):
    conn.execute(
        """INSERT INTO router_logs
           (timestamp, model_used, provider, input_tokens, output_tokens, cost_usd,
            success, workload_type, cost_usd, quality_score, quality_method, is_shadow)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,0)""",
        (
            f"{DAY}T10:00:00+10:00", model, "p", 100, 10, 0.001,
            1, "compression", 0.001, score, method,
        ),
    )


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(_LOGS_SCHEMA)
    # Canonical DDL from rollup.py, so this fixture cannot drift from the
    # production schema the way a hand-copied one would.
    c.executescript(rollup.NEW_TABLES_SQL)
    yield c
    c.close()


def test_rollup_excludes_v1_rows_from_quality_avg(conn):
    """A v1 row must not drag the reported quality_avg."""
    _row(conn, model="m1", method=quality.METHOD_V2, score=0.9)
    _row(conn, model="m1", method=quality.METHOD_V2, score=0.7)
    # v1 measured 0.02 on the same model. Must not appear in the average.
    _row(conn, model="m1", method=quality.METHOD, score=0.02)

    rollup.rollup_day(conn, DAY)

    got = conn.execute(
        "SELECT quality_avg, quality_n FROM daily_findings WHERE model = 'm1'"
    ).fetchone()
    assert got is not None, "rollup wrote no findings row"
    assert got["quality_n"] == 2, f"expected only the 2 v2 rows, got {got['quality_n']}"
    assert got["quality_avg"] == pytest.approx(0.8), got["quality_avg"]


def test_rollup_reports_no_quality_when_only_v1_rows_exist(conn):
    """v1-only history must read as *unmeasured*, not as a real score.

    This is what keeps the optimiser honest: it treats quality_n == 0 as
    'no evidence' and refuses to rank, rather than accepting 0.02 as a floor.
    """
    _row(conn, model="m2", method=quality.METHOD, score=0.02)
    _row(conn, model="m2", method=quality.METHOD, score=0.05)

    rollup.rollup_day(conn, DAY)

    got = conn.execute(
        "SELECT quality_avg, quality_n FROM daily_findings WHERE model = 'm2'"
    ).fetchone()
    assert got["quality_n"] == 0
    assert got["quality_avg"] is None


def test_v1_rows_are_preserved_not_deleted(conn):
    """The fix is going forward only — historical rows stay for audit."""
    _row(conn, model="m3", method=quality.METHOD, score=0.02)
    rollup.rollup_day(conn, DAY)
    n = conn.execute(
        "SELECT COUNT(*) FROM router_logs WHERE quality_method = ?", (quality.METHOD,)
    ).fetchone()[0]
    assert n == 1
