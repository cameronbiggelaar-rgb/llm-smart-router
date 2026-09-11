"""B1 — schema & migration layer for the self-optimising router.

Verifies the new tables/columns exist, that migration is idempotent, and that
migrating an existing populated DB preserves its rows (no data loss, no
destructive rebuild).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from rollup import NEW_LOG_COLUMNS, migrate, new_tables_present  # noqa: E402


def _legacy_db(path: Path) -> None:
    """Create a DB shaped like the pre-migration production router_logs."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE router_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            session_id TEXT NOT NULL,
            model_used TEXT NOT NULL,
            provider TEXT NOT NULL DEFAULT '',
            task_type TEXT NOT NULL DEFAULT 'other',
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            latency_seconds REAL NOT NULL DEFAULT 0,
            estimated_cost_usd REAL NOT NULL DEFAULT 0,
            workload_type TEXT NOT NULL DEFAULT ''
        )"""
    )
    conn.execute(
        "INSERT INTO router_logs (timestamp, session_id, model_used, input_tokens, output_tokens) "
        "VALUES ('2026-09-01T00:00:00+00:00', 'sess-1', 'deepseek-v4.1-flash', 1000, 250)"
    )
    conn.commit()
    conn.close()


def test_migrate_creates_all_new_tables(tmp_path):
    db = tmp_path / "router_logs.db"
    conn = sqlite3.connect(str(db))
    migrate(conn)
    present = new_tables_present(conn)
    assert present == {
        "model_pricing": True,
        "experiments": True,
        "daily_findings": True,
        "rollup_state": True,
    }


def test_migrate_adds_all_new_log_columns(tmp_path):
    db = tmp_path / "router_logs.db"
    conn = sqlite3.connect(str(db))
    migrate(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(router_logs)")}
    for col in NEW_LOG_COLUMNS:
        assert col in cols, f"missing column {col}"


def test_migrate_is_idempotent(tmp_path):
    """Running migrate twice must not raise and must not duplicate anything."""
    db = tmp_path / "router_logs.db"
    conn = sqlite3.connect(str(db))
    migrate(conn)
    before = {r[1] for r in conn.execute("PRAGMA table_info(router_logs)")}
    migrate(conn)  # second run on the already-migrated DB
    after = {r[1] for r in conn.execute("PRAGMA table_info(router_logs)")}
    assert before == after
    # exactly one row per table in sqlite_master
    dupes = conn.execute(
        "SELECT name, COUNT(*) c FROM sqlite_master WHERE type='table' "
        "GROUP BY name HAVING c > 1"
    ).fetchall()
    assert dupes == []


def test_migrate_preserves_existing_rows(tmp_path):
    """The migration is additive: legacy rows survive with defaults filled in."""
    db = tmp_path / "router_logs.db"
    _legacy_db(db)
    conn = sqlite3.connect(str(db))
    migrate(conn)
    row = conn.execute(
        "SELECT session_id, input_tokens, output_tokens, cost_usd, cost_unknown, "
        "is_shadow, quality_score, quality_method FROM router_logs"
    ).fetchone()
    assert row[0] == "sess-1"
    assert row[1] == 1000
    assert row[2] == 250
    # new columns come back with sane defaults
    assert row[3] == 0.0          # cost_usd
    # Pre-instrumentation rows have no recorded price, so they must migrate as
    # cost_unknown=1. Defaulting to 0 would claim 231k legacy rows were priced
    # at $0 — the exact fiction this column exists to prevent.
    assert row[4] == 1            # cost_unknown
    assert row[5] == 0            # is_shadow
    assert row[6] is None         # quality_score NULL == "not measured"
    assert row[7] == ""           # quality_method


def test_migrate_enables_wal(tmp_path):
    db = tmp_path / "router_logs.db"
    conn = sqlite3.connect(str(db))
    migrate(conn)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_migrate_adds_workload_timestamp_index(tmp_path):
    db = tmp_path / "router_logs.db"
    conn = sqlite3.connect(str(db))
    migrate(conn)
    idx = {r[1] for r in conn.execute("PRAGMA index_list(router_logs)")}
    assert "idx_router_logs_workload_ts" in idx


def test_daily_findings_primary_key_is_composite(tmp_path):
    """Daily findings upsert on (day, workload_type, model, call_type)."""
    db = tmp_path / "router_logs.db"
    conn = sqlite3.connect(str(db))
    migrate(conn)
    conn.execute(
        "INSERT INTO daily_findings (day, workload_type, model, call_type, calls) "
        "VALUES ('2026-09-01', 'session_compression', 'glm-5.3', 'compression', 5)"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO daily_findings (day, workload_type, model, call_type, calls) "
            "VALUES ('2026-09-01', 'session_compression', 'glm-5.3', 'compression', 9)"
        )


def test_model_pricing_versions_are_distinct(tmp_path):
    """Same model can hold multiple dated prices; same date cannot repeat."""
    db = tmp_path / "router_logs.db"
    conn = sqlite3.connect(str(db))
    migrate(conn)
    conn.execute(
        "INSERT INTO model_pricing (model, provider, input_usd_per_1m, output_usd_per_1m, "
        "effective_from, source) VALUES ('glm-5.3', 'ollama-cloud', 1.5, 4.5, '2026-09-01', 'list')"
    )
    conn.execute(
        "INSERT INTO model_pricing (model, provider, input_usd_per_1m, output_usd_per_1m, "
        "effective_from, source) VALUES ('glm-5.3', 'ollama-cloud', 1.2, 4.0, '2026-10-01', 'list')"
    )
    assert conn.execute("SELECT COUNT(*) FROM model_pricing").fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO model_pricing (model, provider, input_usd_per_1m, output_usd_per_1m, "
            "effective_from, source) VALUES ('glm-5.3', 'ollama-cloud', 9.9, 9.9, '2026-09-01', 'dup')"
        )
