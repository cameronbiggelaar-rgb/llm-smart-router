"""Rollup, retention and schema migration for the self-optimising router.

This module owns three things:

1. **Schema migration** (this batch) — additively extends ``router_logs`` with
   the cost/quality/experiment columns and creates the new tables
   (``model_pricing``, ``experiments``, ``daily_findings``, ``rollup_state``).
2. **Daily rollup** — aggregates raw ``router_logs`` rows into
   ``daily_findings``, which are small and kept forever.
3. **Retention** — purges raw rows once their rollup is committed.

Batch isolation note: this file is imported by ``tests/test_rollup_schema.py``,
``tests/test_rollup.py``, and ``optimiser.py`` — never by the request hot path.
"""

from __future__ import annotations

import sqlite3
from typing import Dict, Mapping

# ── Log columns added to router_logs ──────────────────────────────────────────
# name -> SQL type/constraint fragment. Kept as an ordered mapping so the
# migration is deterministic and testable.

NEW_LOG_COLUMNS: Mapping[str, str] = {
    "cost_usd": "REAL NOT NULL DEFAULT 0",
    "cost_unknown": "INTEGER NOT NULL DEFAULT 0",
    "pricing_version": "TEXT NOT NULL DEFAULT ''",
    "experiment": "TEXT NOT NULL DEFAULT ''",
    "experiment_arm": "TEXT NOT NULL DEFAULT ''",
    "is_shadow": "INTEGER NOT NULL DEFAULT 0",
    "quality_score": "REAL",
    "quality_method": "TEXT NOT NULL DEFAULT ''",
}

# ── New tables ────────────────────────────────────────────────────────────────

NEW_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS model_pricing (
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    input_usd_per_1m REAL NOT NULL,
    output_usd_per_1m REAL NOT NULL,
    effective_from TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (model, effective_from)
);

CREATE TABLE IF NOT EXISTS experiments (
    name TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 0,
    model TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'shadow',
    percent REAL NOT NULL DEFAULT 0,
    match_json TEXT NOT NULL DEFAULT '{}',
    started TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS daily_findings (
    day TEXT NOT NULL,
    workload_type TEXT NOT NULL,
    model TEXT NOT NULL,
    call_type TEXT NOT NULL DEFAULT '',
    calls INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    cost_unknown_calls INTEGER NOT NULL DEFAULT 0,
    success_calls INTEGER NOT NULL DEFAULT 0,
    escalated_calls INTEGER NOT NULL DEFAULT 0,
    latency_p50 REAL NOT NULL DEFAULT 0,
    latency_p95 REAL NOT NULL DEFAULT 0,
    quality_avg REAL,
    quality_n INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, workload_type, model, call_type)
);

CREATE INDEX IF NOT EXISTS idx_daily_findings_day ON daily_findings(day);

CREATE TABLE IF NOT EXISTS rollup_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_router_logs_workload_ts
    ON router_logs(workload_type, timestamp);
"""


def _log_columns(conn: sqlite3.Connection) -> set[str]:
    return {r[1] for r in conn.execute("PRAGMA table_info(router_logs)")}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def new_tables_present(conn: sqlite3.Connection) -> Dict[str, bool]:
    """Report presence of each new table. Used by tests and health checks."""
    return {
        name: _table_exists(conn, name)
        for name in ("model_pricing", "experiments", "daily_findings", "rollup_state")
    }


def migrate(conn: sqlite3.Connection) -> Dict[str, int]:
    """Apply the schema migration idempotently.

    Additive only: existing rows are preserved and new columns take their
    declared defaults. Safe to run on every process start.

    Returns a small report of what was added.
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # If router_logs does not exist at all, create the baseline shape first so
    # the ADD COLUMN statements below have something to attach to.
    if not _table_exists(conn, "router_logs"):
        conn.execute(
            """CREATE TABLE router_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT '',
                model_used TEXT NOT NULL DEFAULT '',
                provider TEXT NOT NULL DEFAULT '',
                task_type TEXT NOT NULL DEFAULT 'other',
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                latency_seconds REAL NOT NULL DEFAULT 0,
                estimated_cost_usd REAL NOT NULL DEFAULT 0,
                success INTEGER NOT NULL DEFAULT 1,
                workload_type TEXT NOT NULL DEFAULT ''
            )"""
        )

    added_columns = 0
    existing = _log_columns(conn)
    for name, ddl in NEW_LOG_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE router_logs ADD COLUMN {name} {ddl}")
            added_columns += 1

    conn.executescript(NEW_TABLES_SQL)
    conn.commit()
    return {"added_columns": added_columns}


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    """Create indexes that depend on migrated columns (kept out of migrate()
    for a legacy DB where the column may not exist yet)."""
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_router_logs_experiment
            ON router_logs(experiment, experiment_arm);
        """
    )
    conn.commit()
