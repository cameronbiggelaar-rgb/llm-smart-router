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
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

# ── Log columns added to router_logs ──────────────────────────────────────────
# name -> SQL type/constraint fragment. Kept as an ordered mapping so the
# migration is deterministic and testable.

NEW_LOG_COLUMNS: Mapping[str, str] = {
    "cost_usd": "REAL NOT NULL DEFAULT 0",
    "cost_unknown": "INTEGER NOT NULL DEFAULT 1",
    "pricing_version": "TEXT NOT NULL DEFAULT ''",
    "experiment": "TEXT NOT NULL DEFAULT ''",
    "experiment_arm": "TEXT NOT NULL DEFAULT ''",
    "is_shadow": "INTEGER NOT NULL DEFAULT 0",
    "quality_score": "REAL",
    "quality_method": "TEXT NOT NULL DEFAULT ''",
    "finish_reason": "TEXT NOT NULL DEFAULT ''",
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
    -- Shadow calls are real spend but NOT production traffic. Keeping the flag
    -- on the finding is what lets a reader exclude them from a spend figure;
    -- folding them in silently overstates production cost.
    is_shadow INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, workload_type, model, call_type, is_shadow)
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


def _migrate_daily_findings_shadow(conn: sqlite3.Connection) -> int:
    """Bring an existing ``daily_findings`` up to the shadow-aware shape.

    ``is_shadow`` is part of the primary key, because the same model can
    legitimately be both the serving model and an experiment's candidate on the
    same day for the same workload — one row each, and they must not collide.

    SQLite cannot ALTER a primary key, so the table is rebuilt and the existing
    rows copied across (stamped ``is_shadow=0``, which is what they are). This is
    the only safe option: ``daily_findings`` is a derived table, but purge may
    have already deleted the raw rows behind older days, so it is NOT always
    recomputable — the history must be carried over rather than re-rolled.

    Idempotent. Returns 1 if the table was rebuilt, else 0.
    """
    if not _table_exists(conn, "daily_findings"):
        return 0

    cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_findings)")}
    pk_cols = [
        r[1] for r in conn.execute("PRAGMA table_info(daily_findings)") if r[5]
    ]
    if "is_shadow" in cols and "is_shadow" in pk_cols:
        return 0

    conn.execute(
        """CREATE TABLE daily_findings_shadow_migration (
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
            is_shadow INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (day, workload_type, model, call_type, is_shadow)
        )"""
    )
    shadow_expr = "is_shadow" if "is_shadow" in cols else "0"
    conn.execute(
        f"""INSERT INTO daily_findings_shadow_migration
            (day, workload_type, model, call_type, calls, input_tokens,
             output_tokens, cost_usd, cost_unknown_calls, success_calls,
             escalated_calls, latency_p50, latency_p95, quality_avg, quality_n,
             is_shadow)
            SELECT day, workload_type, model, call_type, calls, input_tokens,
                   output_tokens, cost_usd, cost_unknown_calls, success_calls,
                   escalated_calls, latency_p50, latency_p95, quality_avg, quality_n,
                   {shadow_expr}
            FROM daily_findings"""
    )
    conn.execute("DROP TABLE daily_findings")
    conn.execute("ALTER TABLE daily_findings_shadow_migration RENAME TO daily_findings")
    conn.commit()
    return 1


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
                prompt_length INTEGER NOT NULL DEFAULT 0,
                context_length INTEGER NOT NULL DEFAULT 0,
                tool_call_count INTEGER NOT NULL DEFAULT 0,
                contains_code_blocks INTEGER NOT NULL DEFAULT 0,
                has_keywords INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                latency_seconds REAL NOT NULL DEFAULT 0,
                estimated_cost_usd REAL NOT NULL DEFAULT 0,
                success INTEGER NOT NULL DEFAULT 1,
                retry_count INTEGER NOT NULL DEFAULT 0,
                escalated INTEGER NOT NULL DEFAULT 0,
                user_corrected INTEGER NOT NULL DEFAULT 0,
                error_type TEXT,
                complexity_score REAL NOT NULL DEFAULT 0.0,
                is_subagent INTEGER NOT NULL DEFAULT 0,
                model_switched INTEGER NOT NULL DEFAULT 0,
                request_id TEXT NOT NULL DEFAULT '',
                requested_model TEXT NOT NULL DEFAULT '',
                streaming INTEGER NOT NULL DEFAULT 0,
                workload_type TEXT NOT NULL DEFAULT '',
                -- Columns the endpoint's INSERT needs. The endpoint also
                -- self-heals these at first write, but a DB that has only ever
                -- been through migrate() must accept a row on its own —
                -- otherwise "migrated" and "writable" silently disagree and the
                -- logging path fails with "no column named ..." while the
                -- request itself still succeeds.
                compression_level TEXT NOT NULL DEFAULT '',
                compression_savings_pct REAL NOT NULL DEFAULT 0,
                compression_time_ms REAL NOT NULL DEFAULT 0,
                requires_tools INTEGER NOT NULL DEFAULT 0,
                context_tokens INTEGER NOT NULL DEFAULT 0,
                empty_stream INTEGER NOT NULL DEFAULT 0,
                saw_content INTEGER NOT NULL DEFAULT 0,
                saw_tool_calls INTEGER NOT NULL DEFAULT 0,
                final_model TEXT NOT NULL DEFAULT '',
                routing_reason TEXT NOT NULL DEFAULT '',
                finish_reason TEXT NOT NULL DEFAULT ''
            )"""
        )

    added_columns = 0
    existing = _log_columns(conn)
    for name, ddl in NEW_LOG_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE router_logs ADD COLUMN {name} {ddl}")
            added_columns += 1

    added_findings_columns = _migrate_daily_findings_shadow(conn)

    conn.executescript(NEW_TABLES_SQL)
    conn.commit()
    return {"added_columns": added_columns, "added_findings_columns": added_findings_columns}


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


# ── daily rollup ──────────────────────────────────────────────────────────────
#
# Raw rows are rolled up into daily_findings, which are ~1 row per
# (day, workload, model, call_type) — small enough to keep forever. Once a day's
# rollup is committed, its raw rows become eligible for retention purge.

# Default raw-log retention. At ~10k rows/day this keeps ~300k raw rows; the
# daily_findings rollup retains the durable history indefinitely, so changing
# this trades disk against forensic detail, not against reporting.
DEFAULT_KEEP_DAYS = 30


@dataclass(frozen=True)
class Finding:
    """One rolled-up (day, workload, model, call_type) aggregate."""

    day: str
    workload_type: str
    model: str
    call_type: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_unknown_calls: int
    success_calls: int
    escalated_calls: int
    latency_p50: float
    latency_p95: float
    quality_avg: Optional[float]
    quality_n: int


@dataclass(frozen=True)
class PurgePlan:
    """Outcome of a retention purge."""

    keep_days: int
    cutoff: str
    candidates: int
    deleted: int
    refused: bool = False
    reason: str = ""


def _set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO rollup_state (key, value) VALUES (?, ?)", (key, value)
    )
    conn.commit()


def _get_state(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM rollup_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def last_rolled_up_day(conn: sqlite3.Connection) -> str:
    return _get_state(conn, "last_rolled_up_day", "")


def last_purged_day(conn: sqlite3.Connection) -> str:
    return _get_state(conn, "last_purged_day", "")


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Nearest-rank percentile. Deterministic and dependency-free."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    # nearest-rank: ceil(pct/100 * N) clamped to [1, N]
    import math

    rank = max(1, min(len(sorted_values), math.ceil(pct / 100.0 * len(sorted_values))))
    return float(sorted_values[rank - 1])


def rollup_day(conn: sqlite3.Connection, day: str) -> int:
    """Aggregate one day's raw rows into ``daily_findings``. Idempotent.

    Implemented as delete-then-insert for the day so late-arriving rows are
    picked up on a re-run, and so a re-run can never double-count. Returns the
    number of finding rows written.
    """
    rows = conn.execute(
        """
        SELECT
            COALESCE(NULLIF(workload_type, ''), 'unknown') AS workload_type,
            model_used,
            COALESCE(NULLIF(workload_type, ''), 'unknown') AS call_type,
            COUNT(*),
            COALESCE(SUM(input_tokens), 0),
            COALESCE(SUM(output_tokens), 0),
            COALESCE(SUM(cost_usd), 0),
            COALESCE(SUM(cost_unknown), 0),
            COALESCE(SUM(success), 0),
            COALESCE(SUM(escalated), 0),
            AVG(quality_score),
            COUNT(quality_score),
            is_shadow
        FROM router_logs
        WHERE substr(timestamp, 1, 10) = substr(?, 1, 10)
        GROUP BY workload_type, model_used, is_shadow
        """,
        (day,),
    ).fetchall()

    if not rows:
        # Still record the watermark: the day is legitimately empty, and
        # refusing to mark it would block retention forever.
        conn.execute("DELETE FROM daily_findings WHERE day = ?", (day,))
        conn.commit()
        _set_state(conn, "last_rolled_up_day", day)
        return 0

    conn.execute("DELETE FROM daily_findings WHERE day = ?", (day,))

    written = 0
    for r in rows:
        (workload_type, model, call_type, calls, in_tok, out_tok, cost,
         unknown_calls, success_calls, escalated_calls, q_avg, q_n, is_shadow) = r
        is_shadow = int(is_shadow or 0)
        # Latency percentiles must be computed over the same shadow/non-shadow
        # slice, or a slow candidate would appear to slow production down.
        latencies = [
            float(x[0])
            for x in conn.execute(
                "SELECT latency_seconds FROM router_logs "
                "WHERE substr(timestamp,1,10)=substr(?,1,10) AND model_used=? "
                "AND COALESCE(NULLIF(workload_type,''),'unknown')=? "
                "AND is_shadow=? "
                "AND latency_seconds > 0 ORDER BY latency_seconds",
                (day, model, workload_type, is_shadow),
            )
        ]
        conn.execute(
            "INSERT INTO daily_findings (day, workload_type, model, call_type, calls, "
            "input_tokens, output_tokens, cost_usd, cost_unknown_calls, success_calls, "
            "escalated_calls, latency_p50, latency_p95, quality_avg, quality_n, is_shadow) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                day, workload_type, model, call_type, int(calls),
                int(in_tok), int(out_tok), float(cost), int(unknown_calls),
                int(success_calls), int(escalated_calls),
                _percentile(latencies, 50.0), _percentile(latencies, 95.0),
                float(q_avg) if q_avg is not None else None, int(q_n or 0),
                is_shadow,
            ),
        )
        written += 1

    conn.commit()
    _set_state(conn, "last_rolled_up_day", day)
    return written


def rollup_range(conn: sqlite3.Connection, start: str, end: str) -> int:
    """Roll up every day from ``start`` to ``end`` inclusive."""
    from datetime import date, timedelta

    def _parse(s: str) -> date:
        return date.fromisoformat(s[:10])

    d = _parse(start)
    stop = _parse(end)
    if d > stop:
        return 0
    total = 0
    while d <= stop:
        total += rollup_day(conn, d.isoformat())
        d += timedelta(days=1)
    return total


def findings(conn: sqlite3.Connection, days: int = 7) -> List[Finding]:
    """Return rolled-up findings for the last ``days`` days."""
    from datetime import date, timedelta

    cutoff = (date.today() - timedelta(days=days)).isoformat()
    out: List[Finding] = []
    for r in conn.execute(
        "SELECT day, workload_type, model, call_type, calls, input_tokens, output_tokens, "
        "cost_usd, cost_unknown_calls, success_calls, escalated_calls, latency_p50, "
        "latency_p95, quality_avg, quality_n FROM daily_findings "
        "WHERE day >= ? ORDER BY day DESC, cost_usd DESC",
        (cutoff,),
    ):
        out.append(Finding(*r))
    return out


# ── retention ─────────────────────────────────────────────────────────────────

def purge_raw(
    conn: sqlite3.Connection,
    keep_days: int,
    now: Optional[str] = None,
    dry_run: bool = True,
    batch: int = 50_000,
) -> PurgePlan:
    """Delete raw ``router_logs`` older than ``keep_days``. Rolled-up days only.

    **Ordering is inviolable: a day is purged only if its rollup is committed.**
    Deleting raw rows for an un-rolled-up day is irreversible data loss, so this
    refuses rather than proceeds.

    Deletion is batched so a large DB does not hold a write lock for minutes.
    ``dry_run=True`` (the default) reports the plan without deleting anything.
    """
    from datetime import date, timedelta

    today = (now or date.today().isoformat())[:10]
    cutoff = (date.fromisoformat(today) - timedelta(days=keep_days)).isoformat()

    # Days eligible by age...
    eligible = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT substr(timestamp,1,10) AS d FROM router_logs "
            "WHERE d < ? ORDER BY d",
            (cutoff,),
        )
    ]
    # ...intersected with days that actually have committed findings.
    rolled = {
        r[0] for r in conn.execute("SELECT DISTINCT day FROM daily_findings")
    }
    unrolled = [d for d in eligible if d not in rolled]

    if unrolled:
        return PurgePlan(
            keep_days=keep_days,
            cutoff=cutoff,
            candidates=len(eligible),
            deleted=0,
            refused=True,
            reason=(
                f"refusing to purge {len(unrolled)} day(s) with no committed rollup: "
                + ", ".join(unrolled[:5])
                + ("..." if len(unrolled) > 5 else "")
            ),
        )

    if not eligible:
        return PurgePlan(keep_days=keep_days, cutoff=cutoff, candidates=0, deleted=0)

    candidate_rows = conn.execute(
        "SELECT COUNT(*) FROM router_logs WHERE substr(timestamp,1,10) < ?",
        (cutoff,),
    ).fetchone()[0]

    if dry_run:
        return PurgePlan(
            keep_days=keep_days, cutoff=cutoff,
            candidates=int(candidate_rows), deleted=0,
        )

    deleted = 0
    while deleted < batch:
        cur = conn.execute(
            "DELETE FROM router_logs WHERE id IN ("
            "  SELECT id FROM router_logs WHERE substr(timestamp,1,10) < ? LIMIT ?"
            ")",
            (cutoff, batch - deleted),
        )
        if cur.rowcount <= 0:
            break
        deleted += cur.rowcount
        conn.commit()

    # Advance the watermark only for days that are now fully gone.
    remaining = conn.execute(
        "SELECT MIN(substr(timestamp,1,10)) FROM router_logs "
        "WHERE substr(timestamp,1,10) < ?",
        (cutoff,),
    ).fetchone()[0]
    purged_through = max(eligible)
    if remaining is None:
        _set_state(conn, "last_purged_day", purged_through)

    return PurgePlan(
        keep_days=keep_days, cutoff=cutoff,
        candidates=int(candidate_rows), deleted=deleted,
    )


def vacuum_if_needed(conn: sqlite3.Connection, min_free_pages: int = 2000) -> bool:
    """VACUUM when enough free pages have accumulated. Returns True if it ran.

    Deleting rows frees pages but does not shrink the file; without this the DB
    stays at its high-water mark forever.
    """
    free = conn.execute("PRAGMA freelist_count").fetchone()[0]
    if free is None or free < min_free_pages:
        return False
    # VACUUM cannot run inside a transaction.
    try:
        conn.commit()
        conn.execute("VACUUM")
        return True
    except sqlite3.Error:
        return False
