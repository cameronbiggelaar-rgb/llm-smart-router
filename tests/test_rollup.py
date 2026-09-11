"""B5 — daily rollup, findings, and raw-log retention.

The requirement: roll logs up into findings regularly, and purge/clean records
so the DB does not grow forever.

The one inviolable rule: **never purge a day whose rollup is not committed.**
These tests prove that ordering, plus idempotency and bounded deletion.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from rollup import (  # noqa: E402
    findings,
    last_rolled_up_day,
    purge_raw,
    rollup_day,
    rollup_range,
    vacuum_if_needed,
)
from rollup import migrate  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "router_logs.db"))
    migrate(conn)
    return conn


def _log(conn, day, model, workload, in_tok, out_tok, cost, latency=1.0,
         success=1, escalated=0, quality=None, cost_unknown=0):
    conn.execute(
        "INSERT INTO router_logs (timestamp, session_id, model_used, workload_type, "
        "input_tokens, output_tokens, cost_usd, cost_unknown, latency_seconds, "
        "success, escalated, quality_score) "
        "VALUES (?,'s',?,?,?,?,?,?,?,?,?,?)",
        (f"{day}T10:00:00+00:00", model, workload, in_tok, out_tok, cost,
         cost_unknown, latency, success, escalated, quality),
    )
    conn.commit()


# ── rollup ────────────────────────────────────────────────────────────────────

def test_rollup_day_aggregates_correctly(db):
    _log(db, "2026-09-01", "glm-5.3", "session_compression", 1000, 100, 0.30)
    _log(db, "2026-09-01", "glm-5.3", "session_compression", 2000, 200, 0.60)
    n = rollup_day(db, "2026-09-01")
    assert n == 1
    row = db.execute(
        "SELECT calls, input_tokens, output_tokens, cost_usd FROM daily_findings "
        "WHERE day='2026-09-01' AND model='glm-5.3'"
    ).fetchone()
    assert row == (2, 3000, 300, pytest.approx(0.90))


def test_rollup_day_separates_call_types(db):
    _log(db, "2026-09-01", "glm-5.3", "session_compression", 1000, 100, 0.30)
    _log(db, "2026-09-01", "glm-5.3", "normal_chat", 100, 10, 0.01)
    rollup_day(db, "2026-09-01")
    types = {r[0] for r in db.execute(
        "SELECT call_type FROM daily_findings WHERE day='2026-09-01'")}
    assert types == {"session_compression", "normal_chat"}


def test_rollup_day_is_idempotent(db):
    """Re-running a day must REPLACE its findings, not double them."""
    _log(db, "2026-09-01", "glm-5.3", "session_compression", 1000, 100, 0.30)
    rollup_day(db, "2026-09-01")
    rollup_day(db, "2026-09-01")
    rollup_day(db, "2026-09-01")
    row = db.execute(
        "SELECT calls, cost_usd FROM daily_findings WHERE day='2026-09-01'"
    ).fetchone()
    assert row == (1, pytest.approx(0.30))


def test_rollup_day_reflects_late_arriving_rows(db):
    """A re-run after more rows land must pick them up (upsert, not skip)."""
    _log(db, "2026-09-01", "glm-5.3", "session_compression", 1000, 100, 0.30)
    rollup_day(db, "2026-09-01")
    _log(db, "2026-09-01", "glm-5.3", "session_compression", 1000, 100, 0.20)
    rollup_day(db, "2026-09-01")
    row = db.execute(
        "SELECT calls, cost_usd FROM daily_findings WHERE day='2026-09-01'"
    ).fetchone()
    assert row == (2, pytest.approx(0.50))


def test_rollup_day_counts_unknown_cost_and_escalations(db):
    _log(db, "2026-09-01", "mystery", "normal_chat", 100, 10, 0.0, cost_unknown=1, escalated=1)
    rollup_day(db, "2026-09-01")
    row = db.execute(
        "SELECT cost_unknown_calls, escalated_calls, success_calls FROM daily_findings"
    ).fetchone()
    assert row[0] == 1
    assert row[1] == 1


def test_rollup_day_computes_latency_percentiles(db):
    for i, lat in enumerate([1.0, 2.0, 3.0, 4.0, 100.0]):
        _log(db, "2026-09-01", "m", "normal_chat", 100, 10, 0.01, latency=lat)
    rollup_day(db, "2026-09-01")
    p50, p95 = db.execute("SELECT latency_p50, latency_p95 FROM daily_findings").fetchone()
    assert p50 == pytest.approx(3.0)
    assert p95 == pytest.approx(100.0)


def test_rollup_day_records_quality_only_when_measured(db):
    _log(db, "2026-09-01", "m", "normal_chat", 100, 10, 0.01, quality=0.9)
    _log(db, "2026-09-01", "m", "normal_chat", 100, 10, 0.01, quality=None)
    rollup_day(db, "2026-09-01")
    avg, n = db.execute("SELECT quality_avg, quality_n FROM daily_findings").fetchone()
    assert n == 1
    assert avg == pytest.approx(0.9)


def test_rollup_day_no_rows_returns_zero(db):
    assert rollup_day(db, "2026-09-01") == 0


def test_rollup_day_advances_watermark(db):
    _log(db, "2026-09-01", "m", "normal_chat", 100, 10, 0.01)
    rollup_day(db, "2026-09-01")
    assert last_rolled_up_day(db) == "2026-09-01"


def test_rollup_range_covers_each_day(db):
    _log(db, "2026-09-01", "m", "normal_chat", 100, 10, 0.01)
    _log(db, "2026-09-02", "m", "normal_chat", 100, 10, 0.02)
    _log(db, "2026-09-03", "m", "normal_chat", 100, 10, 0.03)
    total = rollup_range(db, "2026-09-01", "2026-09-03")
    assert total == 3
    days = {r[0] for r in db.execute("SELECT day FROM daily_findings")}
    assert days == {"2026-09-01", "2026-09-02", "2026-09-03"}


# ── findings ──────────────────────────────────────────────────────────────────

def test_findings_returns_rolled_up_days(db):
    _log(db, "2026-09-01", "glm-5.3", "session_compression", 1000, 100, 0.30)
    rollup_day(db, "2026-09-01")
    f = findings(db, days=3650)
    assert any(x.model == "glm-5.3" and x.call_type == "session_compression" for x in f)


def test_findings_survive_raw_purge(db):
    """Findings are the durable record — purging raw rows must not lose them."""
    _log(db, "2026-01-01", "glm-5.3", "session_compression", 1000, 100, 0.30)
    rollup_day(db, "2026-01-01")
    purge_raw(db, keep_days=1, now="2026-09-01", dry_run=False)
    assert db.execute("SELECT COUNT(*) FROM router_logs").fetchone()[0] == 0
    f = [x for x in findings(db, days=3650) if x.day == "2026-01-01"]
    assert f and f[0].cost_usd == pytest.approx(0.30)


# ── retention ─────────────────────────────────────────────────────────────────

def test_purge_refuses_when_day_not_rolled_up(db):
    """THE critical safety rule: no committed rollup -> no purge."""
    _log(db, "2026-01-01", "m", "normal_chat", 100, 10, 0.01)
    plan = purge_raw(db, keep_days=1, now="2026-09-01", dry_run=False)
    assert plan.refused is True
    assert plan.deleted == 0
    assert db.execute("SELECT COUNT(*) FROM router_logs").fetchone()[0] == 1


def test_purge_dry_run_deletes_nothing(db):
    _log(db, "2026-01-01", "m", "normal_chat", 100, 10, 0.01)
    rollup_day(db, "2026-01-01")
    plan = purge_raw(db, keep_days=1, now="2026-09-01", dry_run=True)
    assert plan.refused is False
    assert plan.candidates == 1
    assert db.execute("SELECT COUNT(*) FROM router_logs").fetchone()[0] == 1


def test_purge_deletes_only_rows_beyond_retention(db):
    _log(db, "2026-01-01", "m", "normal_chat", 100, 10, 0.01)   # old -> gone
    _log(db, "2026-08-31", "m", "normal_chat", 100, 10, 0.01)   # recent -> kept
    rollup_day(db, "2026-01-01")
    rollup_day(db, "2026-08-31")
    plan = purge_raw(db, keep_days=1, now="2026-09-01", dry_run=False)
    assert plan.deleted == 1
    remaining = {r[0] for r in db.execute("SELECT timestamp FROM router_logs")}
    assert remaining == {"2026-08-31T10:00:00+00:00"}


def test_purge_advances_purge_watermark(db):
    _log(db, "2026-01-01", "m", "normal_chat", 100, 10, 0.01)
    rollup_day(db, "2026-01-01")
    purge_raw(db, keep_days=1, now="2026-09-01", dry_run=False)
    from rollup import last_purged_day
    assert last_purged_day(db) == "2026-01-01"


def test_purge_is_idempotent(db):
    _log(db, "2026-01-01", "m", "normal_chat", 100, 10, 0.01)
    rollup_day(db, "2026-01-01")
    first = purge_raw(db, keep_days=1, now="2026-09-01", dry_run=False)
    second = purge_raw(db, keep_days=1, now="2026-09-01", dry_run=False)
    assert first.deleted == 1
    assert second.deleted == 0


def test_purge_respects_batch_bound(db):
    """Deletion is bounded so a large DB does not lock for minutes."""
    for _ in range(50):
        _log(db, "2026-01-01", "m", "normal_chat", 100, 10, 0.01)
    rollup_day(db, "2026-01-01")
    plan = purge_raw(db, keep_days=1, now="2026-09-01", dry_run=False, batch=10)
    assert plan.deleted == 10
    assert db.execute("SELECT COUNT(*) FROM router_logs").fetchone()[0] == 40


def test_vacuum_if_needed_returns_bool(db):
    _log(db, "2026-01-01", "m", "normal_chat", 100, 10, 0.01)
    rollup_day(db, "2026-01-01")
    purge_raw(db, keep_days=1, now="2026-09-01", dry_run=False)
    assert isinstance(vacuum_if_needed(db), bool)
