"""B44c — plumb ``governor_fail_open`` from the decision to the row, and report it.

B44a/b closed the leak and stopped the reason string from asserting a verdict that
never happened. Neither is *countable*: a blind window and a deliberate allow both
still land as ``plan_budget_denied = 0`` with a class stamped. This batch makes a
blind governor visible in the row data, which is the whole point of B44 — the
defect was never a wrong answer, it was an unobservable one.

RED before GREEN. These fail on the pre-B44c tree because the column does not
exist, ``_log_request_to_db`` has no such parameter, and the report has no
``blind`` bucket.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402
import rollup  # noqa: E402
import router_ops  # noqa: E402


# ── Schema: the column must exist in both migration paths ────────────────────

def test_rollup_declares_the_column():
    assert "governor_fail_open" in rollup.NEW_LOG_COLUMNS


def test_endpoint_schema_map_declares_the_column():
    assert "governor_fail_open" in ep._STREAM_OBS_COLUMNS or \
        "governor_fail_open" in ep._COST_OBS_COLUMNS


def test_rollup_migration_adds_the_column_to_a_legacy_db(tmp_path):
    """The existing production router_logs.db must gain the column in place."""
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    # The documented pre-B1 legacy shape: migrate() adds columns on top of it,
    # but it also creates indexes/views that reference these base columns, so a
    # thinner table is not a valid stand-in for production.
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
        "INSERT INTO router_logs (timestamp, session_id, model_used) "
        "VALUES ('2026-09-01T00:00:00+00:00', 'sess-1', 'deepseek-v4.1-flash')"
    )
    conn.commit()

    rollup.migrate(conn)

    cols = {r[1] for r in conn.execute("PRAGMA table_info(router_logs)").fetchall()}
    assert "governor_fail_open" in cols
    # Migration is additive: the pre-existing row survives and reads as governed.
    row = conn.execute("SELECT model_used, governor_fail_open FROM router_logs").fetchone()
    assert row[0] == "deepseek-v4.1-flash"
    assert row[1] == 0
    conn.close()


# ── Write path: the value must actually reach the row ────────────────────────

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "router_logs.db"
    conn = sqlite3.connect(str(db_path))
    # Base production shape; _ensure_log_columns adds the observability columns
    # (including governor_fail_open) on top of it, as it does in production.
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
            success INTEGER NOT NULL DEFAULT 1,
            escalated INTEGER NOT NULL DEFAULT 0,
            error_type TEXT,
            complexity_score REAL NOT NULL DEFAULT 0.0,
            compression_level TEXT NOT NULL DEFAULT 'off',
            compression_savings_pct REAL NOT NULL DEFAULT 0.0,
            compression_time_ms REAL NOT NULL DEFAULT 0.0
        )"""
    )
    conn.commit()
    monkeypatch.setattr(ep, "_sqlite_lock", threading.Lock())
    monkeypatch.setattr(
        ep, "_get_db_connection", lambda: (ep._ensure_log_columns(conn), conn)[1]
    )
    yield conn
    conn.close()


def _log(conn, **kw):
    ep._log_request_to_db(
        model_used=kw.pop("model_used", "gpt-5.5"),
        provider="openai-codex",
        task_type="planning",
        complexity_score=0.9,
        input_tokens=100,
        output_tokens=50,
        latency_seconds=1.0,
        routing_time_ms=5,
        **kw,
    )
    return conn.execute(
        "SELECT governor_fail_open, plan_budget_denied, budget_class "
        "FROM router_logs ORDER BY id DESC LIMIT 1"
    ).fetchone()


def test_blind_window_is_recorded_in_the_row(temp_db):
    """The Oct-4/5 signature: served, classed, not denied — and now flagged."""
    blind, denied, klass = _log(
        temp_db,
        plan_budget_denied=False,
        budget_class="routine",
        governor_fail_open=True,
    )
    assert blind == 1
    assert denied == 0
    assert klass == "routine"


def test_normal_allow_is_not_flagged_as_blind(temp_db):
    """A verdict actually reached must not be smeared into the blind count."""
    blind, denied, klass = _log(
        temp_db,
        plan_budget_denied=False,
        budget_class="architectural",
        governor_fail_open=False,
    )
    assert blind == 0
    assert denied == 0
    assert klass == "architectural"


def test_ungoverned_row_defaults_to_not_blind(temp_db):
    """Historical/ungoverned rows must not read as a blind governor (B37i rule)."""
    blind, denied, klass = _log(temp_db)
    assert blind == 0
    assert denied == 0
    assert klass == ""


def test_endpoint_passes_the_flag_from_the_decision():
    """Every site that writes plan_budget_denied must also write the blind flag.

    A call site that forgets logs the request as governed-and-allowed, which is
    the exact invisibility this batch exists to remove.
    """
    src = (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()
    assert src.count("plan_budget_denied=") == src.count("governor_fail_open=")
    assert src.count("governor_fail_open=") >= 5


# ── Reporting: blind must be distinguishable from denied and from served ─────

def _report_db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "report.db"))
    conn.execute(
        "CREATE TABLE router_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "timestamp TEXT NOT NULL, session_id TEXT NOT NULL, model_used TEXT NOT NULL, "
        "input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0, "
        "plan_budget_denied INTEGER NOT NULL DEFAULT 0, budget_class TEXT NOT NULL DEFAULT '', "
        "governor_fail_open INTEGER NOT NULL DEFAULT 0)"
    )
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    rows = [
        ("gpt-5.5", 0, "routine", 1),          # blind: the leak, now visible
        ("gpt-5.5", 0, "routine", 1),          # blind
        ("gpt-5.6-terra", 0, "architectural", 0),  # served deliberately
        ("glm-5.3", 1, "routine", 0),          # denied
        ("deepseek-v4-pro", 0, "", 0),         # ungoverned
    ]
    for model, denied, klass, blind in rows:
        conn.execute(
            "INSERT INTO router_logs (timestamp, session_id, model_used, input_tokens, "
            "output_tokens, plan_budget_denied, budget_class, governor_fail_open) "
            "VALUES (?, 's', ?, 10, 5, ?, ?, ?)",
            (now, model, denied, klass, blind),
        )
    conn.commit()
    return conn


def test_report_counts_blind_rows_separately(tmp_path):
    out = router_ops.format_plan_budget(_report_db(tmp_path), days=7)
    assert "blind" in out.lower()
    assert "2" in out  # the two blind rows are counted
    assert "blind (governor unreadable): 2" in out


def test_report_does_not_claim_the_leak_is_closed_in_a_blind_window(tmp_path):
    out = router_ops.format_plan_budget(_report_db(tmp_path), days=7)
    assert "1" in out  # one genuine denial
    assert "leak closed" not in out.lower(), (
        "a window with blind rows must not report the leak as closed"
    )
    assert "BLIND" in out or "blind window" in out.lower()


def test_report_json_carries_the_blind_count(tmp_path):
    conn = _report_db(tmp_path)
    cols = router_ops._log_columns(conn)
    assert "governor_fail_open" in cols
