"""Tests that the endpoint logs the routing reason (which lane fired) so the
report can measure rethink/rearchitect vs escalation vs force_model usage.

The routing reason is the only signal that distinguishes WHY a strong model
was chosen — e.g. an explicit rethink/rearchitect trigger (tier 13) vs a
failure escalation vs a force_model override. Without it the report can only
infer the lane from model_used + escalated + requested_model.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the endpoint's DB connection at a fresh temp DB."""
    db_path = tmp_path / "router_logs.db"
    conn = sqlite3.connect(str(db_path))
    # Create the base schema (mirrors models.SCHEMA_SQL's router_logs).
    conn.executescript(
        """
        CREATE TABLE router_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            session_id TEXT NOT NULL,
            model_used TEXT NOT NULL,
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
            instruction_count INTEGER NOT NULL DEFAULT 0,
            has_format_constraint INTEGER NOT NULL DEFAULT 0,
            has_niche_references INTEGER NOT NULL DEFAULT 0,
            is_subagent INTEGER NOT NULL DEFAULT 0,
            parent_model TEXT NOT NULL DEFAULT '',
            delegation_depth INTEGER NOT NULL DEFAULT 0,
            user_correction_count INTEGER NOT NULL DEFAULT 0,
            model_switched INTEGER NOT NULL DEFAULT 0,
            session_message_count INTEGER NOT NULL DEFAULT 0,
            cheaper_model_would_work INTEGER NOT NULL DEFAULT 0,
            recommended_model TEXT NOT NULL DEFAULT '',
            compression_level TEXT NOT NULL DEFAULT 'off',
            compression_savings_pct REAL NOT NULL DEFAULT 0.0,
            compression_time_ms REAL NOT NULL DEFAULT 0.0
            );
        """
    )
    conn.commit()
    import threading
    monkeypatch.setattr(ep, "_sqlite_lock", threading.Lock())
    monkeypatch.setattr(ep, "_get_db_connection", lambda: (ep._ensure_log_columns(conn), conn)[1])
    yield conn
    conn.close()


def _log(conn, **kw):
    ep._log_request_to_db(
        model_used=kw.pop("model_used", "gpt-5.6-sol"),
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
        "SELECT routing_reason, model_used, escalated FROM router_logs ORDER BY id DESC LIMIT 1"
    ).fetchone()


def test_routing_reason_column_is_migrated(temp_db):
    # The column must exist after _ensure_log_columns runs (via _get_db_connection).
    ep._ensure_log_columns(temp_db)
    cols = {r[1] for r in temp_db.execute("PRAGMA table_info(router_logs)").fetchall()}
    assert "routing_reason" in cols


def test_rethink_reason_is_logged(temp_db):
    row = _log(
        temp_db,
        routing_reason="capability tier 13 needed, selected gpt-5.6-sol (tier 13)",
    )
    assert row[0] == "capability tier 13 needed, selected gpt-5.6-sol (tier 13)"
    assert row[1] == "gpt-5.6-sol"


def test_escalation_reason_is_logged(temp_db):
    row = _log(
        temp_db,
        model_used="gpt-5.6-sol",
        escalated=True,
        routing_reason="escalation from gpt-5.5 (timeout)",
    )
    assert row[0] == "escalation from gpt-5.5 (timeout)"
    assert row[2] == 1


def test_force_model_reason_is_logged(temp_db):
    row = _log(
        temp_db,
        model_used="gpt-6-astra",
        routing_reason="forced model: gpt-6-astra",
    )
    assert row[0] == "forced model: gpt-6-astra"
    assert row[1] == "gpt-6-astra"


def test_routing_reason_defaults_to_empty(temp_db):
    # A call that doesn't pass routing_reason must still work (default "").
    row = _log(temp_db, model_used="deepseek-v4-flash")
    assert row[0] == ""
