"""Tests for the routing effectiveness report's success-rate computation.

The report must surface a real completion success rate that EXCLUDES
in-flight streaming start rows (error_type='streaming_in_progress'), which
are not failures. This guards against the misleading ~50% success that the
raw `success` column produces when in-flight rows are counted as failures.
"""

from __future__ import annotations

import io
import importlib.util
import sqlite3
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# The report is a hyphenated script (check-routing-stats.py), so it cannot be
# imported by name — load it from its path.
_SPEC = importlib.util.spec_from_file_location(
    "check_routing_stats", SCRIPTS_DIR / "check-routing-stats.py"
)
crs = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(crs)


def _make_state_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            started_at REAL,
            parent_session_id TEXT
        );
        CREATE TABLE session_model_usage (
            session_id TEXT,
            model TEXT,
            billing_provider TEXT,
            api_call_count INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            first_seen REAL,
            last_seen REAL
        );
        """
    )
    now = datetime.now(timezone.utc).timestamp()
    conn.execute(
        "INSERT INTO sessions (id, started_at, parent_session_id) VALUES (?, ?, NULL)",
        ("s1", now),
    )
    conn.execute(
        """INSERT INTO session_model_usage
           (session_id, model, billing_provider, api_call_count,
            input_tokens, output_tokens, first_seen, last_seen)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("s1", "deepseek-v4-flash:cloud", "cloud", 4, 1000, 200, now, now),
    )
    conn.commit()
    conn.close()


def _make_router_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE router_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, session_id TEXT, model_used TEXT, provider TEXT,
            task_type TEXT, prompt_length INTEGER, context_length INTEGER,
            tool_call_count INTEGER, contains_code_blocks INTEGER,
            has_keywords INTEGER, input_tokens INTEGER, output_tokens INTEGER,
            latency_seconds REAL, estimated_cost_usd REAL, success INTEGER,
            retry_count INTEGER, escalated INTEGER, user_corrected INTEGER,
            error_type TEXT, complexity_score REAL, instruction_count INTEGER,
            has_format_constraint INTEGER, has_niche_references INTEGER,
            is_subagent INTEGER, parent_model TEXT, delegation_depth INTEGER,
            user_correction_count INTEGER, model_switched INTEGER,
            session_message_count INTEGER, cheaper_model_would_work INTEGER,
            recommended_model TEXT, compression_level TEXT,
            compression_savings_pct REAL, compression_time_ms REAL,
            request_id TEXT, requested_model TEXT, streaming INTEGER,
            workload_type TEXT, requires_tools INTEGER, context_tokens INTEGER,
            empty_stream INTEGER, saw_content INTEGER, saw_tool_calls INTEGER,
            final_model TEXT
        );
        """
    )
    ts = datetime.now(timezone.utc).isoformat()
    rows = [
        # 2 successful completions
        (ts, "s1", "deepseek-v4-flash", "cloud", "coding", 1, 1, 1, 1, 1,
         100, 50, 1.0, 0.0, 1, 0, 0, 0, "", 0.5, 0, 0, 0, 0, "", 0, 0, 0, 0,
         0, "", "off", 0.0, 0.0, "r1", "biggie-router", 0, "normal_chat", 0,
         0, 0, 1, 0, "deepseek-v4-flash"),
        (ts, "s1", "deepseek-v4-flash", "cloud", "coding", 1, 1, 1, 1, 1,
         100, 50, 1.0, 0.0, 1, 0, 0, 0, "", 0.5, 0, 0, 0, 0, "", 0, 0, 0, 0,
         0, "", "off", 0.0, 0.0, "r2", "biggie-router", 0, "normal_chat", 0,
         0, 0, 1, 0, "deepseek-v4-flash"),
        # 1 failed completion (real failure)
        (ts, "s1", "deepseek-v4-flash", "cloud", "coding", 1, 1, 1, 1, 1,
         100, 0, 1.0, 0.0, 0, 0, 0, 0, "timeout", 0.5, 0, 0, 0, 0, "", 0, 0,
         0, 0, 0, "", "off", 0.0, 0.0, "r3", "biggie-router", 0, "normal_chat",
         0, 0, 0, 0, 0, "deepseek-v4-flash"),
        # 1 in-flight streaming start row — must be EXCLUDED from the rate
        (ts, "s1", "deepseek-v4-flash", "cloud", "coding", 1, 1, 1, 1, 1,
         100, 0, 0.0, 0.0, 0, 0, 0, 0, "streaming_in_progress", 0.5, 0, 0, 0,
         0, "", 0, 0, 0, 0, 0, "", "off", 0.0, 0.0, "r4", "biggie-router", 1,
         "normal_chat", 0, 0, 0, 0, 0, "deepseek-v4-flash"),
    ]
    conn.executemany(
        """INSERT INTO router_logs (
            timestamp, session_id, model_used, provider, task_type,
            prompt_length, context_length, tool_call_count, contains_code_blocks,
            has_keywords, input_tokens, output_tokens, latency_seconds,
            estimated_cost_usd, success, retry_count, escalated, user_corrected,
            error_type, complexity_score, instruction_count, has_format_constraint,
            has_niche_references, is_subagent, parent_model, delegation_depth,
            user_correction_count, model_switched, session_message_count,
            cheaper_model_would_work, recommended_model, compression_level,
            compression_savings_pct, compression_time_ms, request_id,
            requested_model, streaming, workload_type, requires_tools,
            context_tokens, empty_stream, saw_content, saw_tool_calls, final_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    conn.close()


def _run_report(state_db: str, router_db: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        crs.STATE_DB = state_db
        crs.ROUTER_LOGS = router_db
        crs.main()
    return buf.getvalue()


def test_report_success_rate_excludes_inflight(tmp_path):
    state_db = str(tmp_path / "state.db")
    router_db = str(tmp_path / "router_logs.db")
    _make_state_db(state_db)
    _make_router_db(router_db)

    out = _run_report(state_db, router_db)

    # 2 ok + 1 failed = 3 completions; the in-flight row is excluded.
    assert "Completion success rate: 66.67%" in out
    assert "2 ok / 1 failed" in out
    # The in-flight row must not be counted as a failure.
    assert "3 failed" not in out
    # The real failure type is surfaced.
    assert "timeout" in out


def test_report_json_has_completion_block(tmp_path):
    state_db = str(tmp_path / "state.db")
    router_db = str(tmp_path / "router_logs.db")
    _make_state_db(state_db)
    _make_router_db(router_db)

    buf = io.StringIO()
    with redirect_stdout(buf):
        crs.STATE_DB = state_db
        crs.ROUTER_LOGS = router_db
        sys.argv = ["check-routing-stats.py", "--json"]
        crs.main()
    sys.argv = ["check-routing-stats.py"]

    import json as _json
    report = _json.loads(buf.getvalue())
    comp = report["completion"]
    assert comp["success"] == 2
    assert comp["failures"] == 1
    assert comp["success_rate_pct"] == 66.67
    assert comp["failure_by_type"] == {"timeout": 1}
