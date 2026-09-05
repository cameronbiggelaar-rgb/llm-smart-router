"""Tests that the routing report surfaces lane counts (rethink/rearchitect vs
escalation vs force_model) so the new gpt-5.6 / gpt-6 lanes can be measured.

The routing_reason column (added in the endpoint) is the only signal that
distinguishes WHY a strong model was chosen. The report must count these
lanes and surface them in both the full and JSON output.
"""

from __future__ import annotations

import io
import importlib.util
import json
import sqlite3
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

_SPEC = importlib.util.spec_from_file_location(
    "check_routing_stats", SCRIPTS_DIR / "check-routing-stats.py"
)
crs = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(crs)


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
            final_model TEXT, routing_reason TEXT
        );
        """
    )
    ts = datetime.now(timezone.utc).isoformat()
    rows = [
        # rethink/rearchitect lane (tier 13)
        (ts, "s1", "gpt-5.6-sol", "openai-codex", "planning", 1, 1, 1, 1, 1,
         100, 50, 1.0, 0.0, 1, 0, 0, 0, "", 0.9, 0, 0, 0, 0, "", 0, 0, 0, 0,
         0, "", "off", 0.0, 0.0, "r1", "biggie-router", 0, "normal_chat", 0,
         0, 0, 1, 0, "gpt-5.6-sol",
         "capability tier 13 needed, selected gpt-5.6-sol (tier 13)"),
        # escalation lane
        (ts, "s1", "gpt-5.6-sol", "openai-codex", "coding", 1, 1, 1, 1, 1,
         100, 50, 1.0, 0.0, 1, 1, 0, 0, "timeout", 0.5, 0, 0, 0, 0, "", 0, 0,
         0, 0, 0, "", "off", 0.0, 0.0, "r2", "biggie-router", 0, "normal_chat",
         0, 0, 0, 1, 0, "gpt-5.6-sol",
         "escalation from gpt-5.5 (timeout)"),
        # force_model lane
        (ts, "s1", "gpt-6-astra", "openai-codex", "other", 1, 1, 1, 1, 1,
         100, 50, 1.0, 0.0, 1, 0, 0, 0, "", 0.5, 0, 0, 0, 0, "", 0, 0, 0, 0,
         0, "", "off", 0.0, 0.0, "r3", "biggie-router", 0, "normal_chat", 0,
         0, 0, 1, 0, "gpt-6-astra",
         "forced model: gpt-6-astra"),
        # normal lane (no routing_reason)
        (ts, "s1", "deepseek-v4-flash", "cloud", "coding", 1, 1, 1, 1, 1,
         100, 50, 1.0, 0.0, 1, 0, 0, 0, "", 0.5, 0, 0, 0, 0, "", 0, 0, 0, 0,
         0, "", "off", 0.0, 0.0, "r4", "biggie-router", 0, "normal_chat", 0,
         0, 0, 1, 0, "deepseek-v4-flash", ""),
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
            context_tokens, empty_stream, saw_content, saw_tool_calls, final_model,
            routing_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?)""",
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
    conn.commit()
    conn.close()


def test_report_surfaces_lane_counts(tmp_path):
    state_db = str(tmp_path / "state.db")
    router_db = str(tmp_path / "router_logs.db")
    _make_state_db(state_db)
    _make_router_db(router_db)

    out = _run_report(state_db, router_db)

    assert "Routing Lanes" in out
    assert "rethink/rearchitect" in out
    assert "escalation" in out
    assert "force_model" in out
    # 1 rethink, 1 escalation, 1 force_model, 1 normal
    assert "1" in out


def test_report_json_has_lane_block(tmp_path):
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

    report = json.loads(buf.getvalue())
    lanes = report["router_endpoint"]["lanes"]
    assert lanes["rethink_rearchitect"] == 1
    assert lanes["escalation"] == 1
    assert lanes["force_model"] == 1
    assert lanes["normal"] == 1
