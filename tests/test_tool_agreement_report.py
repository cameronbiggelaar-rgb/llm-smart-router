"""B34 — report_tool_agreement pairs shadow↔incumbent and scores tool choice.

The report is the go/no-go surface: it turns the new tool_names/tool_args_sig
columns into a single number (flash tool-set agreement vs pro). These tests run
against a throwaway SQLite DB via BIGGIE_ROUTER_DB, never the production file.
"""

import os
import sqlite3
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import report_tool_agreement as rta  # noqa: E402


def _mk_db(tmp_path, rows):
    db = tmp_path / "r.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE router_logs (
            request_id TEXT, tool_names TEXT, tool_args_sig TEXT,
            finish_reason TEXT, saw_tool_calls INTEGER, model_used TEXT,
            final_model TEXT, cost_usd REAL, is_shadow INTEGER, error_type TEXT,
            timestamp TEXT
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO router_logs
        (request_id, tool_names, tool_args_sig, finish_reason, saw_tool_calls,
         model_used, final_model, cost_usd, is_shadow, error_type, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()
    return str(db)


def _row(req, tools, sig, finish, saw, is_shadow, ts="2026-09-20T00:00:00+00:00"):
    model = "deepseek-v4.1-flash" if is_shadow else "deepseek-v4-pro"
    return (req, tools, sig, finish, saw, model, model, 0.0, is_shadow, "", ts)


def test_full_agreement(tmp_path, monkeypatch):
    rows = [
        _row("r1", "search_files", "a1", "tool_calls", 1, 1),
        _row("r1", "search_files", "a1", "tool_calls", 1, 0),
    ]
    db = _mk_db(tmp_path, rows)
    monkeypatch.setenv("BIGGIE_ROUTER_DB", db)
    conn = sqlite3.connect(db); conn.row_factory = sqlite3.Row
    pairs = rta._pairs(conn, 0)
    s = rta._score(pairs)
    assert s["total_pairs"] == 1
    assert s["both_have_tools"] == 1
    assert s["tool_agreement"] == 1
    assert s["tool_agreement_rate"] == 1.0
    assert s["arg_agreement"] == 1


def test_tool_mismatch_detected(tmp_path, monkeypatch):
    rows = [
        _row("r1", "search_files", "a1", "tool_calls", 1, 1),
        _row("r1", "read_file", "a1", "tool_calls", 1, 0),
    ]
    db = _mk_db(tmp_path, rows)
    monkeypatch.setenv("BIGGIE_ROUTER_DB", db)
    conn = sqlite3.connect(db); conn.row_factory = sqlite3.Row
    pairs = rta._pairs(conn, 0)
    s = rta._score(pairs)
    assert s["both_have_tools"] == 1
    assert s["tool_agreement"] == 0
    assert s["tool_agreement_rate"] == 0.0
    assert len(s["disagree_examples"]) == 1


def test_arg_mismatch_keeps_tool_agreement(tmp_path, monkeypatch):
    rows = [
        _row("r1", "write_file", "a1", "tool_calls", 1, 1),
        _row("r1", "write_file", "a2", "tool_calls", 1, 0),
    ]
    db = _mk_db(tmp_path, rows)
    monkeypatch.setenv("BIGGIE_ROUTER_DB", db)
    conn = sqlite3.connect(db); conn.row_factory = sqlite3.Row
    pairs = rta._pairs(conn, 0)
    s = rta._score(pairs)
    assert s["tool_agreement"] == 1
    assert s["arg_agreement"] == 0


def test_empty_content_counted(tmp_path, monkeypatch):
    rows = [
        _row("r1", "", "", "", 0, 1),       # shadow empty, no tool call
        _row("r1", "search_files", "a1", "tool_calls", 1, 0),
    ]
    db = _mk_db(tmp_path, rows)
    monkeypatch.setenv("BIGGIE_ROUTER_DB", db)
    conn = sqlite3.connect(db); conn.row_factory = sqlite3.Row
    pairs = rta._pairs(conn, 0)
    s = rta._score(pairs)
    assert s["shadow_empty_content"] == 1
    assert s["shadow_empty_content_rate"] == 1.0


def test_incumbent_stream_marker_excluded(tmp_path, monkeypatch):
    # The incumbent row with error_type='streaming_in_progress' must be skipped,
    # so the real completion row is used for comparison.
    rows = [
        _row("r1", "search_files", "a1", "tool_calls", 1, 1),
        _row("r1", "", "", "streaming_in_progress", 0, 0),
        _row("r1", "search_files", "a1", "tool_calls", 1, 0),
    ]
    # mark the middle row as the streaming marker
    rows = [
        (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], "streaming_in_progress" if i == 1 else "", r[10])
        for i, r in enumerate(rows)
    ]
    db = _mk_db(tmp_path, rows)
    monkeypatch.setenv("BIGGIE_ROUTER_DB", db)
    conn = sqlite3.connect(db); conn.row_factory = sqlite3.Row
    pairs = rta._pairs(conn, 0)
    assert len(pairs) == 1
    s = rta._score(pairs)
    assert s["tool_agreement"] == 1


def test_no_pairs_returns_exit_1(tmp_path, monkeypatch):
    db = _mk_db(tmp_path, [])
    monkeypatch.setenv("BIGGIE_ROUTER_DB", db)
    monkeypatch.setattr(sys, "argv", ["report_tool_agreement.py", "--since", "0"])
    assert rta.main() == 1
