"""B34 — tool-name capture for flash-vs-pro cutover agreement.

The shadow lane proved flash *can* emit tool calls, but "can" is not "does the
same job as pro". A valid-JSON tool call to the WRONG tool is invisible when the
only stored signal is ``saw_tool_calls`` (a bool). Cutover is gated on
*agreement*: does flash pick the same tool (and equivalent args) that pro picks
on the same request?

This file covers the capture primitives that make agreement measurable:

  * two new router_logs columns (``tool_names``, ``tool_args_sig``) migrate
    in both the endpoint's own migration and rollup.migrate();
  * extraction from a non-streaming assistant message / response;
  * SSE tool-call delta fragmentation into an index-keyed accumulator, and
    finalization into the ordered comma-joined names + a bounded args signature.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402
import rollup  # noqa: E402


# ── schema migration ──────────────────────────────────────────────────────────

def test_endpoint_migration_adds_tool_name_columns():
    """Both columns must exist after the endpoint's own migration, or a live
    production DB that only self-heals via the endpoint would reject the INSERT."""
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE router_logs (id INTEGER PRIMARY KEY, model_used TEXT)"
    )
    db.commit()
    ep._ensure_log_columns(db)
    cols = {r[1] for r in db.execute("PRAGMA table_info(router_logs)").fetchall()}
    assert "tool_names" in cols
    assert "tool_args_sig" in cols


def test_rollup_migration_adds_tool_name_columns():
    """rollup.migrate() is the authoritative offline migration; it must agree."""
    db = sqlite3.connect(":memory:")
    rollup.migrate(db)
    cols = {r[1] for r in db.execute("PRAGMA table_info(router_logs)").fetchall()}
    assert "tool_names" in cols
    assert "tool_args_sig" in cols
    assert "tool_names" in rollup.NEW_LOG_COLUMNS
    assert "tool_args_sig" in rollup.NEW_LOG_COLUMNS


# ── non-streaming extraction ─────────────────────────────────────────────────

def test_tool_names_from_message_orders_and_dedupes_names():
    msg = {
        "tool_calls": [
            {"function": {"name": "terminal", "arguments": '{"c":"ls"}'}},
            {"function": {"name": "search", "arguments": '{"q":"x"}'}},
        ]
    }
    names, sig = ep._tool_names_from_message(msg)
    assert names == "terminal,search"
    assert sig  # non-empty when args present


def test_tool_names_from_message_empty_when_no_tool_calls():
    assert ep._tool_names_from_message({"content": "plain"}) == ("", "")
    assert ep._tool_names_from_message(None) == ("", "")
    assert ep._tool_names_from_message({"tool_calls": []}) == ("", "")


def test_tool_names_from_response_reads_first_choice():
    resp = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "read_file", "arguments": '{"path":"/x"}'}},
    ]}}]}
    names, sig = ep._tool_names_from_response(resp)
    assert names == "read_file"
    assert sig


def test_tool_names_sig_is_stable_across_arg_key_order():
    """Args that differ only in JSON key order must produce the same signature."""
    a = {"tool_calls": [{"function": {"name": "t", "arguments": '{"b":2,"a":1}'}}]}
    b = {"tool_calls": [{"function": {"name": "t", "arguments": '{"a":1,"b":2}'}}]}
    assert ep._tool_names_from_message(a)[1] == ep._tool_names_from_message(b)[1]


# ── streaming fragmentation ───────────────────────────────────────────────────

def _sse_delta(tool_calls):
    return __import__("json").dumps(
        {"choices": [{"delta": {"tool_calls": tool_calls}}]}
    )


def test_sse_tool_call_fragments_merge_by_index():
    acc = {}
    ep._accumulate_tool_calls(acc, _sse_delta(
        [{"index": 0, "function": {"name": "terminal", "arguments": '{"c"'}}]
    ))
    ep._accumulate_tool_calls(acc, _sse_delta(
        [{"index": 0, "function": {"arguments": ': "ls"}'}}]
    ))
    names, sig = ep._finalize_tool_calls(acc)
    assert names == "terminal"
    assert sig


def test_sse_fragments_handle_multiple_tool_calls_out_of_order():
    acc = {}
    ep._accumulate_tool_calls(acc, _sse_delta(
        [{"index": 1, "function": {"name": "search", "arguments": '{}'}}]
    ))
    ep._accumulate_tool_calls(acc, _sse_delta(
        [{"index": 0, "function": {"name": "terminal", "arguments": '{}'}}]
    ))
    names, sig = ep._finalize_tool_calls(acc)
    assert names == "terminal,search"  # ordered by index, not arrival
