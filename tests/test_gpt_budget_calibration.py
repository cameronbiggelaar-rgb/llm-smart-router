"""Batch B37k (RED) — calibration replay.

Why this module exists
----------------------
The governor's policy is a *prediction*: "routine never reaches the plan, and the
architectural lane is never starved."§5.7 requires that prediction be tested
against recorded history before the policy is armed, because the alternative is
discovering an over-denial in production.

The honest limit
----------------
Prompts are not stored — ``router_logs`` keeps ``prompt_length``, not the text —
so prompt-based classification **cannot** be replayed. Replay therefore reads the
logged ``routing_reason``, which records the tier the router *needed*
("capability tier 13 needed"), the force_model override, and whether the request
was tool-bearing.

The consequence is a deliberate bias, and it must be stated wherever the numbers
are: replay can only *under*-detect architectural requests, never over-detect
them. So its denial count is an **upper bound**, and the one result that would be
alarming — the architectural lane being denied — is the result replay is most
likely to produce spuriously. A replay that reports architectural denials is a
prompt to check the classifier, not proof of a policy failure.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import gpt_budget as gb  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(
    ts: str,
    model: str,
    reason: str,
    *,
    task_type: str = "coding",
    requires_tools: int = 0,
    escalated: int = 0,
    in_tokens: int = 1000,
    out_tokens: int = 500,
) -> tuple:
    return (ts, model, reason, task_type, requires_tools, escalated, in_tokens, out_tokens)


def _db(tmp_path, rows):
    db = str(tmp_path / "logs.db")
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE router_logs (
            timestamp TEXT NOT NULL, final_model TEXT, routing_reason TEXT,
            task_type TEXT, requires_tools INTEGER, escalated INTEGER,
            input_tokens INTEGER, output_tokens INTEGER
        )
        """
    )
    conn.executemany(
        "INSERT INTO router_logs (timestamp, final_model, routing_reason, task_type,"
        " requires_tools, escalated, input_tokens, output_tokens)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db


# ── Classification from stored history ────────────────────────────────────────

def test_forced_model_row_replays_as_architectural():
    """The operator escape hatch must survive replay (P0 reserve)."""
    klass = gb.classify_stored_request(
        reason="forced model: gpt-5.5", model="gpt-5.5", escalated=False
    )
    assert klass == gb.CLASS_ARCHITECTURAL


def test_tier_13_needed_replays_as_architectural():
    klass = gb.classify_stored_request(
        reason="capability tier 13 needed, selected gpt-5.6-sol (tier 13)",
        model="gpt-5.6-sol",
        escalated=False,
    )
    assert klass == gb.CLASS_ARCHITECTURAL


def test_tool_branch_row_replays_as_routine():
    """The leak, stated directly: the tool branch carries no tier clause."""
    klass = gb.classify_stored_request(
        reason="tool-bearing request — selected proven tool-capable model gpt-5.5 (tier 10)",
        model="gpt-5.5",
        escalated=False,
    )
    assert klass == gb.CLASS_ROUTINE


def test_escalation_inherits_and_is_NOT_reclassified_as_architectural():
    """Escalation is tool-failover, not architecture (§3.1).

    A tier-10 gpt model on an escalation row must not be read as architectural
    just because it is a plan model.
    """
    klass = gb.classify_stored_request(
        reason="tool escalation from deepseek-v4-pro (timeout) — fell to most-capable tool model gpt-5.6-sol",
        model="gpt-5.6-sol",
        escalated=True,
    )
    assert klass == gb.CLASS_ROUTINE


def test_selected_tier_is_not_mistaken_for_needed_tier():
    """`(tier 13)` in the SELECTED clause must not look like `13 needed`.

    This is the parse that keeps replay honest: the tool branch names the tier it
    *chose*, which for gpt-5.6-sol is 13 even though nothing was needed above 8.
    Reading that as architecture would make the replay claim the governor spares
    traffic it actually denies.
    """
    klass = gb.classify_stored_request(
        reason="tool-bearing request — selected proven tool-capable model gpt-5.6-sol (tier 13)",
        model="gpt-5.6-sol",
        escalated=False,
    )
    assert klass == gb.CLASS_ROUTINE


def test_lower_tier_needed_replays_as_routine():
    klass = gb.classify_stored_request(
        reason="capability tier 10 needed, selected gpt-5.5 (tier 10)",
        model="gpt-5.5",
        escalated=False,
    )
    assert klass == gb.CLASS_ROUTINE


def test_empty_reason_replays_as_routine():
    """Legacy no-reason rows must not be silently promoted to architectural."""
    assert (
        gb.classify_stored_request(reason="", model="gpt-5.5", escalated=False)
        == gb.CLASS_ROUTINE
    )


# ── The replay report ─────────────────────────────────────────────────────────

def _replay(tmp_path, rows, *, since="2026-09-01T00:00:00+00:00"):
    import calibrate_plan_budget as cal

    db = _db(tmp_path, rows)
    conn = sqlite3.connect(db)
    try:
        return cal.replay(conn, since=since)
    finally:
        conn.close()


def test_replay_reports_routine_denied_per_day(tmp_path):
    """§5.7: denial counts tracked per day so over-denial is visible."""
    rows = [
        _row("2026-09-05T01:00:00+00:00", "gpt-5.5",
             "tool-bearing request — selected proven tool-capable model gpt-5.5 (tier 10)"),
        _row("2026-09-05T02:00:00+00:00", "gpt-5.5",
             "capability tier 10 needed, selected gpt-5.5 (tier 10)"),
        _row("2026-09-06T01:00:00+00:00", "gpt-5.5",
             "tool-bearing request — selected proven tool-capable model gpt-5.5 (tier 10)"),
    ]
    out = _replay(tmp_path, rows)
    assert out["days"]["2026-09-05"]["routine_denied"] == 2
    assert out["days"]["2026-09-06"]["routine_denied"] == 1
    assert out["totals"]["routine_denied"] == 3


def test_replay_never_denies_the_architectural_lane(tmp_path):
    """The success criterion: architectural work keeps the plan."""
    rows = [
        _row("2026-09-05T01:00:00+00:00", "gpt-5.6-sol",
             "capability tier 13 needed, selected gpt-5.6-sol (tier 13)",
             task_type="planning"),
        _row("2026-09-05T02:00:00+00:00", "gpt-5.5", "forced model: gpt-5.5"),
    ]
    out = _replay(tmp_path, rows)
    assert out["totals"]["architectural_served"] == 2
    assert out["totals"]["architectural_denied"] == 0
    assert out["architectural_lane_denied"] is False


def test_replay_flags_architectural_denial_when_it_occurs(tmp_path):
    """If the detector is too narrow, replay must say so loudly (§5.7).

    Constructed by denying the architectural class outright, which is what a
    narrowed keyword list would look like.
    """
    import calibrate_plan_budget as cal

    db = _db(tmp_path, [
        _row("2026-09-05T01:00:00+00:00", "gpt-5.6-sol",
             "capability tier 13 needed, selected gpt-5.6-sol (tier 13)",
             task_type="planning"),
    ])
    conn = sqlite3.connect(db)
    try:
        out = cal.replay(conn, since="2026-09-01T00:00:00+00:00",
                         allowlist=[gb.CLASS_ROUTINE])
    finally:
        conn.close()
    assert out["totals"]["architectural_denied"] == 1
    assert out["architectural_lane_denied"] is True


def test_replay_smooths_the_boom_bust(tmp_path):
    """§5.7 success shape: no 700-then-6. The bust days collapse to deny-all."""
    rows = []
    # A boom day: 30 routine tool-bearing calls.
    for i in range(30):
        rows.append(_row(f"2026-09-05T{i:02d}:00:00+00:00", "gpt-5.5",
                         "tool-bearing request — selected proven tool-capable model gpt-5.5 (tier 10)"))
    # A bust day: 1 call (the window was drained).
    rows.append(_row("2026-09-07T01:00:00+00:00", "gpt-5.5",
                     "tool-bearing request — selected proven tool-capable model gpt-5.5 (tier 10)"))
    out = _replay(tmp_path, rows)
    assert out["days"]["2026-09-05"]["routine_denied"] == 30
    assert out["days"]["2026-09-07"]["routine_denied"] == 1
    # Post-policy the plan serves the architectural lane only — the routine
    # swing no longer reaches it at all.
    assert out["totals"]["architectural_served"] == 0


def test_replay_reports_the_prompt_limitation(tmp_path):
    """The bias must travel with the numbers, not live only in a docstring."""
    out = _replay(tmp_path, [
        _row("2026-09-05T01:00:00+00:00", "gpt-5.5", "capability tier 8 needed, selected gpt-5.5"),
    ])
    assert out["limitation"], "replay must state that prompts are not stored"
    assert "upper bound" in out["limitation"].lower()


def test_replay_handles_empty_history_without_raising(tmp_path):
    """A fresh DB must not crash the calibration job."""
    out = _replay(tmp_path, [])
    assert out["totals"]["routine_denied"] == 0
    assert out["days"] == {}


def test_replay_ignores_ungoverned_models(tmp_path):
    """Non-plan models are not the governor's business; replay must not count them."""
    out = _replay(tmp_path, [
        _row("2026-09-05T01:00:00+00:00", "deepseek-v4-pro",
             "capability tier 8 needed, selected deepseek-v4-pro (tier 8)"),
    ])
    assert out["totals"].get("routine_denied", 0) == 0
    assert out["totals"].get("architectural_served", 0) == 0
