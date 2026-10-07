"""Batch B37j — the nightly digest line for plan-budget activity.

Why this module exists
----------------------
The governor can only be trusted over time if someone sees its behaviour without
going looking. The nightly lanes cron already runs and already reports routing
lane counts; this adds the plan-budget line to it: window used %, denials by
class, and architectural calls.

Two hard requirements, both about not making things worse:

* **A digest must never fail to produce a report.** The report is the operator's
  only routine view of the system, so a missing column (un-migrated DB), an
  unreadable probe, or an absent `gpt_budget` module must degrade to a note —
  never a traceback, and never a non-zero exit that reads as "the system broke".
* **Absent data must not read as zero.** "0 denials" and "no data" are opposite
  facts. Rendering the first when the second is true would tell the operator the
  governor is armed and healthy when it has never run.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sqlite3
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location(
    "check_routing_stats", SCRIPTS_DIR / "check-routing-stats.py"
)
crs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crs)

import gpt_budget as gb  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db(tmp_path, rows, *, with_governor_cols=True):
    """Build a minimal router_logs + sessions DB for the report."""
    state_db = str(tmp_path / "state.db")
    router_db = str(tmp_path / "router_logs.db")

    conn = sqlite3.connect(state_db)
    conn.executescript(
        """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at REAL, parent_session_id TEXT);
        CREATE TABLE session_model_usage (
            session_id TEXT, model TEXT, billing_provider TEXT, api_call_count INTEGER,
            input_tokens INTEGER, output_tokens INTEGER, first_seen REAL, last_seen REAL
        );
        """
    )
    conn.commit()
    conn.close()

    gov_cols = (
        ", budget_class TEXT NOT NULL DEFAULT '', plan_budget_denied INTEGER NOT NULL DEFAULT 0"
        ", governor_fail_open INTEGER NOT NULL DEFAULT 0"
        if with_governor_cols
        else ""
    )
    conn = sqlite3.connect(router_db)
    conn.execute(
        f"""
        CREATE TABLE router_logs (
            timestamp TEXT NOT NULL, session_id TEXT, model_used TEXT, provider TEXT,
            task_type TEXT, success INTEGER DEFAULT 1, escalated INTEGER DEFAULT 0,
            error_type TEXT, request_id TEXT, requested_model TEXT, streaming INTEGER DEFAULT 0,
            workload_type TEXT DEFAULT '', requires_tools INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
            final_model TEXT DEFAULT '', routing_reason TEXT DEFAULT ''
            {gov_cols}
        )
        """
    )
    for row in rows:
        cols = [
            "timestamp", "session_id", "model_used", "provider", "task_type",
            "success", "escalated", "request_id", "requested_model", "final_model",
            "routing_reason",
        ]
        if with_governor_cols:
            cols += ["budget_class", "plan_budget_denied", "governor_fail_open"]
        conn.execute(
            f"INSERT INTO router_logs ({', '.join(cols)})"
            f" VALUES ({','.join('?' * len(cols))})",
            row,
        )
    conn.commit()
    conn.close()
    return state_db, router_db


def _run_json(state_db, router_db, monkeypatch, reading=None):
    if reading is not None:
        monkeypatch.setattr(gb, "fetch_reading", lambda cfg=None, **kw: reading)
    buf = io.StringIO()
    with redirect_stdout(buf):
        crs.STATE_DB = state_db
        crs.ROUTER_LOGS = router_db
        sys.argv = ["check-routing-stats.py", "--json"]
        crs.main()
    sys.argv = ["check-routing-stats.py"]
    return json.loads(buf.getvalue())


def _reading(used_pct, allowed=True, source="live"):
    return gb.Reading(
        source=source,
        allowed=allowed,
        primary=gb.WindowReading(
            used_percent=used_pct, limit_window_seconds=18000, reset_after_seconds=165300
        ),
        secondary=gb.WindowReading(
            used_percent=used_pct, limit_window_seconds=604800, reset_after_seconds=165300
        ),
    )


def _governed_rows():
    return [
        (_now(), "s1", "deepseek-v4-pro", "cloud", "coding", 1, 0, "r1", "biggie-router",
         "deepseek-v4-pro", "tool-bearing request — plan account denied to class 'routine'",
         "routine", 1, 0),
        (_now(), "s1", "deepseek-v4-pro", "cloud", "coding", 1, 0, "r2", "biggie-router",
         "deepseek-v4-pro", "tool-bearing request — plan account denied to class 'routine'",
         "routine", 1, 0),
        (_now(), "s1", "gpt-5.6-sol", "openai-codex", "planning", 1, 0, "r3", "biggie-router",
         "gpt-5.6-sol", "capability tier 13 needed, selected gpt-5.6-sol (tier 13)",
         "architectural", 0, 0),
    ]


def _blind_rows():
    """Two routine rows served while the reading was unreadable (fail-open)."""
    return [
        (_now(), "s1", "gpt-5.5", "openai-codex", "coding", 1, 0, "r1", "biggie-router",
         "gpt-5.5", "tool-bearing request — plan governor UNVERIFIED (unreadable)",
         "routine", 0, 1),
        (_now(), "s1", "gpt-5.5", "openai-codex", "coding", 1, 0, "r2", "biggie-router",
         "gpt-5.5", "tool-bearing request — plan governor UNVERIFIED (unreadable)",
         "routine", 0, 1),
    ]


# ── The digest block ─────────────────────────────────────────────────────────

def test_json_report_carries_a_plan_budget_block(tmp_path, monkeypatch):
    state_db, router_db = _db(tmp_path, _governed_rows())
    report = _run_json(state_db, router_db, monkeypatch, _reading(41.2))
    pb = report["router_endpoint"]["plan_budget"]
    assert pb["routine_denied"] == 2, pb
    assert pb["architectural_served"] == 1, pb
    assert pb["window_used_percent"] == pytest.approx(41.2), pb


def test_plan_budget_reports_ungoverned_rows_separately(tmp_path, monkeypatch):
    rows = _governed_rows() + [
        (_now(), "s1", "deepseek-v4-flash", "cloud", "coding", 1, 0, "r4", "biggie-router",
         "deepseek-v4-flash", "capability tier 3 needed", "", 0, 0),
    ]
    state_db, router_db = _db(tmp_path, rows)
    pb = _run_json(state_db, router_db, monkeypatch, _reading(0.0))["router_endpoint"]["plan_budget"]
    assert pb["ungoverned"] == 1, pb
    assert pb["routine_denied"] == 2, pb


def test_missing_governor_columns_reports_unavailable_not_zero(tmp_path, monkeypatch):
    """An un-migrated DB must say so — 'unavailable', not a false zero."""
    state_db, router_db = _db(tmp_path, [], with_governor_cols=False)
    report = _run_json(state_db, router_db, monkeypatch, _reading(10.0))
    pb = report["router_endpoint"]["plan_budget"]
    assert pb["available"] is False, pb
    # Critically: no denial counts invented.
    assert pb.get("routine_denied") in (None, 0), pb


def test_unreadable_probe_does_not_break_the_report(tmp_path, monkeypatch):
    """A failed probe degrades to a note; the report must still render."""
    state_db, router_db = _db(tmp_path, _governed_rows())
    report = _run_json(
        state_db, router_db, monkeypatch,
        gb.Reading(source="unavailable", allowed=None, primary=None, secondary=None),
    )
    pb = report["router_endpoint"]["plan_budget"]
    assert pb["window_used_percent"] is None, pb
    assert pb["routine_denied"] == 2, pb


def test_fetch_reading_exception_does_not_break_the_report(tmp_path, monkeypatch):
    """gpt_budget raising must not abort the nightly report."""
    def _boom(cfg=None, **kw):
        raise RuntimeError("probe exploded")

    state_db, router_db = _db(tmp_path, _governed_rows())
    monkeypatch.setattr(gb, "fetch_reading", _boom)
    report = _run_json(state_db, router_db, monkeypatch)
    assert report["router_endpoint"]["plan_budget"]["routine_denied"] == 2


def test_text_report_prints_a_plan_budget_section(tmp_path, monkeypatch):
    state_db, router_db = _db(tmp_path, _governed_rows())
    monkeypatch.setattr(gb, "fetch_reading", lambda cfg=None, **kw: _reading(41.2))
    buf = io.StringIO()
    with redirect_stdout(buf):
        crs.STATE_DB = state_db
        crs.ROUTER_LOGS = router_db
        sys.argv = ["check-routing-stats.py"]
        crs.main()
    sys.argv = ["check-routing-stats.py"]
    out = buf.getvalue()
    assert "Plan Budget" in out, out
    assert "Routine denied" in out, out
    assert "Architectural calls served" in out, out
    assert "Window used" in out, out


def test_exit_code_stays_zero_with_no_governor_data(tmp_path, monkeypatch):
    """The nightly cron reads a crash as failure — it must not fire.

    `main()` returns None by design (it is a script, not a library entry point),
    so the contract the cron actually depends on is "no exception raised". A
    raised exception is what would surface as a non-zero exit.
    """
    state_db, router_db = _db(tmp_path, [], with_governor_cols=False)
    monkeypatch.setattr(gb, "fetch_reading", lambda cfg=None, **kw: _reading(0.0))
    buf = io.StringIO()
    with redirect_stdout(buf):
        crs.STATE_DB = state_db
        crs.ROUTER_LOGS = router_db
        sys.argv = ["check-routing-stats.py", "--json"]
        crs.main()  # must not raise
    sys.argv = ["check-routing-stats.py"]
    report = json.loads(buf.getvalue())
    assert report["router_endpoint"]["plan_budget"]["available"] is False


def test_digest_line_carries_the_three_required_facts(tmp_path, monkeypatch):
    """Design §3.6: window used %, denial count by class, architectural calls."""
    state_db, router_db = _db(tmp_path, _governed_rows())
    pb = _run_json(state_db, router_db, monkeypatch, _reading(41.2))["router_endpoint"]["plan_budget"]
    for key in ("window_used_percent", "routine_denied", "architectural_served"):
        assert key in pb, f"digest missing {key}: {pb}"


# ── B44d: a blind window must not read as "closed" ───────────────────────────

def test_json_report_carries_a_blind_count(tmp_path, monkeypatch):
    """The digest must report how many requests were served blind."""
    state_db, router_db = _db(tmp_path, _blind_rows())
    pb = _run_json(state_db, router_db, monkeypatch, _reading(41.2))["router_endpoint"]["plan_budget"]
    assert pb["blind"] == 2, pb
    # And those two rows are the failure mode that used to read as "closed":
    assert pb["routine_denied"] == 0, pb


def test_text_report_refuses_closure_on_blind(tmp_path, monkeypatch):
    """A blind window must not print 'leak closed'."""
    state_db, router_db = _db(tmp_path, _blind_rows())
    monkeypatch.setattr(gb, "fetch_reading", lambda cfg=None, **kw: _reading(41.2))
    buf = io.StringIO()
    with redirect_stdout(buf):
        crs.STATE_DB = state_db
        crs.ROUTER_LOGS = router_db
        sys.argv = ["check-routing-stats.py"]
        crs.main()
    sys.argv = ["check-routing-stats.py"]
    out = buf.getvalue()
    assert "leak closed" not in out, out
    assert "NOT provably closed" in out, out
    assert "Blind (governor unreadable)" in out, out


def test_text_report_claims_closure_when_clean(tmp_path, monkeypatch):
    """A clean window still says 'leak closed' — the honest case."""
    state_db, router_db = _db(tmp_path, _governed_rows())
    monkeypatch.setattr(gb, "fetch_reading", lambda cfg=None, **kw: _reading(41.2))
    buf = io.StringIO()
    with redirect_stdout(buf):
        crs.STATE_DB = state_db
        crs.ROUTER_LOGS = router_db
        sys.argv = ["check-routing-stats.py"]
        crs.main()
    sys.argv = ["check-routing-stats.py"]
    out = buf.getvalue()
    assert "Routine denied (leak closed)" in out, out
    assert "NOT provably closed" not in out, out
