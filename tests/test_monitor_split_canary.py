"""Tests for the split-canary monitor in the post-flip (no-control) era.

B36d flipped ``v41flash-tool-canary`` to percent=100, so there is no live
control arm. The monitor must still answer "is flash behaving like the pro
baseline it replaced?" by comparing against a pre-flip pro baseline scoped to
the same eligible cohort, and must not error when no control rows exist.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import monitor_split_canary as mon  # noqa: E402

CANARY = "v41flash-tool-canary"


def _make_db(tmp_path: Path) -> str:
    """Minimal router_logs shaped like production for the columns we read."""
    db = tmp_path / "router_logs.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE router_logs (
            timestamp TEXT,
            experiment TEXT,
            experiment_arm TEXT,
            is_shadow INTEGER,
            escalated INTEGER,
            error_type TEXT,
            success INTEGER,
            cost_usd REAL,
            cached_tokens INTEGER,
            input_tokens INTEGER,
            task_type TEXT,
            workload_type TEXT,
            requires_tools INTEGER,
            final_model TEXT,
            model_used TEXT,
            session_id TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    return str(db)


def _row(db: str, **kw) -> None:
    cols = {
        "timestamp": "2026-09-23T23:00:00+00:00",
        "experiment": "",
        "experiment_arm": "",
        "is_shadow": 0,
        "escalated": 0,
        "error_type": "",
        "success": 1,
        "cost_usd": 0.005,
        "cached_tokens": 0,
        "input_tokens": 1000,
        "task_type": "planning",
        "workload_type": "normal_chat",
        "requires_tools": 1,
        "final_model": "deepseek-v4.1-flash:cloud",
        "model_used": "deepseek-v4.1-flash:cloud",
        "session_id": "s1",
    }
    cols.update(kw)
    conn = sqlite3.connect(db)
    conn.execute(
        f"INSERT INTO router_logs ({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})",
        tuple(cols.values()),
    )
    conn.commit()
    conn.close()


def _collect(db: str, since: str = "24h") -> dict:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return mon._collect(conn, since)


# --- flash-only reporting -------------------------------------------------


def test_flash_only_window_reports_flash_health(tmp_path):
    db = _make_db(tmp_path)
    for _ in range(40):
        _row(db, experiment=CANARY, experiment_arm="treatment")

    stats = _collect(db)
    assert stats["treatment"]["requests"] == 40
    assert stats["arm_mode"] == "flash_only"
    assert stats["treatment"]["hard_failure_rate"] == 0.0


def test_arm_mode_is_split_when_control_rows_present(tmp_path):
    db = _make_db(tmp_path)
    _row(db, experiment=CANARY, experiment_arm="treatment")
    _row(db, experiment=CANARY, experiment_arm="control",
         final_model="deepseek-v4-pro")

    stats = _collect(db)
    assert stats["arm_mode"] == "split"


def test_no_canary_rows_returns_zero_exit_and_empty_stdout(tmp_path, capsys, monkeypatch):
    db = _make_db(tmp_path)
    monkeypatch.setenv("BIGGIE_ROUTER_DB", db)
    monkeypatch.setattr(sys, "argv", ["monitor_split_canary.py"])

    rc = mon.main()
    out = capsys.readouterr()
    assert rc == 0
    assert out.out == ""


# --- baseline -------------------------------------------------------------


def test_baseline_scoped_to_eligible_cohort_and_excludes_treatment(tmp_path):
    db = _make_db(tmp_path)
    # Eligible pro rows -> baseline.
    for _ in range(40):
        _row(db, experiment="", experiment_arm="", final_model="deepseek-v4-pro",
             model_used="deepseek-v4-pro", cost_usd=0.007)
    # Ineligible pro rows -> must NOT enter the baseline.
    for _ in range(40):
        _row(db, experiment="", experiment_arm="", final_model="deepseek-v4-pro",
             model_used="deepseek-v4-pro", task_type="debugging", cost_usd=0.5)
    # Canary treatment rows -> must NOT enter the baseline.
    for _ in range(40):
        _row(db, experiment=CANARY, experiment_arm="treatment", cost_usd=0.004)

    stats = _collect(db)
    base = stats["baseline"]
    assert base["requests"] == 40
    assert base["cost_per_completed"] == pytest.approx(0.007, rel=1e-6)


def test_baseline_reports_its_own_sample_size(tmp_path):
    db = _make_db(tmp_path)
    for _ in range(7):
        _row(db, final_model="deepseek-v4-pro", model_used="deepseek-v4-pro")
    for _ in range(5):
        _row(db, experiment=CANARY, experiment_arm="treatment")

    stats = _collect(db)
    assert stats["baseline"]["requests"] == 7


# --- guardrails -----------------------------------------------------------


def test_delta_guardrail_fires_when_escalation_exceeds_baseline(tmp_path):
    db = _make_db(tmp_path)
    # Baseline: 100 pro rows, 1 escalation (1%).
    for i in range(100):
        _row(db, final_model="deepseek-v4-pro", model_used="deepseek-v4-pro",
             escalated=1 if i == 0 else 0)
    # Flash: 40 rows, 12 escalated (30%) -> +29pp over baseline.
    for i in range(40):
        _row(db, experiment=CANARY, experiment_arm="treatment",
             escalated=1 if i < 12 else 0)

    stats = _collect(db)
    report = mon._report(stats)
    assert "escalation" in report.lower()
    assert "baseline" in report.lower()
    assert stats["treatment"]["escalation_rate"] == pytest.approx(0.30)


def test_delta_guardrail_silent_when_baseline_thin(tmp_path):
    db = _make_db(tmp_path)
    # Thin baseline (5 rows) -> delta comparison must be suppressed.
    for _ in range(5):
        _row(db, final_model="deepseek-v4-pro", model_used="deepseek-v4-pro")
    for i in range(40):
        _row(db, experiment=CANARY, experiment_arm="treatment",
             escalated=1 if i < 12 else 0)

    stats = _collect(db)
    flags = mon._guardrail_flags(stats)
    assert not any("baseline" in f.lower() for f in flags)


def test_absolute_guardrails_still_fire(tmp_path):
    db = _make_db(tmp_path)
    for _ in range(40):
        _row(db, experiment=CANARY, experiment_arm="treatment", escalated=1)

    stats = _collect(db)
    flags = mon._guardrail_flags(stats)
    assert any("escalation" in f.lower() for f in flags)


def test_no_flags_on_clean_flash_traffic(tmp_path):
    db = _make_db(tmp_path)
    for _ in range(100):
        _row(db, final_model="deepseek-v4-pro", model_used="deepseek-v4-pro")
    for _ in range(100):
        _row(db, experiment=CANARY, experiment_arm="treatment")

    flags = mon._guardrail_flags(_collect(db))
    assert flags == []


# --- json surface ---------------------------------------------------------


def test_json_output_carries_arm_mode_and_baseline(tmp_path, capsys, monkeypatch):
    db = _make_db(tmp_path)
    for _ in range(40):
        _row(db, final_model="deepseek-v4-pro", model_used="deepseek-v4-pro")
    for _ in range(40):
        _row(db, experiment=CANARY, experiment_arm="treatment")

    monkeypatch.setenv("BIGGIE_ROUTER_DB", db)
    monkeypatch.setattr(sys, "argv", ["monitor_split_canary.py", "--json"])
    rc = mon.main()
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["arm_mode"] == "flash_only"
    assert payload["baseline"]["requests"] == 40
