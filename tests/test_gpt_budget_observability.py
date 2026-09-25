"""Batch B37i — observability: columns, events, `router_ops plan-budget`.

Why this module exists
----------------------
The governor's justification is that it removes a *measurable* leak. A routing
reason string is prose and cannot be aggregated, so the refusal and the
classification must be real columns: one to count what was stopped, one to prove
the architectural lane still gets served.

Two failure modes these tests pin:

1. **A silent migration.** `rollup.migrate()` and the endpoint's own startup
   migration must agree, or a column exists in one process and not the other and
   the insert path fails at runtime — which, on the live endpoint, means routing
   logs stop recording.
2. **A report that cannot distinguish "no governor" from "governor allowed it".**
   Historical rows have ``budget_class = ''``. If those read as class
   ``routine``/``architectural`` the operator sees a fabricated policy history.
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import rollup  # noqa: E402
import router_ops  # noqa: E402
import gpt_budget as gb  # noqa: E402


# ── Schema ───────────────────────────────────────────────────────────────────

def test_rollup_declares_both_governor_columns():
    """The authoritative schema source must carry both columns."""
    assert "plan_budget_denied" in rollup.NEW_LOG_COLUMNS
    assert "budget_class" in rollup.NEW_LOG_COLUMNS
    # Ungoverned defaults: historical rows must never read as a denial.
    assert "DEFAULT 0" in rollup.NEW_LOG_COLUMNS["plan_budget_denied"]
    assert "DEFAULT ''" in rollup.NEW_LOG_COLUMNS["budget_class"]


def test_endpoint_declares_the_same_governor_columns():
    """The endpoint's own migration must not drift from rollup's."""
    src = (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()
    for col in ("plan_budget_denied", "budget_class"):
        assert f'"{col}"' in src, f"endpoint migration missing {col}"


def test_migrate_is_idempotent_and_adds_the_columns(tmp_path):
    """A fresh DB must come out of migrate() with both columns."""
    db = str(tmp_path / "logs.db")
    conn = sqlite3.connect(db)
    rollup.migrate(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(router_logs)")}
    assert "plan_budget_denied" in cols
    assert "budget_class" in cols
    # Idempotent: a second run must not raise.
    rollup.migrate(conn)
    conn.close()


def test_migrated_rows_default_to_ungoverned_not_denied(tmp_path):
    """A row written before the governor must read as ungoverned, not denied.

    Simulates the real upgrade path: rows exist first, the columns arrive
    later. The old row must not appear in any denial count — reading it as a
    refusal would fabricate a policy history that never happened.
    """
    db = str(tmp_path / "logs.db")
    conn = sqlite3.connect(db)
    rollup.migrate(conn)
    # Roll the schema BACK to genuine pre-governor shape: drop the two columns,
    # then write a row as the old code would have. This is the real upgrade
    # path rather than a simulation of it.
    conn.execute("ALTER TABLE router_logs DROP COLUMN plan_budget_denied")
    conn.execute("ALTER TABLE router_logs DROP COLUMN budget_class")
    conn.execute(
        "INSERT INTO router_logs (timestamp, model_used, routing_reason) VALUES (?, ?, ?)",
        ("2026-01-01T00:00:00+00:00", "deepseek-v4-pro", "capability tier 8 needed"),
    )
    conn.commit()
    conn.close()

    # Re-open and migrate, as the endpoint restart would.
    conn = sqlite3.connect(db)
    rollup.migrate(conn)
    row = conn.execute(
        "SELECT plan_budget_denied, budget_class FROM router_logs"
    ).fetchone()
    assert row == (0, ""), row
    conn.close()


def test_insert_path_writes_the_governor_fields():
    """The insert must actually bind both columns, not just declare them."""
    src = (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()
    assert "plan_budget_denied, budget_class" in src, "column list not extended"
    assert "1 if plan_budget_denied else 0" in src, "value not bound"
    assert "budget_class or \"\"" in src, "value not bound"


def test_decision_fields_are_bound_at_the_escalation_call_site():
    """Escalation rows must carry the inherited class too."""
    src = (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()
    assert "plan_budget_denied=decision.plan_budget_denied" in src
    assert "plan_budget_denied=escalation.plan_budget_denied" in src


# ── router_ops plan-budget ───────────────────────────────────────────────────

def _db_with_governor_rows(rows) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE router_logs (
            timestamp TEXT, input_tokens INTEGER, output_tokens INTEGER,
            budget_class TEXT, plan_budget_denied INTEGER
        )
        """
    )
    conn.executemany(
        "INSERT INTO router_logs (timestamp, input_tokens, output_tokens, budget_class, plan_budget_denied)"
        " VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_plan_budget_report_counts_denials_and_served_classes():
    conn = _db_with_governor_rows([
        (_now(), 100, 50, "routine", 1),
        (_now(), 100, 50, "routine", 1),
        (_now(), 200, 100, "architectural", 0),
    ])
    out = router_ops.format_plan_budget(conn, days=7)
    assert "routine denied (leak closed): 2" in out
    assert "architectural served: 1" in out
    conn.close()


def test_plan_budget_report_separates_ungoverned_from_governed():
    """Blank class = pre-governor row: must not be counted as a class."""
    conn = _db_with_governor_rows([
        (_now(), 100, 50, "", 0),
        (_now(), 100, 50, "routine", 1),
    ])
    out = router_ops.format_plan_budget(conn, days=7)
    assert "ungoverned" in out, out
    assert "routine denied (leak closed): 1" in out
    conn.close()


def test_plan_budget_warns_when_architecture_is_starved():
    """The operator's core fear: routine denied AND nothing architectural served."""
    conn = _db_with_governor_rows([(_now(), 100, 50, "routine", 1)])
    out = router_ops.format_plan_budget(conn, days=7)
    assert "architectural lane served 0 calls" in out, out
    conn.close()


def test_plan_budget_helpful_when_columns_absent():
    """An un-migrated DB must explain itself, not traceback."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE router_logs (timestamp TEXT)")
    out = router_ops.format_plan_budget(conn, days=7)
    assert "not present yet" in out, out
    conn.close()


def test_plan_budget_json_mode():
    conn = _db_with_governor_rows([
        (_now(), 100, 50, "routine", 1),
        (_now(), 200, 100, "architectural", 0),
    ])
    buf = io.StringIO()
    args = type("A", (), {"json": True, "days": 7})()
    with redirect_stdout(buf):
        rc = router_ops.cmd_plan_budget(conn, args)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["available"] is True
    classes = {b["class"]: b for b in payload["buckets"]}
    assert classes["routine"]["denied"] is True
    assert classes["routine"]["calls"] == 1
    assert classes["architectural"]["denied"] is False
    conn.close()


def test_plan_budget_is_registered_as_a_subcommand():
    src = (SCRIPTS_DIR / "router_ops.py").read_text()
    assert 'sub.add_parser("plan-budget"' in src
    assert '"plan-budget": cmd_plan_budget' in src


def test_plan_budget_ignores_rows_outside_the_window():
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    conn = _db_with_governor_rows([
        (old, 100, 50, "routine", 1),
        (_now(), 100, 50, "routine", 1),
    ])
    out = router_ops.format_plan_budget(conn, days=7)
    assert "routine denied (leak closed): 1" in out, out
    conn.close()
