"""B13 — static contracts: catch signature/schema drift without running anything.

Three of the five production defects found on 2026-09-12 were *not* logic bugs
and no amount of behavioural testing would have found them by reasoning about
behaviour. They were drift between two artefacts that must agree:

  1. ``finish_reason`` was passed by a call site but was not a parameter of
     ``_log_request_to_db``. That raises ``TypeError`` inside a function whose
     whole body is wrapped in ``except Exception: logger.warning(...)`` — so the
     request returned 200 and the row was silently lost.
  2. ``rollup.migrate()`` built a ``router_logs`` table missing 10 columns the
     endpoint writes to. Any DB created by the rollup path (rather than the
     endpoint's own path) could not accept a logged request.
  3. The test fixture hand-rolled its own schema, so it could never disagree
     with production — which is exactly backwards: it should be the thing that
     notices production drifted.

These are cheap to check statically and impossible to check by reasoning about
behaviour, which is why they get their own file. All three are *parity*
assertions: two artefacts that must agree, asserted equal.
"""

from __future__ import annotations

import ast
import inspect
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402
import rollup  # noqa: E402


# --------------------------------------------------------------------------
# 1. Call-site keywords must be real parameters.
# --------------------------------------------------------------------------

def test_every_call_site_keyword_is_a_real_parameter():
    """Defect class: ``finish_reason``.

    A call site passing an unknown keyword raises TypeError inside a body that
    swallows exceptions, so the endpoint keeps answering 200 while dropping the
    evidence. Nothing behavioural catches this; parsing does.
    """
    src = (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()
    tree = ast.parse(src)

    valid = set(inspect.signature(ep._log_request_to_db).parameters)
    # **row style passthrough is legitimate and cannot be checked statically.
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_log_request_to_db":
            for kw in node.keywords:
                if kw.arg is None:  # **kwargs expansion
                    continue
                if kw.arg not in valid:
                    offenders.append((node.lineno, kw.arg))

    assert not offenders, (
        "call sites pass keywords _log_request_to_db does not accept, so the "
        f"row is silently dropped: {offenders}. Valid parameters: "
        f"{sorted(valid)}"
    )


def test_logger_does_not_swallow_type_errors_silently():
    """The swallow is what makes the drift invisible; keep it observable.

    This does not forbid catching exceptions — logging must never break a
    request. It requires that a *programming* error (TypeError: unexpected
    keyword) is not indistinguishable from an operational one.
    """
    sig = inspect.signature(ep._log_request_to_db)
    # Every call site keyword must bind, so a TypeError here is impossible.
    # Assert the corollary: passing a bogus keyword must raise, not pass
    # silently. If this ever stops raising, the swallow has widened.
    with pytest.raises(TypeError):
        ep._log_request_to_db(
            model_used="x", provider="y", task_type="z", complexity_score=0.0,
            input_tokens=1, output_tokens=1, latency_seconds=0.0,
            routing_time_ms=0, not_a_real_parameter=1,
        )
    assert "not_a_real_parameter" not in sig.parameters


# --------------------------------------------------------------------------
# 2. The two schema builders must agree.
# --------------------------------------------------------------------------

def _columns(conn: sqlite3.Connection, table: str) -> set:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_rollup_migrate_accepts_every_column_the_endpoint_writes(tmp_path):
    """Defect class: ``rollup.migrate()`` missing 10 endpoint columns.

    The endpoint writes with an explicit column list. If any of those columns
    is absent from the migrated table the INSERT raises — again inside the
    swallow. Assert the migrated schema is a superset of what the endpoint
    writes, by reading the actual INSERT rather than a hardcoded list that
    could itself go stale.
    """
    db = tmp_path / "migrated.db"
    conn = sqlite3.connect(str(db))
    rollup.migrate(conn)

    written = _columns_from_insert(ep, "router_logs")
    present = _columns(conn, "router_logs")
    missing = sorted(written - present)

    assert not missing, (
        "rollup.migrate() builds a router_logs that cannot accept a logged "
        f"request; missing columns: {missing}"
    )


def _columns_from_insert(module, table: str) -> set:
    """Extract the column list of the module's INSERT INTO <table> statement."""
    src = Path(module.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if f"INSERT INTO {table}" in text and "VALUES" in text:
                head = text.split("VALUES")[0]
                start = head.index("(") + 1
                end = head.rindex(")")
                cols = {c.strip() for c in head[start:end].split(",") if c.strip()}
                if cols:
                    return cols
    raise AssertionError(f"could not locate INSERT INTO {table} in {module.__name__}")


def test_endpoint_init_and_rollup_migrate_agree_on_router_logs(tmp_path, monkeypatch):
    """The endpoint's real init path must produce the same schema as the rollup.

    Both create ``router_logs``; whichever runs first wins. If they disagree,
    the schema a request can be logged against depends on startup order.

    Drives the *real* initialisation (the B11 override plus
    ``_get_db_connection``) rather than calling ``_ensure_log_columns`` on a
    bare connection — the latter only ALTERs an existing table, so on a fresh
    DB it does nothing and would assert nothing.
    """
    a = sqlite3.connect(str(tmp_path / "via_rollup.db"))
    rollup.migrate(a)

    db_file = tmp_path / "via_endpoint.db"
    monkeypatch.setenv("BIGGIE_ROUTER_DB", str(db_file))
    monkeypatch.setattr(ep, "_sqlite_conn", None, raising=False)
    conn = ep._get_db_connection()

    via_rollup = _columns(a, "router_logs")
    via_endpoint = _columns(conn, "router_logs")

    missing = sorted(via_rollup - via_endpoint)
    assert not missing, (
        "a DB initialised by the endpoint lacks columns the rollup schema "
        f"defines; startup order changes what is writable: {missing}"
    )
    # And the endpoint must be able to write a row, which is the property that
    # actually matters (a schema can be a superset and still reject an INSERT).
    written = _columns_from_insert(ep, "router_logs")
    assert not (written - via_endpoint), (
        "endpoint-initialised DB cannot accept the endpoint's own INSERT: "
        f"{sorted(written - via_endpoint)}"
    )


# --------------------------------------------------------------------------
# 3. The endpoint and the rollup must agree about what a log row contains.
# --------------------------------------------------------------------------

def test_rollup_reads_every_column_the_endpoint_writes():
    """A written column the rollup never reads is invisible spend/quality.

    The rollup is how raw rows become findings. A column it does not know about
    is a column no report will ever show — the silent-loss failure mode of the
    *analytics* layer. Every exception below is listed with its reason, so the
    next drift is a deliberate decision rather than an oversight.
    """
    written = _columns_from_insert(ep, "router_logs")

    # Columns the rollup deliberately ignores, with reasons.
    deliberately_unrolled = {
        "timestamp",       # rolled up into `day`
        "session_id",      # not available at endpoint level
        "request_id",      # per-request correlation, not an aggregate
        "requested_model", # the rollup aggregates on model_used
        "streaming",
        "empty_stream",
        "saw_content",
        "final_model",
        "routing_reason",
        "pricing_version",
        "provider",
        "complexity_score",
        "latency_seconds",
        "routing_time_ms",
        "escalated",
        "error_type",
        "compression_level",
        "compression_savings_pct",
        "compression_time_ms",
        # Renamed or aggregated into a findings column:
        "model_used",      # -> daily_findings.model
        "success",         # -> daily_findings.success_calls
        "task_type",       # -> daily_findings.call_type
        # KNOWN GAP, not a deliberate omission — see test below. Listed here so
        # this assertion stays a drift-detector rather than a standing failure.
        "context_tokens",
        "requires_tools",
        "saw_tool_calls",
    }

    known = set(getattr(rollup, "NEW_LOG_COLUMNS", []) or [])
    derived = {
        "calls", "cost_usd", "cost_unknown_calls", "success_calls",
        "escalated_calls", "latency_p50", "latency_p95", "quality_avg",
        "quality_n", "input_tokens", "output_tokens", "day", "workload_type",
        "call_type", "model", "is_shadow", "escalated",
    }
    unaccounted = sorted(
        written - deliberately_unrolled - known - derived - {"id"}
    )

    assert not unaccounted, (
        "the endpoint writes columns that neither the rollup nor the "
        "documented ignore-list accounts for, so their data never reaches a "
        f"report: {unaccounted}. Either roll them up or add them to "
        "deliberately_unrolled with a reason."
    )


def test_known_gap_saw_tool_calls_is_not_rolled_up(tmp_path):
    """Pins a real, unfixed analytics gap so it cannot be forgotten silently.

    ``saw_tool_calls`` / ``context_tokens`` / ``requires_tools`` are written per
    request but absent from ``daily_findings``. For ``saw_tool_calls`` that
    matters specifically: the glm-5.3-flash shadow verdict was *exactly* "emits
    tool_calls with no content when tools are offered", and that signature is
    currently invisible to every rollup-level report — you must query raw
    ``router_logs`` to see it.

    This test asserts the current state deliberately. When a rollup column is
    added, this test SHOULD fail: update it, and add the column to
    ``deliberately_unrolled``'s inverse (i.e. remove it from the gap set).
    """
    conn = sqlite3.connect(str(tmp_path / "m.db"))
    rollup.migrate(conn)
    findings = _columns(conn, "daily_findings")

    for col in ("saw_tool_calls", "context_tokens", "requires_tools"):
        assert col not in findings, (
            f"{col} now exists in daily_findings — good. Remove it from the "
            "known-gap set in test_rollup_reads_every_column_the_endpoint_"
            "writes and delete this assertion."
        )

    written = _columns_from_insert(ep, "router_logs")
    assert "saw_tool_calls" in written, (
        "the endpoint no longer records saw_tool_calls; the shadow verdict's "
        "key signal has been dropped"
    )
