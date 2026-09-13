"""Writer cleanup B23.1 — one row per streaming request.

The marker row is deliberate crash-recovery design: a streaming request writes a
``streaming_in_progress`` start row so that an abandoned stream is still visible
(``_find_abandoned_streams`` depends on it), then writes a completion row with
the same ``request_id``.

That design leaves TWO rows per completed request. Measured on production before
this change:

    router_logs rows                242,964
    marker rows                     118,326   (48.7% of the table)
    completion rows                 123,607
    markers with a completed twin   ~99.x%
    genuinely abandoned (marker-only)  289

Every billable aggregation already excludes the marker via
``rollup.BILLABLE_ROW_SQL``, so the duplication was *paid for* in every consumer
that would otherwise have to remember the filter. The generator, not the sums,
is the defect.

The fix is **delete-on-completion**: when a non-marker row is written for a
request whose marker is present, the marker is removed in the same transaction.
The surviving row is byte-identical to today's completion row, so every existing
consumer sees exactly what it saw before — there is simply one row instead of two.

What this file pins:

  1. a completed streaming request leaves exactly ONE row;
  2. the survivor is the billed completion, not the unpriced marker;
  3. an ABANDONED stream's marker survives (the crash-recovery property);
  4. a completion with an EMPTY request_id deletes nothing — 170 production rows
     share an empty request_id, and a naive ``DELETE ... WHERE request_id = ''``
     would wipe every unpaired marker at once;
  5. the canonical billable sum over one streaming request counts it once.
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


@pytest.fixture()
def fresh_db(monkeypatch, tmp_path):
    monkeypatch.setenv("BIGGIE_ROUTER_DB", str(tmp_path / "ledger.db"))
    monkeypatch.setattr(ep, "_sqlite_conn", None, raising=False)
    conn = ep._get_db_connection()
    yield conn
    monkeypatch.setattr(ep, "_sqlite_conn", None, raising=False)


def _basic(**over):
    kw = dict(
        model_used="deepseek-v4.1-flash",
        provider="ollama-cloud",
        task_type="chat",
        complexity_score=0.5,
        input_tokens=1_000_000,
        output_tokens=1_000,
        latency_seconds=1.0,
        routing_time_ms=5,
        workload_type="session_compression",
        request_id="rid-1",
    )
    kw.update(over)
    return kw


def _rows(conn, request_id):
    return conn.execute(
        "SELECT error_type, success, cost_usd, output_tokens FROM router_logs "
        "WHERE request_id = ? ORDER BY id",
        (request_id,),
    ).fetchall()


# --------------------------------------------------------------------------
# 1-2. One row per completed streaming request, and it is the billed one
# --------------------------------------------------------------------------

def test_completed_streaming_request_leaves_one_row(fresh_db):
    """The marker must not survive its own completion row."""
    ep._log_request_to_db(
        **_basic(request_id="s1", streaming=True, success=False,
                 error_type="streaming_in_progress")
    )
    assert len(_rows(fresh_db, "s1")) == 1, "marker should exist before completion"

    ep._log_request_to_db(**_basic(request_id="s1", streaming=True, success=True))

    rows = _rows(fresh_db, "s1")
    assert len(rows) == 1, (
        f"a completed streaming request left {len(rows)} rows; the marker was not "
        f"merged away: {rows!r}"
    )


def test_surviving_row_is_the_billed_completion_not_the_marker(fresh_db):
    """Merging must keep the priced completion, never the unpriced marker.

    Deleting the wrong row would lose the cost *and* the outcome — strictly
    worse than the duplication being fixed.
    """
    ep._log_request_to_db(
        **_basic(request_id="s2", streaming=True, success=False,
                 error_type="streaming_in_progress")
    )
    ep._log_request_to_db(**_basic(request_id="s2", streaming=True, success=True))

    (error_type, success, cost, out_tok), = _rows(fresh_db, "s2")
    assert error_type == "", (
        f"the surviving row is still the start marker (error_type={error_type!r}); "
        f"the completion row was deleted instead of the marker"
    )
    assert success == 1, "the surviving row must be the successful completion"
    assert cost > 0, (
        "the surviving row carries no cost — the billed row was deleted and the "
        "unpriced marker kept"
    )


# --------------------------------------------------------------------------
# 3. Crash-recovery property: an ABANDONED stream stays visible
# --------------------------------------------------------------------------

def test_abandoned_stream_marker_survives(fresh_db):
    """Only a marker *with* a completion may be removed.

    This is the property the marker exists for. If dedup deleted markers
    unconditionally, a crashed stream would vanish from observability entirely.
    """
    ep._log_request_to_db(
        **_basic(request_id="stuck-1", streaming=True, success=False,
                 error_type="streaming_in_progress")
    )

    assert len(_rows(fresh_db, "stuck-1")) == 1, "the abandoned marker was deleted"

    abandoned = ep._find_abandoned_streams(fresh_db, older_than_seconds=-1)
    assert "stuck-1" in abandoned, (
        "abandoned-stream detection broke: the marker was deleted rather than "
        "only superseded on successful completion"
    )


# --------------------------------------------------------------------------
# 4. The empty-request_id trap
# --------------------------------------------------------------------------

def test_completion_with_empty_request_id_deletes_nothing(fresh_db):
    """An empty request_id is shared by unrelated rows — never a delete key.

    Production has rows with ``request_id = ''``. A delete keyed on equality
    would match every one of them and wipe unrelated in-flight markers.
    """
    ep._log_request_to_db(
        **_basic(request_id="", streaming=True, success=False,
                 error_type="streaming_in_progress")
    )
    ep._log_request_to_db(
        **_basic(request_id="", streaming=True, success=False,
                 error_type="streaming_in_progress")
    )
    before = fresh_db.execute(
        "SELECT COUNT(*) FROM router_logs WHERE request_id = ''"
    ).fetchone()[0]
    assert before == 2

    ep._log_request_to_db(**_basic(request_id="", streaming=True, success=True))

    after = fresh_db.execute(
        "SELECT COUNT(*) FROM router_logs WHERE request_id = ''"
    ).fetchone()[0]
    assert after == 3, (
        f"a completion with an empty request_id deleted unrelated rows "
        f"({before} -> {after}); only the matching marker may ever be removed"
    )


# --------------------------------------------------------------------------
# 5. The canonical sum still counts one streaming request once
# --------------------------------------------------------------------------

def test_canonical_billable_sum_counts_one_streaming_request_once(fresh_db):
    """Dedup must not change what a billable sum reports."""
    ep._log_request_to_db(
        **_basic(request_id="s5", streaming=True, success=False,
                 error_type="streaming_in_progress")
    )
    ep._log_request_to_db(**_basic(request_id="s5", streaming=True, success=True))

    total, n = fresh_db.execute(
        "SELECT COALESCE(SUM(cost_usd), 0), COUNT(*) FROM router_logs WHERE "
        + rollup.BILLABLE_ROW_SQL
    ).fetchone()
    assert n == 1, f"expected one billable row, found {n}"
    assert total > 0


def test_non_streaming_row_is_untouched(fresh_db):
    """The dedup is scoped to superseding a marker; ordinary rows are unaffected."""
    ep._log_request_to_db(**_basic(request_id="plain-1"))
    ep._log_request_to_db(**_basic(request_id="plain-1"))

    n = fresh_db.execute(
        "SELECT COUNT(*) FROM router_logs WHERE request_id = 'plain-1'"
    ).fetchone()[0]
    assert n == 2, (
        "two non-streaming rows share a request_id and both must survive — "
        "there is no marker to supersede and deleting either would lose data"
    )
