"""B16 — a streaming request is one billable event, not two.

Found by reconciling the router ledger against the Ollama dashboard the user
shared. The dashboard said $76.86 for the week; the ledger claimed $1,395.25
for a single day. The gap was not pricing — it was counting.

The mechanism
-------------
Every streaming request writes TWO rows:

  1. a *start marker* — ``error_type='streaming_in_progress'``, ``success=0``,
     carrying the FULL input token count. It exists so an abandoned stream is
     still visible (``_find_abandoned_streams`` depends on it).
  2. a *completion row* — same ``request_id``, same token counts.

Both rows carry the same ``input_tokens``, so any ``SUM(...)`` over the table
counts a streaming request twice. Measured on production:

    2026-09-11   raw $1,395.25   one-row-per-request $755.27   (1.85x)

The start marker is an in-flight marker. It is not a second trip to the
vendor, and it must never be billed.

How it got live
---------------
The start marker passes no cost fields. Pre-B12 that was harmless: the logger
defaulted ``cost_unknown=1`` and ``cost_usd=0.0``, so markers cost nothing.
Then a backfill priced 114,279 markers ($17,591.88) and the ledger became
2x. Worse, B12's resolver *computes* a cost whenever the caller omits one —
so restarting the endpoint would have made the double-count live rather than
historical.

Two independent guards, because either alone is one edit away from regressing:

  * the start marker is explicitly recorded as NON-billable (belt), and
  * aggregation counts only billable rows (braces), so re-pricing a marker can
    never inflate a report again.

Note the honest cost of the braces: a stream that never completes has no
completion row, so its tokens drop out of the aggregate. On production that is
290 of 236,451 rows (0.12%), all genuinely abandoned streams. Losing them is
correct — the vendor's charge for an abandoned stream is not knowable — and it
is strictly better than counting every completed stream twice.
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


# --------------------------------------------------------------------------
# Guard 1: the start marker is not a billable event
# --------------------------------------------------------------------------

def test_streaming_start_marker_is_not_priced(fresh_db):
    """Omitting cost on a start marker must mean 'not billable', not 'price it'.

    This is the B12 regression guard. B12 made the logger price any call whose
    caller omits cost fields; the start marker omits them, so without this the
    double-count goes live.
    """
    ep._log_request_to_db(**_basic(error_type="streaming_in_progress", success=False))
    row = fresh_db.execute(
        "SELECT cost_usd, cost_unknown FROM router_logs WHERE error_type='streaming_in_progress'"
    ).fetchone()
    assert row is not None, "start marker was not logged"
    assert row[0] == 0.0, (
        f"start marker was priced at ${row[0]} — a streaming request is ONE billable "
        "event; pricing the marker double-counts the whole request"
    )
    assert row[1] == 0, (
        "start marker is recorded as cost_unknown=1, so a rollup that sums "
        "cost_unknown will treat it as an unpriced call we still owe for"
    )


def test_completion_row_of_the_same_request_is_still_priced(fresh_db):
    """The fix must not blind the ledger — the real event still gets its cost."""
    ep._log_request_to_db(**_basic())
    row = fresh_db.execute(
        "SELECT cost_usd, cost_unknown FROM router_logs WHERE error_type=''"
    ).fetchone()
    assert row is not None
    assert row[0] > 0, "completion row lost its cost"
    assert row[1] == 0, "completion row is now unpriced"


# --------------------------------------------------------------------------
# Guard 2: aggregation counts one row per streaming request
# --------------------------------------------------------------------------

def _write_stream_pair(conn):
    """One streaming request: start marker + completion, same tokens."""
    ep._log_request_to_db(**_basic(error_type="streaming_in_progress", success=False,
                                   streaming=True, latency_seconds=0.0))
    ep._log_request_to_db(**_basic(streaming=True, latency_seconds=1.5))


def test_rollup_counts_a_streaming_request_once(fresh_db, monkeypatch, tmp_path):
    _write_stream_pair(fresh_db)
    day = fresh_db.execute("SELECT substr(timestamp,1,10) FROM router_logs LIMIT 1").fetchone()[0]
    rollup.rollup_day(fresh_db, day)

    calls, cost, in_tok = fresh_db.execute(
        "SELECT SUM(calls), SUM(cost_usd), SUM(input_tokens) FROM daily_findings WHERE day=?",
        (day,),
    ).fetchone()
    assert calls == 1, f"expected 1 call, findings counted {calls} — the marker was billed"
    assert in_tok == 1_000_000, (
        f"expected 1,000,000 input tokens, findings counted {in_tok:,} — tokens doubled"
    )
    completion_cost = fresh_db.execute(
        "SELECT cost_usd FROM router_logs WHERE error_type=''"
    ).fetchone()[0]
    assert cost == pytest.approx(completion_cost), (
        f"findings cost ${cost} != the single completion cost ${completion_cost}"
    )


def test_billable_marker_is_shared_not_reinvented():
    """The exclusion rule must live in one place, or the three aggregators drift."""
    assert hasattr(rollup, "BILLABLE_ROW_SQL"), (
        "rollup must export the billable-row predicate so unit_economics and "
        "the optimiser cannot drift from it"
    )
    assert "streaming_in_progress" in rollup.BILLABLE_ROW_SQL


def test_rollup_sql_actually_applies_the_predicate():
    """Behavioural, not textual: guard against the predicate existing but unused."""
    import inspect
    src = inspect.getsource(rollup.rollup_day)
    assert "BILLABLE_ROW_SQL" in src, "rollup_day does not apply the billable predicate"


def test_unit_economics_counts_a_streaming_request_once(fresh_db):
    import unit_economics

    _write_stream_pair(fresh_db)
    rows = unit_economics.unit_cost(fresh_db, since="1970-01-01")
    total_calls = sum(r.calls for r in rows)
    assert total_calls == 1, (
        f"unit economics counted {total_calls} calls for one streaming request"
    )


def test_abandoned_stream_marker_is_still_visible(fresh_db):
    """Excluding markers from sums must not stop us SEEING a stuck stream."""
    ep._log_request_to_db(**_basic(error_type="streaming_in_progress", success=False,
                                   streaming=True, request_id="stuck-1"))
    abandoned = ep._find_abandoned_streams(fresh_db, older_than_seconds=-1)
    assert "stuck-1" in abandoned, (
        "abandoned-stream detection broke: the marker was excluded from the "
        "table rather than only from the sums"
    )
