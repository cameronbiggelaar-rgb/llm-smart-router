"""B8 — cost backfill for pre-instrumentation rows.

231k historical rows were logged before cost capture existed, so they carry
``cost_unknown=1`` and ``cost_usd=0``. Their model names and token counts
survive, so the spend is recoverable — but a *reconstructed* figure must never
be presented as a measured one, and the backfill must be idempotent and
honest about what it cannot know.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from rollup import migrate  # noqa: E402
from unit_economics import backfill_costs, seed_prices  # noqa: E402


@pytest.fixture()
def conn(tmp_path):
    c = sqlite3.connect(str(tmp_path / "router_logs.db"))
    migrate(c)
    seed_prices(c)
    return c


def _row(conn, model, in_tok, out_tok, cost_usd=0.0, cost_unknown=1):
    conn.execute(
        "INSERT INTO router_logs (timestamp, session_id, model_used, input_tokens, "
        "output_tokens, cost_usd, cost_unknown, workload_type) "
        "VALUES ('2026-09-01T10:00:00+00:00','s',?,?,?,?,?,'session_compression')",
        (model, in_tok, out_tok, cost_usd, cost_unknown),
    )
    conn.commit()


def test_backfill_prices_unknown_rows_from_token_counts(conn):
    # 1M input @ $1.50 (glm-5.3) = $1.50
    _row(conn, "glm-5.3", 1_000_000, 0)
    res = backfill_costs(conn)
    assert res["updated"] == 1
    cost = conn.execute("SELECT cost_usd FROM router_logs").fetchone()[0]
    assert cost == pytest.approx(1.50)


def test_backfill_marks_rows_as_reconstructed_not_measured(conn):
    """A reconstructed cost must be distinguishable from a measured one."""
    _row(conn, "glm-5.3", 1_000_000, 0)
    backfill_costs(conn)
    pv = conn.execute("SELECT pricing_version FROM router_logs").fetchone()[0]
    assert pv.startswith("backfill")


def test_backfill_is_idempotent(conn):
    _row(conn, "glm-5.3", 1_000_000, 0)
    first = backfill_costs(conn)
    second = backfill_costs(conn)
    assert first["updated"] == 1
    assert second["updated"] == 0        # nothing left to reconstruct
    cost = conn.execute("SELECT cost_usd FROM router_logs").fetchone()[0]
    assert cost == pytest.approx(1.50)   # not doubled


def test_backfill_never_touches_already_priced_rows(conn):
    _row(conn, "glm-5.3", 1_000_000, 0, cost_usd=99.0, cost_unknown=0)
    res = backfill_costs(conn)
    assert res["updated"] == 0
    assert conn.execute("SELECT cost_usd FROM router_logs").fetchone()[0] == 99.0


def test_backfill_leaves_unpriced_models_alone(conn):
    """An unknown model stays unknown — backfill must not invent a price."""
    _row(conn, "never-heard-of-it", 1_000_000, 0)
    res = backfill_costs(conn)
    assert res["updated"] == 0
    assert res["skipped_unpriced"] == 1
    row = conn.execute("SELECT cost_usd, cost_unknown FROM router_logs").fetchone()
    assert row[0] == 0.0
    assert row[1] == 1                   # still flagged unknown


def test_backfill_accepts_provider_suffixed_model_names(conn):
    """Production logs 'glm-5.3:cloud'; pricing is keyed on the bare name."""
    _row(conn, "glm-5.3:cloud", 1_000_000, 0)
    res = backfill_costs(conn)
    assert res["updated"] == 1


def test_backfill_respects_dry_run(conn):
    _row(conn, "glm-5.3", 1_000_000, 0)
    res = backfill_costs(conn, dry_run=True)
    assert res["would_update"] == 1
    assert res["updated"] == 0
    assert conn.execute("SELECT cost_usd FROM router_logs").fetchone()[0] == 0.0


def test_backfill_reports_output_token_gap(conn):
    """Streaming rows have no output tokens — the figure understates output
    cost, and the result must say so rather than imply full accuracy."""
    _row(conn, "glm-5.3", 1_000_000, 0)
    _row(conn, "glm-5.3", 1_000_000, 500_000)
    res = backfill_costs(conn)
    assert res["updated"] == 2
    assert res["rows_missing_output_tokens"] == 1


def test_backfill_batches_without_losing_rows(conn):
    for _ in range(25):
        _row(conn, "glm-5.3", 1000, 0)
    res = backfill_costs(conn, batch=10)
    assert res["updated"] == 25
    n = conn.execute("SELECT COUNT(*) FROM router_logs WHERE cost_unknown = 0").fetchone()[0]
    assert n == 25
