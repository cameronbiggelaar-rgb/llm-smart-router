"""B26.7 — `unit_economics` must not average v1 and v2 quality together.

Same defect class as the rollup (B26.4), found by sweeping every consumer of
`quality_score` after the metric change rather than fixing only the one I knew
about.

`unit_economics.model_unit_costs` computed `AVG(quality_score)` over all rows.
v1 scores top out ~0.24 on real payloads and v2 spans 0.0-1.0, so averaging the
two produces a number on no scale at all — and it feeds `cost_per_quality_point`,
which is the cost/quality comparison used to judge whether a model is worth its
price. A contaminated average there is a wrong economic conclusion, not just a
cosmetic reporting issue.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))

from rollup import migrate  # noqa: E402
from unit_economics import unit_cost  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "router_logs.db"))
    migrate(conn)
    return conn


def _log(conn, model, workload, cost, quality, method, calls=1):
    for _ in range(calls):
        conn.execute(
            "INSERT INTO router_logs (timestamp, session_id, model_used, workload_type, "
            "input_tokens, output_tokens, cost_usd, cost_unknown, success, "
            "quality_score, quality_method) "
            "VALUES ('2026-09-01T10:00:00+00:00','s',?,?,1000,100,?,0,1,?,?)",
            (model, workload, cost, quality, method),
        )
    conn.commit()


def test_v1_rows_are_excluded_from_the_average(db):
    """A v1 row must not drag a v2 average down."""
    _log(db, "m", "session_compression", 0.01, 0.9, "fact_fidelity_v2", calls=2)
    _log(db, "m", "session_compression", 0.01, 0.1, "fact_coverage_v1", calls=2)
    rows = [r for r in unit_cost(db, since="2026-09-01") if r.model == "m"]
    assert len(rows) == 1
    r = rows[0]
    assert r.quality_avg == pytest.approx(0.9), "v1 rows must not be averaged in"
    assert r.quality_n == 2

    # A blended average would have been 0.5; the cost/quality figure must reflect
    # only the comparable v2 measurement.
    assert r.cost_per_quality_point == pytest.approx(r.cost_per_call / 0.9)


def test_only_v1_rows_reports_unmeasured_not_zero(db):
    """Legacy-only data must read as unmeasured, never as a low quality score."""
    _log(db, "old", "session_compression", 0.01, 0.2, "fact_coverage_v1", calls=3)
    rows = [r for r in unit_cost(db, since="2026-09-01") if r.model == "old"]
    r = rows[0]
    assert r.quality_n == 0
    assert r.quality_avg is None
    assert r.cost_per_quality_point is None
