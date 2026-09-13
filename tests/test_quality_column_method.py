"""B26.8 — `quality_column` must not average across scoring methods.

The third consumer of `quality_score` with the method-mixing defect (after the
rollup in B26.4 and `unit_economics` in B26.7). Found by sweeping every
aggregation of the column rather than fixing the ones I happened to know about:

    grep -rn 'AVG(quality_score)|COUNT(quality_score)' scripts/

is the check to re-run whenever the scorer changes.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))

from quality import quality_column  # noqa: E402
from rollup import migrate  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "router_logs.db"))
    migrate(conn)
    return conn


def _log(conn, model, quality, method):
    conn.execute(
        "INSERT INTO router_logs (timestamp, session_id, model_used, workload_type, "
        "input_tokens, output_tokens, cost_usd, cost_unknown, success, "
        "quality_score, quality_method) "
        "VALUES ('2026-09-01T10:00:00+00:00','s',?,'session_compression',"
        "1000,100,0.01,0,1,?,?)",
        (model, quality, method),
    )
    conn.commit()


def test_v1_rows_do_not_enter_the_average(db):
    _log(db, "m", 0.9, "fact_fidelity_v2")
    _log(db, "m", 0.1, "fact_coverage_v1")
    rows = quality_column(db, since="2026-09-01")
    assert len(rows) == 1
    assert rows[0].quality_avg == pytest.approx(0.9), "v1 row must not be averaged in"
    assert rows[0].measured == 1


def test_legacy_only_reads_unmeasured_not_zero(db):
    _log(db, "legacy", 0.2, "fact_coverage_v1")
    rows = quality_column(db, since="2026-09-01")
    assert rows[0].measured == 0
    assert rows[0].quality_avg is None
