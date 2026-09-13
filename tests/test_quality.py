"""B3 — response quality measurement.

The scorer is the established fact-coverage / hallucinated-number metric from
``ab_glm53_flash_compress.py``, promoted into a module so the optimiser and the
ingress sampler use ONE definition of quality. These tests pin its exact
semantics, including the deliberate "ignore numbers < 100" rule.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from quality import (  # noqa: E402
    METHOD,
    METHOD_V2,
    extract_facts,
    quality_column,
    score_summary,
)
from rollup import migrate  # noqa: E402

SOURCE = (
    "We fixed 231217 rows across 46 columns. Latency was 197.6 seconds. "
    "The cost is 1458 dollars per week. Ticket 42 is closed. There are 3 hosts."
)
SUMMARY_GOOD = (
    "## Completed Actions\n"
    "Repaired 231217 rows over 46 columns. Latency 197.6 seconds. Cost 1458 per week.\n"
)


def test_extract_facts_ignores_numbers_under_100():
    """Small integers are too common to be evidence of coverage."""
    facts = extract_facts("There are 3 hosts and 42 tickets, but 1458 dollars and 231217 rows.")
    assert 1458.0 in facts
    assert 231217.0 in facts
    assert 3.0 not in facts
    assert 42.0 not in facts


def test_extract_facts_normalises_thousands_separators():
    assert extract_facts("total 231,217 rows") == {231217.0}


def test_score_summary_full_coverage_no_hallucination():
    s = score_summary(SOURCE, SUMMARY_GOOD)
    assert s.method == METHOD == "fact_coverage_v1"
    # source facts >=100: 231217, 46 is <100? no: 46 < 100 so excluded; 197.6, 1458
    # => {231217.0, 197.6, 1458.0}
    assert s.coverage == pytest.approx(1.0)
    assert s.hallucinated_numbers == 0
    assert s.score == pytest.approx(1.0)


def test_score_summary_missing_numbers_lowers_coverage():
    s = score_summary(SOURCE, "## Completed Actions\nRepaired 231217 rows.\n")
    # 1 of 3 source facts present
    assert s.coverage == pytest.approx(1 / 3, rel=0.01)
    assert s.hallucinated_numbers == 0


def test_score_summary_detects_hallucinated_numbers():
    """A number in the summary that is absent from the source is a hallucination."""
    s = score_summary(SOURCE, "## Completed Actions\nRepaired 231217 rows costing 9999 per week.\n")
    assert s.hallucinated_numbers == 1
    # score is penalised for the invented figure
    assert s.score < 1.0


def test_score_summary_no_source_facts_gives_zero_coverage():
    s = score_summary("nothing numeric here at all", "## Completed Actions\nfine.\n")
    assert s.coverage == 0.0


def test_score_summary_empty_summary_is_zero_quality():
    s = score_summary(SOURCE, "")
    assert s.score == 0.0
    assert s.coverage == 0.0


def test_score_is_bounded_0_to_1():
    s = score_summary(SOURCE, SUMMARY_GOOD + " 123456 789012 345678")
    # 6 summary facts (3 real + 3 invented) -> coverage 1.0, hallucination 3/6 = 0.5
    assert s.coverage == pytest.approx(1.0)
    assert s.hallucinated_numbers == 3
    assert s.score == pytest.approx(0.5)
    s2 = score_summary(SOURCE, SUMMARY_GOOD)
    assert 0.0 <= s2.score <= 1.0


def test_score_summary_is_deterministic():
    a = score_summary(SOURCE, SUMMARY_GOOD)
    b = score_summary(SOURCE, SUMMARY_GOOD)
    assert (a.score, a.coverage, a.hallucinated_numbers) == (
        b.score, b.coverage, b.hallucinated_numbers
    )


# ── the quality column over logged rows ───────────────────────────────────────

def _log(conn, model, workload, quality):
    conn.execute(
        "INSERT INTO router_logs (timestamp, session_id, model_used, workload_type, "
        "input_tokens, output_tokens, quality_score, quality_method) "
        "VALUES ('2026-09-01T10:00:00+00:00','s',?,?,100,10,?,?)",
        (model, workload, quality, METHOD_V2 if quality is not None else ""),
    )


def test_quality_column_averages_measured_rows_only(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    migrate(conn)
    _log(conn, "glm-5.3", "session_compression", 0.5)
    _log(conn, "glm-5.3", "session_compression", 1.0)
    _log(conn, "glm-5.3", "session_compression", None)   # unmeasured
    conn.commit()
    rows = quality_column(conn, since="2026-09-01", until="2026-09-02")
    r = [x for x in rows if x.model == "glm-5.3"][0]
    assert r.measured == 2
    assert r.unmeasured == 1
    assert r.quality_avg == pytest.approx(0.75)


def test_quality_column_null_never_counted_as_zero(tmp_path):
    """An unmeasured call must not be treated as a failed-quality call."""
    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    migrate(conn)
    _log(conn, "m", "normal_chat", None)
    conn.commit()
    rows = [r for r in quality_column(conn, since="2026-09-01", until="2026-09-02") if r.model == "m"]
    assert rows[0].measured == 0
    assert rows[0].quality_avg is None
