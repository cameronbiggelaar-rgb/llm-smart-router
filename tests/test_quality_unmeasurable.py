"""B26.6 — 'not measurable' must never be recorded as 'measured bad'.

The plan's requirement: *unmeasurable is not zero*. A summary carrying no
extractable numeric facts (ordinary prose) has not demonstrated anything either
way. Scoring it 0.0 and recording `measurable = 1` asserts a measurement that was
never made — the false-positive class the user called out explicitly.

Consequence if left: a model whose summaries happen to be prose-only accumulates
a 0.0 average, is excluded by the optimiser's floor, and looks like a bad
summariser when it was never assessed. Worse, it drags `quality_avg` for any
model that mixes prose and numeric output.

Distinguish three states, exactly as the ledger must:
  * not scoreable      -> score NULL, measurable = 0
  * scoreable + good   -> score x, measurable = 1
  * scoreable + empty  -> score 0.0, measurable = 1   (stated nothing, provably)
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))

import quality_probe as qp  # noqa: E402
from quality import extract_facts, score_summary_v2  # noqa: E402

NUMERIC_SOURCE = (
    "The build used 12345 tokens and 250 files with 8000 lines of code. "
    "Cost was 1234 dollars."
)
PROSE_SUMMARY = (
    "The task was completed successfully after careful review of the situation. "
    "No figures were reported."
)


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    qp.migrate(c)
    yield c
    c.close()


class TestFactlessSummaryIsNotMeasurable:
    def test_prose_summary_has_no_facts(self):
        """Precondition: the fixture really is factless."""
        assert extract_facts(PROSE_SUMMARY) == set()

    def test_probe_records_null_and_unmeasurable(self, conn):
        """A factless summary must not be recorded as a 0.0 measurement."""
        qp.record_probe(conn, "req-prose", NUMERIC_SOURCE, PROSE_SUMMARY,
                        model="m", workload="session_compression")
        row = conn.execute(
            "SELECT score, method, measurable FROM quality_probe "
            "WHERE request_id = 'req-prose'"
        ).fetchone()
        assert row["score"] is None, "must be NULL, not 0.0"
        assert row["measurable"] == 0
        assert row["method"] == ""

    def test_metrics_do_not_average_unmeasurable_rows(self, conn):
        """The aggregate must not let a non-measurement drag the average."""
        qp.record_probe(conn, "good", NUMERIC_SOURCE,
                        "Build used 12345 tokens, 250 files, 8000 lines, cost 1234.",
                        model="m", workload="session_compression")
        qp.record_probe(conn, "prose", NUMERIC_SOURCE, PROSE_SUMMARY,
                        model="m", workload="session_compression")
        m = qp.summary_metrics(conn)
        assert m["measured"] == 1
        assert m["unmeasured"] == 1
        assert m["avg_score"] == pytest.approx(
            score_summary_v2(NUMERIC_SOURCE,
                             "Build used 12345 tokens, 250 files, 8000 lines, cost 1234.").score
        )

    def test_empty_summary_still_unmeasurable(self, conn):
        qp.record_probe(conn, "empty", NUMERIC_SOURCE, "", model="m")
        row = conn.execute(
            "SELECT score, measurable FROM quality_probe WHERE request_id = 'empty'"
        ).fetchone()
        assert row["score"] is None and row["measurable"] == 0

    def test_scorer_reports_zero_summary_numbers(self):
        """The scorer's own fields must make the distinction checkable."""
        qs = score_summary_v2(NUMERIC_SOURCE, PROSE_SUMMARY)
        assert qs.summary_numbers == 0
        assert qs.score == 0.0
