"""Batch 3 (RED) — compression quality must be MEASURED, not assumed.

The cheap rung is carrying ~99% of compression traffic unmeasured:
`quality_score` is populated only in the shadow path, and the shadow
experiment is disabled, so nothing scores the summaries that actually serve.
Without a quality signal a cost regression is indistinguishable from a
quality regression - which is exactly how a cheaper model "wins" by getting
worse.

These tests pin the properties that make the measurement trustworthy:

* scoring is SAMPLED and OFF the user's critical path (it runs post-response;
  the response is already delivered, so scoring can never delay it),
* a scored row records its method, so a future scorer can be distinguished,
* an unscored row stays NULL rather than defaulting to a number ("not
  measured" must never look like "measured bad"),
* a scorer failure can never fail the request.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import quality_probe as qp  # noqa: E402
from quality import METHOD, score_summary  # noqa: E402
from rollup import migrate  # noqa: E402

SOURCE = "The build used 12,345 tokens and 250 files with 8,000 lines of code."
GOOD = "Build: 12,345 tokens, 8,000 lines."
POOR = "Build completed successfully."


@pytest.fixture()
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "q.db")
    migrate(c)
    return c


def test_rate_zero_measures_nothing():
    """Sampling must be genuinely off at rate 0 - no accidental hot path."""
    assert qp.should_measure("abc", rate=0.0) is False


def test_rate_one_measures_everything():
    assert qp.should_measure("abc", rate=1.0) is True


def test_decision_is_stable_per_request_id():
    """A request must not be measured twice or flap between decisions."""
    decisions = {qp.should_measure("req-42", rate=0.5) for _ in range(50)}
    assert len(decisions) == 1


def test_only_compression_is_measured_by_default():
    """Normal chat is not a summarisation task; scoring it would be noise."""
    assert qp.is_measurable_workload("session_compression") is True
    assert qp.is_measurable_workload("normal_chat") is False


def test_score_reflects_real_coverage_difference():
    """A summary that drops facts must score below one that keeps them."""
    good = score_summary(SOURCE, GOOD)
    poor = score_summary(SOURCE, POOR)
    assert good.score > poor.score
    assert poor.score == pytest.approx(0.0)


def test_probe_records_method_so_scorers_are_distinguishable(conn):
    qp.record_probe(conn, "req-1", SOURCE, GOOD)
    row = conn.execute(
        "SELECT score, method FROM quality_probe WHERE request_id = 'req-1'"
    ).fetchone()
    assert row[0] == pytest.approx(score_summary(SOURCE, GOOD).score)
    assert row[1] == METHOD


def test_unscored_is_null_not_zero(conn):
    """'Not measured' must be distinguishable from 'measured bad'."""
    qp.record_probe(conn, "req-2", SOURCE, "")
    row = conn.execute(
        "SELECT score, method, measurable FROM quality_probe WHERE request_id = 'req-2'"
    ).fetchone()
    assert row[2] == 0, "an unscoreable attempt must be flagged, not scored 0"
    assert row[0] is None


def test_unscoreable_input_does_not_raise(conn):
    """A scorer failure must never propagate into the request path."""
    assert qp.record_probe(conn, "req-3", None, None) is None


def test_a_raising_input_cannot_break_the_request(conn):
    """The contract is exception-safety, so test it with input that RAISES.

    Passing None/None proves nothing (that path never raises). A source whose
    ``__str__`` explodes is what a malformed upstream payload can actually look
    like, and the probe must swallow it: the response is already delivered, and
    a measurement must never turn a served request into a failed one.
    """

    class Exploding:
        def __str__(self):
            raise RuntimeError("malformed payload")

        __repr__ = __str__

    # score_summary would blow up on this; record_probe must return None, not raise.
    assert qp.record_probe(conn, "req-3b", Exploding(), "summary") is None
    assert qp.record_probe(conn, "req-3c", "source", Exploding()) is None


def test_a_failing_scorer_does_not_poison_the_connection(conn):
    """After a swallowed failure the connection must still be usable."""

    class Exploding:
        def __str__(self):
            raise RuntimeError("boom")

        __repr__ = __str__

    qp.record_probe(conn, "bad", Exploding(), "x")
    # a normal probe must still work afterwards
    assert qp.record_probe(conn, "good", SOURCE, GOOD) is not None
    assert qp.summary_metrics(conn)["measured"] == 1


def test_probe_is_idempotent_per_request(conn):
    """Re-running a probe must update, not duplicate - counts stay honest."""
    qp.record_probe(conn, "req-4", SOURCE, GOOD)
    qp.record_probe(conn, "req-4", SOURCE, GOOD)
    n = conn.execute(
        "SELECT COUNT(*) FROM quality_probe WHERE request_id = 'req-4'"
    ).fetchone()[0]
    assert n == 1


def test_summary_metrics_aggregate_only_measured_rows(conn):
    """The aggregate must report measured vs unmeasured separately."""
    qp.record_probe(conn, "a", SOURCE, GOOD)
    qp.record_probe(conn, "b", SOURCE, "")
    m = qp.summary_metrics(conn)
    assert m["measured"] == 1
    assert m["unmeasured"] == 1
    assert m["avg_score"] == pytest.approx(score_summary(SOURCE, GOOD).score)


def test_no_measurements_reports_none_not_zero(conn):
    """With nothing measured, the average is unknown - not 0.0."""
    m = qp.summary_metrics(conn)
    assert m["measured"] == 0
    assert m["avg_score"] is None


def test_model_comparison_separates_measured_from_unmeasured(conn):
    """The fleet question: which model summarises well, and at what coverage."""
    qp.record_probe(conn, "m1", SOURCE, GOOD, model="deepseek-v4.1-flash")
    qp.record_probe(conn, "m2", SOURCE, POOR, model="glm-5.3")
    rows = {r["model"]: r for r in qp.by_model(conn)}
    assert rows["deepseek-v4.1-flash"]["avg_score"] > rows["glm-5.3"]["avg_score"]
    assert rows["deepseek-v4.1-flash"]["measured"] == 1
