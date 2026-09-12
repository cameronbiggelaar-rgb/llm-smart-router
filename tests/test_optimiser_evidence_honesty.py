"""B14 — the optimiser must not assert a conclusion the evidence cannot support.

Real defect, found while answering "how does self-optimisation decide?":

``optimiser.format_proposal`` prints, whenever no candidate is emitted:

    "No cheaper qualified routing identified — incumbent is optimal."

On production that sentence is currently **false as stated**. Zero of 235,000+
``router_logs`` rows carry a measured ``quality_score``, so ``rank_models``
returns nothing for every workload and ``propose`` skips each one at
``if not ranked: continue``. The optimiser therefore concluded "incumbent is
optimal" from *no evidence at all*.

This is the same defect class as the rest of this batch: a system reporting
confidence it has not earned. The distinction that matters is:

  * "no model has measured quality yet, so nothing is eligible to be ranked"
    -> the router cannot self-optimise, and must say so
  * "quality is measured, and the incumbent already wins"
    -> genuinely optimal, no change needed
  * "a cheaper qualified model exists"
    -> a real proposal

All three must read differently. A user acting on the second message stops
looking; a user acting on the first knows to fix measurement.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import optimiser  # noqa: E402
import rollup  # noqa: E402


@pytest.fixture()
def conn(tmp_path):
    c = sqlite3.connect(str(tmp_path / "opt.db"))
    rollup.migrate(c)
    yield c
    c.close()


def _finding(conn, workload, model, day, calls, cost, quality, qn, is_shadow=0):
    conn.execute(
        """INSERT INTO daily_findings
           (day, workload_type, model, calls, input_tokens, output_tokens,
            cost_usd, quality_avg, quality_n, is_shadow)
           VALUES (?, ?, ?, ?, 0, 0, ?, ?, ?, ?)""",
        (day, workload, model, calls, cost, quality, qn, is_shadow),
    )
    conn.commit()


def test_no_measured_quality_reports_that_fact_not_optimality(conn):
    """The live state: traffic exists, quality was never measured."""
    _finding(conn, "session_compression", "deepseek-v4.1-flash", "2026-09-11", 500, 6.0, None, 0)
    _finding(conn, "session_compression", "glm-5.3-flash", "2026-09-11", 40, 0.7, None, 0)

    prop = optimiser.propose(conn, days=14)
    text = optimiser.format_proposal(prop)

    assert prop.candidates == []
    low = text.lower()
    assert "no cheaper qualified routing identified" not in low, (
        "must not claim optimality when nothing could be ranked"
    )
    assert "incumbent is optimal" not in low, (
        "must not claim optimality when nothing could be ranked"
    )
    # It must name the real reason instead: quality is unmeasured. Match the
    # meaning, not one phrasing — "no model has measured quality" and "quality
    # has not been measured" are both correct statements of the same fact.
    assert "quality" in low, f"the report must explain why; got:\n{text}"
    assert ("measured quality" in low or "unmeasured" in low or "quality is unmeasured" in low), (
        f"the report must explain that quality is unmeasured; got:\n{text}"
    )
    # And it must say self-optimisation is therefore blocked, not idle.
    assert "cannot" in low or "unable" in low or "blocked" in low, (
        f"the report must say the optimiser cannot rank; got:\n{text}"
    )


def test_measured_but_incumbent_wins_still_says_optimal(conn):
    """Genuine optimality must still be stated — the fix must not remove it."""
    # Same model on both sides is not the case here: two models, both measured,
    # incumbent cheapest, so no change is warranted.
    _finding(conn, "session_compression", "cheap-model", "2026-09-11", 500, 1.0, 0.90, 500)
    _finding(conn, "session_compression", "pricey-model", "2026-09-11", 100, 5.0, 0.95, 100)

    prop = optimiser.propose(conn, days=14)
    text = optimiser.format_proposal(prop)
    assert prop.candidates == []
    low = text.lower()
    assert "no cheaper qualified routing identified" in low or "optimal" in low, (
        f"with measured quality and a winning incumbent, optimality is the "
        f"correct conclusion; got:\n{text}"
    )
    assert "not measured" not in low, "quality WAS measured here"


def test_a_cheaper_qualified_model_is_still_proposed(conn):
    """The happy path must keep working: real savings, real proposal."""
    # Both measured above the floor; the cheaper one costs a third as much.
    _finding(conn, "session_compression", "pricey-model", "2026-09-11", 900, 9.0, 0.92, 900)
    _finding(conn, "session_compression", "cheap-model", "2026-09-11", 300, 1.0, 0.91, 300)

    prop = optimiser.propose(conn, days=14)
    text = optimiser.format_proposal(prop)
    assert prop.candidates, f"a cheaper qualified model must be proposed; got:\n{text}"
    c = prop.candidates[0]
    assert c.recommended_model == "cheap-model"
    assert c.incumbent_model == "pricey-model"
    assert "cheap-model" in text


def test_proposal_records_why_it_was_empty(conn):
    """The reason must be structured data, not only prose.

    So a caller (CLI, report, CI) can branch on it rather than string-matching
    an English sentence.
    """
    _finding(conn, "session_compression", "deepseek-v4.1-flash", "2026-09-11", 500, 6.0, None, 0)
    prop = optimiser.propose(conn, days=14)
    assert prop.candidates == []
    assert getattr(prop, "workloads_examined", None) is not None, (
        "Proposal must record how many workloads were examined"
    )
    assert prop.workloads_examined >= 1
    assert getattr(prop, "workloads_unrankable", None) is not None, (
        "Proposal must record how many workloads could not be ranked for lack "
        "of measured quality"
    )
    assert prop.workloads_unrankable >= 1, (
        "the session_compression workload has traffic but no measured quality"
    )
