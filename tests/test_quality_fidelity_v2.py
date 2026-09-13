"""B26.2 — `fact_fidelity_v2`: a metric that can actually rank summarisers.

Why v1 is unfit (measured, /tmp/b26_design.txt, /tmp/metric_diag2.txt):

1. **Wrong source** — fed only ``role == "user"`` content: 19,426 of 225,862
   chars. A summary reporting a true fact from a tool/assistant message was
   counted as a *hallucination*. Fixed in B26.1.

2. **Not scale-invariant** — ``score = coverage * (1 - hallucination_rate)``
   where ``coverage = |supported| / |source_facts|``. The denominator grows with
   the source, so an identical-quality summary measured 0.2439 against a 200K
   context and 0.6780 against a 68K one. A metric whose scale depends on the
   size of the input cannot compare models, and this pipeline varies context
   size by 20x.

3. **Ceiling below the decision floor** — the optimiser requires
   ``quality_avg >= 0.80`` (``optimiser.DEFAULT_QUALITY_FLOOR``) and divides
   cost by quality. v1's best real score is ~0.24, so *every* model is
   permanently ineligible and the optimiser can never act.

v2 measures the two things that actually distinguish a good compression summary
of a large context:
* **fidelity** — share of the facts the summary states that are real
  (precision). Inventing figures is the failure mode that damages a downstream
  reader, and it is penalised directly.
* **recall against an achievable budget** — you cannot reproduce 492 facts in a
  6K-char summary, so raw source coverage is unachievable by construction. v2
  asks instead: did the summary spend its own length budget on real facts?
  Normalised by the summary's fact capacity, so the score does not move when the
  source grows.

These tests pin the properties a ranking metric must have, not just its
arithmetic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from quality import (  # noqa: E402
    METHOD_V2,
    extract_facts,
    score_summary,
    score_summary_v2,
    source_text_from_messages,
)

CPF = 60  # chars-per-fact budget constant, mirrors quality.FACT_CHARS_BUDGET


def facts(n: int, start: int = 1000, step: int = 7):
    """n distinct integers >= 100 as a list."""
    return [float(start + i * step) for i in range(n)]


def summary_with(real, invented=(), prose_per_fact: int = 40, target_chars: int = 0):
    """A plausible summary stating `real` + `invented` numbers.

    With ``target_chars`` the body is padded to that exact length, so summaries
    differing only in fact count can be compared at matched length — the
    property v2 normalises on.
    """
    body = "; ".join(f"value {int(v)}" for v in list(real) + list(invented))
    prose = "Reviewed and verified. " * max(1, prose_per_fact)
    out = "## Summary\n" + body + "\n" + prose
    if target_chars and len(out) < target_chars:
        out += "Reviewed and verified. " * ((target_chars - len(out)) // 25 + 1)
        out = out[:target_chars]
    return out


# ── identity ─────────────────────────────────────────────────────────────────

def test_v2_has_its_own_method_name():
    """v1 rows must stay separable from v2 rows — never average across methods."""
    assert METHOD_V2 == "fact_fidelity_v2"
    assert METHOD_V2 != "fact_coverage_v1"
    s = score_summary_v2("fixed 231217 rows", "repaired 231217 rows")
    assert s.method == METHOD_V2


# ── P1: discrimination ───────────────────────────────────────────────────────

def test_dense_faithful_summary_outranks_sparse_faithful_summary():
    """At MATCHED LENGTH, more real facts must rank higher.

    Length is held constant because that is exactly what v2 normalises: a summary
    is judged against its own length budget, not the source's, which is what makes
    the score scale-invariant. Comparing summaries of different lengths would test
    a property v2 deliberately does not have (and v1's failure mode).
    """
    src = " ".join(f"metric {int(v)} observed" for v in facts(300, step=11))
    dense = summary_with(facts(15, step=11), target_chars=8000)
    sparse = summary_with(facts(2, step=11), target_chars=8000)
    q_dense = score_summary_v2(src, dense)
    q_sparse = score_summary_v2(src, sparse)
    assert q_dense.score > q_sparse.score, (q_dense, q_sparse)


# ── P2: truth beats recall ───────────────────────────────────────────────────

def test_fabricating_summary_ranks_below_honest_sparse_summary():
    """A summary that states more facts but invents half of them must lose.

    This is the property that protects against a model that pads its output with
    plausible-looking figures — the exact failure a summariser is prone to.
    """
    src = " ".join(f"metric {int(v)} observed" for v in facts(300, step=11))
    honest = summary_with(facts(20, step=11))
    liar = summary_with(facts(20, step=11), invented=facts(20, start=800000, step=13))
    q_honest = score_summary_v2(src, honest)
    q_liar = score_summary_v2(src, liar)
    assert q_liar.score < q_honest.score, (q_liar, q_honest)


def test_every_stated_fact_invented_scores_zero():
    src = " ".join(f"metric {int(v)} observed" for v in facts(50, step=11))
    all_lies = summary_with([], invented=facts(30, start=900000, step=17))
    q = score_summary_v2(src, all_lies)
    assert q.score == pytest.approx(0.0)


# ── P3: scale invariance ─────────────────────────────────────────────────────

def test_same_quality_summary_scores_the_same_on_small_and_large_source():
    """The score must not depend on how big the source is.

    v1 measured 0.2439 on a 200K context and 0.6780 on a 68K one for the same
    summary. A metric that moves with input size cannot compare models across a
    workload whose contexts vary 20x.
    """
    small_src = " ".join(f"metric {int(v)} observed" for v in facts(60, step=11))
    large_src = " ".join(f"metric {int(v)} observed" for v in facts(600, step=11))
    s = summary_with(facts(40, step=11))
    a = score_summary_v2(small_src, s)
    b = score_summary_v2(large_src, s)
    assert a.score == pytest.approx(b.score, abs=0.02), (a, b)


def test_v1_does_not_have_that_property():
    """Regression guard on the diagnosis itself: v1 is size-dependent.

    If someone "fixes" v1 back to scale invariance, this tells them the finding
    moved and the docs need revisiting.
    """
    small_src = " ".join(f"metric {int(v)} observed" for v in facts(60, step=11))
    large_src = " ".join(f"metric {int(v)} observed" for v in facts(600, step=11))
    s = summary_with(facts(10, step=11))
    a = score_summary(small_src, s)
    b = score_summary(large_src, s)
    assert a.score > b.score * 1.5, (a, b)


# ── boundaries ───────────────────────────────────────────────────────────────

def test_perfect_summary_scores_one():
    src = " ".join(f"metric {int(v)} observed" for v in facts(20, step=11))
    perfect = " ".join(f"metric {int(v)} observed" for v in facts(20, step=11))
    assert score_summary_v2(src, perfect).score == pytest.approx(1.0)


def test_empty_summary_scores_zero():
    assert score_summary_v2("231217 rows", "").score == pytest.approx(0.0)


def test_summary_with_no_numbers_scores_zero():
    """On a numeric source, prose-only carries no verifiable fact."""
    s = score_summary_v2("231217 rows and 1458 dollars", "Some work happened.")
    assert s.score == pytest.approx(0.0)


def test_score_is_bounded_and_deterministic():
    src = " ".join(f"metric {int(v)} observed" for v in facts(100, step=11))
    s = summary_with(facts(30, step=11), invented=facts(5, start=700000, step=3))
    a = score_summary_v2(src, s)
    b = score_summary_v2(src, s)
    assert 0.0 <= a.score <= 1.0
    assert (a.score, a.coverage, a.hallucinated_numbers) == (
        b.score, b.coverage, b.hallucinated_numbers
    )


def test_reaches_the_optimiser_floor_for_a_genuinely_dense_summary():
    """The point of the exercise: a good summary must clear quality_floor=0.80.

    v1's ceiling on real payloads was ~0.24, so no model was ever eligible.
    """
    src = " ".join(f"metric {int(v)} observed" for v in facts(200, step=11))
    dense = summary_with(facts(150, step=11), prose_per_fact=60)
    assert score_summary_v2(src, dense).score >= 0.80


# ── the source-completeness property, under v2 ───────────────────────────────

def test_true_fact_from_assistant_or_tool_message_is_not_a_hallucination():
    """Carries B26.1 forward: the v2 score must see the whole conversation."""
    real = Path(__file__).resolve().parent.parent / "data" / "compression_samples"
    files = sorted(real.glob("*.json"))
    if not files:
        pytest.skip("no captured payloads")
    d = json.loads(files[0].read_text())
    src = source_text_from_messages(d["messages"])
    all_facts = sorted(extract_facts(src))
    assert len(all_facts) >= 10
    cited = all_facts[:10]
    s = score_summary_v2(src, " ".join(f"value {int(v)}" for v in cited))
    assert s.hallucinated_numbers == 0, "a true source fact was counted as invented"
    assert s.score > 0.0
