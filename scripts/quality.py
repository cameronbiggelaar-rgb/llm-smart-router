"""Response quality measurement — ``fact_coverage_v1``.

Promotes the fact-coverage / hallucinated-number scorer that the offline A/B
harnesses already used (``ab_glm53_flash_compress.py`` / ``ab_direct.py``) into a
module, so the optimiser, the reporting layer and any live experiment all share
ONE definition of quality. A second, slightly-different scorer would make every
cost/quality comparison across systems meaningless.

Scoring rule (unchanged from the harness):
* Numbers in the source and the summary are extracted, normalised and compared
  as sets.
* Numbers below 100 are ignored — they are too common in prose to be evidence
  that a fact survived compression.
* ``coverage`` = share of source facts reproduced.
* ``hallucinated_numbers`` = summary numbers with no counterpart in the source.

**This is never called on the request hot path.** Scoring needs the source text
and (for a real signal) a model output to compare against; doing it inline would
pay a second generation per request to measure the first, and would blow the
120 s compression deadline. It runs offline over captured samples, or online
only for sampled production calls.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Set

METHOD = "fact_coverage_v1"

# Numbers with optional thousands separators and decimals.
_NUM = re.compile(r"\b\d[\d,]*\.?\d*\b")

# Minimum magnitude for a number to count as a "fact".
MIN_FACT = 100

# Structural sections the production summariser prompt asks for.
SECTIONS = [
    "Active Task",
    "Goal",
    "Constraints & Preferences",
    "Completed Actions",
    "Errors & Fixes",
    "Key Decisions",
    "Resolved Questions",
    "Relevant Files",
    "Critical Context",
]

# Degeneration marker: the summariser emitting its own deliberation.
_DEGEN = re.compile(r"Let me analyze|Let me analyse|Here is|I'll summarize|The conversation is about")


@dataclass(frozen=True)
class QualityScore:
    """Quality of one summary against its source."""

    score: float
    coverage: float
    hallucinated_numbers: int
    summary_numbers: int
    source_numbers: int
    method: str = METHOD


@dataclass(frozen=True)
class QualityRow:
    """Aggregated measured quality for one (model, call_type)."""

    model: str
    call_type: str
    measured: int
    unmeasured: int
    quality_avg: Optional[float]


def extract_facts(text: str) -> Set[float]:
    """Extract candidate fact-numbers (>= MIN_FACT) from text."""
    out: Set[float] = set()
    for m in _NUM.findall(text or ""):
        try:
            value = float(m.replace(",", ""))
        except ValueError:
            continue
        if value >= MIN_FACT:
            out.add(value)
    return out


def score_summary(source: str, summary: str, reference: Optional[str] = None) -> QualityScore:
    """Score ``summary`` against ``source`` using fact coverage.

    Returns a ``score`` in 0..1 combining coverage and hallucination penalty:

        score = coverage * (1 - hallucination_rate)

    A summary that reproduces every source fact with no invented numbers scores
    1.0. An empty summary scores 0.0. Inventing numbers is penalised
    multiplicatively, so a summary that is exhaustive but invents half its
    figures cannot outrank a faithful one.

    ``reference`` is accepted for API symmetry with reference-based scorers and
    is currently ignored — scoring is source-grounded, not reference-grounded.
    """
    src_facts = extract_facts(source)
    smy_facts = extract_facts(summary)

    if not smy_facts:
        return QualityScore(
            score=0.0,
            coverage=0.0,
            hallucinated_numbers=0,
            summary_numbers=0,
            source_numbers=len(src_facts),
        )

    coverage = (len(src_facts & smy_facts) / len(src_facts)) if src_facts else 0.0
    invented = smy_facts - src_facts
    hallucination_rate = len(invented) / len(smy_facts)

    score = coverage * (1.0 - hallucination_rate)
    # Clamp for safety against float edge cases.
    score = max(0.0, min(1.0, score))

    return QualityScore(
        score=round(score, 6),
        coverage=round(coverage, 6),
        hallucinated_numbers=len(invented),
        summary_numbers=len(smy_facts),
        source_numbers=len(src_facts),
    )


def quality_column(
    conn: sqlite3.Connection,
    since: str,
    until: Optional[str] = None,
    model: Optional[str] = None,
    call_type: Optional[str] = None,
) -> List[QualityRow]:
    """Aggregate measured quality per (model, call_type) over a window.

    Rows with NULL ``quality_score`` are counted as ``unmeasured`` and excluded
    from the average — "not measured" is deliberately distinct from "measured
    bad".
    """
    sql = """
        SELECT model_used,
               COALESCE(NULLIF(workload_type, ''), 'unknown') AS call_type,
               COUNT(quality_score) AS measured,
               SUM(CASE WHEN quality_score IS NULL THEN 1 ELSE 0 END) AS unmeasured,
               AVG(quality_score)  AS quality_avg
        FROM router_logs
        WHERE substr(timestamp, 1, 10) >= substr(?, 1, 10)
    """
    params: List[object] = [since]
    if until is not None:
        sql += " AND substr(timestamp, 1, 10) <= substr(?, 1, 10)"
        params.append(until)
    if model is not None:
        sql += " AND model_used = ?"
        params.append(model)
    if call_type is not None:
        sql += " AND workload_type = ?"
        params.append(call_type)
    sql += " GROUP BY model_used, call_type ORDER BY model_used, call_type"

    rows: List[QualityRow] = []
    for r in conn.execute(sql, params):
        rows.append(
            QualityRow(
                model=r[0],
                call_type=r[1],
                measured=int(r[2] or 0),
                unmeasured=int(r[3] or 0),
                quality_avg=float(r[4]) if r[4] is not None else None,
            )
        )
    return rows


def record_quality(
    conn: sqlite3.Connection,
    request_id: str,
    score: QualityScore,
) -> int:
    """Attach a measured quality score to the logged row for ``request_id``.

    Returns the number of rows updated. Scoring is post-hoc, so the row already
    exists; this never inserts.
    """
    cur = conn.execute(
        "UPDATE router_logs SET quality_score = ?, quality_method = ? WHERE request_id = ?",
        (score.score, score.method, request_id),
    )
    conn.commit()
    return cur.rowcount
