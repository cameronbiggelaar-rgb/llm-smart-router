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
from typing import Any, List, Optional, Set

METHOD = "fact_coverage_v1"

# v2 method name. Recorded per-row in ``quality_method`` so v1 and v2 rows are
# never averaged together: they are not on the same scale.
METHOD_V2 = "fact_fidelity_v2"

# Characters of summary text assumed to carry one extractable numeric fact.
#
# MEASURED, not guessed. Anchored at the 25th percentile of real production
# compaction summaries (200 sampled from state.db): the densest quarter of real
# summaries reach 804 chars/fact, median is 870, p75 is 1280.
#
# Why p25 rather than the mean (~920): the budget sets where ``fact_yield``
# saturates at 1.0. Anchoring at the mean saturates half of all real summaries,
# making them mutually indistinguishable exactly where model ranking matters. At
# p25 only the densest quarter saturates and the rest stay ordered below 1.0.
#
# The original v2 draft used 60, assuming one fact per 60 chars -- ~13x denser
# than real summariser output -- so ``fact_yield`` saturated near 0.07 for genuine
# summaries and no model could ever reach the optimiser's 0.80 floor. The
# optimiser would have stayed inert even after the redesign.
#
# Recalibration procedure: references/plan-quality-metrics-v2.md.
FACT_CHARS_BUDGET = 804.0

# Minimum number of real facts a summary must carry before its score is treated
# as evidence of summariser quality.
#
# ``fact_yield`` is normalised by the summary's OWN length, which creates a
# perverse incentive: a three-fact 1.4KB summary saturates at score 1.0 while a
# realistic fifteen-fact 16KB production summary scores 0.74. Left unguarded, a
# model producing terse output would look like the best summariser and could be
# promoted -- a false positive.
#
# Measured on 200 real production compaction summaries: p05 = 5 facts, p10 = 7,
# median = 15. Five is the point below which real summariser output essentially
# does not fall, so a summary under it has not demonstrated anything. Below the
# floor the score is scaled down proportionally rather than zeroed, so the
# signal degrades smoothly instead of falling off a cliff.
MIN_SUMMARY_FACTS = 5

# Structural markers masquerading as facts: list ordinals ("473."), numbered
# section headings ("## 3.1 Budget"), and standalone version numbers ("v1.2.3").
# Measured on real summaries these are a mean of 22.3% of extracted "facts" (max
# 83%) and are layout, not claims. Counting them inflates precision's
# denominator (depressing precision) and marks them hallucinated when they are
# absent from the source -- both for reasons unrelated to summary quality.
STRUCTURAL_MARKER_PATTERNS = (
    r"(?m)^\s*\d{1,4}\.\s",  # "473. REPAIRED"
    r"(?m)^#{1,6}\s*\d+(?:\.\d+)*\b",  # "## 3.1 Budget", "### 1520 Overhead"
)

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
    """Quality of one summary against its source.

    ``precision`` and ``fact_yield`` are the v2 components; they default to 0.0
    so v1 call sites and stored v1 rows keep their original shape.
    """

    score: float
    coverage: float
    hallucinated_numbers: int
    summary_numbers: int
    source_numbers: int
    method: str = METHOD
    precision: float = 0.0
    fact_yield: float = 1.0


@dataclass(frozen=True)
class QualityRow:
    """Aggregated measured quality for one (model, call_type)."""

    model: str
    call_type: str
    measured: int
    unmeasured: int
    quality_avg: Optional[float]


def structural_marker_numbers(text: str) -> Set[float]:
    """Numbers that appear ONLY as layout markers, never as a claim.

    List ordinals (``473. REPAIRED``) and numbered headings (``## 3.1``) are
    typography. Measured on real compaction summaries they are a mean of 22.3% of
    all extracted "facts" (up to 83%), and because the source conversation rarely
    contains the same ordinals they were additionally counted as *hallucinated*
    numbers — penalising a summary for its numbering.

    A number is treated as a marker only when every one of its occurrences in
    this text sits in a marker position. A figure that also appears in prose
    ("473 calls") is a real claim and is kept.
    """
    if not text:
        return set()

    marker_nums: Set[float] = set()
    for pattern in STRUCTURAL_MARKER_PATTERNS:
        for m in re.finditer(pattern, text):
            for tok in _NUM.findall(m.group(0)):
                try:
                    value = float(tok.replace(",", ""))
                except ValueError:
                    continue
                if value >= MIN_FACT:
                    marker_nums.add(value)

    if not marker_nums:
        return set()

    # Keep only those with no non-marker occurrence anywhere in the text.
    all_nums = set()
    for tok in _NUM.findall(text):
        try:
            value = float(tok.replace(",", ""))
        except ValueError:
            continue
        if value >= MIN_FACT:
            all_nums.add(value)

    # Occurrences outside marker spans: strip every marker span from the text,
    # then anything still extractable is a genuine in-prose claim.
    stripped = text
    for pattern in STRUCTURAL_MARKER_PATTERNS:
        stripped = re.sub(pattern, " ", stripped)
    outside: Set[float] = set()
    for tok in _NUM.findall(stripped):
        try:
            value = float(tok.replace(",", ""))
        except ValueError:
            continue
        if value >= MIN_FACT:
            outside.add(value)

    return {v for v in all_nums if v not in outside}


def extract_facts(text: str) -> Set[float]:
    """Extract candidate fact-numbers (>= MIN_FACT) from text.

    Structural markers are excluded — see ``structural_marker_numbers``.
    """
    out: Set[float] = set()
    for m in _NUM.findall(text or ""):
        try:
            value = float(m.replace(",", ""))
        except ValueError:
            continue
        if value >= MIN_FACT:
            out.add(value)
    return out - structural_marker_numbers(text or "")


def content_to_text(content: Any) -> str:
    """Flatten a message ``content`` field to text.

    Providers send content either as a bare string or as a list of typed parts
    (``[{"type": "text", "text": "..."}]``). Both shapes occur in captured
    payloads, so scoring must accept both rather than silently dropping the
    structured form.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                # Text parts carry "text"; tool results carry "content".
                for key in ("text", "content"):
                    val = part.get(key)
                    if isinstance(val, str) and val:
                        parts.append(val)
                        break
        return "\n".join(parts)
    return str(content)


def source_text_from_messages(messages: Any) -> str:
    """Build the scoring source from EVERY message in the conversation.

    This is the single definition of "what text the summary was asked to
    compress". It must include all roles.

    Why this exists: the v1 scorer was fed only ``role == "user"`` content, which
    on a real captured payload is 19,426 of 225,862 chars. 87% of the true fact
    set was therefore invisible, so a summary faithfully reporting a fact that
    originated in an assistant or tool message was counted as a *hallucinated
    number* and penalised. That inverts the metric — it rewarded summaries that
    ignored the conversation and punished the ones that reported it.
    """
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return ""
    parts: List[str] = []
    for m in messages:
        if isinstance(m, str):
            parts.append(m)
            continue
        if not isinstance(m, dict):
            continue
        parts.append(content_to_text(m.get("content")))
        # A tool message's payload may sit in a separate field.
        if isinstance(m.get("function_call"), dict):
            parts.append(str(m["function_call"].get("arguments") or ""))
    return "\n".join(p for p in parts if p)


def source_text_user_only(messages: Any) -> str:
    """The v1 source builder: ``role == "user"`` content only.

    RETIRED from scoring — kept solely so diagnostics and regression tests can
    reproduce what v1 measured and quantify the difference. Never use this to
    score a production call: it hides the majority of the fact set and counts
    true facts as hallucinations (see ``source_text_from_messages``).
    """
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        content_to_text(m.get("content"))
        for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
    )


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


def score_summary_v2(source: str, summary: str, reference: Optional[str] = None) -> QualityScore:
    """Score ``summary`` against ``source`` — ``fact_fidelity_v2``.

        score = precision * fact_yield

    * ``precision`` = share of the numbers the summary states that really occur
      in the source. This is the safety property: it is what stops a
      plausible-looking fabricated figure from ranking highly.
    * ``fact_yield`` = facts carried, as a fraction of what a summary of this
      length could carry (``len(summary) / FACT_CHARS_BUDGET``), capped at 1.0.
      Judged against the summary's own budget, not the source's size, which is
      what makes the score scale-invariant — and lets a genuinely dense 6K
      summary of a 200K context reach 1.0, where v1 capped it at ~0.24.

    Both components are in 0..1, so the product is too. Multiplying means a
    summary must be *both* truthful and substantive: perfect precision alone
    scores 0 if it states nothing, and high yield alone cannot rescue invention.

    Reference to v1 (``score_summary``): v1's ``coverage`` divides by the source
    fact count, which on a 200K context is 492 facts — unreproducible inside a
    6K-char summary by construction. v2 keeps v1's ``coverage`` field populated
    for continuity in reporting, but does not rank on it.
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
            method=METHOD_V2,
            precision=0.0,
            fact_yield=0.0,
        )

    supported = src_facts & smy_facts
    invented = smy_facts - src_facts

    coverage = (len(supported) / len(src_facts)) if src_facts else 0.0
    precision = len(supported) / len(smy_facts)

    capacity = max(1.0, len(summary) / FACT_CHARS_BUDGET)
    fact_yield = min(1.0, len(supported) / capacity)

    # Thin summaries must not score as evidence of quality: fact_yield is
    # normalised by the summary's own length, so a three-fact stub saturates at
    # 1.0 while real multi-fact output scores lower. Scale down below the floor
    # rather than zeroing, so the signal degrades smoothly.
    substance = min(1.0, len(supported) / MIN_SUMMARY_FACTS) if MIN_SUMMARY_FACTS else 1.0

    score = max(0.0, min(1.0, precision * fact_yield * substance))

    return QualityScore(
        score=round(score, 6),
        coverage=round(coverage, 6),
        hallucinated_numbers=len(invented),
        summary_numbers=len(smy_facts),
        source_numbers=len(src_facts),
        method=METHOD_V2,
        precision=round(precision, 6),
        fact_yield=round(fact_yield, 6),
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
