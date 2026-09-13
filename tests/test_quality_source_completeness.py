"""B26.1 (RED) — the source must be the WHOLE conversation.

`fact_coverage_v1` was fed only `role == "user"` content. On a real captured
payload that is 19,426 of 225,862 chars — **87% of the true fact set is
invisible**, so a summary that faithfully reports a fact originating in an
assistant or tool message is scored as a HALLUCINATION and heavily penalised.

That inverts the metric: it rewards summaries that ignore the conversation and
punishes the ones that report it. These tests pin the property that the source
used for scoring is the complete input, whatever role a message carries.

The tests deliberately use a compressor-like message shape (system + user +
assistant + tool) because the production payload is exactly that mix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import quality_probe as qp  # noqa: E402
from quality import score_summary, source_text_from_messages  # noqa: E402


# A conversation where the load-bearing facts live in NON-user messages, which
# is what real captured payloads look like (12 user / 180 assistant / 177 tool).
CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Please summarise our progress."},
    {
        "role": "assistant",
        "content": "I fixed 231217 rows and the latency was 197.6 seconds.",
    },
    {
        "role": "tool",
        "content": "cost is 1458 dollars per week across 46 columns",
    },
]

# Facts that exist ONLY in assistant/tool messages.
FACT_FROM_ASSISTANT = 231217
FACT_FROM_TOOL = 1458


def test_source_text_includes_every_role():
    """The source is the whole conversation, not just the user's turns."""
    src = source_text_from_messages(CONVERSATION)
    assert "231217" in src, "a fact from an assistant message must be in the source"
    assert "1458" in src, "a fact from a tool message must be in the source"


def test_source_text_handles_content_parts_lists():
    """Some providers send content as a list of parts, not a bare string."""
    msgs = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "repaired 231217 rows"}],
        }
    ]
    assert "231217" in source_text_from_messages(msgs)


def test_summary_reporting_a_true_fact_is_not_a_hallucination():
    """The exact defect: a TRUE fact scored as invented because v1 hid it.

    This is the assertion that fails on v1 and drives the fix.
    """
    src = source_text_from_messages(CONVERSATION)
    summary = "Fixed 231217 rows; cost 1458 per week."

    s = score_summary(src, summary)
    assert s.hallucinated_numbers == 0, (
        "231217 and 1458 are both present in the real input — neither is invented"
    )


def test_the_v1_style_source_would_have_failed_this():
    """Non-vacuity control: prove the assertion above discriminates.

    Reproduces v1's user-only extraction and shows it DOES hallucinate the same
    summary. If this control ever stops failing, the test above has stopped
    testing anything.
    """
    v1_src = "\n".join(
        str(m.get("content") or "")
        for m in CONVERSATION
        if isinstance(m, dict) and m.get("role") == "user"
    )
    s = score_summary(v1_src, "Fixed 231217 rows; cost 1458 per week.")
    assert s.hallucinated_numbers > 0, "the user-only source must penalise true facts"
    assert s.score < 0.5


def test_probe_uses_the_whole_conversation_as_source(tmp_path):
    """The probe path must inherit the complete source, not re-implement v1."""
    from rollup import migrate

    conn = tmp_path / "q.db"
    import sqlite3

    c = sqlite3.connect(str(conn))
    migrate(c)

    qp.record_probe(
        c,
        "req-full-context",
        CONVERSATION,
        # Reproduces every extractable source fact (231217, 1458, 197.6).
        "Fixed 231217 rows; cost 1458 per week; latency 197.6 seconds.",
    )
    row = c.execute(
        "SELECT coverage, measurable FROM quality_probe WHERE request_id = 'req-full-context'"
    ).fetchone()
    assert row[1] == 1, "a real summary of a real conversation must be measurable"
    # Every source fact is in the summary -> full recall, no invention penalty.
    assert row[0] == pytest.approx(1.0), (
        "coverage must reflect the true fact set, not the user-only subset"
    )


def test_no_second_source_extraction_implementation():
    """One definition only.

    A second extraction site is how the shadow path and the probe drifted apart
    in the first place (endpoint.py:581 vs quality_probe.py:110). Guard against
    a third copy appearing.
    """
    scripts = SCRIPTS_DIR
    offenders = []
    for path in scripts.glob("*.py"):
        if path.name in {"quality.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if 'role") == "user"' in line and "for m in" not in line:
                # A user-only join used to build a SCORING source.
                window = "\n".join(text.splitlines()[max(0, i - 6):i + 3])
                if "score_summary" in window or "record_probe" in window:
                    offenders.append(f"{path.name}:{i}")
    assert not offenders, f"scoring source built from user-only messages: {offenders}"
