"""B26.4 — structural markers must not count as facts.

Measured on 200 real production compaction summaries: a MEAN of 22.3% of all
numbers `extract_facts` pulled out were list ordinals ("473. REPAIRED") or
numbered headings ("## 3.1 Budget"), reaching 83% in the worst case. These are
typography, not claims. Two harms, both unrelated to summary quality:

1. They inflate the denominator of ``precision``, depressing it for well-numbered
   summaries.
2. The source conversation rarely shares the same ordinals, so they were counted
   as *hallucinated* — punishing a summary for its numbering scheme.

The fix must not go too far: a genuine figure that also happens to sit in a list
position must survive, because it is a real claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))

from quality import extract_facts, score_summary_v2, structural_marker_numbers  # noqa: E402


class TestStructuralMarkersExcluded:
    def test_list_ordinal_in_prose_is_kept(self):
        """A number appearing in prose as well as in a list position is a claim."""
        text = "473. The router logged 473 compression calls."
        assert 473.0 in extract_facts(text), "in-prose figure must survive"

    def test_pure_ordinal_is_dropped(self):
        text = "473. REPAIRED the writer.\n474. COMMITTED the fix."
        facts = extract_facts(text)
        assert 473.0 not in facts and 474.0 not in facts

    def test_numbered_heading_is_dropped(self):
        text = "## 1520 Overhead\nSome prose with no figures."
        assert extract_facts(text) == set()

    def test_real_figure_alongside_ordinals_survives(self):
        text = (
            "## Historical Task Snapshot\n"
            "1. Fixed 243041 rows.\n"
            "2. Cost was 18674 dollars.\n"
        )
        facts = extract_facts(text)
        assert 243041.0 in facts
        assert 18674.0 in facts
        assert 1.0 not in facts and 2.0 not in facts

    def test_marker_helper_returns_only_pure_markers(self):
        # Ordinals above MIN_FACT (100) are the real-world case: a summary
        # numbered 473, 474, ... . A 1-digit ordinal is never a "fact" anyway.
        text = "473. saw 999 rows\n474. nothing here"
        markers = structural_marker_numbers(text)
        assert 473.0 in markers and 474.0 in markers
        assert 999.0 not in markers


class TestMarkerFilterImprovesScoring:
    """The end-to-end consequence: numbering must not depress a real score."""

    def test_same_facts_score_equal_regardless_of_numbering(self):
        """Renumbering the same summary must not change its quality."""
        source = "The run processed 243041 rows and cost 18674 dollars."
        plain = "Processed 243041 rows. Cost 18674 dollars."
        numbered = "1. Processed 243041 rows.\n2. Cost 18674 dollars."
        a = score_summary_v2(source, plain).score
        b = score_summary_v2(source, numbered).score
        assert a == pytest.approx(b), f"numbering changed score {a} -> {b}"

    def test_numbering_is_not_counted_as_hallucination(self):
        source = "The run processed 243041 rows and cost 18674 dollars."
        numbered = "1. Processed 243041 rows.\n2. Cost 18674 dollars."
        qs = score_summary_v2(source, numbered)
        assert qs.hallucinated_numbers == 0
