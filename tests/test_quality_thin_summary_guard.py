"""B26.5 — a thin summary must not be promoted as evidence of quality.

`fact_yield` is normalised by the summary's OWN length, so without an absolute
floor a three-fact stub saturates at score 1.0 while a realistic fifteen-fact
production summary scores ~0.74. The optimiser would then rank the terse model
highest and could promote it — a false positive of exactly the kind this pipeline
is built to avoid.

Measured on 200 real production compaction summaries: p05 = 5 facts, p10 = 7,
median = 15. MIN_SUMMARY_FACTS is set at that p05 boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))

from quality import (  # noqa: E402
    MIN_SUMMARY_FACTS,
    extract_facts,
    score_summary_v2,
)

SOURCE = " ".join(f"metric {1000 + i * 11} observed" for i in range(300))


def at_length(n_facts: int, target: int = 8000) -> str:
    body = "; ".join(f"value {1000 + i * 11}" for i in range(n_facts)) + ". "
    return pad(body, target)


def pad(body: str, target: int = 8000) -> str:
    """Pad a summary body to an exact length, so only fact content varies."""
    while len(body) < target:
        body += "Reviewed and verified. "
    return body[:target]


class TestThinSummariesAreNotRewarded:
    def test_stub_does_not_beat_realistic_output(self):
        """The core guard: 3 facts must not outrank a realistic summary."""
        stub = at_length(3)
        real = at_length(20)
        q_stub = score_summary_v2(SOURCE, stub)
        q_real = score_summary_v2(SOURCE, real)
        assert q_stub.score < q_real.score, (q_stub, q_real)

    def test_below_floor_is_scaled_not_zeroed(self):
        """Degradation is smooth, so partial signal survives."""
        scores = [
            score_summary_v2(SOURCE, at_length(n)).score for n in (1, 2, 3, 4, 5)
        ]
        assert all(b > a for a, b in zip(scores, scores[1:])), scores
        assert scores[-1] > 0.0

    def test_floor_is_a_real_world_boundary(self):
        """The constant must match the measured p05 of production summaries."""
        assert MIN_SUMMARY_FACTS == 5

    def test_at_and_above_floor_is_unpenalised(self):
        """5 facts is the floor: it must score full yield at its density."""
        q = score_summary_v2(SOURCE, at_length(5, target=400))
        assert q.fact_yield == pytest.approx(1.0)

    def test_monotone_in_fact_count_at_matched_length(self):
        """More real facts at the same length is never worse."""
        scores = [
            score_summary_v2(SOURCE, at_length(n)).score
            for n in (5, 8, 12, 16, 24, 32)
        ]
        assert all(b >= a for a, b in zip(scores, scores[1:])), scores
        assert scores[-1] > scores[0]

    def test_fabrication_lowers_the_score_at_equal_true_facts(self):
        """Padding a summary with invented figures must strictly lower its score.

        True-fact count and length are held constant so only fabrication varies —
        that isolates the safety property. (Comparing a thin honest summary against
        a longer lying one compares true-fact counts, which is a different thing.)
        """
        true_facts = "; ".join(f"value {1000 + i * 11}" for i in range(15)) + ". "
        lies = "; ".join(f"value {999000 + i}" for i in range(15)) + ". "
        honest = pad(true_facts)
        padded = pad(true_facts + lies)
        q_honest = score_summary_v2(SOURCE, honest)
        q_padded = score_summary_v2(SOURCE, padded)
        assert q_padded.score < q_honest.score, (q_padded, q_honest)
        assert q_padded.hallucinated_numbers == 15

    def test_heavy_fabrication_cannot_clear_the_promotion_floor(self):
        """The optimiser floor is 0.80; a half-fabricated summary must not reach it.

        This is the safety mechanism that stops a model padding its output with
        plausible figures from being promoted on this metric.
        """
        true_facts = "; ".join(f"value {1000 + i * 11}" for i in range(15)) + ". "
        lies = "; ".join(f"value {999000 + i}" for i in range(15)) + ". "
        q = score_summary_v2(SOURCE, pad(true_facts + lies, target=4000))
        assert q.score < 0.80, q
