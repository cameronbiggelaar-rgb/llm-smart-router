"""Batch B37b (RED) — the request classifier must find the architectural lane.

Why this module exists
----------------------
The governor's policy is an allowlist: the ChatGPT plan is reserved for the
architectural lane only. That policy is only as good as the detector that
decides what *is* architectural — and the existing detector is broken for this
purpose in two measured ways:

  1. ``match_sub_type(task_type, prompt)`` is keyed on ``task_type``. The same
     architectural prompt returns tier 13 under ``planning`` but ``None`` under
     ``debugging`` — and ``debugging`` is ~30% of traffic. An allowlist built
     on the current detector would silently deny real architectural asks.
  2. The genuine tier-13 clause has fired *once in 208,530 logged calls*, so
     the signal must be read from the prompt text, scanning EVERY task type,
     not from the task_type the classifier happened to pick.

These tests pin ``classify_request`` to scan all task types for a tier ≥13
sub-type match, and to treat ``force_model`` as an unconditional architectural
signal (the escape hatch the operator can always pull).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import gpt_budget as gb  # noqa: E402

# The routing table's real tier-13 sub-type keywords (routing_table.yaml,
# planning sub-type). These are the ONLY strings that reach tier 13 today.
TIER13_KEYWORDS = [
    "rethink",
    "rearchitect",
    "redesign the architecture",
    "find failure states",
    "update the plans",
    "review the architecture",
]

ARCH = "architectural"
ROUTINE = "routine"


# ── The linchpin: same prompt, different task_type ────────────────────────────

@pytest.mark.parametrize("task_type", ["debugging", "testing", "coding", "planning", "unknown"])
def test_architectural_prompt_is_architectural_regardless_of_task_type(task_type):
    """The defect: match_sub_type('debugging', arch_prompt) → None.

    classify_request must NOT inherit that. The same architectural prompt must
    classify architectural no matter what task_type the router tagged it with.
    """
    prompt = "rearchitect the system design to eliminate the failure states"

    assert gb.classify_request(task_type=task_type, prompt=prompt, force_model=False) == ARCH


def test_canonical_architectural_ask_is_a_known_gap_not_a_false_positive():
    """The user's own wording scores tier 8 under the current keyword list.

    This is the documented residual risk (design §3.2): the keyword list is
    narrower than real architectural phrasing, so this exact ask classifies
    ROUTINE and the allowlist would deny it. The classifier must NOT paper over
    this by matching a bare 'architecture' substring — that would re-open the
    routine lane. The escape hatch is ``force_model`` (asserted below).
    """
    prompt = "design the architecture for it and implementation and test plans"

    # Document the gap honestly: it is NOT a tier-13 sub-type match today.
    assert gb.classify_request(task_type="planning", prompt=prompt, force_model=False) == ROUTINE

    # ...and the escape hatch must catch it.
    assert gb.classify_request(task_type="planning", prompt=prompt, force_model=True) == ARCH


def test_routine_prompt_is_routine_under_any_task_type():
    prompt = "fix the typo in the test fixture"

    assert gb.classify_request(task_type="planning", prompt=prompt, force_model=False) == ROUTINE


def test_empty_prompt_is_routine():
    assert gb.classify_request(task_type="planning", prompt="", force_model=False) == ROUTINE


def test_none_prompt_is_routine():
    assert gb.classify_request(task_type="planning", prompt=None, force_model=False) == ROUTINE


# ── force_model is the unconditional escape hatch ─────────────────────────────

def test_force_model_is_architectural_regardless_of_prompt():
    assert gb.classify_request(task_type="coding", prompt="add a comment", force_model=True) == ARCH


def test_force_model_is_architectural_even_with_empty_prompt():
    assert gb.classify_request(task_type="coding", prompt="", force_model=True) == ARCH


def test_force_model_false_with_routine_prompt_is_routine():
    assert gb.classify_request(task_type="coding", prompt="add a comment", force_model=False) == ROUTINE


# ── Keyword matching semantics ────────────────────────────────────────────────

def test_substring_keyword_match_is_case_insensitive():
    assert gb.classify_request(task_type="coding", prompt="we need to RETHINK this", force_model=False) == ARCH


def test_exact_keyword_fires_only_on_the_full_phrase():
    """'deep review' exact_keyword must not fire on a bare 'review'."""
    # 'review the architecture' is itself a tier-13 keyword; use a phrasing that
    # only matches via the exact keyword path to prove it is honoured.
    prompt = "please perform a deep review of the module"
    assert gb.classify_request(task_type="planning", prompt=prompt, force_model=False) == ARCH

    # A bare incidental mention must NOT classify architectural.
    assert gb.classify_request(task_type="planning", prompt="a quick review please", force_model=False) == ROUTINE


def test_non_architectural_keyword_is_routine():
    """'refactor' is a real keyword but tier < 13 → routine."""
    assert gb.classify_request(task_type="coding", prompt="refactor this function", force_model=False) == ROUTINE


# ── The classifier scans every task type's sub-types ─────────────────────────

def test_architectural_keyword_in_a_non_planning_task_type_is_found():
    """Prove the scan crosses task-type boundaries (the core fix).

    'rearchitect' lives under planning; a debugging-tagged prompt carrying it
    must still classify architectural because the scan walks the whole table.
    """
    prompt = "this debug session shows we need to rearchitect the retry logic"

    assert gb.classify_request(task_type="debugging", prompt=prompt, force_model=False) == ARCH


def test_classify_returns_one_of_the_two_classes():
    # Sanity: the classifier only ever emits the two canonical class tokens.
    for prompt in ("x", "rearchitect now", "perform a deep review"):
        k = gb.classify_request(task_type="planning", prompt=prompt, force_model=False)
        assert k in {ARCH, ROUTINE}
