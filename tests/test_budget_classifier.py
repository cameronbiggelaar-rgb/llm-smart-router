"""Batch B37h (RED) — the shared lane classifier.

Why this module exists
----------------------
Two consumers need to answer "which routing lane fired?": the router (at
enforcement time) and `check-routing-stats.py` (reporting). Before this batch
each answered it independently — reporting from the logged `routing_reason`
string, routing from the prompt, because at enforcement time the reason string
has not been written yet and the tool branch never records a tier clause.

Two implementations of one question drift, and a drifted report is worse than no
report: the whole point of the governor's observability is to show the operator
what was denied, and a classifier that disagrees with the router would show a
leak that is not there (or hide one that is).

The lane names are the reporting vocabulary already in use — `force_model`,
`escalation`, `rethink_rearchitect`, `normal` — so this extraction must not
change any existing count. Precedence is significant and asserted: force_model
beats escalation, which beats the tier-13 architectural trigger, which beats
normal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import gpt_budget as gb  # noqa: E402


# ── The reporting lanes (existing vocabulary, unchanged counts) ──────────────

def test_forced_model_reason_is_the_force_model_lane():
    assert (
        gb.classify_lane(reason="forced model: gpt-5.5", tier=10) == gb.LANE_FORCE_MODEL
    )


def test_escalation_reason_is_the_escalation_lane():
    assert (
        gb.classify_lane(reason="escalation from deepseek-v4-pro (timeout)", tier=10)
        == gb.LANE_ESCALATION
    )


def test_escalated_flag_alone_marks_the_escalation_lane():
    """The `escalated` column is authoritative even if the reason is bare."""
    assert (
        gb.classify_lane(reason="capability tier 10 needed", tier=10, escalated=True)
        == gb.LANE_ESCALATION
    )


def test_tier_13_with_no_other_marker_is_the_architectural_lane():
    assert (
        gb.classify_lane(reason="capability tier 13 needed", tier=13)
        == gb.LANE_RETHINK_REARCHITECT
    )


def test_ordinary_request_is_the_normal_lane():
    assert gb.classify_lane(reason="capability tier 8 needed", tier=8) == gb.LANE_NORMAL


# ── Precedence (must match the existing reporting counts exactly) ───────────

def test_force_model_beats_escalation():
    """Precedence is load-bearing: changing it silently re-buckets history."""
    assert (
        gb.classify_lane(reason="forced model: gpt-5.5; escalation", tier=13, escalated=True)
        == gb.LANE_FORCE_MODEL
    )


def test_escalation_beats_tier_13():
    assert (
        gb.classify_lane(reason="escalation from gpt-5.5 (rate_limit)", tier=13, escalated=True)
        == gb.LANE_ESCALATION
    )


def test_tier_13_beats_normal():
    assert (
        gb.classify_lane(reason="capability tier 13 needed, selected gpt-5.6-sol", tier=13)
        == gb.LANE_RETHINK_REARCHITECT
    )


# ── Governor-aware lane names ───────────────────────────────────────────────

def test_governor_allowed_architectural_reason_is_still_the_architectural_lane():
    """The new reason strings must land in the right bucket, not `normal`.

    The governor appends "; plan governor allowed class 'architectural'" to the
    ordinary capability reason. If the classifier ignored that, every governed
    architectural request would silently report as ordinary and the operator
    would see the lane as dormant — the exact opposite of the truth.
    """
    assert (
        gb.classify_lane(
            reason=(
                "capability tier 13 needed, selected gpt-5.6-sol (tier 13); "
                "plan governor allowed class 'architectural'"
            ),
            tier=13,
        )
        == gb.LANE_RETHINK_REARCHITECT
    )


def test_denied_reason_is_not_silently_bucketed_as_an_architectural_success():
    """A denial must never read as the architectural lane firing."""
    lane = gb.classify_lane(
        reason=(
            "tool-bearing request — plan account denied to class 'routine' by plan "
            "governor (architectural allowlist; gpt is reserved for architectural "
            "loads) — selected deepseek-v4-pro (tier 8)"
        ),
        tier=8,
    )
    assert lane == gb.LANE_NORMAL


def test_denied_reason_mentioning_tier_13_is_not_architectural():
    """The word 'architectural' in a DENIAL must not promote the lane.

    Denials quote the allowlist ("architectural allowlist"), so a naive
    substring test on the word would mislabel routine denials as architectural
    work. This case is deliberately constructed at tier 13 — the tier that
    normally IS the architectural lane — because that is the only configuration
    where the two implementations disagree. A denial is a refusal by policy, not
    evidence of architectural work, so the tier must not be able to promote it.
    """
    lane = gb.classify_lane(
        reason=(
            "capability tier 13 needed — plan account denied to class 'routine' by "
            "plan governor (architectural allowlist; gpt is reserved for "
            "architectural loads) — selected deepseek-v4-pro (tier 8)"
        ),
        tier=13,
    )
    assert lane == gb.LANE_NORMAL


# ── Reason-free callers ─────────────────────────────────────────────────────

def test_empty_reason_defaults_to_normal():
    assert gb.classify_lane(reason="", tier=8) == gb.LANE_NORMAL


def test_legacy_rows_with_no_reason_are_normal_not_architectural():
    """Legacy no-reason rows must not inflate the architectural lane."""
    assert gb.classify_lane(reason="", tier=0) == gb.LANE_NORMAL


# ── Round trip: the reasons the ROUTER emits must classify correctly ─────────
#
# The drift this batch exists to prevent is between the router's reason strings
# and the report's reads of them. Asserting on hand-written literals would not
# catch a reason string that changes shape, so these tests drive the real
# `route_task` and feed its actual output through the classifier.

@pytest.fixture
def clean_state():
    import router

    models = [
        "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra",
        "glm-5.3", "deepseek-v4-pro", "deepseek-v4.1-flash", "llama3.1:8b",
    ]
    saved = {m: router._MODEL_STATUSES.get(m) for m in models}
    for m in models:
        router._MODEL_STATUSES[m] = router.ModelStatus(model=m, available=True)
    yield
    for m, st in saved.items():
        if st is None:
            router._MODEL_STATUSES.pop(m, None)
        else:
            router._MODEL_STATUSES[m] = st


def _arm_governor(monkeypatch):
    monkeypatch.setattr(
        gb,
        "load_governor_config",
        lambda force=False: gb.GovernorConfig(
            enabled=True,
            allowlist=[gb.CLASS_ARCHITECTURAL],
            routine_ceiling_percent=0,
            hard_cap_percent=95.0,
        ),
    )
    monkeypatch.setattr(
        gb,
        "fetch_reading",
        lambda cfg=None, **kw: gb.Reading(
            source="live",
            allowed=True,
            primary=gb.WindowReading(
                used_percent=0.0, limit_window_seconds=18000, reset_after_seconds=165300
            ),
            secondary=gb.WindowReading(
                used_percent=0.0, limit_window_seconds=604800, reset_after_seconds=165300
            ),
        ),
    )


def test_router_architectural_reason_classifies_as_the_architectural_lane(
    clean_state, monkeypatch
):
    import router

    _arm_governor(monkeypatch)
    d = router.route_task(
        complexity_score=0.9,
        task_type="planning",
        prompt="rearchitect the budget governor from first principles",
        requires_tools=False,
    )
    assert d.selected_model == "gpt-5.6-sol", d.reason
    tier = router.MODEL_CAPABILITY_TIERS.get(d.selected_model, 0)
    assert gb.classify_lane(d.reason, tier) == gb.LANE_RETHINK_REARCHITECT, d.reason


def test_router_routine_denial_reason_classifies_as_normal(clean_state, monkeypatch):
    """A denied routine request must report as ordinary, not architectural."""
    import router

    _arm_governor(monkeypatch)
    d = router.route_task(
        complexity_score=0.85,
        task_type="coding",
        prompt="refactor the parse loop",
        requires_tools=True,
    )
    assert d.plan_budget_denied is True, d.reason
    tier = router.MODEL_CAPABILITY_TIERS.get(d.selected_model, 0)
    assert gb.classify_lane(d.reason, tier) == gb.LANE_NORMAL, d.reason


def test_router_force_model_reason_classifies_as_the_force_model_lane(
    clean_state, monkeypatch
):
    import router

    _arm_governor(monkeypatch)
    d = router.route_task(
        complexity_score=0.2,
        task_type="coding",
        prompt="refactor the parse loop",
        requires_tools=False,
        force_model="gpt-5.5",
    )
    assert d.selected_model == "gpt-5.5", d.reason
    assert gb.classify_lane(d.reason, 10) == gb.LANE_FORCE_MODEL, d.reason


def test_escalation_reason_from_router_classifies_as_escalation(clean_state, monkeypatch):
    import router

    _arm_governor(monkeypatch)
    d = router.escalate_on_failure(
        failed_model="deepseek-v4-pro",
        error_type="timeout",
        requires_tools=True,
        budget_class=gb.CLASS_ARCHITECTURAL,
    )
    assert gb.classify_lane(d.reason, 10, escalated=True) == gb.LANE_ESCALATION, d.reason
