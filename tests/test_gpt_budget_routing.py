"""Batch B37f (RED) — Hook 2: the ``_select_model(min_tier)`` path.

Why this module exists
----------------------
B37e closed the tool-bearing leak (97.6% of the plan's burn). This batch closes
the *second*, smaller one on the ordinary capability path: ``_select_model``
walks ``get_available_models()`` and returns the first model meeting the needed
tier, with no notion of entitlement. For a routine request scoring tier 10 that
first model is ``gpt-5.5`` — a plan model. Measured at 0.4 Mtok (1.2%) of the
clean-era plan burn.

The mirror requirement is just as important: the architectural lane must still
reach ``gpt-5.6-sol``. A governor that starves the class it exists to protect
has failed at its whole job, so ``planning`` + a tier-13 trigger stays on the
plan.

Both the entitlement matrix (allowed/denied/closed) and the fail-open invariant
are the same as the tool path (§5.1: "non-tool path — same matrix").
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import gpt_budget as gb  # noqa: E402
import router  # noqa: E402

CODEX_MODELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra"]
OLLAMA_MODELS = ["glm-5.3", "deepseek-v4-pro", "deepseek-v4.1-flash"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_state():
    saved = {
        m: router._MODEL_STATUSES.get(m)
        for m in CODEX_MODELS + OLLAMA_MODELS + ["llama3.1:8b"]
    }
    for m in CODEX_MODELS + OLLAMA_MODELS + ["llama3.1:8b"]:
        router._MODEL_STATUSES[m] = router.ModelStatus(model=m, available=True)
    yield
    for m, st in saved.items():
        if st is None:
            router._MODEL_STATUSES.pop(m, None)
        else:
            router._MODEL_STATUSES[m] = st


def _reading(used_pct=0.0, allowed=True, source="live"):
    return gb.Reading(
        source=source,
        allowed=allowed,
        primary=gb.WindowReading(
            used_percent=used_pct,
            limit_window_seconds=18000,
            reset_after_seconds=165300,
        ),
        secondary=gb.WindowReading(
            used_percent=used_pct,
            limit_window_seconds=604800,
            reset_after_seconds=165300,
        ),
    )


def _cfg(enabled=True, allowlist=None):
    return gb.GovernorConfig(
        enabled=enabled,
        allowlist=list(allowlist if allowlist is not None else [gb.CLASS_ARCHITECTURAL]),
        routine_ceiling_percent=0,
        hard_cap_percent=95.0,
    )


def _govern(monkeypatch, used_pct=0.0, allowed=True, enabled=True, source="live"):
    """Install a governor with real ``evaluate_policy`` and a stubbed reading."""
    monkeypatch.setattr(gb, "load_governor_config", lambda force=False: _cfg(enabled=enabled))
    monkeypatch.setattr(
        gb, "fetch_reading", lambda cfg=None, **kw: _reading(used_pct, allowed, source)
    )


# ── The leak ──────────────────────────────────────────────────────────────────

def test_non_tool_routine_does_not_take_gpt55(clean_state, monkeypatch):
    """THE leak on this path: routine tier-10 work must not land on gpt-5.5.

    Complexity 0.85 on an ordinary coding task needs tier 10. Before the
    governor, ``_select_model`` returned the first available model at that tier
    — ``gpt-5.5``, a plan model — for work with no architectural content at all.
    """
    _govern(monkeypatch)
    d = router.route_task(
        complexity_score=0.85,
        task_type="coding",
        prompt="refactor the parse loop",
        requires_tools=False,
    )
    assert d.selected_model != "gpt-5.5", d.reason
    assert router.get_model_account(d.selected_model) != "openai-codex", d.reason
    assert d.plan_budget_denied is True, d.reason
    assert d.budget_class == gb.CLASS_ROUTINE, d.reason


def test_non_tool_routine_high_complexity_does_not_take_plan(clean_state, monkeypatch):
    """The same must hold at the top of the complexity range."""
    _govern(monkeypatch)
    d = router.route_task(
        complexity_score=0.99,
        task_type="debugging",
        prompt="why did the nightly job fail overnight",
        requires_tools=False,
    )
    assert router.get_model_account(d.selected_model) != "openai-codex", d.reason


def test_denied_routine_still_gets_a_capable_model(clean_state, monkeypatch):
    """Denial re-selects, it does not degrade to nothing.

    Design §131: degrade PREDICTABLY to deepseek-v4-pro (tier 8) — the
    downgrade already proven at 99.97% success — rather than fail closed. On the
    ordinary capability path `min_tier` is a quality floor, not a capability
    wall, and trading a real answer for an error on the largest class of traffic
    would be a worse failure than serving it at tier 8.
    """
    _govern(monkeypatch)
    d = router.route_task(
        complexity_score=0.85,
        task_type="coding",
        prompt="refactor the parse loop",
        requires_tools=False,
    )
    assert d.selected_model == "deepseek-v4-pro", d.reason
    assert router.MODEL_CAPABILITY_TIERS.get(d.selected_model, 0) >= 6


# ── The architectural lane must survive ───────────────────────────────────────

def test_non_tool_architectural_still_reaches_sol(clean_state, monkeypatch):
    """planning + tier-13 trigger is the lane the plan exists for."""
    _govern(monkeypatch)
    d = router.route_task(
        complexity_score=0.9,
        task_type="planning",
        prompt="rearchitect the budget governor from first principles",
        requires_tools=False,
    )
    assert d.selected_model == "gpt-5.6-sol", d.reason
    assert d.budget_class == gb.CLASS_ARCHITECTURAL, d.reason
    assert d.plan_budget_denied is False, d.reason


def test_architectural_is_not_reported_as_denied(clean_state, monkeypatch):
    _govern(monkeypatch)
    d = router.route_task(
        complexity_score=0.5,
        task_type="planning",
        prompt="update the plans for the router",
        requires_tools=False,
    )
    assert d.plan_budget_denied is False, d.reason


# ── Capacity vs entitlement ───────────────────────────────────────────────────

def test_hard_cap_closes_even_for_architectural(clean_state, monkeypatch):
    """At/above the hard cap the plan is shut to every class, architectural too."""
    _govern(monkeypatch, used_pct=96.0)
    d = router.route_task(
        complexity_score=0.9,
        task_type="planning",
        prompt="rearchitect the budget governor",
        requires_tools=False,
    )
    assert router.get_model_account(d.selected_model) != "openai-codex", d.reason
    assert d.plan_budget_denied is True, d.reason


def test_account_verdict_false_closes_the_plan(clean_state, monkeypatch):
    """``allowed=False`` is authoritative and overrides the allowlist."""
    _govern(monkeypatch, used_pct=10.0, allowed=False)
    d = router.route_task(
        complexity_score=0.9,
        task_type="planning",
        prompt="rearchitect the budget governor",
        requires_tools=False,
    )
    assert router.get_model_account(d.selected_model) != "openai-codex", d.reason


# ── Fail-open invariant ───────────────────────────────────────────────────────

def test_unreadable_reading_fails_open_to_the_plan(clean_state, monkeypatch):
    """An unreadable probe must never starve the plan (§3.5 fail-open)."""
    _govern(monkeypatch, used_pct=99.0, source="unavailable")
    d = router.route_task(
        complexity_score=0.85,
        task_type="coding",
        prompt="refactor the parse loop",
        requires_tools=False,
    )
    assert d.selected_model == "gpt-5.5", d.reason
    assert d.plan_budget_denied is False, d.reason


def test_governor_exception_fails_open_not_closed(clean_state, monkeypatch):
    """A governor bug must never wedge live traffic."""
    def _boom(force=False):
        raise RuntimeError("policy file exploded")

    monkeypatch.setattr(gb, "load_governor_config", _boom)
    d = router.route_task(
        complexity_score=0.85,
        task_type="coding",
        prompt="refactor the parse loop",
        requires_tools=False,
    )
    assert d.selected_model == "gpt-5.5", d.reason


# ── Disabled == today's behaviour (rollback lever) ────────────────────────────

def test_governor_disabled_is_byte_identical_to_pre_b37(clean_state, monkeypatch):
    """``enabled: false`` must reproduce the old decision exactly.

    This is the rollback lever, so it is asserted as an exact model match, not a
    loose "not a plan model" check.
    """
    _govern(monkeypatch, enabled=False)
    d = router.route_task(
        complexity_score=0.85,
        task_type="coding",
        prompt="refactor the parse loop",
        requires_tools=False,
    )
    assert d.selected_model == "gpt-5.5", d.reason
    assert d.budget_class == "", d.reason
    assert d.plan_budget_denied is False, d.reason


def test_governor_disabled_reason_is_unchanged(clean_state, monkeypatch):
    """Nothing is reported as governed when the governor is off."""
    _govern(monkeypatch, enabled=False)
    d = router.route_task(
        complexity_score=0.85,
        task_type="coding",
        prompt="refactor the parse loop",
        requires_tools=False,
    )
    assert d.reason == (
        "capability tier 10 needed, selected gpt-5.5 (tier 10)"
    ), d.reason


# ── Operator escape hatch ─────────────────────────────────────────────────────

def test_force_model_gpt55_is_honoured_despite_the_allowlist(clean_state, monkeypatch):
    """Operator intent beats the allowlist — the P0 reserve (§3.3).

    Naming a plan model explicitly is the escape hatch: the operator must never
    be blocked from the plan on demand.
    """
    _govern(monkeypatch)
    d = router.route_task(
        complexity_score=0.2,
        task_type="coding",
        prompt="refactor the parse loop",
        requires_tools=False,
        force_model="gpt-5.5",
    )
    assert d.selected_model == "gpt-5.5", d.reason
    assert d.plan_budget_denied is False, d.reason


# ── Regression guards (§5.5) ──────────────────────────────────────────────────

def test_flash_canary_unaffected_on_non_tool_path(clean_state, monkeypatch):
    """Simple routine work keeps routing to flash; the canary stays valid."""
    _govern(monkeypatch)
    d = router.route_task(
        complexity_score=0.1,
        task_type="coding",
        prompt="add a docstring",
        requires_tools=False,
    )
    assert router.get_model_account(d.selected_model) != "openai-codex", d.reason
    assert d.plan_budget_denied is False, d.reason


def test_astra_never_auto_selected_even_when_plan_open(clean_state, monkeypatch):
    """gpt-6-astra stays force_model-only with the governor on."""
    _govern(monkeypatch)
    d = router.route_task(
        complexity_score=1.0,
        task_type="planning",
        prompt="rearchitect and redesign the architecture for everything",
        requires_tools=False,
    )
    assert d.selected_model != "gpt-6-astra", d.reason


def test_future_plan_model_is_governed_by_account(clean_state, monkeypatch):
    """Governance keys on the ACCOUNT, so a new plan model needs no edit here.

    A synthetic registry entry on the governed account must be denied to routine
    without the governor having ever heard of that model name — this is what
    makes the policy survive a model refresh (luna/terra/sol all appeared after
    the governor's design).
    """
    import models

    _govern(monkeypatch)
    monkeypatch.setitem(router.MODEL_CAPABILITY_TIERS, "gpt-9-future", 10)
    monkeypatch.setitem(
        models.MODEL_REGISTRY,
        "gpt-9-future",
        {"provider": "openai-codex", "ratio": 40.0, "input": 20.0, "output": 80.0, "tier": 10},
    )
    router.MODEL_COST_ORDER.append("gpt-9-future")
    try:
        assert router.get_model_account("gpt-9-future") == "openai-codex"
        assert router._is_plan_model("gpt-9-future") is True
        assert router._plan_denied_for_class(gb.CLASS_ROUTINE) is True
    finally:
        router.MODEL_CAPABILITY_TIERS.pop("gpt-9-future", None)
        if "gpt-9-future" in router.MODEL_COST_ORDER:
            router.MODEL_COST_ORDER.remove("gpt-9-future")
