"""Batch B37g (RED) — escalation inherits the originating request's class.

Why this module exists
----------------------
Escalation is tool-failover, not architecture (design §3.1). A gpt call that
returned a 429 is evidence about *availability*, never about the nature of the
work — so an escalation must inherit the class of the request that spawned it,
it must not be re-classified from the error.

That inheritance is what keeps the leak shut under failure. Both escalation
ladders in `escalate_on_failure` pick the MOST CAPABLE model available:

* the tool ladder (``best_tier``) — would pick ``gpt-5.6-sol`` from tier 13;
* the capability ladder (``tier > failed_tier``) — walks UP, so a routine
  request failing on ``deepseek-v4-pro`` (tier 8) escalates straight into the
  plan at tier 10.

So a routine request that fails once re-enters the plan through failover unless
the originating class is threaded down and enforced. The mirror requirement
matters just as much: a failed ARCHITECTURAL request must still be able to
escalate into the plan — that is the lane the plan exists for, and starving it
on its first hiccup would defeat the whole change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import gpt_budget as gb  # noqa: E402
import router  # noqa: E402

CODEX_MODELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra"]
OLLAMA_MODELS = ["glm-5.3", "deepseek-v4-pro", "deepseek-v4.1-flash"]


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
            used_percent=used_pct, limit_window_seconds=18000, reset_after_seconds=165300
        ),
        secondary=gb.WindowReading(
            used_percent=used_pct, limit_window_seconds=604800, reset_after_seconds=165300
        ),
    )


@pytest.fixture
def governor_open(monkeypatch):
    """Governor armed, plan healthy at 0% — architectural would be allowed."""
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
    monkeypatch.setattr(gb, "fetch_reading", lambda cfg=None, **kw: _reading(0.0))


# ── The leak under failure ────────────────────────────────────────────────────

def test_routine_tool_escalation_does_not_land_on_plan(clean_state, governor_open):
    """A failed routine tool request must not fail over INTO the plan.

    The tool ladder picks the most-capable tool model, which is gpt-5.6-sol.
    Inheritance must keep the routine class off it.
    """
    d = router.escalate_on_failure(
        failed_model="deepseek-v4-pro",
        error_type="timeout",
        requires_tools=True,
        budget_class=gb.CLASS_ROUTINE,
    )
    assert d.selected_model, d.reason
    assert router.get_model_account(d.selected_model) != "openai-codex", d.reason


def test_routine_non_tool_escalation_does_not_land_on_plan(clean_state, governor_open):
    """The capability ladder walks up in tier; routine must not follow it up."""
    d = router.escalate_on_failure(
        failed_model="deepseek-v4-pro",
        error_type="timeout",
        requires_tools=False,
        budget_class=gb.CLASS_ROUTINE,
    )
    assert d.selected_model, d.reason
    assert router.get_model_account(d.selected_model) != "openai-codex", d.reason


def test_routine_escalation_reports_its_class(clean_state, governor_open):
    """The decision must carry the inherited class for observability."""
    d = router.escalate_on_failure(
        failed_model="deepseek-v4-pro",
        error_type="timeout",
        requires_tools=False,
        budget_class=gb.CLASS_ROUTINE,
    )
    assert d.budget_class == gb.CLASS_ROUTINE, d.reason


def test_routine_escalation_chain_contains_no_plan_model(clean_state, governor_open):
    """The escalated decision's own chain must be contained too."""
    d = router.escalate_on_failure(
        failed_model="deepseek-v4-pro",
        error_type="timeout",
        requires_tools=False,
        budget_class=gb.CLASS_ROUTINE,
    )
    for model in d.fallback_chain:
        assert router.get_model_account(model) != "openai-codex", d.fallback_chain


# ── Inheritance, not re-classification ───────────────────────────────────────

def test_architectural_escalation_still_reaches_the_plan(clean_state, governor_open):
    """A failed architectural request MAY escalate into the plan (§9.2).

    Starving the protected lane on its first hiccup would defeat the change.
    """
    d = router.escalate_on_failure(
        failed_model="gpt-5.5",
        error_type="timeout",
        requires_tools=False,
        budget_class=gb.CLASS_ARCHITECTURAL,
    )
    assert d.selected_model, d.reason
    assert d.budget_class == gb.CLASS_ARCHITECTURAL, d.reason


def test_escalation_does_not_re_classify_from_the_error(clean_state, governor_open):
    """Escalation is tool-failover, not architecture (§3.1).

    A rate_limit error on a plan model is evidence about availability, not about
    the nature of the work: the class must be inherited from the argument, not
    inferred from the failure.
    """
    d = router.escalate_on_failure(
        failed_model="gpt-5.6-sol",
        error_type="rate_limit",
        requires_tools=True,
        budget_class=gb.CLASS_ROUTINE,
    )
    assert d.budget_class == gb.CLASS_ROUTINE, d.reason
    assert router.get_model_account(d.selected_model) != "openai-codex", d.reason


# ── Backward compatibility ───────────────────────────────────────────────────

def test_empty_class_non_explicit_cannot_reach_plan(clean_state, governor_open):
    """No class supplied (legacy caller) ⇒ NOT entitled, never reaches the plan.

    The CLI path (``router.py escalate --fail …``) and older callers pass no
    class. Under B45 a missing class is not entitled — that empty class was the
    431-row hole (escalations reaching ``gpt-5.6-sol`` with a blank class), so
    escalation must not land on a plan model.
    """
    d = router.escalate_on_failure(
        failed_model="deepseek-v4-pro",
        error_type="timeout",
        requires_tools=True,
    )
    assert d.budget_class == "", d.reason
    assert router.get_model_account(d.selected_model) != "openai-codex", d.reason


def test_governor_disabled_preserves_escalation_behaviour(clean_state, monkeypatch):
    """With the governor off (rollback), escalation is unchanged."""
    monkeypatch.setattr(
        gb,
        "load_governor_config",
        lambda force=False: gb.GovernorConfig(
            enabled=False, allowlist=[gb.CLASS_ARCHITECTURAL],
            routine_ceiling_percent=0, hard_cap_percent=95.0,
        ),
    )
    monkeypatch.setattr(gb, "fetch_reading", lambda cfg=None, **kw: _reading(0.0))
    d = router.escalate_on_failure(
        failed_model="deepseek-v4-pro",
        error_type="timeout",
        requires_tools=True,
        budget_class=gb.CLASS_ROUTINE,
    )
    assert d.selected_model == "gpt-5.6-sol", d.reason


# ── No empty decision ────────────────────────────────────────────────────────

def test_routine_escalation_fails_closed_when_only_plan_models_remain(
    clean_state, governor_open
):
    """Only plan models left ⇒ fail closed, never leak (§5.1 no empty decision).

    This is the escalation analogue of the routing-path test: the governor must
    report exhaustion honestly rather than quietly using the reserved plan.
    """
    for m in ("deepseek-v4-pro", "glm-5.3", "deepseek-v4.1-flash", "glm-5.2", "glm-5.1"):
        st = router._MODEL_STATUSES.setdefault(m, router.ModelStatus(model=m, available=True))
        st.available = False
        st.rate_limited_until = 10**12
    d = router.escalate_on_failure(
        failed_model="deepseek-v4-pro",
        error_type="timeout",
        requires_tools=True,
        budget_class=gb.CLASS_ROUTINE,
    )
    assert d.selected_model == "", d.reason
    assert d.all_exhausted is True, d.reason


def test_prefer_model_cannot_smuggle_routine_onto_the_plan(clean_state, governor_open):
    """The canary's declared fallback must not override the class (§5.5).

    `prefer_model` exists so a canary miss lands on its declared fallback. If an
    experiment were ever configured with a plan model as its fallback, a routine
    request would be routed straight onto the reserved plan by preference.
    """
    d = router.escalate_on_failure(
        failed_model="deepseek-v4-pro",
        error_type="timeout",
        requires_tools=True,
        prefer_model="gpt-5.5",
        budget_class=gb.CLASS_ROUTINE,
    )
    assert router.get_model_account(d.selected_model) != "openai-codex", d.reason


def test_prefer_model_still_honoured_for_architectural(clean_state, governor_open):
    """Containment must not break the legitimate preference path."""
    d = router.escalate_on_failure(
        failed_model="deepseek-v4-pro",
        error_type="timeout",
        requires_tools=True,
        prefer_model="gpt-5.5",
        budget_class=gb.CLASS_ARCHITECTURAL,
    )
    assert d.selected_model == "gpt-5.5", d.reason
