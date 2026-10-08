"""Batch B45a (RED) — entitlement is the explicit ask, not the classifier.

Why this module exists
----------------------
B44 closed the governor's silent fail-open, but left the entitlement rule as
"architectural class ⇒ entitled on the plan". The classifier behind that rule is
a bare substring match, so ordinary phrasing classifies architectural:

    "rethink the wording of this email"  -> architectural
    "rearchitect my resume"              -> architectural
    "update the plans for the weekend"   -> architectural

Tightening the keywords would fix those examples, not the class of defect: any
substring heuristic can be tripped by ordinary language. The operator's
direction (2026-10-08) is to remove the classifier from the trust path entirely
— make the EXPLICIT ASK the entitlement, so a false positive can no longer admit
a request to the plan.

Policy under test (``explicit_only``):

* ``true``  → entitled iff the request explicitly named a plan model. The
  allowlist is moot for automatic paths.
* ``false`` → today's behaviour (``klass in allowlist``). Default, so the change
  is opt-in and reversible by config alone.

Plan: references/plan-b45-gpt-by-request-only.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import gpt_budget as gb  # noqa: E402

ARCH = gb.CLASS_ARCHITECTURAL
ROUTINE = gb.CLASS_ROUTINE
OPEN = gb.VERDICT_OPEN
CLOSED = gb.VERDICT_CLOSED
DENIED = gb.VERDICT_DENIED
UNVERIFIED = gb.VERDICT_UNVERIFIED


def _reading(*, used: float, allowed: bool = True, limit: int = 604800):
    return gb.Reading(
        source="live",
        allowed=allowed,
        primary=gb.WindowReading(
            used_percent=used,
            limit_window_seconds=limit,
            reset_after_seconds=int(limit * (1.0 - used / 100.0)),
            reset_at=0,
        ),
        secondary=None,
    )


def _blind():
    """The probe could not be read — the leak's own escape route (B44)."""
    return gb.Reading(source="unavailable", allowed=None, primary=None, secondary=None)


def _cfg(**overrides):
    defaults = dict(
        enabled=True,
        allowlist=[ARCH],
        routine_ceiling_percent=0,
        hard_cap_percent=95,
    )
    defaults.update(overrides)
    return gb.GovernorConfig(**defaults)


# ── The policy knob must exist and default to OFF ───────────────────────────


def test_explicit_only_defaults_to_off():
    """Opt-in: a GovernorConfig built without the flag keeps B44 behaviour."""
    assert _cfg().explicit_only is False


def test_explicit_only_is_read_from_the_policy_file():
    """The knob is operator-tunable, like every other governor setting."""
    text = (
        "governor:\n"
        "  enabled: true\n"
        "  accounts:\n"
        "    openai-codex:\n"
        "      explicit_only: true\n"
        "      allowlist: [architectural]\n"
        "      hard_cap_percent: 95\n"
    )
    cfg = gb._parse_policy(text)
    assert cfg is not None
    assert cfg.explicit_only is True


def test_missing_knob_parses_to_off_not_on():
    """A policy file predating B45 must not silently become explicit-only."""
    text = (
        "governor:\n"
        "  enabled: true\n"
        "  accounts:\n"
        "    openai-codex:\n"
        "      allowlist: [architectural]\n"
        "      hard_cap_percent: 95\n"
    )
    cfg = gb._parse_policy(text)
    assert cfg is not None
    assert cfg.explicit_only is False


# ── The point of B45: classification alone grants nothing ───────────────────


def test_auto_architectural_is_denied_under_explicit_only():
    """THE point. A classifier false positive must not reach the plan."""
    verdict = gb.evaluate_policy(
        _reading(used=10.0), _cfg(explicit_only=True), klass=ARCH, explicit=False
    )
    assert verdict == DENIED


def test_explicit_request_is_served_under_explicit_only():
    """The escape hatch survives: naming a plan model is always honoured."""
    verdict = gb.evaluate_policy(
        _reading(used=10.0), _cfg(explicit_only=True), klass=ARCH, explicit=True
    )
    assert verdict == OPEN


def test_explicit_request_is_served_even_when_class_is_routine():
    """Entitlement is the ASK, not the class — an explicit routine ask is served."""
    verdict = gb.evaluate_policy(
        _reading(used=10.0), _cfg(explicit_only=True), klass=ROUTINE, explicit=True
    )
    assert verdict == OPEN


def test_auto_routine_is_denied_under_explicit_only():
    verdict = gb.evaluate_policy(
        _reading(used=10.0), _cfg(explicit_only=True), klass=ROUTINE, explicit=False
    )
    assert verdict == DENIED


# ── A blind probe must not become the way in ─────────────────────────────────


def test_blind_probe_does_not_admit_a_non_explicit_request():
    """B44's UNVERIFIED is for entitled work.

    Under explicit_only a non-explicit request is not entitled, so the blind
    probe must not be a route onto the plan — that is the leak B44 closed,
    approached from the other side.
    """
    verdict = gb.evaluate_policy(_blind(), _cfg(explicit_only=True), klass=ARCH, explicit=False)
    assert verdict == DENIED


def test_blind_probe_flags_an_explicit_request_served():
    """An explicit ask is served, but the unverifiable window is still reported."""
    verdict = gb.evaluate_policy(_blind(), _cfg(explicit_only=True), klass=ARCH, explicit=True)
    assert verdict == UNVERIFIED


# ── Capacity and rollback still outrank entitlement ──────────────────────────


def test_explicit_request_is_refused_at_the_hard_cap():
    """The reactive backstop applies to every class, explicit or not."""
    verdict = gb.evaluate_policy(
        _reading(used=99.0), _cfg(explicit_only=True), klass=ARCH, explicit=True
    )
    assert verdict == CLOSED


def test_explicit_request_is_refused_when_the_account_says_no():
    verdict = gb.evaluate_policy(
        _reading(used=10.0, allowed=False),
        _cfg(explicit_only=True),
        klass=ARCH,
        explicit=True,
    )
    assert verdict == CLOSED


def test_disabled_governor_opens_everything():
    """The rollback lever outranks explicit_only."""
    cfg = _cfg(explicit_only=True, enabled=False)
    for klass in (ARCH, ROUTINE):
        for explicit in (True, False):
            assert gb.evaluate_policy(
                _reading(used=99.0, allowed=False), cfg, klass=klass, explicit=explicit
            ) == OPEN


# ── Regression: explicit_only=false keeps exactly the B44 truth table ───────


def test_allowlist_still_governs_when_explicit_only_is_off():
    cfg = _cfg(explicit_only=False)
    assert gb.evaluate_policy(_reading(used=10.0), cfg, klass=ARCH, explicit=False) == OPEN
    assert gb.evaluate_policy(_reading(used=10.0), cfg, klass=ROUTINE, explicit=False) == DENIED


# ── Routing under explicit_only (plan cases 7, 8, 10) ────────────────────────
#
# The policy layer above proves the entitlement rule; these prove the routing
# layer honours it end-to-end through ``route_task`` and ``escalate_on_failure``,
# so a decision can never disagree with the policy.

import router as _router  # noqa: E402

CODEX_MODELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra"]
OLLAMA_MODELS = ["glm-5.3", "deepseek-v4-pro", "deepseek-v4.1-flash"]


@pytest.fixture
def clean_state():
    saved = {
        m: _router._MODEL_STATUSES.get(m)
        for m in CODEX_MODELS + OLLAMA_MODELS + ["llama3.1:8b"]
    }
    for m in CODEX_MODELS + OLLAMA_MODELS + ["llama3.1:8b"]:
        _router._MODEL_STATUSES[m] = _router.ModelStatus(model=m, available=True)
    yield
    for m, st in saved.items():
        if st is None:
            _router._MODEL_STATUSES.pop(m, None)
        else:
            _router._MODEL_STATUSES[m] = st


def _govern(monkeypatch, used_pct=0.0, allowed=True, source="live"):
    """Install a governor with ``explicit_only=True`` and a stubbed reading."""
    cfg = _cfg(explicit_only=True)
    monkeypatch.setattr(gb, "load_governor_config", lambda force=False: cfg)
    monkeypatch.setattr(
        gb,
        "fetch_reading",
        lambda cfg=None, **kw: _reading(used=used_pct, allowed=allowed),
    )


def test_auto_architectural_prompt_cannot_reach_a_plan_model(clean_state, monkeypatch):
    """Plan case 7: the whole point of B45 — classification stops granting.

    A prompt that classifies architectural (substring match) but does NOT name a
    plan model must not reach the plan, even at the top of the complexity range.
    """
    _govern(monkeypatch)
    d = _router.route_task(
        complexity_score=0.99,
        task_type="planning",
        prompt="rearchitect the budget governor from first principles",
        requires_tools=False,
    )
    assert _router.get_model_account(d.selected_model) != "openai-codex", d.reason


def test_explicit_force_model_reaches_plan_and_is_recorded(clean_state, monkeypatch):
    """Plan case 8: naming a plan model is the entitlement and is visible."""
    _govern(monkeypatch)
    d = _router.route_task(
        complexity_score=0.5,
        task_type="qa",
        prompt="summarise this email",
        requires_tools=False,
        force_model="gpt-5.5",
    )
    assert d.selected_model == "gpt-5.5", d.reason
    assert d.plan_explicit_request is True, d.reason
    assert d.budget_class != "", d.reason


def test_explicit_escalation_with_empty_class_may_reach_plan(clean_state, monkeypatch):
    """Plan case 10: explicit ask survives escalation even with a blank class."""
    _govern(monkeypatch)
    d = _router.escalate_on_failure(
        failed_model="deepseek-v4-pro",
        error_type="timeout",
        requires_tools=True,
        plan_explicit=True,
    )
    assert d.selected_model, d.reason
    assert _router.get_model_account(d.selected_model) == "openai-codex", d.reason
