"""Batch 1 — a forced model must respect the account hold.

Why this exists
---------------
The account-wide cap short-circuit (B28) works: one `usage_limit_reached` holds
every model on the `openai-codex` account, and normal capability routing
correctly avoids them (measured: `route_task(normal_workload)` never returns a
held codex model).

But `route_task()`'s `force_model` branch early-returns a RoutingDecision for
the requested model with **no availability check at all**:

    if force_model:
        return RoutingDecision(
            selected_model=force_model,
            ...
            reason=f"forced model: {force_model}",
        )

and the endpoint turns ANY client-supplied model name that is not the router
sentinel into a `force_model` override:

    force_model = "" if requested_model in ("", "biggie-router", "biggie-llm") else requested_model

So a caller that sends `{"model": "gpt-5.5"}` reaches a model already held
83h out, re-hitting the same capped ChatGPT account. Live evidence: the account
cap fires ~1-3x/min while `get_available_models()` reports zero eligible codex
models — i.e. traffic is being dispatched by the force path, not by routing.

The hold must win. A forced model that is unavailable must NOT be selected;
the caller gets the same fail-closed signal as any other exhausted route
(empty `selected_model` -> the endpoint answers 503 all_models_exhausted),
rather than being sent into a wall.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import router  # noqa: E402


@pytest.fixture
def held_codex_account():
    """Hold the whole openai-codex account the way a real cap does."""
    saved = dict(router._MODEL_STATUSES)
    resets = int(time.time() + 300_000)  # ~83h, matching the live cap
    affected = router.mark_account_rate_limited("openai-codex", resets_at_s=resets)
    yield affected, resets
    router._MODEL_STATUSES.clear()
    router._MODEL_STATUSES.update(saved)


def test_forced_model_that_is_held_is_not_selected(held_codex_account):
    """A forced model under an account cap must not be returned as selected."""
    affected, _ = held_codex_account
    assert "gpt-5.5" in affected, "fixture must hold gpt-5.5"

    decision = router.route_task(force_model="gpt-5.5")

    assert decision.selected_model != "gpt-5.5", (
        "force_model returned a model held by an active account cap — the "
        "hold must dominate an explicit client override, or the capped "
        "account keeps being called (live: ~1-3 cap events/min)"
    )
    assert not router.is_model_available("gpt-5.5")


def test_forced_model_fails_closed_with_all_exhausted(held_codex_account):
    """Fail closed: no silent substitution, the caller is told it is exhausted."""
    affected, _ = held_codex_account
    assert "gpt-5.6-sol" in affected

    decision = router.route_task(force_model="gpt-5.6-sol")

    assert decision.selected_model == "", (
        f"expected fail-closed empty selection, got {decision.selected_model!r}"
    )
    assert decision.all_exhausted is True, (
        "an unavailable forced model must report all_exhausted so the endpoint "
        "returns 503 rather than dispatching into the cap"
    )


def test_forced_model_still_works_when_available():
    """The explicit lane must survive for a genuinely available model."""
    saved = dict(router._MODEL_STATUSES)
    try:
        router.mark_available("gpt-5.5")
        decision = router.route_task(force_model="gpt-5.5")
        assert decision.selected_model == "gpt-5.5"
        assert decision.all_exhausted is False
    finally:
        router._MODEL_STATUSES.clear()
        router._MODEL_STATUSES.update(saved)


def test_forced_local_model_unaffected_by_codex_account_cap(held_codex_account):
    """Holding one account must not break forcing a model on another.

    Establishes its own precondition: other tests in the suite leave local
    models held, so 'local is available' must be set up, not assumed — the
    guard under test fails closed on a genuinely held local, which is correct
    and would otherwise make this test depend on suite ordering.
    """
    router.mark_available("llama3.1:8b")
    decision = router.route_task(force_model="llama3.1:8b")
    assert decision.selected_model == "llama3.1:8b", (
        "local must remain forceable while the codex account is capped"
    )


def test_endpoint_does_not_force_a_held_codex_model(held_codex_account):
    """The endpoint's requested_model -> force_model path must not bypass the hold.

    This is the live dispatcher: any non-sentinel client model name becomes a
    force override. With the account capped, the router must not hand back a
    held codex model for such a request.
    """
    decision = router.route_task(force_model="gpt-5.6-terra")
    assert decision.selected_model != "gpt-5.6-terra", (
        "endpoint force path dispatched to a held codex model"
    )


def test_escalation_cannot_re_enter_a_held_family(held_codex_account):
    """Escalation must not climb back onto a capped account.

    Pins the existing safe behaviour: `escalate_on_failure` derives its
    candidate list from `get_available_models()`, which is filtered by
    `is_model_available()`, so a held family is unreachable by escalation.
    This is asserted rather than assumed because the force_model hole showed
    that a bypass is easy to introduce one branch away.
    """
    decision = router.escalate_on_failure(
        "deepseek-v4.1-flash", complexity_score=0.9, error_type="error",
    )
    codex = {"gpt-5.5", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra"}
    assert decision.selected_model not in codex, (
        f"escalation reached a capped codex model: {decision.selected_model!r}"
    )
    assert not (set(decision.fallback_chain) & codex), (
        f"capped codex model present in fallback chain: {decision.fallback_chain}"
    )

