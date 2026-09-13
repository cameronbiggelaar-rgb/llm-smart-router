"""B28 batch 2b — escalation must respect account caps.

Two behaviours that together turn a 75-failure cascade into a single event:
  1. escalation never selects another model on the SAME capped account
  2. a rate_limit escalation never SHORTENS an authoritative cooldown
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import models  # noqa: E402
import router  # noqa: E402

CODEX_MODELS = router.models_in_account("openai-codex")


@pytest.fixture()
def clean_state(monkeypatch, tmp_path):
    """Isolate model state + persisted DB writes."""
    monkeypatch.setattr(router, "_MODEL_STATUSES", {
        m: router.ModelStatus(model=m) for m in models.MODEL_REGISTRY
    }, raising=False)
    monkeypatch.setattr(router, "_save_model_state", lambda *a, **k: None,
                        raising=False)
    monkeypatch.setattr(router, "_log_recovery_event", lambda *a, **k: None,
                        raising=False)
    monkeypatch.setattr(router, "_LIMP_HOME_ACTIVE", False, raising=False)
    yield


def test_escalation_never_picks_a_model_on_the_capped_account(clean_state):
    """gpt-5.5 caps -> escalation must leave the openai-codex account entirely."""
    router.mark_account_rate_limited("openai-codex", resets_at_s=time.time() + 3600)

    decision = router.escalate_on_failure(
        failed_model="gpt-5.5", complexity_score=0.9, error_type="rate_limit",
    )

    assert decision.selected_model, "escalation found nothing at all"
    assert router.get_model_account(decision.selected_model) != "openai-codex", (
        f"escalated to {decision.selected_model}, still on the capped account"
    )


def test_escalation_skips_every_capped_sibling(clean_state):
    """No sibling of a capped account may appear in an escalation chain."""
    router.mark_account_rate_limited("openai-codex", resets_at_s=time.time() + 3600)
    decision = router.escalate_on_failure(
        failed_model="gpt-5.5", complexity_score=0.9, error_type="rate_limit",
    )
    assert all(router.get_model_account(m) != "openai-codex"
               for m in decision.fallback_chain), decision.fallback_chain


def test_rate_limit_escalation_does_not_shorten_upstream_cooldown(clean_state):
    """A correct multi-hour upstream hold must survive a rate_limit escalation."""
    far_future = time.time() + 4 * 3600
    router.mark_rate_limited("gpt-5.5", resets_at_s=far_future)
    before = router._MODEL_STATUSES["gpt-5.5"].rate_limited_until

    router.escalate_on_failure(
        failed_model="gpt-5.5", complexity_score=0.9, error_type="rate_limit",
    )

    after = router._MODEL_STATUSES["gpt-5.5"].rate_limited_until
    assert after >= before - 1, (
        f"cooldown was shortened: {before} -> {after}"
    )


def test_account_cap_does_not_inflate_sibling_failure_counts(clean_state):
    """The 75-failure cascade: one account cap must not count 5 failures."""
    router.mark_account_rate_limited("openai-codex", resets_at_s=time.time() + 3600)
    for m in CODEX_MODELS:
        assert router._MODEL_STATUSES[m].consecutive_failures == 0, m
