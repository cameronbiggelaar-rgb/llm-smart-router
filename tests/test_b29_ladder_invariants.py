"""B29 — invariants that keep the summariser ladder honest.

These pin the properties the fix depends on, so a later policy edit cannot
silently undo it:

* the ladder never contains a ChatGPT-plan model (item 4);
* escalation never lands on the plan, even when every curated rung is down;
* total exhaustion fails closed with an empty selection rather than promoting;
* the import-time validator rejects unregistered or reserved rungs.
"""

import os
import sys
import time

import pytest

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(SKILL_DIR, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import router  # noqa: E402
from router import ModelStatus  # noqa: E402


@pytest.fixture()
def clean_statuses():
    """Snapshot and restore model status so tests cannot leak health state."""
    saved = dict(router._MODEL_STATUSES)
    yield
    router._MODEL_STATUSES.clear()
    router._MODEL_STATUSES.update(saved)


def test_ladder_contains_no_chatgpt_plan_model():
    """Item 4: the plan must never be a summariser rung."""
    for ctx in (0, 50_000, 120_000, 400_000):
        ladder = router._session_compression_ladder(ctx)
        leaked = [m for m in ladder if router._is_reserved_from_compression(m)]
        assert leaked == [], f"plan model(s) in summariser ladder at {ctx} tokens: {leaked}"


def test_ladder_is_never_empty():
    assert router._session_compression_ladder(0)
    assert router._session_compression_ladder(120_000)


def test_all_ladder_entries_are_registered_models():
    from models import MODEL_REGISTRY

    for model in router._session_compression_ladder(0):
        assert model in MODEL_REGISTRY, f"{model} is not in MODEL_REGISTRY"


def test_reserved_set_covers_whole_gpt5_family():
    """Item 4 is a policy over the whole plan, not a single hardcoded model."""
    for model in ("gpt-5.5", "gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        assert router._is_reserved_from_compression(model), model


def test_every_chatgpt_plan_model_is_reserved_by_provider():
    """The durable form of item 4: reservation follows the *provider*.

    A new plan model added to MODEL_REGISTRY must be excluded from compression
    automatically — the hand-maintained name list is not the mechanism.
    """
    from models import MODEL_REGISTRY

    plan_models = [
        name for name, cfg in MODEL_REGISTRY.items()
        if cfg["provider"] in router.COMPRESSION_RESERVED_PROVIDERS
    ]
    assert plan_models, "expected at least one plan model in the registry"
    for model in plan_models:
        assert router._is_reserved_from_compression(model), (
            f"{model} draws on a reserved provider but is eligible for compression"
        )


def test_non_plan_models_are_not_reserved():
    for model in ("deepseek-v4.1-flash", "glm-5.3", "qwen3.5", "deepseek-v3.1:671b"):
        assert not router._is_reserved_from_compression(model), model


def test_unregistered_plan_alias_is_reserved():
    """An unregistered plan-family name (reachable via force_model) is still
    excluded — the prefix rule closes the gap the provider rule cannot see."""
    for model in ("gpt-5.6", "gpt-6", "gpt-6.1", "gpt-5.5-pro"):
        assert router._is_reserved_from_compression(model), model


def test_escalation_never_lands_on_plan_when_all_rungs_down(clean_statuses):
    """Even a total curatated-ladder outage must not promote to the plan."""
    ladder = router._session_compression_ladder(120_000)
    assert ladder, "expected a non-empty ladder for this test to be meaningful"
    for model in ladder:
        router._MODEL_STATUSES[model] = ModelStatus(
            model=model, available=False, consecutive_failures=99
        )

    decision = router.escalate_on_failure(
        failed_model="deepseek-v4.1-flash:cloud",
        complexity_score=0.9,
        error_type="empty_content",
        requires_tools=False,
        workload_type="session_compression",
        context_tokens=120_000,
    )
    if decision.selected_model:
        assert not router._is_reserved_from_compression(decision.selected_model)


def test_total_exhaustion_fails_closed(clean_statuses, monkeypatch):
    """No summariser at all -> empty selection (fail closed), not a promotion.

    ``_check_limp_home`` runs a recovery probe before escalation, so simply
    flagging models unavailable is not enough: the probe is stubbed out to model
    a genuine outage (every upstream still down), which is the condition under
    which fail-closed must hold.
    """
    monkeypatch.setattr(router, "probe_all_recovering", lambda *a, **k: None)
    for status in router._MODEL_STATUSES.values():
        status.available = False
        status.rate_limited_until = time.time() + 3600
        status.circuit_open_until = time.time() + 3600

    decision = router.escalate_on_failure(
        failed_model="deepseek-v4.1-flash:cloud",
        complexity_score=0.9,
        error_type="empty_content",
        requires_tools=False,
        workload_type="session_compression",
        context_tokens=120_000,
    )
    assert decision.selected_model == ""
    assert decision.all_exhausted is True
    assert not router._is_reserved_from_compression(decision.selected_model or "")


def test_escalation_advances_along_the_ladder(clean_statuses):
    """A failed rung retries the *next* rung, not a more capable model."""
    ladder = router._session_compression_ladder(0)
    first = ladder[0]
    for model in ladder:
        if model != first:
            router._MODEL_STATUSES[model] = ModelStatus(
                model=model, available=False, consecutive_failures=99
            )

    decision = router.escalate_on_failure(
        failed_model=first,
        complexity_score=0.9,
        error_type="empty_content",
        requires_tools=True,  # even a tool-flagged compaction stays on the ladder
        workload_type="session_compression",
        context_tokens=0,
    )
    assert decision.selected_model != first
    assert not router._is_reserved_from_compression(decision.selected_model)
    assert decision.selected_model in ladder


def test_validator_rejects_reserved_rung(monkeypatch):
    """A future edit adding a plan model to the ladder fails loudly."""
    monkeypatch.setattr(
        router,
        "_session_compression_ladder",
        lambda ctx=0: ["deepseek-v4-flash", "gpt-5.5"],
    )
    with pytest.raises(AssertionError, match="reserved"):
        router._validate_curated_ladder()


def test_validator_rejects_unregistered_rung(monkeypatch):
    monkeypatch.setattr(
        router,
        "_session_compression_ladder",
        lambda ctx=0: ["deepseek-v4-flash", "no-such-model"],
    )
    with pytest.raises(AssertionError, match="unregistered"):
        router._validate_curated_ladder()


def test_validator_rejects_empty_ladder(monkeypatch):
    monkeypatch.setattr(router, "_session_compression_ladder", lambda ctx=0: [])
    with pytest.raises(AssertionError, match="empty"):
        router._validate_curated_ladder()


def test_tool_flagged_compression_still_escalates_on_ladder(clean_statuses):
    """Regression: requires_tools=True must not win over the compression branch.

    This is the exact overnight defect — a compaction request carried the parent
    session's tool schemas and was promoted onto the ChatGPT plan.
    """
    decision = router.escalate_on_failure(
        failed_model="deepseek-v4.1-flash:cloud",
        complexity_score=1.0,
        error_type="empty_content",
        requires_tools=True,
        workload_type="session_compression",
        context_tokens=150_000,
    )
    if decision.selected_model:
        assert not router._is_reserved_from_compression(decision.selected_model)
    assert "gpt-5" not in (decision.selected_model or "")
