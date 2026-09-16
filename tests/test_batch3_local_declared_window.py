"""Batch 3 — declare local's real window so oversized work is not handed to it.

Why this exists
---------------
`llama-server` runs `-c 4096 --context-shift` (CPU-only, 1 slot), but Ollama's
own `/api/show` reports `llama.context_length = 131072` from the model's
training metadata. The router trusted that declared number, so compaction and
last-resort escalation handed local prompts far larger than it can hold — 28
requests over 4,096 tokens in a single day. `--context-shift` then silently
drops the *oldest* tokens, which is how a legitimate summary comes back
truncated or empty rather than as an obvious error.

Fix (Option A, chosen over raising the server window): record local's real
window as 4,096 in `MODEL_CONTEXT_CEILING` — the existing single source of
truth for per-model context limits — and refuse to route work that exceeds it.
No behaviour change for anything that fits.

The existing glm-5.3 ceiling is deliberately re-asserted here: it proves the
new entries extend that policy rather than replacing it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import router  # noqa: E402

LOCAL_MODELS = ("llama3.1:8b", "dolphin3")
OVERSIZED = 131_072  # the number Ollama wrongly advertised
SMALL = 1_000


def test_local_declares_its_real_window():
    """Local must be declared at 4,096, not Ollama's 131,072 metadata."""
    for model in LOCAL_MODELS:
        ceiling = router.MODEL_CONTEXT_CEILING.get(model)
        assert ceiling == 4096, (
            f"{model} declares {ceiling!r} — llama-server runs -c 4096, so "
            "anything larger is a window the model cannot honour"
        )


def test_glm53_ceiling_still_enforced():
    """Regression: the pre-existing per-model ceiling must survive."""
    assert router.MODEL_CONTEXT_CEILING.get("glm-5.3") == 150_000


def test_within_declared_window_helper():
    """The gate is reusable and treats 0/absent as 'no declared limit'."""
    assert router._within_declared_window("llama3.1:8b", SMALL) is True
    assert router._within_declared_window("llama3.1:8b", OVERSIZED) is False
    assert router._within_declared_window("glm-5.3", 100_000) is True
    assert router._within_declared_window("glm-5.3", 200_000) is False
    # Unknown/unlimited models are unaffected.
    assert router._within_declared_window("gpt-5.5", OVERSIZED) is True
    # A suffix must not dodge the declared window.
    assert router._within_declared_window("llama3.1:8b:cloud", OVERSIZED) is False


def test_fallback_chain_omits_local_for_oversized_context():
    """The chain must not offer local as an escape hatch for a huge prompt."""
    router.mark_available("llama3.1:8b")

    chain = router._build_fallback_chain("glm-5.3-flash", context_tokens=OVERSIZED)

    assert "llama3.1:8b" not in chain, (
        "an oversized request was offered local, which would silently shift "
        "away the oldest tokens"
    )


def test_fallback_chain_keeps_local_for_a_prompt_that_fits():
    """Nothing changes for work local can actually serve."""
    router.mark_available("llama3.1:8b")

    chain = router._build_fallback_chain("glm-5.3-flash", context_tokens=SMALL)

    assert "llama3.1:8b" in chain, (
        "local should remain the last resort for prompts within its window"
    )


def test_fallback_chain_default_preserves_existing_behaviour():
    """No context supplied (0) must not change any existing caller's chain."""
    router.mark_available("llama3.1:8b")

    chain = router._build_fallback_chain("glm-5.3-flash")

    assert "llama3.1:8b" in chain, (
        "default callers pass no context; their chain must be unchanged"
    )


def test_escalation_does_not_use_local_for_oversized_context():
    """Last-resort local must respect the declared window, and fail closed."""
    saved = dict(router._MODEL_STATUSES)
    try:
        # Every cloud model out: escalation's only remaining move is local.
        for m in router.MODEL_COST_ORDER:
            if m not in LOCAL_MODELS:
                router.mark_rate_limited(m, retry_after_s=3600, count_failure=False)
        router.mark_available("llama3.1:8b")

        decision = router.escalate_on_failure(
            "glm-5.3-flash",
            complexity_score=0.4,
            error_type="error",
            context_tokens=OVERSIZED,
        )

        assert decision.selected_model != "llama3.1:8b", (
            "escalation routed an oversized prompt to local as last resort"
        )
    finally:
        router._MODEL_STATUSES.clear()
        router._MODEL_STATUSES.update(saved)


def test_escalation_still_uses_local_for_a_prompt_that_fits():
    """The last-resort lane must remain reachable for work local can serve."""
    saved = dict(router._MODEL_STATUSES)
    try:
        for m in router.MODEL_COST_ORDER:
            if m not in LOCAL_MODELS:
                router.mark_rate_limited(m, retry_after_s=3600, count_failure=False)
        router.mark_available("llama3.1:8b")

        decision = router.escalate_on_failure(
            "glm-5.3-flash",
            complexity_score=0.4,
            error_type="error",
            context_tokens=SMALL,
        )

        assert decision.selected_model == "llama3.1:8b", (
            f"local last resort regressed: got {decision.selected_model!r}"
        )
    finally:
        router._MODEL_STATUSES.clear()
        router._MODEL_STATUSES.update(saved)


def test_endpoint_passes_context_into_the_chain():
    """The endpoint knows the request size; it must not drop it on the floor.

    Without this the chain is built as if every request were unbounded, which
    is how oversized work reached local in the first place.
    """
    src = (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()
    assert "context_tokens=" in src, (
        "endpoint never passes context_tokens, so the router cannot apply its "
        "declared-window policy to this request"
    )
