"""B28 batch 2 — account-level (billing) rate limiting.

A 429 that says "this ACCOUNT is capped" must take down every model sharing
that account, not just the one that happened to fail first. Otherwise the
escalation ladder walks the family (gpt-5.5 -> 5.6-luna -> terra -> sol) and
burns each one against the same wall — gpt-5.5 reached 75 consecutive
failures on 2026-09-13 for exactly this reason.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402
import models  # noqa: E402
import router  # noqa: E402

CODEX_MODELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra"]
OLLAMA_MODELS = ["glm-5.3", "deepseek-v4-pro", "qwen3.5"]


@pytest.fixture
def clean_state():
    saved = {m: router._MODEL_STATUSES.get(m) for m in CODEX_MODELS + OLLAMA_MODELS + ["llama3.1:8b"]}
    for m in CODEX_MODELS + OLLAMA_MODELS:
        router._MODEL_STATUSES[m] = router.ModelStatus(model=m)
    yield
    for m, st in saved.items():
        if st is None:
            router._MODEL_STATUSES.pop(m, None)
        else:
            router._MODEL_STATUSES[m] = st


def test_account_group_maps_models_to_billing_accounts():
    assert router.get_model_account("gpt-5.5") == "openai-codex"
    assert router.get_model_account("gpt-5.6-sol") == "openai-codex"
    assert router.get_model_account("glm-5.3") == "ollama-cloud"
    # local models have no quota-bearing account
    assert router.get_model_account("llama3.1:8b") == ""


def test_account_cap_marks_every_model_in_the_account(clean_state):
    soon = time.time() + 7200
    router.mark_account_rate_limited("openai-codex", resets_at_s=soon)
    for m in CODEX_MODELS:
        st = router._MODEL_STATUSES[m]
        assert st.rate_limited_until is not None, m
        assert abs(st.rate_limited_until - soon) < 2, m
        assert not router.is_model_available(m), m


def test_account_cap_does_not_touch_other_accounts(clean_state):
    router.mark_account_rate_limited("openai-codex", resets_at_s=time.time() + 7200)
    for m in OLLAMA_MODELS:
        st = router._MODEL_STATUSES[m]
        assert not st.rate_limited_until, m
        assert st.available, m
        assert router.is_model_available(m), m


def test_local_models_are_never_group_marked(clean_state):
    router.mark_account_rate_limited("ollama-cloud", resets_at_s=time.time() + 7200)
    assert router.is_model_available("llama3.1:8b")


def test_account_cap_detector_recognises_both_upstream_shapes():
    # ChatGPT Plus account-wide cap
    codex = ('{"error": {"message": "You have hit your usage limit", '
             '"code": "usage_limit_reached", "plan_type": "plus", "resets_at": 1789316836}}')
    assert ep._extract_429_account_cap(codex) is True

    # ollama billing cap
    ollama = ('{"error": {"message": "usage credits auto reload monthly max reached, '
              'increase your monthly max or add usage credits"}}')
    assert ep._extract_429_account_cap(ollama) is True


def test_account_cap_detector_ignores_per_model_throttles():
    # A plain per-model TPM throttle is NOT an account cap.
    assert ep._extract_429_account_cap('{"error": {"message": "rate limit exceeded"}}') is False
    assert ep._extract_429_account_cap("") is False


def test_codex_account_cap_429_takes_down_the_whole_family_end_to_end(clean_state):
    """The regression that burned gpt-5.5 to 75 failures."""
    resets = int(time.time() + 14400)
    body = ('{"error": {"message": "usage limit reached", "code": "usage_limit_reached", '
            f'"plan_type": "plus", "resets_at": {resets}}}}}')
    ep._handle_backend_status("openai-codex", "gpt-5.5", 429, error_text=body)
    # Every sibling is unavailable, so escalation cannot walk into the same wall.
    for m in CODEX_MODELS:
        assert not router.is_model_available(m), m


def test_per_model_throttle_still_only_marks_that_model(clean_state):
    ep._handle_backend_status(
        "openai-codex", "gpt-5.5", 429,
        error_text='{"error": {"message": "rate limit exceeded"}}',
    )
    assert not router.is_model_available("gpt-5.5")
    assert router.is_model_available("gpt-5.6-sol")


def test_account_group_derived_from_registry_has_no_drift():
    """Every quota-bearing provider in the registry must have a group."""
    for name, cfg in models.MODEL_REGISTRY.items():
        acct = router.get_model_account(name)
        if cfg["provider"] in models.ACCOUNT_QUOTA_PROVIDERS:
            assert acct == cfg["provider"], name
        else:
            assert acct == "", name
