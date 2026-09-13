"""B28 batch 2c — an account cap must hold the failing model itself.

Regression: account grouping holds registered siblings, but a backend name
that is not a registry member (new/renamed model) was left available, so the
next request walked straight back into the same cap.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402
import router  # noqa: E402


@pytest.fixture
def unknown_model():
    model = "brand-new-unregistered-model"
    router._MODEL_STATUSES[model] = router.ModelStatus(model=model)
    yield model
    router._MODEL_STATUSES.pop(model, None)


CODEX_CAP_BODY = (
    '{"error": {"message": "usage limit reached", "code": "usage_limit_reached", '
    '"plan_type": "plus", "resets_at": %d}}'
)


def test_unregistered_model_is_held_on_account_cap(unknown_model):
    resets = int(time.time() + 7200)
    ep._handle_backend_status(
        "openai-codex", unknown_model, 429, error_text=CODEX_CAP_BODY % resets,
    )
    st = router._MODEL_STATUSES[unknown_model]
    assert not st.available, "unregistered model was left available under an account cap"
    assert st.rate_limited_until and abs(st.rate_limited_until - resets) < 2
    assert not router.is_model_available(unknown_model)


def test_unregistered_model_held_even_without_reset_time(unknown_model):
    """Account cap with no reset hint still holds the failing model."""
    body = '{"error": {"code": "insufficient_quota"}}'
    ep._handle_backend_status("openai-codex", unknown_model, 429, error_text=body)
    assert not router._MODEL_STATUSES[unknown_model].available


def test_registered_model_is_not_double_marked(unknown_model):
    """A registry member is held once by the group, not re-marked as a failure."""
    resets = int(time.time() + 3600)
    ep._handle_backend_status(
        "openai-codex", "gpt-5.5", 429, error_text=CODEX_CAP_BODY % resets,
    )
    st = router._MODEL_STATUSES["gpt-5.5"]
    assert st.consecutive_failures == 0
    assert st.rate_limited_until and abs(st.rate_limited_until - resets) < 2
