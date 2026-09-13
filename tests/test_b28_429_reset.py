"""B28 — 429 reset-time parsing, account grouping, finish_reason-aware empty guard.

Batch 1: mark_rate_limited honours an explicit upstream reset time.
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
def isolated_model():
    """Use a throwaway model name so state mutations never touch real models."""
    model = "b28-probe-model"
    status = router.ModelStatus(model=model)
    router._MODEL_STATUSES[model] = status
    yield model
    router._MODEL_STATUSES.pop(model, None)


def _call_mark_rate_limited(model, **kwargs):
    # Capture the cooldown that would be applied by reading state after the call.
    router.mark_rate_limited(model, **kwargs)
    return router._MODEL_STATUSES[model]


def test_explicit_retry_after_sets_exact_cooldown(isolated_model):
    status = router.ModelStatus(model=isolated_model)
    router._MODEL_STATUSES[isolated_model] = status
    now = time.time()
    router.mark_rate_limited(isolated_model, retry_after_s=37)
    got = router._MODEL_STATUSES[isolated_model].rate_limited_until
    # Must honour the explicit 37s, not exponential backoff (60s at failure #1).
    assert abs((got - now) - 37) < 1.0


def test_explicit_resets_at_epoch_sets_exact_cooldown(isolated_model):
    router._MODEL_STATUSES[isolated_model] = router.ModelStatus(model=isolated_model)
    now = time.time()
    resets_at = now + 500
    router.mark_rate_limited(isolated_model, resets_at_s=resets_at)
    got = router._MODEL_STATUSES[isolated_model].rate_limited_until
    assert abs((got - now) - 500) < 1.0


def test_no_explicit_time_falls_back_to_backoff(isolated_model):
    router._MODEL_STATUSES[isolated_model] = router.ModelStatus(model=isolated_model)
    now = time.time()
    router.mark_rate_limited(isolated_model)
    got = router._MODEL_STATUSES[isolated_model].rate_limited_until
    # Fallback: 60s * 2^(1-1) = 60s on first failure.
    assert abs((got - now) - 60) < 1.0
