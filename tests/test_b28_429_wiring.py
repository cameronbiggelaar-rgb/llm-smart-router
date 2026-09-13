"""B28 batch 1 — integration: a 429 body reaches mark_rate_limited's reset time.

Proves the wiring (call site -> _handle_backend_status -> mark_rate_limited),
not just the unit-level extractor.
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
def isolated_backend_model():
    model = "b28-integration-model"
    router._MODEL_STATUSES[model] = router.ModelStatus(model=model)
    yield model
    router._MODEL_STATUSES.pop(model, None)


def test_codex_429_body_sets_reset_cooldown_end_to_end(isolated_backend_model):
    now = time.time()
    resets = int(now + 7200)
    body = (
        '{"error": {"message": "usage limit reached", '
        f'"code": "usage_limit_reached", "plan_type": "plus", "resets_at": {resets}}}}}'
    )
    ep._handle_backend_status("openai-codex", isolated_backend_model, 429, error_text=body)
    got = router._MODEL_STATUSES[isolated_backend_model].rate_limited_until
    assert got is not None
    # Honoured the 2h upstream reset, not the 60s backoff.
    assert abs(got - resets) < 2


def test_ollama_429_without_reset_uses_backoff(isolated_backend_model):
    """A plain per-model throttle (no reset hint, no account cap) uses backoff."""
    now = time.time()
    body = '{"error": {"message": "rate limit exceeded, slow down"}}'
    ep._handle_backend_status("ollama-cloud", isolated_backend_model, 429, error_text=body)
    got = router._MODEL_STATUSES[isolated_backend_model].rate_limited_until
    assert abs((got - now) - 60) < 2


def test_retry_after_header_takes_precedence_via_endpoint(isolated_backend_model):
    now = time.time()
    ep._handle_backend_status(
        "ollama-cloud", isolated_backend_model, 429,
        error_text="", headers={"retry-after": "900"},
    )
    got = router._MODEL_STATUSES[isolated_backend_model].rate_limited_until
    assert abs((got - now) - 900) < 5
