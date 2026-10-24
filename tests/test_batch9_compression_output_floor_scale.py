"""Batch 9 — the session-compression output floor scales with input size.

Why this exists
---------------
The endpoint injects a fixed ``max_tokens=16384`` floor onto native
session-compression requests (which Hermes sends with NO cap, deliberately, so
the summariser has room to reason). ``deepseek-v4.1-flash`` emits a hidden
reasoning trace before visible content, so a 16,384 budget that must cover
*reasoning + summary* can run out before any summary content lands, yielding
``finish_reason=length`` and a truncated/empty summary that Hermes refuses to
commit (``summary_generation_aborted``) — which is what left the main DM session
stuck at ~335k tokens with every auto-compression failing.

The floor cannot simply be removed: it is load-bearing against reasoning-only
empty completions at SMALL context (its own docstring + the llm-smart-router
skill both record this). The fix is to make the floor *scale with input size*
so a genuinely large session gets a budget the summariser can finish under,
while small sessions keep the safe minimum.

This batch asserts the scaling contract and a non-vacuity probe.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402


def _reload():
    importlib.reload(ep)


@pytest.fixture(autouse=True)
def _defaults():
    """Reset module env-derived constants between tests."""
    _reload()
    yield
    _reload()


def test_floor_is_min_at_zero_input():
    body: dict = {}
    changed = ep.apply_session_compression_output_floor(body, "session_compression", input_tokens=0)
    assert changed is True
    assert body["max_tokens"] == ep.SESSION_COMPRESSION_MIN_MAX_TOKENS


def test_floor_scales_up_for_mid_input():
    min_v = ep.SESSION_COMPRESSION_MIN_MAX_TOKENS
    max_v = ep.SESSION_COMPRESSION_MAX_MAX_TOKENS
    mid = ep.SESSION_COMPRESSION_INPUT_FOR_MAX // 2
    body: dict = {}
    ep.apply_session_compression_output_floor(body, "session_compression", input_tokens=mid)
    assert min_v < body["max_tokens"] < max_v


def test_floor_reaches_max_at_or_above_input_for_max():
    max_v = ep.SESSION_COMPRESSION_MAX_MAX_TOKENS
    body: dict = {}
    ep.apply_session_compression_output_floor(
        body, "session_compression", input_tokens=ep.SESSION_COMPRESSION_INPUT_FOR_MAX
    )
    assert body["max_tokens"] == max_v
    body2: dict = {}
    ep.apply_session_compression_output_floor(
        body2, "session_compression", input_tokens=ep.SESSION_COMPRESSION_INPUT_FOR_MAX * 2
    )
    assert body2["max_tokens"] == max_v


def test_explicit_caller_cap_above_floor_is_preserved():
    body = {"max_tokens": 200000}
    changed = ep.apply_session_compression_output_floor(
        body, "session_compression", input_tokens=999999
    )
    assert changed is False
    assert body["max_tokens"] == 200000


def test_explicit_caller_cap_below_floor_is_raised():
    body = {"max_tokens": 2048}
    changed = ep.apply_session_compression_output_floor(
        body, "session_compression", input_tokens=0
    )
    assert changed is True
    assert body["max_tokens"] == ep.SESSION_COMPRESSION_MIN_MAX_TOKENS


def test_non_session_compression_untouched():
    body = {}
    changed = ep.apply_session_compression_output_floor(body, "normal_chat", input_tokens=500000)
    assert changed is False
    assert body == {}


def test_input_tokens_defaults_to_min_floor():
    """Backward-compatible: omitted input_tokens -> floor = MIN (old behaviour)."""
    body = {}
    changed = ep.apply_session_compression_output_floor(body, "session_compression")
    assert changed is True
    assert body["max_tokens"] == ep.SESSION_COMPRESSION_MIN_MAX_TOKENS


def test_env_override_scales_the_floor(monkeypatch):
    import os
    monkeypatch.setenv("BIGGIE_SESSION_COMPRESSION_MAX_MAX_TOKENS", "65536")
    monkeypatch.setenv("BIGGIE_SESSION_COMPRESSION_INPUT_FOR_MAX", "100000")
    _reload()
    body = {}
    ep.apply_session_compression_output_floor(
        body, "session_compression", input_tokens=100000
    )
    assert body["max_tokens"] == 65536


def test_numerically_consistent_monotonic_scale():
    """Floor is monotonic non-decreasing and bounded [MIN, MAX]."""
    min_v = ep.SESSION_COMPRESSION_MIN_MAX_TOKENS
    max_v = ep.SESSION_COMPRESSION_MAX_MAX_TOKENS
    prev = -1
    for inp in (0, 1000, 50000, 100000, 200000, 299999, 300000, 400000):
        body: dict = {}
        ep.apply_session_compression_output_floor(body, "session_compression", input_tokens=inp)
        assert min_v <= body["max_tokens"] <= max_v
        assert body["max_tokens"] >= prev
        prev = body["max_tokens"]


def test_floor_calibration_clears_truncation_window():
    """Regression: the raised floor must clear the observed truncation window.

    The diagnosed truncation cluster emitted 16,973–19,314 output tokens at an
    average failing context of ~24,643 because the old MIN (16,384) scaled to only
    ~17,730 there. The 2026-10-24 calibration must grant strictly more than the
    largest observed truncated output at that context.
    """
    assert ep.SESSION_COMPRESSION_MIN_MAX_TOKENS >= 24576
    assert ep.SESSION_COMPRESSION_MAX_MAX_TOKENS >= 49152
    body: dict = {}
    ep.apply_session_compression_output_floor(body, "session_compression", input_tokens=24643)
    assert body["max_tokens"] > 19314
