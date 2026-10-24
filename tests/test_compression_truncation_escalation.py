"""Compression truncation escalation — `finish_reason=length` must escalate to the
next summariser rung (never the plan), not surface as a user-visible failure.

The truncation class was diagnosed as the endpoint's own output floor: it granted
only ~17-18K output tokens at real compression sizes while `deepseek-v4.1-flash`
(99.6% of compression calls) burns a large hidden reasoning trace before the summary,
so the visible summary ran out of budget mid-generation and ended
`finish_reason=length`. The endpoint then deliberately did NOT escalate on `length`
(treating it as a caller budget problem) — but the caller was the endpoint's own floor.

This batch pins the escalation contract across all four layers:

  1. `_response_is_truncated` (non-streaming) recognises `finish_reason=length`;
  2. the streaming preflight drains a session-compression stream to completion and
     reports `truncated` (so `proxy_to_backend_streaming` escalates before commit);
  3. `_classify_failure_detail` maps the truncation detail to `error_type="truncation"`;
  4. `escalate_on_failure(workload_type="session_compression")` stays on the curated
     summariser ladder and never lands on a ChatGPT-plan model.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(SKILL_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import biggie_llm_endpoint as ep  # noqa: E402
import router  # noqa: E402
from router import ModelStatus  # noqa: E402


@pytest.fixture()
def clean_statuses():
    """Snapshot and restore model status so tests cannot leak health state."""
    saved = dict(router._MODEL_STATUSES)
    yield
    router._MODEL_STATUSES.clear()
    router._MODEL_STATUSES.update(saved)


# ── 1. non-streaming truncation helper ──────────────────────────────────────


def test_response_is_truncated_detects_length():
    resp = {"choices": [{"index": 0, "finish_reason": "length", "message": {"content": "partial"}}]}
    assert ep._response_is_truncated(resp) is True


def test_response_is_truncated_false_for_stop_and_others():
    for reason in ("stop", "tool_calls", None, ""):
        resp = {"choices": [{"index": 0, "finish_reason": reason, "message": {"content": "x"}}]}
        assert ep._response_is_truncated(resp) is False, reason


def test_response_is_truncated_false_for_non_dict_and_empty_choices():
    assert ep._response_is_truncated("not a dict") is False
    assert ep._response_is_truncated({}) is False
    assert ep._response_is_truncated({"choices": []}) is False


# ── 2. streaming preflight full-drain (truncation detection) ─────────────────


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp

    def build_request(self, *a, **k):
        return object()

    async def send(self, *a, **k):
        return self._resp


class _FakeStreamResp:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b""

    async def aclose(self):
        pass


def _preflight(lines, monkeypatch, workload_type="session_compression"):
    monkeypatch.setattr(ep, "_get_httpx_client", lambda: _FakeClient(_FakeStreamResp(lines)))
    return asyncio.run(
        ep._preflight_openai_stream(
            {"base_url": "http://upstream", "api_key": "", "backend_model": "m", "provider": "cloud"},
            [{"role": "user", "content": "summarise this"}],
            {},
            workload_type=workload_type,
        )
    )


def test_preflight_reports_truncated_for_length_stream(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"part one"}}]}',
        'data: {"choices":[{"delta":{"content":"part two"}}]}',
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}',
        "data: [DONE]",
    ]
    pf = _preflight(lines, monkeypatch)
    assert pf.status == "truncated", f"expected truncated, got {pf.status!r}"


def test_preflight_reports_ok_for_clean_stream(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"part one"}}]}',
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    pf = _preflight(lines, monkeypatch)
    assert pf.status == "ok", f"expected ok, got {pf.status!r} (err={pf.error!r})"
    # The full stream must be buffered so _resume_stream can replay it without
    # re-reading the (now exhausted) iterator.
    assert pf.buffered[-1] == "data: [DONE]"


def test_preflight_still_bounded_for_non_compression(monkeypatch):
    """Non-compression keeps the bounded preflight: commit at first content."""
    lines = [
        'data: {"choices":[{"delta":{"content":"first"}}]}',
        'data: {"choices":[{"delta":{"content":"second"}}]}',
        "data: [DONE]",
    ]
    pf = _preflight(lines, monkeypatch, workload_type="normal_chat")
    assert pf.status == "ok"
    # Bounded: buffer holds only the first content chunk, not [DONE].
    assert pf.buffered[-1] != "data: [DONE]"


def test_preflight_reports_empty_for_contentless_compression_stream(monkeypatch):
    pf = _preflight(["data: [DONE]"], monkeypatch)
    assert pf.status == "empty"


# ── 3. failure classification ───────────────────────────────────────────────


def test_classify_maps_truncated_to_truncation():
    assert ep._classify_failure_detail("Backend m returned truncated session compression summary") == "truncation"


def test_classify_preserves_existing_mappings():
    assert ep._classify_failure_detail("Backend m returned empty content") == "empty_content"
    assert ep._classify_failure_detail("malformed tool_call") == "malformed_tool_call"
    assert ep._classify_failure_detail("degenerated") == "degeneration"
    assert ep._classify_failure_detail("some other error") == "error"


# ── 4. the fence: truncation escalation never reaches a pro/plan model ───────


def test_truncation_is_in_no_breaker_trip_tuple():
    assert "truncation" in router._QUALITY_ERROR_TYPES, (
        "truncation must escalate WITHOUT tripping the circuit breaker "
        "(it is an output-budget/quality issue, not a backend defect)"
    )


def test_truncation_escalation_advances_along_the_ladder(clean_statuses):
    """A truncated rung retries the *next* summariser rung — never the plan."""
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
        error_type="truncation",
        requires_tools=True,  # even a tool-flagged compaction stays on the ladder
        workload_type="session_compression",
        context_tokens=0,
    )
    assert decision.selected_model != first
    assert not router._is_reserved_from_compression(decision.selected_model)
    assert decision.selected_model in ladder


def test_truncation_escalation_caps_below_pro_tier(clean_statuses):
    """The anti-over-escalation fence: compaction retries never reach a pro model.

    deepseek-v4-pro is the ladder's top rung (tier 8) and a deliberate
    *first-attempt* summariser for very large contexts — but it is out of bounds
    for escalation. When the cheap summarisers are down, escalation must land on
    another sub-pro model or fail closed, never deepseek-v4-pro / minimax-m3 /
    gpt-5.x.
    """
    pro = "deepseek-v4-pro"
    assert router.MODEL_CAPABILITY_TIERS.get(pro, 0) >= router.COMPRESSION_ESCALATION_MAX_TIER

    # Take the cheap summarisers offline, leaving deepseek-v4-pro as the only
    # "natural" next ladder rung — the fence must still refuse it.
    for model in ("glm-5.3", "glm-5.2"):
        router._MODEL_STATUSES[model] = ModelStatus(
            model=model, available=False, consecutive_failures=99
        )

    decision = router.escalate_on_failure(
        failed_model="deepseek-v4.1-flash",
        complexity_score=0.9,
        error_type="truncation",
        requires_tools=True,
        workload_type="session_compression",
        context_tokens=0,
    )
    assert decision.selected_model != pro
    if decision.selected_model:
        tier = router.MODEL_CAPABILITY_TIERS.get(decision.selected_model, 0)
        assert tier < router.COMPRESSION_ESCALATION_MAX_TIER, (
            f"escalation landed on {decision.selected_model} (tier {tier}), a pro-tier model"
        )
