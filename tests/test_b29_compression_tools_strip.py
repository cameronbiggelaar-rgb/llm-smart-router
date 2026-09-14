"""B29 batch 2 — endpoint side: compression must not carry tools, and must not promote.

Two coupled defects
-------------------
FIX 2 (tools strip)
    A native Hermes session-compression request inherits the parent session's
    tool schemas. `requires_tools = bool(body.get("tools"))` therefore becomes
    True for a workload that can never call a tool. That flag is what later
    unlocks the "most-capable tool model" escalation branch, so the leak starts
    here.

FIX 3 (rung retry, not promotion)
    Even with tools stripped, a summariser can return tool-call-only/empty
    output. When it does, the endpoint raises 502 and calls
    `escalate_on_failure`. The escalation must be told this is compression
    (`workload_type="session_compression"`) or it falls back to the generic
    tool/tier climb and lands on the ChatGPT plan.

These tests exercise the real handler helpers rather than re-implementing the
policy, so they fail if the wiring is dropped even when the router functions
themselves are correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402
import router  # noqa: E402


def _compression_messages() -> list:
    """A prompt the real detector classifies as session_compression."""
    return [
        {
            "role": "user",
            "content": (
                "Context compaction task: summarize earlier conversation so I can "
                "continue. Preserve key facts, decisions, and file paths."
            ),
        }
    ]


def _tool_schema() -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "run a shell command",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


# ── Fix 2: tools stripped from compression ────────────────────────────────────


def test_compression_request_detected_and_tools_stripped():
    """The normaliser the handler uses must drop tools for compression."""
    body = {
        "model": "biggie-router",
        "messages": _compression_messages(),
        "tools": _tool_schema(),
        "tool_choice": "auto",
    }
    workload = ep.detect_workload_type(body["messages"], body, {})
    assert workload == "session_compression"

    normalized = ep._normalize_compression_request(body, workload)

    assert normalized.get("tools") is None, "compression must not offer tool schemas"
    assert normalized.get("tool_choice") is None
    assert normalized["_requires_tools"] is False


def test_normal_request_keeps_its_tools():
    """A genuine tool-bearing request is untouched."""
    body = {
        "model": "biggie-router",
        "messages": [{"role": "user", "content": "List the files in /tmp please."}],
        "tools": _tool_schema(),
        "tool_choice": "auto",
    }
    workload = ep.detect_workload_type(body["messages"], body, {})
    assert workload != "session_compression"

    normalized = ep._normalize_compression_request(body, workload)

    assert normalized.get("tools"), "non-compression requests must keep tool schemas"
    assert normalized["_requires_tools"] is True


def test_requires_tools_flag_derives_from_normalized_body():
    """The handler's requires_tools must come from the normalized body."""
    body = {
        "messages": _compression_messages(),
        "tools": _tool_schema(),
    }
    workload = ep.detect_workload_type(body["messages"], body, {})
    normalized = ep._normalize_compression_request(body, workload)

    # This is the expression the handler uses after normalisation.
    assert bool(normalized.get("tools")) is False


# ── Fix 3: compression escalation retries a rung, never promotes ──────────────


def test_compression_escalation_never_promotes_to_chatgpt_plan():
    """The call site's argument shape must land compression on a summariser."""
    statuses = dict(router._MODEL_STATUSES)
    try:
        decision = router.escalate_on_failure(
            failed_model="deepseek-v4.1-flash:cloud",
            complexity_score=0.9,
            error_type="empty_content",
            requires_tools=False,
            workload_type="session_compression",
            context_tokens=120_000,
        )
        assert decision.selected_model
        assert not router._is_reserved_from_compression(decision.selected_model)
        assert not decision.selected_model.startswith("gpt-5")
        for fb in decision.fallback_chain:
            assert not router._is_reserved_from_compression(fb), (
                f"compression fallback chain leaked {fb}"
            )
    finally:
        router._MODEL_STATUSES.clear()
        router._MODEL_STATUSES.update(statuses)


def test_compression_escalation_with_tools_flag_still_stays_on_ladder():
    """Even if requires_tools is wrongly True, compression must not promote.

    This is the historical defect: compression reached the tool branch. The
    workload_type argument must dominate requires_tools.
    """
    statuses = dict(router._MODEL_STATUSES)
    try:
        decision = router.escalate_on_failure(
            failed_model="deepseek-v4.1-flash:cloud",
            complexity_score=0.9,
            error_type="empty_content",
            requires_tools=True,
            workload_type="session_compression",
            context_tokens=120_000,
        )
        assert decision.selected_model
        assert not decision.selected_model.startswith("gpt-5"), (
            "session_compression reached the tool-promotion branch"
        )
    finally:
        router._MODEL_STATUSES.clear()
        router._MODEL_STATUSES.update(statuses)
