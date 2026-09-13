"""B28 batch 3 — empty-content guard must be finish_reason-aware.

A 200 with blank content is degenerate ONLY when the model actually stopped
(``finish_reason`` ``stop``/``tool_calls``). When ``finish_reason`` is
``length`` the model was still thinking and the token budget ran out — that is
a caller budget problem, not a backend defect, and must not be escalated as a
502 ("empty content"). The shadow/eval path already made this distinction;
the live path did not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import biggie_llm_endpoint as ep  # noqa: E402


def _resp(content, finish_reason, tool_calls=None):
    message = {"content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }]
    }


def test_empty_content_with_length_is_not_a_defect():
    """Reasoning tokens ate the budget — budget problem, not empty output."""
    assert ep._response_has_empty_content(_resp("", "length")) is False


def test_empty_content_with_stop_is_still_a_defect():
    """The model finished on its own and produced nothing — real defect."""
    assert ep._response_has_empty_content(_resp("", "stop")) is True


def test_empty_content_with_tool_calls_finish_is_still_a_defect():
    """No tool_calls payload but finish_reason claims tool_calls — defect."""
    assert ep._response_has_empty_content(_resp("", "tool_calls")) is True


def test_empty_content_with_missing_finish_reason_is_a_defect():
    """Unknown/absent finish_reason keeps the original conservative behaviour."""
    r = _resp("", None)
    del r["choices"][0]["finish_reason"]
    assert ep._response_has_empty_content(r) is True


def test_real_tool_calls_never_empty_regardless_of_finish_reason():
    r = _resp("", "tool_calls", tool_calls=[{"id": "1", "function": {"name": "f"}}])
    assert ep._response_has_empty_content(r) is False


def test_nonempty_content_with_length_is_not_empty():
    assert ep._response_has_empty_content(_resp("hello", "length")) is False


def test_empty_guard_reason_helper_reports_length():
    """The helper exposes WHY it is not a defect, for the log line."""
    assert ep._empty_content_reason(_resp("", "length")) == "length"
    assert ep._empty_content_reason(_resp("", "stop")) == ""
    assert ep._empty_content_reason(_resp("ok", "stop")) == ""
