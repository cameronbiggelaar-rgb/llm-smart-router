"""Batch 8 — a tool-bearing request is never demoted by text markers alone.

Why this exists
---------------
Batch 7 scoped the marker scan to the last user message, which fixes the
reported defect (a compaction handoff sitting in EARLIER history). This batch
closes the remaining seam: a marker phrase appearing in the LAST user message of
a request that nonetheless carries a `tools` array.

Two opposite failure modes must both be held:

1. FALSE POSITIVE (the reported bug). If a *user* message happens to contain a
   marker phrase -- e.g. the user pastes a compaction handoff, or asks "summarize
   the conversation so I can continue" while genuinely wanting tool use -- the
   request loses its tools and goes to a non-tool model. Belt-and-braces: a body
   that carries `tools` is not demoted by TEXT ALONE.

2. FALSE NEGATIVE (B29's purpose). A genuine Hermes compaction request inherits
   the parent session's tool schemas but a summariser never calls a tool. Those
   MUST still be stripped, or `requires_tools` unlocks the tool-capability route
   and the "most-capable tool model" escalation branch, and offered tools
   provoke tool-call-only responses from a summariser.

Resolution: explicit signalling (header/metadata) always wins. Text markers only
demote a request when it carries NO tool schemas. A genuine first-time compaction
prompt carries no `tools`, so it keeps being stripped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402

TOOLS = [
    {"type": "function", "function": {"name": "skill_view", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}},
]

# The real first-time compaction prompt (agent/context_compressor.py).
FIRST_TIME_COMPACTION = (
    "You are a summarization agent creating a context checkpoint. Treat the "
    "conversation turns below as source material for a compact record of prior "
    "work. Produce only the structured summary; do not add a greeting.\n\n"
    "Create a structured checkpoint summary for the conversation after earlier "
    "turns are compacted.\n\nTURNS TO SUMMARIZE:\n(turns)\n"
)
ITERATIVE_COMPACTION = (
    "You are a summarization agent creating a context checkpoint.\n\n"
    "You are updating a context compaction summary. A previous compaction "
    "produced the summary below.\n\nPREVIOUS SUMMARY:\n(prev)\n"
)


#  Mode 1: tool-bearing requests survive marker text 

@pytest.mark.parametrize(
    "user_text",
    [
        "Here is the context compaction summary you asked for; now load archify.",
        "Please summarize the conversation so I can continue -- then run the tests.",
        "Preserve key facts, decisions and open tasks, then fix the failing test.",
        "The compressed summary is attached. Use the terminal to check disk space.",
    ],
)
def test_tool_bearing_request_is_not_demoted_by_text_markers(user_text):
    body = {"model": "biggie-router", "tools": TOOLS, "tool_choice": "auto"}
    assert ep.detect_workload_type([{"role": "user", "content": user_text}], body, {}) == "normal_chat"


def test_tools_in_a_body_survive_a_marker_in_the_last_message():
    body = {"model": "biggie-router", "tools": TOOLS}
    workload = ep.detect_workload_type(
        [{"role": "user", "content": "summarize the conversation so I can continue"}], body, {}
    )
    out = ep._normalize_compression_request(dict(body), workload)
    assert out.get("tools"), "tool schemas stripped from a tool-bearing request"
    assert out.get("_requires_tools") is True


def test_empty_tools_list_does_not_defeat_the_marker():
    """An empty `tools: []` is not a tool-bearing request."""
    body = {"model": "biggie-router", "tools": []}
    assert (
        ep.detect_workload_type([{"role": "user", "content": ITERATIVE_COMPACTION}], body, {})
        == "session_compression"
    )


#  Mode 2: B29's purpose preserved -- genuine compaction is still stripped 

def test_genuine_first_time_compaction_still_classifies_without_tools():
    assert (
        ep.detect_workload_type([{"role": "user", "content": FIRST_TIME_COMPACTION}], {}, {})
        == "session_compression"
    )


def test_genuine_iterative_compaction_still_classifies_without_tools():
    assert (
        ep.detect_workload_type([{"role": "user", "content": ITERATIVE_COMPACTION}], {}, {})
        == "session_compression"
    )


def test_inherited_tool_schemas_still_stripped_when_explicitly_signalled():
    """The real dangerous case: a compaction request that inherited the parent
    session's tool schemas. Explicit metadata must still strip them."""
    body = {"model": "biggie-router", "tools": TOOLS, "metadata": {"hermes_task": "compression"}}
    workload = ep.detect_workload_type(
        [{"role": "user", "content": FIRST_TIME_COMPACTION}], body, {}
    )
    out = ep._normalize_compression_request(dict(body), workload)
    assert workload == "session_compression"
    assert "tools" not in out
    assert out.get("_requires_tools") is False


def test_explicit_header_beats_tools_when_not_actually_tool_bearing():
    """A compaction request with inherited schemas AND an explicit header: the
    header is the authoritative signal and must strip."""
    body = {"model": "biggie-router", "tools": TOOLS}
    workload = ep.detect_workload_type([{"role": "user", "content": "hi"}], body, {
        "x-hermes-auxiliary-task": "compression"
    })
    out = ep._normalize_compression_request(dict(body), workload)
    assert workload == "session_compression"
    assert "tools" not in out


#  Regression guards from batch 7 

def test_handoff_in_earlier_history_still_ignored():
    messages = [
        {"role": "user", "content": "[CONTEXT COMPACTION - REFERENCE ONLY] prior work..."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Load the archify skill."},
    ]
    body = {"tools": TOOLS}
    assert ep.detect_workload_type(messages, body, {}) == "normal_chat"


def test_ordinary_chat_without_tools_is_unaffected():
    assert ep.detect_workload_type([{"role": "user", "content": "hello there"}], {}, {}) == "normal_chat"


def test_empty_history_with_tools_is_safe():
    body = {"tools": TOOLS}
    assert ep.detect_workload_type([], body, {}) == "normal_chat"
