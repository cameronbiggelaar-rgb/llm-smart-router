"""Batch 10 — the streaming path must harvest and log ``finish_reason``.

Why this exists
---------------
The non-streaming path already captures ``finish_reason`` from the backend
response and writes it to the router_logs ledger. The streaming path does NOT:
``_resume_stream`` harvests token usage and content/tool-call flags from the
SSE stream, but never the terminal ``finish_reason`` chunk, and the streaming
completion loggers never pass a ``finish_reason`` value. The result is that
every streamed request — including every ``session_compression`` request, the
busiest workload — lands in the ledger with ``finish_reason = ''``.

That makes the single most important signal for the truncation class of bugs
(``finish_reason == "length"``) unobservable: a truncated summary and a clean
stop are indistinguishable in the ledger. This batch closes that gap.

Pinned (three layers, following the writer-cleanup / usage-capture test style):

  1. a parser reads ``finish_reason`` from the terminal SSE chunk;
  2. ``_resume_stream`` harvests it into ``stats`` (what the logger reads);
  3. the streaming completion loggers forward it to ``_log_request_to_db``
     (AST-asserted, since the loggers are nested closures inside the request
     handler and not unit-callable directly).
"""
from __future__ import annotations

import ast
import asyncio
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402


def _endpoint_src() -> str:
    return (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()


# ── 1. parser unit tests ─────────────────────────────────────────────────────


def test_parser_exists():
    assert hasattr(ep, "_sse_finish_reason"), "no parser for the streamed finish_reason chunk"


def test_parser_reads_terminal_finish_reason_chunk():
    chunk = '{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
    assert ep._sse_finish_reason(chunk) == "stop"


def test_parser_reads_length():
    chunk = '{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}'
    assert ep._sse_finish_reason(chunk) == "length"


def test_parser_reads_tool_calls():
    chunk = '{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
    assert ep._sse_finish_reason(chunk) == "tool_calls"


def test_parser_prefers_first_non_empty_across_choices():
    chunk = (
        '{"choices":[{"index":0,"delta":{},"finish_reason":null},'
        '{"index":1,"delta":{},"finish_reason":"stop"}]}'
    )
    assert ep._sse_finish_reason(chunk) == "stop"


def test_parser_silent_on_chunks_without_finish_reason():
    for chunk in (
        '{"choices":[{"delta":{"content":"hi"}}]}',
        '{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}',
        "[DONE]",
        "",
        "not json at all",
        '{"choices":[{"delta":{}}]}',
    ):
        assert ep._sse_finish_reason(chunk) == "", f"chunk {chunk!r} produced a reason"


# ── 2. harvest into stats ────────────────────────────────────────────────────


class _FakeResp:
    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b""

    async def aclose(self):
        pass


def test_resume_stream_harvests_finish_reason_into_stats():
    """The terminal finish_reason chunk must reach ``stats``, not be dropped.

    The final chunk carries ``finish_reason`` with an EMPTY delta — a parser
    that only looks at ``delta.content`` never sees it, exactly the class of
    drop that hid usage for the whole streaming path until B23.
    """
    content = 'data: {"choices":[{"delta":{"content":"hi"}}]}'
    finish = 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}'

    async def _iter():
        for line in (finish, "data: [DONE]"):
            yield line

    pf = ep._StreamPreflight(
        status="ok",
        buffered=[content],
        iterator=_iter(),
        response=_FakeResp([]),
        accumulated="hi",
    )
    stats = {}

    async def _drain():
        async for _ in ep._resume_stream(pf, {}, [], {}, stats=stats):
            pass

    asyncio.run(_drain())
    assert stats.get("finish_reason") == "length", (
        f"the streamed finish_reason never reached stats: {stats!r}"
    )


def test_resume_stream_leaves_finish_reason_unset_when_absent():
    """A clean stream with no terminal reason must not invent one."""

    async def _iter():
        yield "data: [DONE]"

    pf = ep._StreamPreflight(
        status="ok",
        buffered=[],
        iterator=_iter(),
        response=_FakeResp([]),
        accumulated="",
    )
    stats = {}

    async def _drain():
        async for _ in ep._resume_stream(pf, {}, [], {}, stats=stats):
            pass

    asyncio.run(_drain())
    assert stats.get("finish_reason", "") == "", (
        f"an absent finish_reason should not be synthesized: {stats!r}"
    )


# ── 3. streaming loggers forward it ──────────────────────────────────────────


def test_streaming_completion_loggers_forward_finish_reason():
    """Both streaming completion loggers must pass finish_reason to the DB.

    The loggers are nested closures inside the request handler, so this asserts
    on the AST: every ``_log_request_to_db`` call inside a streaming completion
    logger must pass a ``finish_reason`` keyword. Without this, the ledger keeps
    filling with ``finish_reason = ''`` for the entire streaming workload.
    """
    tree = ast.parse(_endpoint_src())
    loggers = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "_log_request_to_db":
                    kws = {k.arg for k in sub.keywords}
                    err = next(
                        (k.value for k in sub.keywords if k.arg == "error_type"),
                        None,
                    )
                    is_start_marker = (
                        isinstance(err, ast.Constant) and err.value == "streaming_in_progress"
                    )
                    # Only completion loggers (not the in-flight start marker).
                    if not is_start_marker:
                        loggers.append((node.name, sub.lineno, kws))
    streaming_loggers = [(n, ln, kws) for n, ln, kws in loggers if "stream" in n.lower()]
    assert streaming_loggers, "no streaming completion logger found to check"
    for name, lineno, kws in streaming_loggers:
        assert "finish_reason" in kws, (
            f"{name} (line {lineno}) does not forward finish_reason to the ledger"
        )
