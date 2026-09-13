"""B27 — streaming session-compression channel regressions."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402


class _Resp:
    def __init__(self, lines):
        self.lines = list(lines)
        self.closed = False
        self.status_code = 200

    async def aiter_lines(self):
        for line in self.lines:
            yield line

    async def aclose(self):
        self.closed = True


def test_session_compression_tool_call_only_stream_is_empty(monkeypatch):
    """A compaction must not be accepted because the model emitted a tool call.

    Production logs show the failure signature: saw_content=0, saw_tool_calls=1,
    output_tokens>0. That is a paid non-summary, so preflight must escalate
    rather than return a successful streamed compaction.
    """
    line = 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x"}]}}]}'
    done = "data: [DONE]"
    resp = _Resp([line, done])

    class _Client:
        def build_request(self, *a, **kw):
            return object()

        async def send(self, req, stream=True):
            return resp

    monkeypatch.setattr(ep, "_get_httpx_client", lambda: _Client())

    pf = asyncio.run(ep._preflight_openai_stream(
        {"base_url": "http://backend", "provider": "ollama-cloud", "backend_model": "deepseek-v4.1-flash"},
        [{"role": "user", "content": "compress"}],
        {},
        workload_type="session_compression",
    ))

    assert pf.status == "empty"
    assert pf.saw_tool_calls is True
    assert pf.saw_content is False
    assert "tool-call-only" in pf.error
    assert resp.closed is True


def test_tool_call_stream_is_ok_for_tool_workload(monkeypatch):
    """The guard is scoped: normal tool-capable calls still stream tool_calls."""
    line = 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x"}]}}]}'
    resp = _Resp([line])

    class _Client:
        def build_request(self, *a, **kw):
            return object()

        async def send(self, req, stream=True):
            return resp

    monkeypatch.setattr(ep, "_get_httpx_client", lambda: _Client())

    pf = asyncio.run(ep._preflight_openai_stream(
        {"base_url": "http://backend", "provider": "ollama-cloud", "backend_model": "tool-model"},
        [{"role": "user", "content": "use a tool"}],
        {"tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]},
        workload_type="normal_chat",
    ))

    assert pf.status == "ok"
    assert pf.saw_tool_calls is True
    assert pf.response is resp
    assert resp.closed is False


def test_channel_trace_logs_content_reasoning_and_tools(caplog):
    first = 'data: {"choices":[{"delta":{"content":"visible"}}]}'
    second = 'data: {"choices":[{"delta":{"reasoning":"hidden"}}]}'
    third = 'data: {"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}'
    done = "data: [DONE]"
    resp = _Resp([second, third, done])
    it = resp.aiter_lines()

    async def run():
        pf = ep._StreamPreflight(
            status="ok", backend_model="deepseek-v4.1-flash", provider="ollama-cloud",
            buffered=[first], saw_content=True, response=resp, iterator=it,
        )
        out = []
        async for evt in ep._resume_stream(pf, {}, [], {}, trace_channels=True):
            out.append(evt)
        return out

    with caplog.at_level("ERROR", logger="biggie_llm_endpoint"):
        events = asyncio.run(run())

    assert any(e.startswith("data: [DONE]") for e in events)
    msgs = [r.getMessage() for r in caplog.records]
    trace = next(m for m in msgs if "session_compression_channel_trace" in m)
    assert "content_chars=7" in trace
    assert "reasoning_chars=6" in trace
    assert "tool_call_deltas=1" in trace
