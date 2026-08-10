"""TDD: _resume_stream must NOT open a fresh connection.

The preflight probe and the downstream stream must be the SAME connection.
Opening a fresh connection causes:
  1. Dropped characters (fresh generation splits tokens differently)
  2. 10+ second pauses (fresh round-trip to ollama-cloud)
  3. Potential duplication (fresh generation may differ from preflight buffer)
"""
import sys, asyncio, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

class TestResumeStreamUsesPreflightConnection:
    def test_resume_stream_does_not_open_fresh_connection(self, monkeypatch):
        from biggie_llm_endpoint import _StreamPreflight, _resume_stream

        # Track if _open_fresh_stream is called
        fresh_calls = []
        async def fake_fresh(*a, **kw):
            fresh_calls.append(1)
            raise RuntimeError("_open_fresh_stream should NOT be called")
        monkeypatch.setattr("biggie_llm_endpoint._open_fresh_stream", fake_fresh)

        # The preflight's response — kept alive, yields remaining chunks
        all_chunks = [
            'data: {"choices":[{"delta":{"content":"Good"}}]}',
            'data: {"choices":[{"delta":{"content":" morning"}}]}',
            'data: [DONE]',
        ]

        class _KeepAliveResp:
            def __init__(self):
                self._lines = list(all_chunks)
                self._closed = False
            async def aiter_lines(self):
                # Single-use iterator: yield ALL lines. Preflight consumes line 0
                # (buffered); _resume_stream continues the SAME iterator from line 1.
                for line in self._lines:
                    yield line
            async def aclose(self):
                self._closed = True

        keep_alive = _KeepAliveResp()
        backend = {"base_url": "http://fake", "api_key": "", "backend_model": "glm-5.2:cloud", "provider": "ollama-cloud"}

        async def run():
            # Preflight consumes line 0 (buffered) and captures the SAME
            # single-use iterator, now positioned after line 0.
            it = keep_alive.aiter_lines()
            first = await it.__anext__()
            assert first == all_chunks[0]

            pf = _StreamPreflight(
                status="ok",
                backend_model="glm-5.2:cloud",
                provider="ollama-cloud",
                buffered=[all_chunks[0]],  # first chunk buffered by preflight
                saw_content=True,
                response=keep_alive,
                iterator=it,  # the SAME iterator, already past line 0
            )

            out = []
            async for evt in _resume_stream(pf, backend, [], {}):
                out.append(evt)
            return out

        events = asyncio.run(run())
        assert fresh_calls == [], f"_open_fresh_stream was called {len(fresh_calls)} times — must use preflight connection"

        content = "".join(
            json.loads(e[6:])["choices"][0]["delta"].get("content", "")
            for e in events
            if e.startswith("data: ") and e[6:].strip() != "[DONE]"
        )
        assert content == "Good morning", f"Expected 'Good morning', got {content!r}"


class TestResumeStreamReusesSingleIterator:
    """Regression: _resume_stream must resume the SAME aiter_lines() iterator
    captured during preflight — it must NOT call resp.aiter_lines() a second
    time. httpx streaming responses are single-use; a second aiter_lines()
    call on the same response yields nothing (or raises), silently dropping
    the rest of the stream after the buffered first chunk."""

    def test_aiter_lines_called_exactly_once(self, monkeypatch):
        from biggie_llm_endpoint import _StreamPreflight, _resume_stream

        aiter_calls = []

        class _SingleUseResp:
            def __init__(self):
                self._lines = [
                    'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                    'data: {"choices":[{"delta":{"content":" world"}}]}',
                    "data: [DONE]",
                ]
                self._closed = False

            async def aiter_lines(self):
                aiter_calls.append(1)
                # A real httpx response can only be iterated once. Simulate that:
                # the FIRST call yields all lines; any SECOND call yields nothing.
                if len(aiter_calls) > 1:
                    return
                for line in self._lines:
                    yield line

            async def aclose(self):
                self._closed = True

        resp = _SingleUseResp()
        backend = {"base_url": "http://fake", "api_key": "", "backend_model": "glm-5.2:cloud", "provider": "ollama-cloud"}

        async def run():
            # Preflight consumes the first line (buffered it) and captures the
            # SAME single-use iterator, now positioned after line 0.
            it = resp.aiter_lines()
            first = await it.__anext__()
            assert first == resp._lines[0]

            pf = _StreamPreflight(
                status="ok",
                backend_model="glm-5.2:cloud",
                provider="ollama-cloud",
                buffered=[resp._lines[0]],
                saw_content=True,
                response=resp,
                iterator=it,  # the SAME iterator, already past line 0
            )

            out = []
            async for evt in _resume_stream(pf, backend, [], {}):
                out.append(evt)
            return out

        events = asyncio.run(run())

        # aiter_lines() must be called exactly once — the preflight's iterator
        # is resumed, never re-created.
        assert len(aiter_calls) == 1, (
            f"aiter_lines() called {len(aiter_calls)} times — must be called exactly once "
            "and the SAME iterator resumed. A second call on a single-use httpx "
            "response silently drops the rest of the stream."
        )

        content = "".join(
            json.loads(e[6:])["choices"][0]["delta"].get("content", "")
            for e in events
            if e.startswith("data: ") and e[6:].strip() != "[DONE]"
        )
        assert content == "Hello world", f"Expected 'Hello world', got {content!r}"
