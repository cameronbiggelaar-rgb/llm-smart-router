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
                # Preflight consumed line 0; yield the rest
                for line in self._lines[1:]:
                    yield line
            async def aclose(self):
                self._closed = True

        keep_alive = _KeepAliveResp()

        pf = _StreamPreflight(
            status="ok",
            backend_model="glm-5.2:cloud",
            provider="ollama-cloud",
            buffered=[all_chunks[0]],  # first chunk buffered by preflight (full "data: ..." line)
            saw_content=True,
            response=keep_alive,
        )

        backend = {"base_url": "http://fake", "api_key": "", "backend_model": "glm-5.2:cloud", "provider": "ollama-cloud"}

        async def collect():
            out = []
            async for evt in _resume_stream(pf, backend, [], {}):
                out.append(evt)
            return out

        events = asyncio.run(collect())
        assert fresh_calls == [], f"_open_fresh_stream was called {len(fresh_calls)} times — must use preflight connection"

        content = "".join(
            json.loads(e[6:])["choices"][0]["delta"].get("content", "")
            for e in events
            if e.startswith("data: ") and e[6:].strip() != "[DONE]"
        )
        assert content == "Good morning", f"Expected 'Good morning', got {content!r}"
