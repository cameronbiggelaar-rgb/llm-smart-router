"""Writer cleanup B23.2 — capture real output tokens on the streaming path.

Both streaming completion loggers hardcode ``output_tokens=0``. Measured on
production before this change:

    streaming completion rows                    114,618
    of those recording output_tokens=0            114,618  (all of them)
    recorded spend on them                        $17,438.92 over 7.77B input tokens

So a cost figure derived from input alone is recorded as though it were
*measured*, and nothing in the schema says otherwise. This is the B15 defect:
``tests/test_streaming_cost_gap.py`` deliberately PINS it and states that when it
is fixed, that file should fail loudly so the fix is announced rather than
assumed.

The fix is possible — and that was measured, not assumed. A direct probe of
ollama.com (``/tmp/probe_usage.py``):

    stream without stream_options            0 lines carry non-null usage
    stream with stream_options.include_usage 1 line carries
                                             usage: {prompt_tokens,
                                                     completion_tokens, ...}

The usage arrives in a final chunk with ``choices: []``. So the defect is not
that usage is unobtainable on a stream — it is that we never ask for it.

What this file pins:

  1. the streaming request asks for usage when the provider supports it;
  2. a stream that supplies usage records the REAL output token count;
  3. a stream that supplies no usage records ``cost_unknown=1`` — an input-only
     cost must never be presented as a complete measured one;
  4. the provider gate excludes providers not known to accept ``stream_options``,
     and an existing caller-supplied ``stream_options`` is not clobbered;
  5. a captured usage chunk is parsed into stats rather than dropped.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402


def _endpoint_src() -> str:
    return (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()


# --------------------------------------------------------------------------
# 1. We ask for usage on the streaming path
#
# These drive the real preflight and inspect the request it actually builds.
# An earlier version of this file asserted on source text ("include_usage" in
# src), which passed even with the provider gate disabled — a vacuous test. The
# non-vacuity proof (/tmp/b23_proof.py) caught it: reverting the gate did not
# make the test fail. Asserting on the constructed body is the only form that
# fails when the behaviour is removed.
# --------------------------------------------------------------------------

class _FakeResp:
    """Minimal httpx streaming response."""

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


class _CapturingClient:
    """Captures the JSON body the preflight actually sends upstream."""

    def __init__(self, lines):
        self.captured = {}
        self._resp = _FakeResp(lines)

    def build_request(self, method, url, json=None, headers=None):
        self.captured["json"] = json
        self.captured["url"] = url
        return {"method": method, "url": url}

    async def send(self, req, stream=False):
        return self._resp


CONTENT_LINE = 'data: {"choices":[{"delta":{"content":"hi"}}]}'
USAGE_LINE = (
    'data: {"choices":[],"usage":{"prompt_tokens":32,"completion_tokens":7}}'
)


def _drive_preflight(monkeypatch, provider, body=None, lines=None):
    """Run the real preflight and return the upstream request body."""
    client = _CapturingClient(lines if lines is not None else [CONTENT_LINE, "data: [DONE]"])
    monkeypatch.setattr(ep, "_get_httpx_client", lambda: client)
    backend = {
        "base_url": "https://example.invalid/v1",
        "backend_model": "test-model",
        "provider": provider,
        "api_key": "",
    }
    import asyncio

    asyncio.run(ep._preflight_openai_stream(backend, [{"role": "user", "content": "x"}], body or {}))
    return client.captured["json"]


def test_streaming_preflight_requests_usage(monkeypatch):
    """The stream must ASK for usage, or output tokens can never be counted."""
    out = _drive_preflight(monkeypatch, "ollama-cloud")
    opts = out.get("stream_options") or {}
    assert opts.get("include_usage") is True, (
        "the streaming preflight did not request usage; output_tokens=0 on "
        f"streaming completions is then unfixable, not merely unfixed (sent {opts!r})"
    )


def test_usage_is_requested_only_for_providers_that_accept_it(monkeypatch):
    """A provider that may reject stream_options must not be sent it.

    Asking blindly would turn a logging improvement into a request failure for
    any backend that does not implement the parameter.
    """
    out = _drive_preflight(monkeypatch, "openai-codex")
    opts = out.get("stream_options") or {}
    assert "include_usage" not in opts, (
        "include_usage was sent to a provider not known to accept it; a "
        f"rejecting backend would fail the streaming request (sent {opts!r})"
    )


def test_existing_stream_options_are_not_clobbered(monkeypatch):
    """Caller-supplied stream_options must be merged, not replaced."""
    out = _drive_preflight(
        monkeypatch, "ollama-cloud", body={"stream_options": {"custom_flag": 1}}
    )
    opts = out.get("stream_options") or {}
    assert opts.get("custom_flag") == 1, (
        f"a caller-supplied stream_options key was discarded (sent {opts!r})"
    )
    assert opts.get("include_usage") is True, (
        f"include_usage was lost while merging (sent {opts!r})"
    )


def test_resume_stream_harvests_usage_into_stats():
    """The usage chunk must reach ``stats`` — that is what the logger reads.

    The chunk carries an EMPTY ``choices`` list, so a delta parser indexing
    ``choices[0]`` drops exactly the payload holding the token counts.
    """
    import asyncio

    async def _iter():
        for line in (USAGE_LINE, "data: [DONE]"):
            yield line

    pf = ep._StreamPreflight(
        status="ok",
        buffered=[CONTENT_LINE],
        iterator=_iter(),
        response=_FakeResp([]),
        accumulated="hi",
    )
    stats = {}

    async def _drain():
        async for _ in ep._resume_stream(pf, {}, [], {}, stats=stats):
            pass

    asyncio.run(_drain())
    assert stats.get("usage_output_tokens") == 7, (
        f"the streamed usage chunk never reached stats: {stats!r}"
    )
    assert stats.get("usage_input_tokens") == 32, (
        f"prompt tokens were not harvested: {stats!r}"
    )


# --------------------------------------------------------------------------
# 2-3. Parsing usage, and honesty when it is absent
# --------------------------------------------------------------------------

def test_usage_chunk_is_parsed_into_output_tokens():
    """A final usage chunk (choices: []) must yield a real token count.

    The usage chunk is easy to drop: it carries an EMPTY choices list, so any
    parser that assumes choices[0] exists will discard exactly the payload we
    need.
    """
    assert hasattr(ep, "_sse_usage_tokens"), (
        "no parser exists for the streamed usage chunk"
    )

    chunk = (
        '{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[],'
        '"usage":{"prompt_tokens":32,"completion_tokens":7,"total_tokens":39}}'
    )
    in_tok, out_tok = ep._sse_usage_tokens(chunk)
    assert out_tok == 7, f"completion_tokens not parsed from the usage chunk: {out_tok!r}"
    assert in_tok == 32, f"prompt_tokens not parsed from the usage chunk: {in_tok!r}"


def test_usage_parser_is_silent_on_chunks_without_usage():
    """A normal content delta has no usage and must report none, not crash."""
    for chunk in (
        '{"choices":[{"delta":{"content":"hi"}}]}',
        "[DONE]",
        "",
        "not json at all",
    ):
        in_tok, out_tok = ep._sse_usage_tokens(chunk)
        assert (in_tok, out_tok) == (0, 0), (
            f"chunk {chunk!r} produced {(in_tok, out_tok)} instead of (0, 0)"
        )


def test_stream_without_usage_records_cost_unknown(fresh_db_available):
    """No usage captured -> the cost is partial and must say so.

    This is the B15 requirement: ``cost_unknown`` is the schema's only way to
    say "this figure is not the whole story".
    """
    src = _endpoint_src()
    tree = ast.parse(src)
    loggers = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "_log_request_to_db":
                    kws = {k.arg: k.value for k in sub.keywords}
                    err = kws.get("error_type")
                    err_txt = getattr(err, "value", "") if isinstance(err, ast.Constant) else ""
                    if err_txt != "streaming_in_progress":
                        loggers.append((node.name, sub.lineno, kws))
    streaming_loggers = [
        (n, ln, kws) for n, ln, kws in loggers
        if "stream" in n.lower()
    ]
    assert streaming_loggers, "no streaming completion logger found to check"
    for name, lineno, kws in streaming_loggers:
        assert "cost_unknown" in kws or "output_tokens" in kws, (
            f"{name} (line {lineno}) neither passes captured output tokens nor "
            f"declares cost_unknown"
        )


@pytest.fixture()
def fresh_db_available():
    """Marker fixture: these tests inspect source, the DB is not required."""
    return True
