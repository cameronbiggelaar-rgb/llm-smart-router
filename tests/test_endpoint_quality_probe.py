"""Batch 3b — the endpoint actually probes compression quality.

`quality_probe` being correct is not the deliverable; the deliverable is that
the RUNNING endpoint measures the summaries it serves. These tests exercise the
endpoint's own hook, so a future refactor that drops the callback fails here
instead of silently returning the fleet to unmeasured.

The hook must also respect the cost/latency contract: it is sampled, it ignores
non-summarisation work, and it can never raise into a request that has already
been served.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402
import quality_probe as qp  # noqa: E402
from rollup import migrate  # noqa: E402

SOURCE = "Build used 12,345 tokens across 250 files with 8,000 lines."
GOOD = "Build: 12,345 tokens, 8,000 lines."


@pytest.fixture()
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "e2e.db")
    migrate(c)
    qp.migrate(c)
    return c


def test_endpoint_hook_records_a_probe(conn, monkeypatch):
    """The endpoint's own helper must write a measurement."""
    monkeypatch.setattr(ep, "_get_db_connection", lambda: conn)
    monkeypatch.setenv("BIGGIE_QUALITY_PROBE_RATE", "1.0")

    ep._probe_compression_quality(
        request_id="req-hook-1",
        workload_type="session_compression",
        source_messages=[{"role": "user", "content": SOURCE}],
        summary_text=GOOD,
        model="deepseek-v4.1-flash",
    )
    row = conn.execute(
        "SELECT score, model, workload FROM quality_probe WHERE request_id = ?",
        ("req-hook-1",),
    ).fetchone()
    assert row is not None, "the endpoint hook did not record a probe"
    assert row[0] is not None and row[0] > 0
    assert row[1] == "deepseek-v4.1-flash"
    assert row[2] == "session_compression"


def test_endpoint_hook_ignores_non_compression_workloads(conn, monkeypatch):
    """Ordinary chat is not a summarisation task and must not be scored."""
    monkeypatch.setattr(ep, "_get_db_connection", lambda: conn)
    monkeypatch.setenv("BIGGIE_QUALITY_PROBE_RATE", "1.0")

    ep._probe_compression_quality(
        request_id="req-chat-1",
        workload_type="normal_chat",
        source_messages=[{"role": "user", "content": SOURCE}],
        summary_text=GOOD,
        model="glm-5.3",
    )
    n = conn.execute("SELECT COUNT(*) FROM quality_probe").fetchone()[0]
    assert n == 0


def test_endpoint_hook_respects_sampling(conn, monkeypatch):
    """At rate 0 the hot path must do nothing at all."""
    monkeypatch.setattr(ep, "_get_db_connection", lambda: conn)
    monkeypatch.delenv("BIGGIE_QUALITY_PROBE_RATE", raising=False)
    monkeypatch.setattr(qp, "DEFAULT_RATE", 0.0)

    for i in range(20):
        ep._probe_compression_quality(
            request_id=f"req-s{i}",
            workload_type="session_compression",
            source_messages=[{"role": "user", "content": SOURCE}],
            summary_text=GOOD,
            model="deepseek-v4.1-flash",
        )
    n = conn.execute("SELECT COUNT(*) FROM quality_probe").fetchone()[0]
    assert n == 0


def test_endpoint_hook_never_raises_into_a_served_request(monkeypatch):
    """A broken connection must not propagate - the response is already sent."""

    def exploding_conn():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(ep, "_get_db_connection", exploding_conn)
    monkeypatch.setenv("BIGGIE_QUALITY_PROBE_RATE", "1.0")

    # must return normally, not raise
    ep._probe_compression_quality(
        request_id="req-boom",
        workload_type="session_compression",
        source_messages=[{"role": "user", "content": SOURCE}],
        summary_text=GOOD,
        model="deepseek-v4.1-flash",
    )


def test_endpoint_hook_skips_an_empty_summary(conn, monkeypatch):
    """An empty summary is an escalation case, not a quality measurement."""
    monkeypatch.setattr(ep, "_get_db_connection", lambda: conn)
    monkeypatch.setenv("BIGGIE_QUALITY_PROBE_RATE", "1.0")

    ep._probe_compression_quality(
        request_id="req-empty",
        workload_type="session_compression",
        source_messages=[{"role": "user", "content": SOURCE}],
        summary_text="   ",
        model="deepseek-v4.1-flash",
    )
    n = conn.execute("SELECT COUNT(*) FROM quality_probe").fetchone()[0]
    assert n == 0


def test_streaming_path_passes_the_callback_through(monkeypatch):
    """The streaming relay must forward on_summary to the resume generator.

    Without this the hook exists but is never invoked: every probe test would
    still pass while production measured nothing. Assert the wiring, not just
    the helper.
    """
    import asyncio

    captured = {}

    async def fake_resume(pf, backend, messages, body, on_complete=None, on_summary=None):
        captured["on_summary"] = on_summary
        if False:
            yield ""

    monkeypatch.setattr(ep, "_resume_stream", fake_resume)

    class FakePF:
        status = "ok"
        buffered = ["data: {}"]
        saw_content = True
        saw_tool_calls = False
        response = None
        iterator = None
        provider = "ollama-cloud"
        backend_model = "deepseek-v4.1-flash"

    async def fake_preflight(backend, messages, body):
        return FakePF()

    monkeypatch.setattr(ep, "_preflight_openai_stream", fake_preflight)

    sentinel = lambda s: None  # noqa: E731

    async def drive():
        resp = await ep.proxy_to_backend_streaming(
            {"base_url": "http://x", "backend_model": "m", "provider": "ollama-cloud"},
            [{"role": "user", "content": "hi"}],
            {},
            on_complete=None,
            stats={},
            on_summary=sentinel,
        )
        # A StreamingResponse body is the async generator; draining it is what
        # actually executes the relay and therefore the forwarding under test.
        async for _ in resp.body_iterator:
            pass
        return resp

    resp = asyncio.run(drive())
    assert resp is not None
    assert captured.get("on_summary") is sentinel, "on_summary was not forwarded"
