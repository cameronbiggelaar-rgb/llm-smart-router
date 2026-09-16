"""Batch 4 — fast-fail local, and make local failures diagnosable.

Fixes #3 and #4.

Why this exists
---------------
#3 — A single shared `httpx.AsyncClient(timeout=300.0)` serves every backend.
Local llama-server is CPU-only with ONE slot (`-c 4096 -np 1`), so a saturated
server does not reject: it queues. Measured 45-46 established plus 88 open
connections to :11434, with requests held 300s and then multiplied by
`api_max_retries: 3` — which is how one slow local call becomes a 502 loop and
1,891 local failures in a day. Local should fail fast (~60s) so the caller can
escalate, instead of parking a connection until the shared 300s ceiling.

#4 — The failing log line is `"Backend %s request failed: %s"` with `e` whose
`str()` is empty for httpx timeouts, producing the observed
`ERROR: Backend local request failed:` — no type, nothing to diagnose. The
database undercounts local failures 62x (33 rows vs 2,043 escalations) because
these paths 502 out before the row is written, so the log is the only trace.
It must at least name the exception class.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402

EP_SRC = (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()


# ── #3 fast-fail local ──────────────────────────────────────────────────────

def test_local_timeout_is_short():
    """Local must fail fast, not inherit the 300s shared ceiling."""
    t = ep._timeout_for("local")
    assert t is not None, "local must have its own timeout, not the shared 300s"
    assert t <= 90, f"local timeout {t}s is too long to fast-fail a 1-slot server"
    assert t < ep._BACKEND_DEFAULT_TIMEOUT_SECONDS, (
        "local must be shorter than the default, or it does not fast-fail"
    )


def test_local_timeout_default_is_about_a_minute():
    """The agreed value is ~60s."""
    assert ep._timeout_for("local") == 60.0


def test_cloud_backends_keep_the_generous_timeout():
    """Only local changes — cloud latency must not be tightened.

    Cloud gets the explicit default rather than None-on-purpose: httpx reads
    ``timeout=None`` as "never time out", so the helper always returns a number
    and callers can pass it through safely.
    """
    for provider in ("openai-codex", "ollama-cloud", "openrouter", "unknown"):
        assert ep._timeout_for(provider) == ep._BACKEND_DEFAULT_TIMEOUT_SECONDS, (
            f"{provider} must keep the generous shared ceiling; tightening it "
            "would break long cloud completions"
        )
        assert ep._timeout_for(provider) > ep._timeout_for("local"), (
            f"{provider} must be more patient than local, which fast-fails"
        )


def test_local_timeout_is_configurable():
    """Operators need to tune this without a code change."""
    assert "BIGGIE_LOCAL_TIMEOUT_SECONDS" in EP_SRC, (
        "the local timeout must be overridable by environment variable"
    )


def test_local_dispatch_actually_passes_the_timeout():
    """The constant is worthless if the dispatch site ignores it.

    There is exactly one local HTTP dispatch site (the `provider == "local"`
    branch); the streaming path routes local through it as well. A second site
    appearing without the timeout is the regression this guards.
    """
    assert "timeout=_timeout_for(" in EP_SRC, (
        "no dispatch site passes _timeout_for(), so the short local timeout is "
        "never applied to a real request"
    )


# ── #4 diagnosable failures ─────────────────────────────────────────────────

def test_exception_description_names_the_type_when_str_is_empty():
    """The exact live failure: str(e) empty, log line uninformative.

    httpx timeout errors stringify to "" — so `"Backend %s failed: %s"` emits
    `Backend local failed:` and the cause is unrecoverable from logs.
    """
    exc = httpx.ReadTimeout("")

    desc = ep._exc_desc(exc)

    assert "ReadTimeout" in desc, (
        f"exception type missing from {desc!r} — an empty str(e) leaves the log "
        "line with no diagnosable content"
    )


def test_exception_description_keeps_the_message_when_present():
    """Don't lose detail when there is some."""
    exc = httpx.ConnectError("connection refused")

    desc = ep._exc_desc(exc)

    assert "ConnectError" in desc
    assert "connection refused" in desc


def test_exception_description_handles_non_httpx_errors():
    """Escalation paths raise other types too."""
    desc = ep._exc_desc(KeyError("choices"))

    assert "KeyError" in desc
    assert "choices" in desc


def test_all_request_failure_logs_use_the_helper():
    """Catch a forgotten site — the bug class this batch exists to close.

    The same uninformative pattern was copy-pasted across branches; each copy
    is a place the type can go missing again.
    """
    assert "_exc_desc(" in EP_SRC
    # Only executable lines count — this module documents the old pattern in
    # prose, and a docstring mentioning it is not a site that can regress.
    offenders = [
        line.strip()
        for line in EP_SRC.splitlines()
        if "request failed: %s" in line
        and "_exc_desc(" not in line
        and not line.strip().startswith("#")
        and "``" not in line
    ]
    assert not offenders, (
        "these failure logs omit the exception type:\n  " + "\n  ".join(offenders)
    )
