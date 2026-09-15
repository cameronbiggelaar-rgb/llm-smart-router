"""B34 batch 2 — the guard must not call a non-discriminating key a session.

Why this exists
---------------
Batch 1 stopped the derived key from being a system-prompt hash. That fixed the
worst collision but did NOT make the derived key a real session identity:
measured on production dumps, compaction payloads open with a constant
"[CONTEXT COMPACTION ...]" wrapper, so payloads from DIFFERENT sessions still
hashed to the same key (group pfx:0376044e... held 4 distinct state.db session
ids).

That matters because the guard RELABELS traffic when the cap trips:
``session_compression`` -> ``session_compression_overflow``, which routes away
from the summariser ladder. If the key is shared, a noisy session's churn spends
another session's budget and a healthy conversation loses compaction treatment.
Measured live: 289 overflow rows on ONE key while the guard is enabled.

The user's decision (2026-09-15): never enforce a per-session cap when the key
cannot discriminate. Enforcement is therefore gated on the key's SOURCE, which
is already recorded as ``session_id_source``:

  header / body / cache_key  -> a real identity, enforce the cap
  derived                    -> a prefix hash that collides, do NOT enforce
  none                       -> nothing to attribute, do NOT enforce

Not enforcing must never mean REJECTING: a compaction request is always served,
it is only exempt from the overflow relabel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_guard_state():
    """The guard keeps module-level state; never leak it between tests."""
    ep._COMPRESSION_GUARD_STATE.clear()
    yield
    ep._COMPRESSION_GUARD_STATE.clear()


# --------------------------------------------------------------------------
# B1. which sources are enforceable
# --------------------------------------------------------------------------

def test_enforceable_sources_are_only_real_identities():
    """Only header/body/cache_key identify a session."""
    assert ep._SESSION_ID_ENFORCEABLE_SOURCES == ("header", "body", "cache_key")


def test_real_identity_is_enforceable():
    assert ep._compression_guard_enforceable("sid:abc123", "header") is True
    assert ep._compression_guard_enforceable("pck_deadbeef", "cache_key") is True


def test_derived_key_is_not_enforceable():
    """A prefix hash collides across sessions, so it cannot carry a cap."""
    assert ep._compression_guard_enforceable("pfx:" + "a" * 32, "derived") is False


def test_unknown_source_is_not_enforceable():
    assert ep._compression_guard_enforceable("", "none") is False
    assert ep._compression_guard_enforceable("something", "") is False


# --------------------------------------------------------------------------
# B2. the cap only trips when it is enforceable
# --------------------------------------------------------------------------

def test_cap_does_not_trip_on_a_derived_key():
    """Spamming a derived key must NOT produce overflow relabelling.

    Under the old behaviour a shared key let one noisy conversation exhaust the
    budget for every other conversation sharing it.
    """
    key = "pfx:" + "b" * 32
    for _ in range(ep.COMPRESSION_GUARD_PER_SESSION_MAX * 3):
        allowed = ep._compression_guard_allows(key, "derived")
        assert allowed is True, "a non-discriminating key must never trip the cap"


def test_cap_still_trips_on_a_real_identity():
    """The runaway protection must survive for genuine session ids."""
    key = "sid:real-session-1"
    results = [
        ep._compression_guard_allows(key, "header")
        for _ in range(ep.COMPRESSION_GUARD_PER_SESSION_MAX + 3)
    ]
    # first MAX attempts allowed, then refused
    assert results[: ep.COMPRESSION_GUARD_PER_SESSION_MAX] == [True] * ep.COMPRESSION_GUARD_PER_SESSION_MAX
    assert results[ep.COMPRESSION_GUARD_PER_SESSION_MAX] is False
    assert all(r is False for r in results[ep.COMPRESSION_GUARD_PER_SESSION_MAX:])


def test_derived_traffic_does_not_consume_a_real_sessions_budget():
    """The two must not share a pot."""
    shared = "pfx:" + "c" * 32
    for _ in range(ep.COMPRESSION_GUARD_PER_SESSION_MAX * 2):
        ep._compression_guard_allows(shared, "derived")
    # a real session must still have its full allowance
    key = "sid:untouched-session"
    for i in range(ep.COMPRESSION_GUARD_PER_SESSION_MAX):
        assert ep._compression_guard_allows(key, "header") is True, f"blocked at {i}"


# --------------------------------------------------------------------------
# B3. never reject — a compaction is always served
# --------------------------------------------------------------------------

def test_guard_allows_never_raises_on_weird_input():
    """Guard bookkeeping must never break a live request."""
    for key in (None, "", "pfx:unkeyable"):
        for src in (None, "", "derived", "header", "bogus"):
            assert ep._compression_guard_allows(key, src) in (True, False)


def test_overflow_workload_is_relabel_not_reject():
    """The overflow label routes cheap; it must not be an error rejection."""
    assert ep.COMPRESSION_GUARD_OVERFLOW_WORKLOAD == "session_compression_overflow"
    assert ep.COMPRESSION_GUARD_OVERFLOW_ERROR != ""


# --------------------------------------------------------------------------
# B4. the CALL SITE must pass the resolved identity and its source
# --------------------------------------------------------------------------
# Found while implementing: the call site derived its own key from
# _compression_session_key(messages, ...) and called
# _compression_guard_allows(_guard_key) with no source. Threading `source`
# through the helpers without fixing the call site would default to "" and
# exempt EVERY request, silently disabling the runaway guard while the unit
# tests stayed green. These assert the wiring, not the helper.

def _endpoint_source():
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent / "scripts" / "biggie_llm_endpoint.py").read_text()


def test_call_site_passes_a_source_argument():
    """`_compression_guard_allows` must be called WITH a source."""
    src = _endpoint_source()
    assert "_compression_guard_allows(_guard_key, _guard_source)" in src, (
        "the guard call site must pass the resolved source, or the cap is "
        "silently exempted for every request"
    )
    assert "_compression_guard_allows(_guard_key)" not in src, (
        "a source-less call site remains - the gate would never activate"
    )


def test_call_site_prefers_a_real_identity_over_the_derived_key():
    """A resolvable real id must be used as the guarded key."""
    src = _endpoint_source()
    assert "if session_id_source in _SESSION_ID_ENFORCEABLE_SOURCES:" in src, (
        "the call site must branch on whether the resolved identity is real"
    )
    assert 'f"sid:{session_id}"' in src, (
        "a real identity must become the guard key (sid:<id>)"
    )


def test_call_site_falls_back_to_the_derived_key_non_enforceably():
    """Without a real id the guard still keys on the derived value."""
    src = _endpoint_source()
    assert "_guard_key = _compression_session_key(messages)" in src
    assert '_guard_source = session_id_source or "derived"' in src, (
        "the fallback must record a non-enforceable source, not a real one"
    )
