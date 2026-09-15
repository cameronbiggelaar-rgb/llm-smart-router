"""B31 batch 1 — a runaway compaction loop must be capped per session.

Why this exists
---------------
Measured 2026-09-14 over 24h: 6,059 ``session_compression`` calls carrying
798,171,007 input tokens — 99.6% of ALL router input — against 3,360,284 tokens
of real chat. Average input per compaction call was ~132,000 tokens producing
~1,371 output tokens, i.e. a summariser being asked to re-read a huge prompt
thousands of times a day for a ~3% reported saving.

These are *distinct* requests (amplification 1.00x — 6,068 of 6,068 request_ids
were unique), so this is not retry amplification; the caller genuinely issues
thousands of compactions. Live logs show why: ``Session hygiene: compressed 650
-> 650 msgs`` — 237 of 697 logged compactions removed ZERO messages and 265 did
not reduce tokens at all, because the configured threshold (60,000) sits below
the incompressible floor (system prompt + tool schemas). The pass cannot get
under the line, so the next turn compacts again, forever.

Hermes core owns that loop and its hygiene ``threshold_pct`` is hardcoded, so
the containment point is the router: count compactions per session in a rolling
window and, past the cap, stop treating them as a compaction — route them cheap
and label them, but NEVER reject the request.

These tests exercise the real helper rather than re-implementing the policy, so
they fail if the wiring is dropped even when the counter itself is correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402


def _compression_messages() -> list:
    """A prompt the real detector classifies as session_compression."""
    return [
        {
            "role": "user",
            "content": (
                "Context compaction task: summarize earlier conversation so I can "
                "continue. Preserve key facts, decisions, and file paths."
            ),
        }
    ]


@pytest.fixture(autouse=True)
def _reset_guard():
    """Each test starts from a clean budget table."""
    ep._COMPRESSION_GUARD_STATE.clear()
    yield
    ep._COMPRESSION_GUARD_STATE.clear()


def test_compression_guard_records_and_counts_calls():
    """The first N compactions for a session are allowed and counted."""
    sid = "sess-alpha"
    for i in range(3):
        assert ep._compression_guard_allows(sid, "header") is True, f"call {i} should be allowed"
    assert ep._compression_call_count(sid) == 3


def test_compression_guard_denies_past_the_cap():
    """Once the per-session cap is reached, further compactions are flagged."""
    sid = "sess-beta"
    cap = ep.COMPRESSION_GUARD_PER_SESSION_MAX
    for _ in range(cap):
        assert ep._compression_guard_allows(sid, "header") is True
    # The cap+1'th call is the first denied one.
    assert ep._compression_guard_allows(sid, "header") is False
    assert ep._compression_guard_allows(sid, "header") is False


def test_compression_guard_is_per_session_not_global():
    """One thrashing session must not consume another session's budget."""
    a, b = "sess-a", "sess-b"
    cap = ep.COMPRESSION_GUARD_PER_SESSION_MAX
    for _ in range(cap):
        ep._compression_guard_allows(a, "header")
    assert ep._compression_guard_allows(a, "header") is False
    # A different session still has its own full budget.
    assert ep._compression_guard_allows(b, "header") is True


def test_compression_guard_window_expiry_resets_budget():
    """Budget is a ROLLING window: old calls age out and the session recovers.

    A hard lifetime cap would permanently cripple a long-lived session, so the
    window must be time-based.
    """
    sid = "sess-gamma"
    cap = ep.COMPRESSION_GUARD_PER_SESSION_MAX
    for _ in range(cap):
        ep._compression_guard_allows(sid, "header")
    assert ep._compression_guard_allows(sid, "header") is False

    # Age every recorded call past the window.
    now = ep._compression_guard_now()
    window = ep.COMPRESSION_GUARD_WINDOW_SECONDS
    ep._COMPRESSION_GUARD_STATE[sid] = [
        t - (window + 1) for t in ep._COMPRESSION_GUARD_STATE[sid]
    ]
    assert ep._compression_guard_allows(sid, "header") is True, "expired calls must free budget"


def test_guard_is_disabled_when_configured_off(monkeypatch):
    """`compression_guard.enabled: false` must make the guard inert."""
    monkeypatch.setattr(ep, "COMPRESSION_GUARD_ENABLED", False)
    sid = "sess-delta"
    for _ in range(ep.COMPRESSION_GUARD_PER_SESSION_MAX * 3):
        assert ep._compression_guard_allows(sid, "header") is True


def test_session_key_is_stable_across_turns_and_distinct_between_sessions():
    """The guard key must identify a session without a session_id.

    Every compaction row in `router_logs` has a BLANK session_id (verified:
    `session_compression||4929`), so the key has to be derived from the prompt
    prefix. It must be stable for the same session across turns, or the cap
    would never trigger, and different for different sessions, or one session
    would throttle another.
    """
    msgs = _compression_messages()
    k1 = ep._compression_session_key(msgs, session_id="")
    k2 = ep._compression_session_key(msgs, session_id="")
    assert k1 == k2, "same prefix must give the same key"

    other = [
        {
            "role": "user",
            "content": (
                "Context compaction task: summarize a COMPLETELY different earlier "
                "conversation about marine biology. Preserve key facts."
            ),
        }
    ]
    k3 = ep._compression_session_key(other, session_id="")
    assert k3 != k1, "different sessions must not collide"


def test_explicit_session_id_wins_over_prefix_hash():
    """When a caller does supply a session id, it is authoritative."""
    msgs = _compression_messages()
    assert ep._compression_session_key(msgs, session_id="abc") == "sid:abc"


def test_overflow_is_labelled_not_rejected():
    """Past the cap the request is STILL SERVED — labelled, never failed.

    Rejecting a compaction would wedge the calling session with no way to
    shrink, so the guard must degrade to "route cheap + record", not "error".
    """
    sid = "sess-eps"
    cap = ep.COMPRESSION_GUARD_PER_SESSION_MAX
    for _ in range(cap):
        ep._compression_guard_allows(sid, "header")
    assert ep._compression_guard_allows(sid, "header") is False
    # The overflow marker is what the request path attaches to the log row.
    assert ep.COMPRESSION_GUARD_OVERFLOW_WORKLOAD == "session_compression_overflow"
    assert ep.COMPRESSION_GUARD_OVERFLOW_ERROR == "compression_budget_exceeded"


def test_guard_helpers_never_raise_on_malformed_input():
    """Guard bookkeeping must never break a live request."""
    assert ep._compression_guard_allows("", "header") in (True, False)
    assert isinstance(ep._compression_session_key(None, session_id=""), str)
    assert isinstance(ep._compression_session_key([], session_id=""), str)
    assert isinstance(ep._compression_session_key(["not-a-dict"], session_id=""), str)


def test_guard_is_wired_into_the_request_path():
    """The guard must actually be CALLED by the handler, not merely defined.

    A passing unit test on `_compression_guard_allows` proves the counter works
    but not that anything consults it — the original defect was a working
    mechanism nobody invoked. Assert the handler source references the guard.
    """
    import inspect

    src = inspect.getsource(ep)
    assert "_compression_session_key(" in src
    assert "_compression_guard_allows(" in src
    # The overflow label must be applied to the workload, not just computed.
    assert "COMPRESSION_GUARD_OVERFLOW_WORKLOAD" in src
    assert "_compression_guard_overflow" in src


# ── The overflow label must ROUTE like a compaction, not fall through ─────────

def test_overflow_label_routes_as_compression_not_tool_gate():
    """A budget-exceeded compaction must not FAIL CLOSED.

    The endpoint relabels a runaway compaction `session_compression_overflow`.
    Compaction prompts inherit the parent session's tool schemas, so if that
    label missed the session_compression branch it would fall through to the
    tool-capability gate and fail closed when no tool model is free — wedging
    the very session the guard exists to protect. This pins the routing.
    """
    import inspect

    import router

    src = inspect.getsource(router)
    assert "session_compression_overflow" in src, (
        "router must recognise the overflow label"
    )
    # It must appear in BOTH the routing branch and the escalation branch.
    assert src.count("session_compression_overflow") >= 4, (
        "the overflow label must be handled in both the route and escalate paths"
    )


def test_overflow_label_uses_the_summariser_ladder():
    """It must escalate on the summariser ladder, never the ChatGPT plan."""
    import inspect

    import router

    src = inspect.getsource(router)
    idx = src.find("session_compression_overflow")
    assert idx != -1
    window = src[max(0, idx - 400): idx + 2000]
    assert "_next_session_compression_rung" in window or "_select_session_compression_model" in window, (
        "the overflow label must stay on the summariser ladder"
    )


def test_source_less_key_is_exempt_from_the_cap():
    """B34 contract change, pinned deliberately.

    The guard used to apply to ANY key, including the derived prompt-prefix key.
    That key is not a session identity - it collides across sessions sharing an
    opening wrapper - so the cap spent one session's budget on another's churn
    (measured: 289 overflow rows landing on ONE key).

    User decision 2026-09-15: never enforce a per-session cap on a key that
    cannot discriminate. A source-less call is therefore EXEMPT, not capped.

    Consequence, stated plainly: for traffic that carries no real session id on
    the wire (the chat-completions path sends none), the overflow relabel no
    longer fires. Runaway protection for that traffic now depends on fixing the
    cause (B33), not on this band-aid.
    """
    sid = "sess-source-less"
    cap = ep.COMPRESSION_GUARD_PER_SESSION_MAX
    for _ in range(cap * 3):
        assert ep._compression_guard_allows(sid) is True, (
            "a key with no real identity must never be capped"
        )
    for _ in range(cap * 3):
        assert ep._compression_guard_allows("pfx:" + "d" * 32, "derived") is True


def test_cap_still_applies_when_a_real_identity_is_present():
    """The runaway protection must survive for genuine session ids."""
    sid = "sess-real"
    cap = ep.COMPRESSION_GUARD_PER_SESSION_MAX
    for _ in range(cap):
        assert ep._compression_guard_allows(sid, "header") is True
    assert ep._compression_guard_allows(sid, "header") is False
