"""B32 — measurement integrity: session attribution + timestamp safety.

Why this exists
---------------
Three cost analyses in the 2026-09-14/15 session produced WRONG numbers, each from
the same class of flaw: comparing values whose format or window did not line up.

1. "$562/day from compression" — a billing artifact. ``cached_tokens`` was
   hardcoded 0, so every cache hit was invoiced at the full input rate.
2. "428/hr -> 2.4/hr, a 26x cost drop" — compared an IDLE 5-hour window against a
   busy one. Nothing had improved.
3. "compaction escalates to gpt-5.6-sol, ~$840/day" — the SQL filter used
   ``'2026-09-14 18:20'`` (SPACE) against stored ``'2026-09-14T18:20'`` (T).
   Because ``'T' > ' '``, every row compared greater: a "6-hour window" silently
   matched 7,205 rows, about 24 hours. Measured truth was $0 post-fix.

Plus a timezone error (logs are AEST+10, the DB is UTC, compared directly) and a
genuine data gap: ``session_id`` was hardcoded blank with the comment "not
available at endpoint level" — which is false, the value is available at the call
site. That blank made "940 compactions from ONE session" an inference from the
*blank key*, not a measurement.

These tests pin the two fixes so the same errors cannot recur silently. They
exercise the real helpers rather than re-implementing policy, so they fail if the
wiring is dropped even when the helper itself is correct.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402


# --------------------------------------------------------------------------
# A1. session attribution
# --------------------------------------------------------------------------

def test_session_id_resolved_from_header_wins():
    """An explicit session header is authoritative."""
    got, src = ep.resolve_session_identity(
        headers={"x-hermes-session-id": "20260915_101010_abcdef"},
        body={},
        messages=[{"role": "user", "content": "hello"}],
    )
    assert got == "20260915_101010_abcdef"
    assert src == "header"


def test_session_id_resolved_from_body_when_no_header():
    """Hermes' own body field is second choice."""
    got, src = ep.resolve_session_identity(
        headers={},
        body={"session_id": "body-sid-123"},
        messages=[{"role": "user", "content": "hello"}],
    )
    assert got == "body-sid-123"
    assert src == "body"


def test_session_id_falls_back_to_body_user_field():
    """Some callers only set 'user'."""
    got, src = ep.resolve_session_identity(
        headers={},
        body={"user": "user-field-999"},
        messages=[{"role": "user", "content": "hello"}],
    )
    assert got == "user-field-999"
    assert src == "body"


def test_session_id_derives_prefix_key_when_nothing_provided():
    """With no identity on the wire, derive a stable key from the prompt prefix."""
    msgs = [{"role": "user", "content": "Context compaction task: summarize earlier conversation " * 30}]
    got, src = ep.resolve_session_identity(headers={}, body={}, messages=msgs)
    assert src == "derived"
    assert got.startswith("pfx:")
    # stable across identical prefixes
    again, _ = ep.resolve_session_identity(headers={}, body={}, messages=msgs)
    assert again == got


def test_derived_key_is_stable_across_growing_turns():
    """The whole point of a prefix key: it must not change as the session grows."""
    opening = "Context compaction task: summarize earlier conversation. " * 40
    turn1 = [{"role": "user", "content": opening}]
    turn2 = [
        {"role": "user", "content": opening},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "more later content that grows the session"},
    ]
    k1, s1 = ep.resolve_session_identity(headers={}, body={}, messages=turn1)
    k2, s2 = ep.resolve_session_identity(headers={}, body={}, messages=turn2)
    assert s1 == s2 == "derived"
    assert k1 == k2, "prefix key must be stable across turns of one session"


def test_derived_keys_differ_between_different_sessions():
    a, _ = ep.resolve_session_identity(headers={}, body={}, messages=[{"role": "user", "content": "session A opening prompt"}])
    b, _ = ep.resolve_session_identity(headers={}, body={}, messages=[{"role": "user", "content": "session B opening prompt"}])
    assert a != b


def test_identical_openings_collide_but_are_labelled_derived():
    """Honest limitation: two sessions sharing an opening prompt share the derived key.

    The provenance label is what makes this safe. A 'derived' row must never be
    read as proof of a distinct session — that was the original attribution error.
    """
    msgs = [{"role": "user", "content": "identical opening prompt for both"}]
    k1, s1 = ep.resolve_session_identity(headers={}, body={}, messages=msgs)
    k2, s2 = ep.resolve_session_identity(headers={}, body={}, messages=msgs)
    assert k1 == k2
    assert s1 == s2 == "derived", "must not claim real session attribution"


def test_resolve_session_identity_never_raises():
    """Guard/logging bookkeeping must never break a live request."""
    for bad in (None, 12345, object(), {"weird": object()}):
        got, src = ep.resolve_session_identity(headers=bad, body=bad, messages=bad)
        assert isinstance(got, str)
        assert src in {"header", "body", "derived", "none"}


def test_log_request_to_db_accepts_and_stores_session_id():
    """The writer must carry the identity through to the row."""
    import inspect

    sig = inspect.signature(ep._log_request_to_db)
    assert "session_id" in sig.parameters, "session_id must be a writer parameter"
    assert "session_id_source" in sig.parameters, (
        "provenance must be stored too, or 'derived' and real ids are conflated"
    )


def test_log_row_no_longer_hardcodes_blank_session_id():
    """Pin the actual defect: the INSERT tuple hardcoded a blank session_id.

    Asserts on the tuple line, not the file text: the file legitimately
    *mentions* the old comment when explaining why it was wrong.
    """
    src = (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()
    assert re.search(r'^\s*""\s*,\s*#\s*session_id', src, re.M) is None, (
        "the INSERT tuple hardcodes a blank session_id again"
    )
    assert re.search(r"^\s*session_id or \"\"\s*,?$", src, re.M) is not None, (
        "the INSERT tuple no longer writes the resolved session_id"
    )


# --------------------------------------------------------------------------
# A2. timestamp safety
# --------------------------------------------------------------------------

def test_db_window_uses_iso_T_separator():
    """The exact bug: a space-separated bound silently matches every row."""
    from timerange import db_window

    lo, hi = db_window(hours=6)
    assert "T" in lo and "T" in hi
    assert " " not in lo and " " not in hi


def test_db_window_bounds_sort_correctly_against_stored_values():
    """SQLite compares timestamps as TEXT, so the bounds must sort correctly."""
    from timerange import db_window

    # Anchor 'now' so the window provably brackets the stored row. Using the
    # real clock would make this assertion time-dependent (it would silently
    # stop testing anything once the row aged out).
    from datetime import datetime, timezone

    lo, hi = db_window(hours=6, now=datetime(2026, 9, 14, 22, 20, tzinfo=timezone.utc))
    stored_lo = "2026-09-14T18:20:00.000000+00:00"
    # THE BUG, demonstrated: a space-separated lower bound sorts BELOW every
    # stored value, so "timestamp >= lo" is true for the whole table.
    assert stored_lo > "2026-09-14 18:20", "precondition: space form sorts low"
    # The T form the helper returns is a real, discriminating comparison.
    assert lo <= stored_lo <= hi, f"real window must contain the stored row: {lo}..{hi}"

    # And the window genuinely excludes something older than itself.
    older = "2026-09-01T00:00:00.000000+00:00"
    assert not (lo <= older), "window must not match rows from weeks ago"


def test_space_separated_bound_is_rejected_loudly():
    """Fail loudly rather than silently returning a 24h window."""
    from timerange import assert_iso_bound

    with pytest.raises(ValueError):
        assert_iso_bound("2026-09-14 18:20")
    # the correct form passes
    assert_iso_bound("2026-09-14T18:20:00")


def test_clock_mismatch_is_detected():
    """Logs are AEST(+10); the DB is UTC. Comparing them directly is the trap."""
    from timerange import assert_same_clock

    # The legitimate case: AEST log line and the UTC DB row for the SAME event
    # (07:45:37+10:00 == 21:45:37Z the previous day) must be accepted.
    assert assert_same_clock(
        "2026-09-15T07:45:37+10:00", "2026-09-14T21:45:37+00:00"
    ) is True

    # aware vs naive must be flagged: you cannot compare a bare log line to UTC.
    with pytest.raises(ValueError):
        assert_same_clock("2026-09-15 07:45:37", "2026-09-14T21:45:37+00:00")

    # The ten-hour skew: reading an AEST log line as if it were UTC.
    with pytest.raises(ValueError):
        assert_same_clock("2026-09-15T07:45:37+00:00", "2026-09-14T21:45:37+00:00")


# --------------------------------------------------------------------------
# A3. drift detector — every writer call site must carry identity
# --------------------------------------------------------------------------

def test_all_log_request_to_db_call_sites_pass_session_id():
    """If a new call site forgets identity, attribution silently degrades again.

    A site is correct if it either passes ``session_id=`` explicitly or splats a
    kwargs dict that carries it (``**_obs`` is the common case).
    """
    src = (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()

    # The shared kwargs dict must carry identity, since 4 sites splat it.
    obs = re.search(r"_obs = \{(.*?)\n    \}", src, re.S)
    assert obs, "could not locate the _obs kwargs dict"
    assert "session_id" in obs.group(1), "_obs must carry session_id"
    assert "session_id_source" in obs.group(1), "_obs must carry session_id_source"

    # Every real call site must pass it explicitly or splat a dict that has it.
    calls = [m.start() for m in re.finditer(r"_log_request_to_db\(", src)]
    real = []
    for pos in calls:
        # skip the definition itself
        if re.search(r"def _log_request_to_db\(", src[pos - 4:pos + 20]):
            continue
        real.append(src[pos:pos + 2500])

    assert real, "expected call sites"

    missing = []
    for chunk in real:
        explicit = "session_id=" in chunk
        splatted = bool(re.search(r"\*\*\w+", chunk)) and "session_id" in obs.group(1)
        if not (explicit or splatted):
            missing.append(chunk[:60].replace("\n", " "))
    assert not missing, f"call sites missing session identity: {missing}"
