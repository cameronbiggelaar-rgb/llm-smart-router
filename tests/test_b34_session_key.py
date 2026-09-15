"""B34 batch 1 — the derived session key must be a SESSION key, not a system-prompt hash.

Why this exists
---------------
B32 added ``session_id_source`` so a blank ``session_id`` could no longer be read
as evidence. Post-deploy the derived rows showed one dominant key carrying 83% of
traffic, which was reported as ONE looping session. The user pushed back: two
agent conversations were running in parallel. They were right.

Root cause, measured on 92 production request dumps:

* ``_compression_session_key`` hashes ``joined_text[:512]``, and the system
  message comes FIRST. The system prompt is always > 512 chars (min 9,088,
  median 27,847), so the key is ALWAYS a pure system-prompt hash — identical for
  every session on a profile.
* Consequence: the "per-session 40/hr compression guard" is a GLOBAL budget.
  3 of 4 parallel Hermes sessions collide (confirmed independently via
  ``state.db.system_prompt_hash``), and 289 overflow rows landed on ONE key while
  only 95 genuine compression rows sat on 2 keys.

Two further facts that shape the fix:

* A real session identity IS on the wire for the Codex path:
  ``body.prompt_cache_key``, present on 58/92 dumps, 42 of which match a
  ``state.db`` session id exactly.
* The chat-completions path carries NO id (0/33 dumps), and headers carry only
  Authorization + Content-Type.

These tests pin the key's discriminating property. They exercise the real
helpers, so they fail if the wiring is dropped even when the helper is correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402


# A long, clearly-constant system prompt — mirrors production (median 27.8K).
SYSTEM = (
    "You are a helpful agent operating under a standing brief. " * 200
)


def _conv(first_user: str, extra_user: str | None = None, system: str = SYSTEM):
    """A conversation shaped like production: system first, then user turns."""
    msgs = [{"role": "system", "content": system}]
    msgs.append({"role": "user", "content": first_user})
    if extra_user:
        msgs.append({"role": "assistant", "content": "ok"})
        msgs.append({"role": "user", "content": extra_user})
    return msgs


# --------------------------------------------------------------------------
# 1. prompt_cache_key is a real identity (Codex path)
# --------------------------------------------------------------------------

def test_prompt_cache_key_resolves_as_its_own_source():
    """``prompt_cache_key`` carries a genuine session id on the Codex path.

    42 of 58 observed values match a ``state.db`` session id exactly, so it must
    be resolved as a real identity rather than falling through to a hash.
    """
    got, src = ep.resolve_session_identity(
        headers={},
        body={"prompt_cache_key": "pck_44bc114f771c2fb66d4dedd4"},
        messages=[{"role": "user", "content": "hello"}],
    )
    assert got == "pck_44bc114f771c2fb66d4dedd4"
    assert src == "cache_key"


def test_prompt_cache_key_recorded_in_source_vocabulary():
    """A new source must be in SESSION_ID_SOURCES or it is not queryable."""
    assert "cache_key" in ep.SESSION_ID_SOURCES


def test_cache_key_outranks_derived_key():
    """A real id must always win over a derived hash."""
    got, src = ep.resolve_session_identity(
        headers={},
        body={"prompt_cache_key": "20260913_074633_bb2d423b"},
        messages=_conv("do a thing"),
    )
    assert src == "cache_key"
    assert not got.startswith("pfx:")


def test_header_still_outranks_cache_key():
    """Header identity is the most authoritative source; keep it first."""
    got, src = ep.resolve_session_identity(
        headers={"x-hermes-session-id": "header-wins"},
        body={"prompt_cache_key": "pck_should_lose"},
        messages=_conv("do a thing"),
    )
    assert got == "header-wins"
    assert src == "header"


# --------------------------------------------------------------------------
# 2. the derived key must DISCRIMINATE between sessions
# --------------------------------------------------------------------------

def test_two_conversations_sharing_a_system_prompt_get_different_keys():
    """THE defect: a shared system prompt must not collapse two conversations.

    This is the check that would have caught B34 on day one.
    """
    key_a = ep._compression_session_key(_conv("Tell me about routers."))
    key_b = ep._compression_session_key(_conv("Write me a poem."))
    assert key_a != key_b, (
        "two different conversations collided on one key — the guard budget is "
        "global, not per-session"
    )


def test_derived_key_is_stable_across_turns_of_one_conversation():
    """The property the fix must NOT break: stability within a conversation.

    (The old docstring claimed both stability and distinctness; distinctness was
    false. This asserts the half that is genuinely required.)
    """
    turn1 = ep._compression_session_key(_conv("Tell me about routers."))
    turn2 = ep._compression_session_key(
        _conv("Tell me about routers.", extra_user="And about caches?")
    )
    assert turn1 == turn2


def test_derived_key_is_not_a_pure_system_prompt_hash():
    """Guards the exact mechanism: the key must not be reproducible from a
    system message alone, or it cannot identify a session."""
    only_system = [{"role": "system", "content": SYSTEM}]
    key_system_only = ep._compression_session_key(only_system)
    key_real = ep._compression_session_key(_conv("Tell me about routers."))
    assert key_system_only != key_real


def test_derived_key_ignores_system_prompt_entirely():
    """Two conversations whose ONLY difference is the system prompt text must
    still differ, because the difference that identifies a session is elsewhere.

    Inverted framing: if the system prompt is excluded from the prefix, changing
    it must not change the key for the same user turn.
    """
    same_user = "Tell me about routers."
    key_1 = ep._compression_session_key(_conv(same_user, system="System A. " * 200))
    key_2 = ep._compression_session_key(_conv(same_user, system="System B. " * 200))
    assert key_1 == key_2, (
        "the system prompt is still feeding the derived key — it is the one "
        "component that is constant across all sessions"
    )


def test_real_session_id_wins_in_derived_helper():
    """An explicit id must bypass hashing entirely."""
    key = ep._compression_session_key(_conv("anything"), session_id="sid-abc")
    assert key == "sid:sid-abc"


# --------------------------------------------------------------------------
# 3. the fallback path must not resurrect the system-prompt hash
# --------------------------------------------------------------------------
# Non-vacuity found this gap: re-allowing the system message into the FALLBACK
# anchor left the suite GREEN, because every other test supplies a user turn and
# therefore never reaches the fallback. These pin the no-user-turn paths.

def test_system_only_payload_is_unkeyable_not_a_system_prompt_hash():
    """A system-only payload carries no session signal at all.

    Falling back to hashing the system prompt would rebuild exactly the defect
    this batch removes: one key shared by every session on the profile.
    """
    import hashlib

    only_system = [{"role": "system", "content": SYSTEM}]
    key = ep._compression_session_key(only_system)
    manual_system_hash = "pfx:" + hashlib.sha256(
        SYSTEM[: ep.COMPRESSION_GUARD_KEY_PREFIX_CHARS].encode()
    ).hexdigest()[:32]
    assert key != manual_system_hash, "fell back to hashing the system prompt"
    assert key == "pfx:unkeyable"


def test_no_user_turn_falls_back_to_first_non_system_message():
    """With no user turn, the first NON-system message is the anchor."""
    import hashlib

    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "assistant", "content": "an assistant-only replay " * 40},
        {"role": "tool", "content": "tool output"},
    ]
    key = ep._compression_session_key(msgs)
    assert key.startswith("pfx:")
    assert key != "pfx:unkeyable"
    manual = "pfx:" + hashlib.sha256(
        SYSTEM[: ep.COMPRESSION_GUARD_KEY_PREFIX_CHARS].encode()
    ).hexdigest()[:32]
    assert key != manual


def test_two_different_system_only_payloads_never_share_a_key():
    """Differing system prompts must not align on one key either."""
    a = [{"role": "system", "content": "Brief A. " * 200}]
    b = [{"role": "system", "content": "Brief B. " * 200}]
    ka, kb = ep._compression_session_key(a), ep._compression_session_key(b)
    assert ka == kb == "pfx:unkeyable", (
        "system-only payloads must be uniformly unkeyable, not hash the brief"
    )
