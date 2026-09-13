"""B28 batch 1 — extract explicit reset times from 429 bodies/headers.

Two upstream 429 shapes exist in the wild here:
  * openai-codex (ChatGPT Plus): JSON body ``{"error": {"type": ...,
    "code": "usage_limit_reached", "message": "... plan_type: plus ...",
    "resets_at": <epoch-seconds>}}``
  * ollama-cloud: message ``"usage credits auto reload monthly max reached ..."``
    (no explicit epoch) — must fall back to backoff.

We also honour a ``Retry-After`` header (integer seconds or HTTP-date).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402
from biggie_llm_endpoint import _extract_429_reset_time  # noqa: E402


def test_resets_at_epoch_from_codex_body():
    now = int(time.time())
    resets = now + 7200
    body = (
        '{"error": {"message": "usage limit reached ... plan_type: plus", '
        f'"code": "usage_limit_reached", "resets_at": {resets}}}}}'
    )
    got = _extract_429_reset_time(body)
    assert got == resets


def test_retry_after_header_integer_seconds():
    got = _extract_429_reset_time("", headers={"retry-after": "120"})
    assert got is not None
    assert abs((got - time.time()) - 120) < 5


def test_retry_after_header_http_date():
    target = time.time() + 300
    import email.utils
    date = email.utils.formatdate(target, usegmt=True)
    got = _extract_429_reset_time("", headers={"retry-after": date})
    assert got is not None
    assert abs((got - time.time()) - 300) < 5


def test_no_explicit_time_returns_none():
    assert _extract_429_reset_time("usage credits auto reload monthly max reached") is None
    assert _extract_429_reset_time("") is None


def test_ollama_no_reset_falls_back():
    # ollama body carries no epoch — extractor must return None so the caller
    # uses exponential backoff, NOT invent a reset time.
    body = '{"error": {"message": "usage credits auto reload monthly max reached"}}'
    assert _extract_429_reset_time(body) is None
