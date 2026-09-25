"""Batch B37a (RED) — the plan-budget reader must be authoritative and fail open.

Why this module exists
----------------------
The ChatGPT plan had no budget governor, so it boom-busted: Sep 20-21 burned a
whole 7-day window inside ~32h (46.7 Mtok) and then went dark for 5.66 days.
Capacity is not modellable — two cap events on the same plan differ 3.5x
(46.7 Mtok on Sep 21 vs 163.4 Mtok on Sep 12) — so the governor must never
*estimate* remaining capacity. It reads the account's own `/usage` verdict.

These tests pin the reader's honesty rules, which are the part that can hurt:
  * a cached value is never labelled ``live``;
  * an unreadable probe treats the plan as OPEN (fail open) — an unreadable
    endpoint must never starve the architectural lane;
  * the governor adds ZERO per-request network calls (one probe per
    ``min_interval_seconds``), because it sits on the hot routing path;
  * the endpoint's own ``allowed=false`` is authoritative and overrides
    everything else — that is the reactive backstop, composed not duplicated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Hermes' own package, for the authoritative usage-fetcher integration tests.
# Resolved relative to this file (not HOME, which conftest redirects for
# production-DB isolation) so it always points at the real checkout.
HERMES_AGENT = Path(__file__).resolve().parents[3] / "hermes-agent"
if str(HERMES_AGENT) not in sys.path:
    sys.path.insert(0, str(HERMES_AGENT))

import gpt_budget as gb  # noqa: E402


# The real payload shape, captured live from the Codex /usage endpoint on
# 2026-09-25 (values reproduced from the actual response; see the design doc
# §3.1). gpt was dark at capture time, hence weekly used_percent=100.
REAL_PAYLOAD = {
    "plan_type": "plus",
    "rate_limit": {
        "allowed": False,
        "limit_reached": True,
        "primary_window": {
            "used_percent": 0,
            "limit_window_seconds": 18000,
            "reset_after_seconds": 18000,
            "reset_at": 1790324503,
        },
        "secondary_window": {
            "used_percent": 100,
            "limit_window_seconds": 604800,
            "reset_after_seconds": 149044,
            "reset_at": 1790455547,
        },
    },
    "additional_rate_limits": None,
    "rate_limit_reset_credits": {"available_count": 3, "applicable_available_count": 3},
    "credits": {"has_credits": False, "unlimited": False, "balance": "0"},
}


def _payload(**overrides):
    """A deep-ish copy of REAL_PAYLOAD with targeted overrides."""
    import copy

    data = copy.deepcopy(REAL_PAYLOAD)
    for path, value in overrides.items():
        node = data
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return data


@pytest.fixture
def tmp_state_db(tmp_path, monkeypatch):
    """Point the reader's persistence at a throwaway state DB."""
    monkeypatch.setattr(gb, "GOVERNOR_STATE_DB", str(tmp_path / "state.db"), raising=False)
    return tmp_path / "state.db"


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_valid_body_parses_both_windows():
    reading = gb.parse_reading(REAL_PAYLOAD, fetched_at=1_000_000.0)

    assert reading.source == "live"
    assert reading.allowed is False
    assert reading.primary is not None and reading.secondary is not None
    assert reading.primary.used_percent == 0.0
    assert reading.primary.limit_window_seconds == 18000
    assert reading.primary.reset_after_seconds == 18000
    assert reading.secondary.used_percent == 100.0
    assert reading.secondary.limit_window_seconds == 604800
    assert reading.secondary.reset_after_seconds == 149044


def test_binding_window_is_the_weekly_not_the_session():
    """The 7-day window is what caps the plan; the 5h one resets hourly.

    Choosing the wrong window would either under-throttle (5h is usually
    near-empty) or permanently deny (the daily window resets constantly).
    """
    reading = gb.parse_reading(REAL_PAYLOAD, fetched_at=1_000_000.0)

    assert gb.binding_window(reading).used_percent == 100.0
    assert gb.binding_window(reading).limit_window_seconds == 604800


def test_binding_window_prefers_the_fullest_across_both():
    payload = _payload(**{"rate_limit.primary_window.used_percent": 91.0,
                          "rate_limit.secondary_window.used_percent": 12.0})
    reading = gb.parse_reading(payload, fetched_at=1_000_000.0)

    assert gb.binding_window(reading).used_percent == 91.0


def test_additional_rate_limits_are_preserved_per_model():
    """A model-scoped allowance at 100% must not be merged away.

    The upstream probe collapses these into one bool and a scoped cap at 100%
    on ONE model then reads as "quota fine" — the exact bug this reader exists
    to avoid repeating.
    """
    payload = _payload(additional_rate_limits=[
        {"model": "gpt-5.6-sol",
         "rate_limit": {"primary_window": {"used_percent": 100.0,
                                           "limit_window_seconds": 604800,
                                           "reset_after_seconds": 100,
                                           "reset_at": 1}}},
    ])
    reading = gb.parse_reading(payload, fetched_at=1_000_000.0)

    assert "gpt-5.6-sol" in reading.scoped
    assert reading.scoped["gpt-5.6-sol"].used_percent == 100.0


def test_scoped_allowance_is_read_even_when_top_level_looks_healthy():
    """No scoped entry => the dict is simply empty, never invented."""
    reading = gb.parse_reading(REAL_PAYLOAD, fetched_at=1_000_000.0)

    assert reading.scoped == {}


def test_missing_secondary_window_primary_governs():
    payload = _payload()
    del payload["rate_limit"]["secondary_window"]

    reading = gb.parse_reading(payload, fetched_at=1_000_000.0)

    assert reading.secondary is None
    assert gb.binding_window(reading) is not None
    assert gb.binding_window(reading).limit_window_seconds == 18000


def test_both_windows_missing_is_unavailable_and_opens():
    payload = _payload()
    payload["rate_limit"].pop("secondary_window", None)
    payload["rate_limit"].pop("primary_window", None)
    payload["rate_limit"].pop("allowed", None)  # no verdict either → truly unreadable

    reading = gb.parse_reading(payload, fetched_at=1_000_000.0)

    assert reading.source == "unavailable"
    assert gb.binding_window(reading) is None
    assert gb.is_open(reading) is True, "unreadable state must never starve the plan"


def test_allowed_false_is_authoritative_and_closes():
    """The endpoint's own verdict outranks our policy — composed backstop."""
    reading = gb.parse_reading(REAL_PAYLOAD, fetched_at=1_000_000.0)

    assert reading.allowed is False
    assert gb.is_open(reading) is False


def test_allowed_false_closes_even_with_an_empty_window():
    """A 429 verdict with a not-yet-refreshed 0% window still means closed."""
    payload = _payload(**{"rate_limit.secondary_window.used_percent": 0.0})
    payload["rate_limit"]["allowed"] = False

    reading = gb.parse_reading(payload, fetched_at=1_000_000.0)

    assert gb.is_open(reading) is False


def test_credits_and_resets_are_surfaced():
    reading = gb.parse_reading(REAL_PAYLOAD, fetched_at=1_000_000.0)

    assert reading.has_credits is False
    assert reading.banked_resets == 3


# --------------------------------------------------------------------------
# Fail open — the safety property
# --------------------------------------------------------------------------

def test_fetch_returns_unavailable_on_network_error(monkeypatch, tmp_state_db):
    def _boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(gb, "_read_usage_payload", _boom, raising=False)
    reading = gb.fetch_reading(use_cache=False)

    assert reading.source == "unavailable"
    assert gb.is_open(reading) is True, "a broken probe must fail OPEN"


def test_fetch_returns_unavailable_when_helper_yields_none(monkeypatch, tmp_state_db):
    """`fetch_account_usage` returns None on any failure — treat as unavailable."""
    monkeypatch.setattr(gb, "_read_usage_payload", lambda: None, raising=False)
    reading = gb.fetch_reading(use_cache=False)

    assert reading.source == "unavailable"
    assert gb.is_open(reading) is True


def test_parse_hard_failure_fails_open(monkeypatch):
    """A malformed body must not raise into the routing path."""
    reading = gb.parse_reading({"rate_limit": "not-a-dict"}, fetched_at=1_000_000.0)

    assert gb.is_open(reading) is True


# --------------------------------------------------------------------------
# Cache and staleness
# --------------------------------------------------------------------------

def test_second_call_inside_min_interval_makes_zero_network_calls(monkeypatch, tmp_state_db):
    """The governor sits on the hot path: one probe per interval, no more."""
    calls = {"n": 0}

    def _count():
        calls["n"] += 1
        return REAL_PAYLOAD

    monkeypatch.setattr(gb, "_read_usage_payload", _count, raising=False)
    cfg = gb.ProbeConfig(min_interval_seconds=300, stale_after_seconds=1800)

    now = {"t": 1_000_000.0}
    first = gb.fetch_reading(cfg=cfg, now=lambda: now["t"])
    second = gb.fetch_reading(cfg=cfg, now=lambda: now["t"])

    assert calls["n"] == 1, "second call inside the interval must not hit the network"
    assert first.source == "live"
    assert second.source == "cache"
    assert second.allowed is False, "the cached verdict must be preserved"


def test_cache_expires_after_the_interval(monkeypatch, tmp_state_db):
    calls = {"n": 0}

    def _count():
        calls["n"] += 1
        return REAL_PAYLOAD

    monkeypatch.setattr(gb, "_read_usage_payload", _count, raising=False)
    cfg = gb.ProbeConfig(min_interval_seconds=300, stale_after_seconds=1800)

    now = {"t": 1_000_000.0}
    gb.fetch_reading(cfg=cfg, now=lambda: now["t"])
    now["t"] += 301
    gb.fetch_reading(cfg=cfg, now=lambda: now["t"])

    assert calls["n"] == 2


def test_stale_cache_beyond_stale_after_is_unavailable_and_opens(monkeypatch, tmp_state_db):
    """Old data is not evidence. Past the staleness bound we must fail open."""
    monkeypatch.setattr(gb, "_read_usage_payload", lambda: REAL_PAYLOAD, raising=False)
    cfg = gb.ProbeConfig(min_interval_seconds=300, stale_after_seconds=1800)

    now = {"t": 1_000_000.0}
    gb.fetch_reading(cfg=cfg, now=lambda: now["t"])

    # Network dies; time moves past staleness.
    def _boom():
        raise OSError("down")

    monkeypatch.setattr(gb, "_read_usage_payload", _boom, raising=False)
    now["t"] += 1801
    reading = gb.fetch_reading(cfg=cfg, now=lambda: now["t"])

    assert reading.source == "unavailable"
    assert gb.is_open(reading) is True


def test_stale_but_usable_cache_is_labelled_cache_not_live(monkeypatch, tmp_state_db):
    """Honesty rule: a remembered value is never presented as a live one."""
    monkeypatch.setattr(gb, "_read_usage_payload", lambda: REAL_PAYLOAD, raising=False)
    cfg = gb.ProbeConfig(min_interval_seconds=300, stale_after_seconds=1800)

    now = {"t": 1_000_000.0}
    gb.fetch_reading(cfg=cfg, now=lambda: now["t"])

    def _boom():
        raise OSError("down")

    monkeypatch.setattr(gb, "_read_usage_payload", _boom, raising=False)
    now["t"] += 600  # past the probe interval, inside staleness
    reading = gb.fetch_reading(cfg=cfg, now=lambda: now["t"])

    assert reading.source == "cache"
    assert reading.source != "live"


def test_reading_survives_a_restart(tmp_state_db):
    """Persisted to router_state.db so a restart does not force a cold probe."""
    reading = gb.parse_reading(REAL_PAYLOAD, fetched_at=1_000_000.0)
    gb.persist_reading(reading)

    restored = gb.load_persisted_reading()

    assert restored is not None
    assert restored.allowed is False
    assert restored.secondary.used_percent == 100.0


def test_persisted_reading_round_trips_scoped(monkeypatch, tmp_state_db):
    payload = _payload(additional_rate_limits=[
        {"model": "gpt-5.5",
         "rate_limit": {"primary_window": {"used_percent": 42.0,
                                           "limit_window_seconds": 604800,
                                           "reset_after_seconds": 10,
                                           "reset_at": 5}}},
    ])
    reading = gb.parse_reading(payload, fetched_at=1_000_000.0)
    gb.persist_reading(reading)

    restored = gb.load_persisted_reading()

    assert restored.scoped["gpt-5.5"].used_percent == 42.0


def test_no_persisted_reading_returns_none(tmp_state_db):
    assert gb.load_persisted_reading() is None


# --------------------------------------------------------------------------
# Integration with the authoritative Hermes helper
# --------------------------------------------------------------------------

def test_read_usage_payload_uses_the_hermes_helper(monkeypatch):
    """We must reuse Hermes's authoritative fetcher, not re-implement HTTP.

    Re-implementing the request would duplicate token refresh, the 401 retry,
    the IPv6/happy-eyeballs client setup and the account-id header — and drift.
    """
    sentinel = {"rate_limit": {"allowed": True}}
    seen = {}

    class _Snap:
        raw = sentinel

    def _fake(provider, **kw):
        seen["provider"] = provider
        return _Snap()

    import agent.account_usage as au

    monkeypatch.setattr(au, "fetch_account_usage", _fake, raising=True)
    got = gb._read_usage_payload()

    assert got == sentinel
    assert seen["provider"] == "openai-codex"


def test_read_usage_payload_returns_none_when_helper_has_no_raw(monkeypatch):
    class _Snap:
        raw = None

    import agent.account_usage as au

    monkeypatch.setattr(au, "fetch_account_usage", lambda *a, **k: _Snap(), raising=True)

    assert gb._read_usage_payload() is None
