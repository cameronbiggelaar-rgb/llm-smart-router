"""Batch B37c (RED) — the policy truth table.

Why this module exists
----------------------
``evaluate_policy`` is the pure function that turns a usage reading + class into
a verdict. It is where the whole governor's correctness lives, so it is pinned
exhaustively before any routing hook exists.

The policy is an ALLOWLIST: the plan is reserved for the architectural lane.
Routine traffic is refused even in a pristine window (it is not entitled to the
plan at all). The hard cap is the reactive backstop for every class, and the
account's own ``allowed=false`` overrides everything. Fail-open is a safety
invariant, never a policy leak.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import gpt_budget as gb  # noqa: E402

ARCH = gb.CLASS_ARCHITECTURAL
ROUTINE = gb.CLASS_ROUTINE
OPEN = "open"
CLOSED = "closed"
DENIED = "denied"


def _reading(*, used: float, allowed: bool = True, limit: int = 604800):
    """A live reading with a single binding window at ``used`` percent."""
    return gb.Reading(
        source="live",
        allowed=allowed,
        primary=gb.WindowReading(
            used_percent=used,
            limit_window_seconds=limit,
            reset_after_seconds=int(limit * (1.0 - used / 100.0)),
            reset_at=0,
        ),
        secondary=None,
    )


def _cfg(**overrides):
    defaults = dict(
        enabled=True,
        allowlist=[ARCH],
        routine_ceiling_percent=0,
        hard_cap_percent=95,
    )
    defaults.update(overrides)
    return gb.GovernorConfig(**defaults)


# ── The allowlist: architectural is entitled, routine is not ─────────────────

@pytest.mark.parametrize("used", [0, 10, 50, 94])
def test_architectural_is_open_below_hard_cap(used):
    v = gb.evaluate_policy(_reading(used=used), _cfg(), klass=ARCH)
    assert v == OPEN


def test_routine_is_denied_even_in_a_pristine_window():
    """THE decisive change: routine is refused at used=0 because it has no
    entitlement to the plan at all — not because the window is full."""
    v = gb.evaluate_policy(_reading(used=0), _cfg(), klass=ROUTINE)
    assert v == DENIED


@pytest.mark.parametrize("used", [0, 10, 50, 94])
def test_routine_is_never_open(used):
    """Regression: routine must not re-acquire a share under any used% below
    the hard cap."""
    v = gb.evaluate_policy(_reading(used=used), _cfg(), klass=ROUTINE)
    assert v == DENIED


# ── The hard cap is the reactive backstop for every class ────────────────────

def test_architectural_is_closed_at_the_hard_cap():
    """used=96 >= hard_cap(95) closes the plan even for the entitled class."""
    v = gb.evaluate_policy(_reading(used=96), _cfg(), klass=ARCH)
    assert v == CLOSED


def test_routine_is_closed_at_the_hard_cap():
    # Routine is denied below the cap and closed at/above it — the distinction
    # matters for observability (denied=policy, closed=capacity).
    v = gb.evaluate_policy(_reading(used=96), _cfg(), klass=ROUTINE)
    assert v == CLOSED


def test_architectural_closed_at_exactly_hard_cap():
    v = gb.evaluate_policy(_reading(used=95), _cfg(), klass=ARCH)
    assert v == CLOSED


# ── The account's own verdict overrides everything ───────────────────────────

def test_allowed_false_closes_architectural_even_at_low_usage():
    v = gb.evaluate_policy(_reading(used=5, allowed=False), _cfg(), klass=ARCH)
    assert v == CLOSED


# ── The real incident, replayed ──────────────────────────────────────────────

def test_real_2026_09_21_state_is_closed():
    """used=100, the actual Sep-21 cap state → closed for the entitled class."""
    r = gb.Reading(
        source="live",
        allowed=False,
        primary=gb.WindowReading(
            used_percent=100.0,
            limit_window_seconds=604800,
            reset_after_seconds=165300,
            reset_at=1790455547,
        ),
        secondary=gb.WindowReading(
            used_percent=100.0,
            limit_window_seconds=604800,
            reset_after_seconds=165300,
            reset_at=1790455547,
        ),
    )
    assert gb.evaluate_policy(r, _cfg(), klass=ARCH) == CLOSED


# ── Division guard + fail open ───────────────────────────────────────────────

def test_zero_limit_window_opens_architectural():
    """limit_window_seconds=0 must never divide-by-zero or throttle."""
    v = gb.evaluate_policy(_reading(used=0, limit=0), _cfg(), klass=ARCH)
    assert v == OPEN


def test_unreadable_reading_is_unverified_not_a_free_pass():
    """B44a reverse: 'could not read the account' is a distinct verdict.

    An ARM-allowlist policy refuses routine on an unreadable probe (an
    unverifiable entitlement is not consent) and reports the named class as
    UNVERIFIED rather than allowed. Failing open here is what let routine ride a
    blind probe onto the plan — 1,083 rows, denied=0, nothing logged.
    """
    r = gb.Reading(source="unavailable", allowed=None, primary=None, secondary=None)
    assert gb.evaluate_policy(r, _cfg(), klass=ARCH) == gb.VERDICT_UNVERIFIED
    assert gb.evaluate_policy(r, _cfg(), klass=ROUTINE) == DENIED


def test_unreadable_reading_respects_the_rollback_lever():
    """enabled=false still opens every class, unreadable probe or not."""
    r = gb.Reading(source="unavailable", allowed=None, primary=None, secondary=None)
    assert gb.evaluate_policy(r, _cfg(enabled=False), klass=ARCH) == OPEN
    assert gb.evaluate_policy(r, _cfg(enabled=False), klass=ROUTINE) == OPEN


def test_fail_open_does_not_leak_when_verdict_is_readable():
    """Fail-open only applies when the state is UNREADABLE. A readable
    allowed=false must still close (composed, not bypassed)."""
    assert gb.evaluate_policy(_reading(used=0, allowed=False), _cfg(), klass=ARCH) == CLOSED


# ── enabled=false == today's behaviour (rollback) ────────────────────────────

def test_disabled_opens_all_classes():
    cfg = _cfg(enabled=False)
    for used in [0, 96, 100]:
        assert gb.evaluate_policy(_reading(used=used), cfg, klass=ARCH) == OPEN
        assert gb.evaluate_policy(_reading(used=used), cfg, klass=ROUTINE) == OPEN


# ── Deleted-concept guards ────────────────────────────────────────────────────

def test_legacy_pacing_keys_have_no_effect():
    """slack/final_stretch/schedule must be GONE. A config that still carries
    them (e.g. a stale hand-edited file) must load and have NO effect."""
    cfg = gb.GovernorConfig(
        enabled=True,
        allowlist=[ARCH],
        routine_ceiling_percent=0,
        hard_cap_percent=95,
        # legacy keys, deliberately present:
        slack=0.1,
        final_stretch_seconds=21600,
    )
    v = gb.evaluate_policy(_reading(used=10), cfg, klass=ARCH)
    assert v == OPEN


def test_governorconfig_rejects_unknown_class_in_allowlist():
    """An unknown class name must not silently widen the allowlist."""
    cfg = _cfg(allowlist=[ARCH, "superuser"])
    # Unknown names are ignored → only architectural remains permitted.
    assert gb.evaluate_policy(_reading(used=0), cfg, klass=ROUTINE) == DENIED
    assert gb.evaluate_policy(_reading(used=0), cfg, klass=ARCH) == OPEN


def test_empty_allowlist_denies_everything():
    """allowlist: [] → no class permitted (would black-hole the plan if the
    loader didn't reject it — asserted at the loader level in B37d)."""
    cfg = _cfg(allowlist=[])
    assert gb.evaluate_policy(_reading(used=0), cfg, klass=ARCH) == DENIED
    assert gb.evaluate_policy(_reading(used=0), cfg, klass=ROUTINE) == DENIED


def test_verdict_is_stable_for_unknown_class_token():
    """An unrecognised class string behaves as routine (never open, never crashes)."""
    v = gb.evaluate_policy(_reading(used=0), _cfg(), klass="whatever")
    assert v == DENIED
