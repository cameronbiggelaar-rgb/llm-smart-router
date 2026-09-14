"""B30 fix 3 - a re-armed circuit breaker must get a real retry budget.

Measured live defect (gpt-5.5, 2026-09-14): ``consecutive_failures`` reached
**4046** and the model was closed for 30 minutes. When that cooldown expired,
``is_model_available()`` set ``available = True`` but never reset the failure
count, so a SINGLE fresh failure immediately satisfied
``consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD`` (3) and re-opened the
breaker. Net effect: a broken model received exactly one attempt per 30-minute
window and the count grew without bound - which is what turned one client
request into 18 upstream 400s, and why gpt-5.5 logged 3,873+ consecutive
failures while its breaker "cycled".

Contract:
 * Expiry of a circuit breaker / rate limit grants a FRESH failure budget.
 * Recovery is recorded as a genuine recovery, not a silent reuse of the
   poisoned counter.
 * A single post-expiry failure must NOT re-open the circuit.
 * ``mark_available`` remains the explicit manual reset.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import router  # noqa: E402


MODEL = "gpt-5.5"


def _force(status, **kw):
    for k, v in kw.items():
        setattr(status, k, v)
    router._save_model_state(status)


def test_expired_circuit_resets_failure_budget(monkeypatch):
    """After cooldown expiry the failure counter must be back to zero."""
    status = router._MODEL_STATUSES[MODEL]
    _force(
        status,
        available=False,
        consecutive_failures=4046,
        circuit_open_until=time.time() - 1,
        rate_limited_until=0.0,
        last_error="error",
    )

    assert router.is_model_available(MODEL) is True

    status = router._MODEL_STATUSES[MODEL]
    assert status.consecutive_failures == 0, (
        "expired breaker did not reset consecutive_failures (%d). A single "
        "new failure will immediately re-open the circuit and the model gets "
        "just one attempt per cooldown window." % status.consecutive_failures
    )


def test_one_failure_after_expiry_does_not_reopen_circuit(monkeypatch):
    """A freshly re-armed model must tolerate CIRCUIT_BREAKER_THRESHOLD failures."""
    status = router._MODEL_STATUSES[MODEL]
    _force(
        status,
        available=False,
        consecutive_failures=4046,
        circuit_open_until=time.time() - 1,
        rate_limited_until=0.0,
        last_error="error",
    )
    router.is_model_available(MODEL)

    opened = []
    monkeypatch.setattr(
        router, "open_circuit",
        lambda model, err, *a, **k: opened.append((model, err)),
    )

    # One ordinary failure - far below the threshold of 3.
    status = router._MODEL_STATUSES[MODEL]
    status.consecutive_failures += 1
    if status.consecutive_failures >= router.CIRCUIT_BREAKER_THRESHOLD:
        router.open_circuit(MODEL, "error")

    assert not opened, (
        "a single failure after breaker expiry re-opened the circuit; the "
        "model is starved to one attempt per cooldown"
    )


def test_expiry_records_recovery_event(monkeypatch):
    """Expiry must be observable as a recovery, not silent."""
    status = router._MODEL_STATUSES[MODEL]
    _force(
        status,
        available=False,
        consecutive_failures=4046,
        circuit_open_until=time.time() - 1,
        rate_limited_until=0.0,
        last_error="error",
    )

    events = []
    monkeypatch.setattr(
        router, "_log_recovery_event",
        lambda model, kind, detail="": events.append((model, kind, detail)),
    )
    router.is_model_available(MODEL)

    assert any(e[1] == "recovered" for e in events), (
        "breaker expiry produced no recovery event: %r" % (events,)
    )


def test_rate_limit_expiry_also_resets_budget(monkeypatch):
    """Same guarantee for the rate-limit expiry path."""
    status = router._MODEL_STATUSES[MODEL]
    _force(
        status,
        available=False,
        consecutive_failures=99,
        circuit_open_until=0.0,
        rate_limited_until=time.time() - 1,
        last_error="rate_limit",
    )

    assert router.is_model_available(MODEL) is True
    status = router._MODEL_STATUSES[MODEL]
    assert status.consecutive_failures == 0, (
        "expired rate limit did not reset consecutive_failures"
    )


def test_mark_available_still_resets_everything():
    """The explicit manual reset must keep working."""
    status = router._MODEL_STATUSES[MODEL]
    _force(
        status,
        available=False,
        consecutive_failures=4046,
        circuit_open_until=time.time() + 9999,
        rate_limited_until=time.time() + 9999,
        last_error="error",
    )

    router.mark_available(MODEL)

    status = router._MODEL_STATUSES[MODEL]
    assert status.available is True
    assert status.consecutive_failures == 0
    assert status.circuit_open_until == 0.0
    assert status.rate_limited_until == 0.0
