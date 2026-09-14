"""B30 fix 2 — a deterministic 4xx must not hot-loop the escalation ladder.

Fix 1 removed the cause (Codex no longer receives sampling params). Fix 2 is
defence in depth: any *future* request-schema mismatch must be classified as
non-retryable so it cannot become a retry storm.

Measured baseline before the fixes: ONE client request carrying
``temperature`` produced **18 upstream Codex 400s** in ~2 minutes, because each
escalation step re-sent the same params to the same rejecting endpoint and the
4xx fed the ordinary retry path.

Contract:
  * ``_is_deterministic_client_error`` recognises the request-schema class
    (``Unsupported parameter``, malformed body, invalid field) and NOT
    transient classes (429 rate limits, 5xx, timeouts).
  * ``_handle_backend_status`` routes such a 4xx away from the retry/escalation
    path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
HERMES_AGENT = ROOT.parents[1] / "hermes-agent"

for _p in (str(SCRIPTS_DIR), str(HERMES_AGENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import biggie_llm_endpoint as ep  # noqa: E402


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Unsupported parameter: temperature",
        "Unsupported parameter: max_tokens",
        "Backend error: Unsupported parameter: top_p",
    ],
)
def test_unsupported_parameter_is_deterministic(text):
    assert ep._is_deterministic_client_error(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Unsupported parameter: temperature",
        '{"error": {"message": "Unsupported parameter: temperature", "type": "invalid_request_error"}}',
    ],
)
def test_codex_400_payload_is_deterministic(text):
    """The real upstream body shape must be recognised."""
    assert ep._is_deterministic_client_error(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "usage_limit_reached",
        '{"error":{"type":"usage_limit_reached"}}',
        "We're currently experiencing high demand, please try again later.",
        "Connection reset by peer",
        "Read timed out",
    ],
)
def test_transient_errors_are_not_deterministic(text):
    """Over-classification would strand models that should recover."""
    assert ep._is_deterministic_client_error(text) is False


def test_function_exists():
    assert hasattr(ep, "_is_deterministic_client_error"), (
        "deterministic-4xx classifier missing — schema faults still feed the "
        "retry/escalation path and can reproduce the 18x amplification"
    )


# --------------------------------------------------------------------------
# Wiring: the 400 path must be handled, not escalated blindly.
# --------------------------------------------------------------------------


def test_handle_backend_status_calls_classifier_for_400(monkeypatch):
    """A 400 carrying an unsupported-parameter body must be detected."""
    seen = {}

    def _fake_classifier(text):
        seen["text"] = text
        return True

    monkeypatch.setattr(ep, "_is_deterministic_client_error", _fake_classifier)
    # Must not raise; classification is best-effort bookkeeping.
    ep._handle_backend_status(
        "openai-codex", "gpt-5.5", 400,
        error_text="Unsupported parameter: temperature",
    )
    assert seen.get("text") == "Unsupported parameter: temperature", (
        "_handle_backend_status did not consult the deterministic-4xx classifier "
        "for a 400 response"
    )


def test_codex_400_does_not_mark_model_rate_limited(monkeypatch):
    """A schema 400 is not a rate limit; marking it so hides the real fault."""
    calls = []
    monkeypatch.setattr(ep, "mark_rate_limited",
                        lambda *a, **k: calls.append(("rate_limited", a, k)))
    monkeypatch.setattr(ep, "mark_credential_expired",
                        lambda *a, **k: calls.append(("cred_expired", a, k)))
    ep._handle_backend_status(
        "openai-codex", "gpt-5.5", 400,
        error_text="Unsupported parameter: temperature",
    )
    kinds = [c[0] for c in calls]
    assert "rate_limited" not in kinds, (
        "a deterministic 400 must not be recorded as a rate limit"
    )
    assert "cred_expired" not in kinds, (
        "a deterministic 400 must not be recorded as a credential failure"
    )

