"""B25.1 — a model excluded from routing must not be reachable via the
compression fail-open.

The defect: ``_select_session_compression_model`` falls back to "the cheapest
available tier-3+ model" when the curated ladder is exhausted. That loop
hardcodes exactly one exclusion (``gpt-5.5``) and never consults
``EXCLUDED_FROM_AUTO_ROUTING``, even though five other selection paths DO.

Measured consequence: 1,400 session-compression calls were served by a gpt-*
model through this path — traffic the curated ladder deliberately excludes for
capacity reasons — because the fail-open had no way to know they were reserved.

Why this matters for a new model: registering a candidate in MODEL_REGISTRY is
the documented way to make it reachable. If the candidate is meant to be
experiment-only, the fail-open would silently promote it into production
summarisation the first time the ladder thinned out. The exclusion set is the
mechanism that is supposed to prevent exactly that; this test makes it apply.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import router as R  # noqa: E402


def _fail_open_source() -> str:
    import inspect

    src = inspect.getsource(R._select_session_compression_model)
    marker = "# If all preferred summariser models are unavailable"
    assert marker in src, (
        "the compression fail-open has been rewritten; re-derive this test "
        "against the new shape rather than deleting it"
    )
    return src[src.index(marker):]


def test_compression_fail_open_consults_the_exclusion_set():
    """The fail-open must honour EXCLUDED_FROM_AUTO_ROUTING, not one hardcoded name.

    Asserting the mechanism (a set lookup) rather than a specific model, so
    adding a model to the exclusion set is enough to keep it out of production
    summarisation — no second edit in a distant function.
    """
    tail = _fail_open_source()
    assert "EXCLUDED_FROM_AUTO_ROUTING" in tail, (
        "the compression fail-open does not consult "
        "EXCLUDED_FROM_AUTO_ROUTING: any model added to the exclusion set is "
        "still eligible to serve production compression the moment the curated "
        "ladder is exhausted (1,400 calls have been served this way by gpt-* "
        "models the ladder deliberately excludes)"
    )


def test_excluded_model_is_not_returned_by_the_fail_open(monkeypatch):
    """Behavioural: with the ladder unavailable, an excluded model is skipped.

    Drives the real function. The preferred ladder is emptied by making no
    model 'available', then the fail-open runs against a patched model set
    whose cheapest tier-3+ entry is EXCLUDED.
    """
    excluded = next(iter(R.EXCLUDED_FROM_AUTO_ROUTING))

    # Make the whole curated ladder look unavailable so the fail-open runs.
    monkeypatch.setattr(R, "is_model_available", lambda m: False if m in set(
        R._select_session_compression_model.__globals__.get(
            "SESSION_COMPRESSION_LADDER_NAMES", ()
        )
    ) else True, raising=False)

    # Deterministic model set: the excluded model is first and cheapest.
    monkeypatch.setattr(R, "get_available_models", lambda: [excluded, "glm-5.3"])
    monkeypatch.setattr(R, "MODEL_CAPABILITY_TIERS", {excluded: 3, "glm-5.3": 6})

    picked = R._select_session_compression_model(context_tokens=1000)
    assert picked != excluded, (
        f"the compression fail-open returned {excluded!r}, which is excluded "
        "from auto routing — an experiment-only model would silently start "
        "serving production compression"
    )


def test_fail_open_still_returns_a_usable_summariser(monkeypatch):
    """Guard the opposite failure: the exclusion must not disable the fallback.

    Declaring exhaustion when a legitimate model is available would turn a
    capacity incident into total compression failure.
    """
    monkeypatch.setattr(R, "is_model_available", lambda m: True)
    monkeypatch.setattr(R, "get_available_models", lambda: ["glm-5.3"])
    monkeypatch.setattr(R, "MODEL_CAPABILITY_TIERS", {"glm-5.3": 6})

    # Force the ladder to miss by emptying it via the ceiling mechanism.
    monkeypatch.setattr(R, "MODEL_CONTEXT_CEILING", {"glm-5.3": 10})
    picked = R._select_session_compression_model(context_tokens=10_000)
    assert picked, (
        "the compression fail-open declared exhaustion while a tier-6 model "
        "was available — a capacity incident would become total compression "
        "failure"
    )
