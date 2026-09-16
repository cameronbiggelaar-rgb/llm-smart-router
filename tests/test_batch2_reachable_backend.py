"""Batch 2 — the reachable-backend fallback must respect availability.

Why this exists
---------------
This is the *actual* source of the codex 429 storm — found in the live journal,
not in routing code:

    WARNING: Selected model deepseek-v3.1:671b not in discovered backends,
             falling back to best reachable
    INFO:    Fell back to reachable backend: gpt-5.5 (tier 10, needed 9)

`gpt-5.5` was held by the account cap at that moment (all five openai-codex
models held ~83h). The fallback walks `discover_backends()`, which lists what
is *configured* — including models held by an account cap or sitting behind an
open circuit — and picks purely by capability tier. It never asks whether the
model may actually be called.

That is why the cap fired ~1-3x/min while `get_available_models()` correctly
reported zero eligible codex models: capability routing was sound, and this
fallback was re-selecting a capped model anyway. It also explains the missing
`router_logs` rows — this path dispatches outside the normal logging seam.

The same tier-walk is duplicated on the escalation path, which is why the bug
exists in two places. Batch 2 extracts it to one helper that takes an
availability predicate, so the check cannot be omitted from one copy only.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402


def _backends(*names):
    """Minimal backend map: discovery returns provider dicts keyed by model."""
    return {
        n: {"provider": "test", "base_url": "http://x", "api_key": ""}
        for n in names
    }


def _none_held(*_a, **_k):
    return True


def test_fallback_does_not_select_a_held_model():
    """A held model must not be chosen, even when it is the most capable.

    Live failure: gpt-5.5 (tier 10) was selected while held by the account
    cap, re-hitting the capped ChatGPT account once a minute.
    """
    backends = _backends("gpt-5.5", "glm-5.3-flash", "deepseek-v4.1-flash")

    chosen = ep._pick_fallback_backend(
        backends,
        needed_tier=9,
        requires_tools=False,
        is_available=lambda m: m != "gpt-5.5",  # the cap
    )

    assert chosen != "gpt-5.5", (
        "fallback selected a model held by an account cap — this is the "
        "one-per-minute codex 429 storm"
    )
    assert chosen in backends, "must still return a usable backend"


def test_fallback_prefers_most_capable_among_available():
    """Unchanged behaviour when nothing is held: most capable wins."""
    backends = _backends("gpt-5.5", "glm-5.3-flash", "deepseek-v4.1-flash")

    chosen = ep._pick_fallback_backend(
        backends, needed_tier=9, requires_tools=False, is_available=_none_held,
    )

    # gpt-5.5 is the highest tier and meets needed_tier, so it is both
    # 'cheapest meeting' and 'best' — it must be selected.
    assert chosen == "gpt-5.5", f"expected gpt-5.5, got {chosen!r}"


def test_fallback_returns_none_when_every_candidate_is_held():
    """Fail closed instead of dispatching into a capped account."""
    backends = _backends("gpt-5.5", "gpt-6-astra")

    chosen = ep._pick_fallback_backend(
        backends, needed_tier=9, requires_tools=False, is_available=lambda m: False,
    )

    assert chosen is None, (
        f"expected no backend when all are unavailable, got {chosen!r} — "
        "dispatching anyway is what re-hits the cap"
    )


def test_fallback_still_enforces_the_tool_capability_gate():
    """The availability check must not weaken the tool-capability gate."""
    backends = _backends("gpt-5.5", "glm-5.3-flash")

    chosen = ep._pick_fallback_backend(
        backends,
        needed_tier=0,
        requires_tools=True,
        is_available=lambda m: m != "gpt-5.5",
    )

    assert chosen in ep.TOOL_CAPABLE_MODELS or chosen is None, (
        f"tool-required work got a non-tool-capable backend: {chosen!r}"
    )


def test_fallback_respects_a_provider_suffixed_alias():
    """`:cloud`/`:local` aliases must be checked by base name too.

    Discovery can hand back `deepseek-v4.1-flash:cloud` while router state is
    keyed `deepseek-v4.1-flash`; a naive availability call would miss the hold.
    """
    backends = _backends("deepseek-v4.1-flash:cloud", "glm-5.3-flash")

    seen = []

    def avail(model):
        seen.append(model)
        return re.sub(r":(cloud|local|ollama)$", "", model) != "deepseek-v4.1-flash"

    chosen = ep._pick_fallback_backend(
        backends, needed_tier=0, requires_tools=False, is_available=avail,
    )

    assert chosen != "deepseek-v4.1-flash:cloud", (
        "a held model held under its base name was still selected via alias"
    )


def test_both_fallback_sites_use_the_shared_helper():
    """Guard against the duplicated tier-walk reappearing.

    The bug was born of copy-paste: two inline tier-walks, one availability
    check missing from both. Any future inline tier-walk here is a regression.
    """
    src = (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()

    assert "falling back to best reachable" in src, "expected call sites still exist"
    # The helper is the ONLY place allowed to compute best/cheapest by tier.
    helper_defs = src.count("def _pick_fallback_backend")
    assert helper_defs == 1, f"expected exactly 1 helper definition, found {helper_defs}"

    inline_walks = src.count("cheapest_meeting_tier = 99")
    assert inline_walks == 1, (
        f"found {inline_walks} inline tier-walks — the selection logic must "
        "live only in _pick_fallback_backend, or the availability check can "
        "again be omitted from a copy"
    )
