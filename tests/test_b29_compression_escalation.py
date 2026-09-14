"""B29 Fix 1 — compression escalation must stay on the summariser ladder.

The defect
----------
A native Hermes session-compression request carries the *parent session's* tool
schemas even though a summariser never calls tools. The endpoint therefore sets
``requires_tools=True`` for compaction traffic.

Routing handles that correctly (compaction is classified before the tool gate),
but **escalation does not**: when flash returns tool-call-only / empty content,
the endpoint calls ``escalate_on_failure(..., requires_tools=True)``, which
branches to "most-capable tool-capable model" — i.e. the ChatGPT plan.

Measured 2026-09-13: 203 codex compression escalations in 8h, every one from
``deepseek-v4.1-flash:cloud (empty_content)``. gpt-5.5 reached 751 consecutive
failures and the Plus plan cap was hit, driven largely by maintenance traffic.

Promoting a *summariser* to a *tool-calling* model is backwards: compression
output is prose, tools are irrelevant. Compression escalation must retry the
next curated summariser rung and must never reach the ChatGPT plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import router as R  # noqa: E402

GPT_PLAN = {
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
}


def _available_only(monkeypatch, models):
    """Make exactly ``models`` available; everything else unavailable."""
    allowed = set(models)
    monkeypatch.setattr(R, "is_model_available", lambda m: m in allowed)
    monkeypatch.setattr(R, "get_available_models", lambda: list(models))


def test_escalation_api_accepts_workload_type():
    """The escalation entry point must be told the workload.

    Without this parameter the escalation path cannot distinguish compaction
    from ordinary work and will keep promoting summarisers onto tool models.
    """
    import inspect

    params = inspect.signature(R.escalate_on_failure).parameters
    assert "workload_type" in params, (
        "escalate_on_failure has no workload_type parameter: the escalation "
        "path cannot tell session_compression apart from ordinary work, so "
        "compression failures keep escalating onto the ChatGPT plan"
    )


def test_compression_escalation_never_reaches_the_chatgpt_plan(monkeypatch):
    """The measured defect, driven through the real function.

    flash fails on tool-call-only output, the full summariser ladder is
    available, and the ChatGPT plan models are also available and *more
    capable*. Escalation must stay on the ladder.
    """
    ladder = ["deepseek-v4.1-flash", "deepseek-v4-flash", "glm-5.3", "glm-5.2"]
    _available_only(monkeypatch, ladder + sorted(GPT_PLAN))
    monkeypatch.setattr(
        R,
        "MODEL_CAPABILITY_TIERS",
        {**{m: 3 for m in ladder}, "gpt-5.6-sol": 8, "gpt-5.5": 7},
    )
    monkeypatch.setattr(R, "MODEL_CONTEXT_CEILING", {})
    monkeypatch.setattr(R, "FLASH_MAX_CONTEXT_TOKENS", 0)
    monkeypatch.setattr(R, "probe_all_recovering", lambda: [])

    decision = R.escalate_on_failure(
        failed_model="deepseek-v4.1-flash",
        complexity_score=0.1,
        error_type="empty_content",
        requires_tools=True,          # exactly what the endpoint passes today
        workload_type="session_compression",
        context_tokens=120_000,
    )

    assert decision.selected_model, (
        "compression escalation returned no model while the summariser ladder "
        "was available — compaction would fail entirely"
    )
    assert decision.selected_model not in GPT_PLAN, (
        f"compression escalation selected {decision.selected_model!r} (ChatGPT "
        "plan) after a flash empty-content failure. Summarisation is prose "
        "work; promoting it to a tool-calling model burns the plan's capacity "
        "on maintenance traffic (measured: 203 such escalations in 8h)"
    )
    assert decision.selected_model in ladder, (
        f"compression escalation selected {decision.selected_model!r}, which is "
        "not on the curated summariser ladder"
    )


def test_compression_escalation_retries_the_next_rung_not_the_same_model(monkeypatch):
    """A failed rung must not be re-selected — that would loop on the failure."""
    ladder = ["deepseek-v4.1-flash", "deepseek-v4-flash", "glm-5.3"]
    _available_only(monkeypatch, ladder)
    monkeypatch.setattr(R, "MODEL_CAPABILITY_TIERS", {m: 3 for m in ladder})
    monkeypatch.setattr(R, "MODEL_CONTEXT_CEILING", {})
    monkeypatch.setattr(R, "FLASH_MAX_CONTEXT_TOKENS", 0)
    monkeypatch.setattr(R, "probe_all_recovering", lambda: [])

    decision = R.escalate_on_failure(
        failed_model="deepseek-v4.1-flash",
        error_type="empty_content",
        requires_tools=True,
        workload_type="session_compression",
        context_tokens=120_000,
    )
    assert decision.selected_model != "deepseek-v4.1-flash", (
        "compression escalation re-selected the model that just failed — the "
        "same empty-content response would be retried indefinitely"
    )


def test_compression_escalation_falls_closed_when_the_ladder_is_exhausted(monkeypatch):
    """The whole point: exhaustion must NOT fail open onto the ChatGPT plan."""
    _available_only(monkeypatch, sorted(GPT_PLAN))
    monkeypatch.setattr(
        R, "MODEL_CAPABILITY_TIERS", {"gpt-5.6-sol": 8, "gpt-5.5": 7}
    )
    monkeypatch.setattr(R, "MODEL_CONTEXT_CEILING", {})
    monkeypatch.setattr(R, "FLASH_MAX_CONTEXT_TOKENS", 0)
    monkeypatch.setattr(R, "probe_all_recovering", lambda: [])

    decision = R.escalate_on_failure(
        failed_model="deepseek-v4.1-flash",
        error_type="empty_content",
        requires_tools=True,
        workload_type="session_compression",
        context_tokens=120_000,
    )
    assert not decision.selected_model or decision.selected_model not in GPT_PLAN, (
        f"with no summariser available, compression escalation selected "
        f"{decision.selected_model!r} — the ChatGPT plan is reserved for "
        "reasoning work, not maintenance summarisation"
    )
    if not decision.selected_model:
        assert decision.all_exhausted, (
            "escalation returned no model without setting all_exhausted — the "
            "caller cannot distinguish exhaustion from a routing bug"
        )


def test_non_compression_escalation_still_honours_the_tool_gate(monkeypatch):
    """Guard the opposite failure: the fix must not disable tool routing.

    Ordinary tool-bearing work that loses its model still needs a proven
    tool-capable model, and still fails closed if none is available.
    """
    monkeypatch.setattr(R, "is_model_available", lambda m: True)
    monkeypatch.setattr(R, "get_available_models", lambda: ["deepseek-v4-pro"])
    monkeypatch.setattr(R, "MODEL_CAPABILITY_TIERS", {"deepseek-v4-pro": 5})
    monkeypatch.setattr(R, "MODEL_CONTEXT_CEILING", {})
    monkeypatch.setattr(R, "FLASH_MAX_CONTEXT_TOKENS", 0)
    monkeypatch.setattr(R, "probe_all_recovering", lambda: [])
    monkeypatch.setattr(R, "TOOL_CAPABLE_MODELS", {"deepseek-v4-pro"})

    decision = R.escalate_on_failure(
        failed_model="glm-5.3",
        error_type="error",
        requires_tools=True,
        workload_type="normal_chat",
    )
    assert decision.selected_model == "deepseek-v4-pro", (
        "ordinary tool-bearing escalation no longer reaches the available "
        "tool-capable model — the compression fix has leaked into general "
        "tool routing"
    )
