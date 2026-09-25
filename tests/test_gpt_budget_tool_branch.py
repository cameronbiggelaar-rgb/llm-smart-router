"""Batch B37e (RED) — Hook 1: the tool-bearing early-return path.

This is the critical batch: the tool branch is where 97.6% of the plan's burn
came from (28.6 of 29.3 Mtok in the clean era), because
``_select_tool_capable_model`` returns the highest-tier available model and
ignores ``min_tier`` entirely. The governor must classify the request from its
*text* (the reason string is not written yet on this path, and the tool branch
never records a tier clause) and deny a routine request that would otherwise
land on the ChatGPT plan.

The single most important test here: a plain tool-bearing request (no
architectural keyword) must NOT select ``gpt-5.6-sol`` — it must re-select to
``deepseek-v4-pro``, the highest-tier non-plan tool-capable model.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import gpt_budget as gb  # noqa: E402
import router  # noqa: E402

CODEX_MODELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra"]
OLLAMA_MODELS = ["glm-5.3", "deepseek-v4-pro", "deepseek-v4.1-flash"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_state():
    saved = {
        m: router._MODEL_STATUSES.get(m)
        for m in CODEX_MODELS + OLLAMA_MODELS + ["llama3.1:8b"]
    }
    for m in CODEX_MODELS + OLLAMA_MODELS + ["llama3.1:8b"]:
        router._MODEL_STATUSES[m] = router.ModelStatus(model=m, available=True)
    yield
    for m, st in saved.items():
        if st is None:
            router._MODEL_STATUSES.pop(m, None)
        else:
            router._MODEL_STATUSES[m] = st


def _reading(used_pct=0.0, allowed=True, source="live"):
    return gb.Reading(
        source=source,
        allowed=allowed,
        primary=gb.WindowReading(
            used_percent=used_pct,
            limit_window_seconds=300,
            reset_after_seconds=165300,
        ),
        secondary=gb.WindowReading(
            used_percent=used_pct,
            limit_window_seconds=604800,
            reset_after_seconds=165300,
        ),
    )


def _architectural_cfg():
    return gb.GovernorConfig(
        enabled=True,
        allowlist=[gb.CLASS_ARCHITECTURAL],
        routine_ceiling_percent=0,
        hard_cap_percent=95.0,
    )


@pytest.fixture
def governor_open(monkeypatch):
    """Governor enabled, plan at 0% used — architectural would be allowed."""
    monkeypatch.setattr(gb, "load_governor_config", lambda force=False: _architectural_cfg())
    monkeypatch.setattr(gb, "fetch_reading", lambda cfg=None, **kw: _reading(0.0))


# ── The decisive test ─────────────────────────────────────────────────────────

def test_tool_bearing_routine_does_not_take_sol(clean_state, governor_open):
    """THE leak: a plain tool-bearing request must not land on gpt-5.6-sol."""
    d = router.route_task(
        complexity_score=0.85,
        task_type="coding",
        prompt="refactor the parse loop to use a deque",
        requires_tools=True,
    )
    assert d.selected_model != "gpt-5.6-sol"
    assert d.selected_model == "deepseek-v4-pro", d.reason
    assert d.budget_class == gb.CLASS_ROUTINE
    assert d.plan_budget_denied is True


def test_tool_bearing_architectural_still_reaches_sol(clean_state, governor_open):
    """Architectural tool-bearing requests are the one thing the plan is for."""
    d = router.route_task(
        complexity_score=0.85,
        task_type="planning",
        prompt="rearchitect the budget governor to be account-keyed",
        requires_tools=True,
    )
    assert d.selected_model == "gpt-5.6-sol", d.reason
    assert d.budget_class == gb.CLASS_ARCHITECTURAL
    assert d.plan_budget_denied is False


def test_tool_bearing_architectural_closed_at_hard_cap(clean_state, monkeypatch):
    """Even architectural tool work is closed when the plan is at the hard cap."""
    monkeypatch.setattr(gb, "load_governor_config", lambda force=False: _architectural_cfg())
    monkeypatch.setattr(gb, "fetch_reading", lambda cfg=None, **kw: _reading(96.0))
    d = router.route_task(
        complexity_score=0.85,
        task_type="planning",
        prompt="rearchitect the budget governor",
        requires_tools=True,
    )
    assert d.selected_model != "gpt-5.6-sol", d.reason
    assert d.selected_model == "deepseek-v4-pro", d.reason
    assert d.plan_budget_denied is True


def test_tool_bearing_force_model_bypasses_allowlist(clean_state, governor_open):
    """force_model is the unconditional escape hatch (§3.3)."""
    d = router.route_task(
        complexity_score=0.5,
        task_type="coding",
        prompt="write a for loop",
        requires_tools=True,
        force_model="gpt-5.5",
    )
    assert d.selected_model == "gpt-5.5", d.reason


def test_governor_disabled_matches_pre_b37_behaviour(clean_state, monkeypatch):
    """enabled=false is the rollback lever — identical to today's behaviour."""
    cfg = gb.GovernorConfig(
        enabled=False,
        allowlist=[gb.CLASS_ARCHITECTURAL],
        hard_cap_percent=95.0,
    )
    monkeypatch.setattr(gb, "load_governor_config", lambda force=False: cfg)
    monkeypatch.setattr(gb, "fetch_reading", lambda cfg=None, **kw: _reading(0.0))
    d = router.route_task(
        complexity_score=0.85,
        task_type="coding",
        prompt="refactor the parse loop",
        requires_tools=True,
    )
    # Pre-B37: highest-tier tool-capable model = gpt-5.6-sol.
    assert d.selected_model == "gpt-5.6-sol", d.reason
    assert d.budget_class == ""
    assert d.plan_budget_denied is False


def test_tool_bearing_denied_but_no_non_plan_model_fails_closed(clean_state, monkeypatch):
    """Only plan models are tool-capable → deny must fail closed, not empty."""
    monkeypatch.setattr(gb, "load_governor_config", lambda force=False: _architectural_cfg())
    monkeypatch.setattr(gb, "fetch_reading", lambda cfg=None, **kw: _reading(0.0))
    # Take the non-plan tool-capable models out of rotation. A far-future
    # cooldown is required: `availability = False` alone is undone by the
    # `probe_all_recovering()` sweep at the top of route_task.
    far_future = time.time() + 3600
    for m in ("deepseek-v4-pro", "glm-5.3"):
        st = router._MODEL_STATUSES[m]
        st.available = False
        st.rate_limited_until = far_future
    d = router.route_task(
        complexity_score=0.85,
        task_type="coding",
        prompt="refactor the parse loop",
        requires_tools=True,
    )
    assert d.all_exhausted is True, d.reason
    assert d.selected_model == "", d.reason


# ── Regression guards (§5.5) ──────────────────────────────────────────────────

def test_tool_fallback_chain_does_not_contain_a_denied_plan_model(clean_state, governor_open):
    """The back door: the tool chain walks UP in tier, straight into the plan.

    For a routine request served by deepseek-v4-pro the chain would otherwise
    contain gpt-5.6-sol — the leak re-entering through failover. Regression
    guard for §5.5.
    """
    # Non-vacuity: assert first that this configuration WOULD leak unfiltered.
    # Without this the test passes trivially whenever the chain is empty.
    unfiltered = router._build_tool_fallback_chain("deepseek-v4-pro")
    leaked = [m for m in unfiltered if router.get_model_account(m) == "openai-codex"]
    assert leaked, (
        "guard is vacuous — the unfiltered chain contains no plan model, so "
        "this test cannot detect the leak it exists to prevent"
    )

    chain = router._build_tool_fallback_chain("deepseek-v4-pro", klass=gb.CLASS_ROUTINE)
    for model in chain:
        assert router.get_model_account(model) != "openai-codex", (
            f"denied plan model {model} reachable via the tool fallback chain"
        )


def test_routine_tool_work_has_no_plan_failover_by_design(clean_state, governor_open):
    """Routine tool work gets NO plan failover — fail closed, not leak (§5.5).

    Documented consequence, not an accident: once the plan is denied to routine,
    a routine tool request served by deepseek-v4-pro has an empty escalation
    chain (every higher-tier tool-capable model is a plan model). The request
    fails visibly rather than quietly escalating onto the reserved plan.
    """
    chain = router._build_tool_fallback_chain("deepseek-v4-pro", klass=gb.CLASS_ROUTINE)
    assert chain == [], chain


def test_tool_fallback_chain_still_reaches_plan_for_architectural(clean_state, governor_open):
    """Containment must not strip the plan from the class entitled to it."""
    chain = router._build_tool_fallback_chain("gpt-5.5", klass=gb.CLASS_ARCHITECTURAL)
    assert chain, "architectural tool work lost its escalation path entirely"


def test_flash_canary_untouched_by_the_governor(clean_state, governor_open):
    """The governor touches plan-account models only (§5.5).

    The flash canary is at percent 100 on live traffic; a governor that altered
    non-plan routing would silently invalidate it.
    """
    d = router.route_task(
        complexity_score=0.2,
        task_type="coding",
        prompt="refactor the parse loop",
        requires_tools=False,
    )
    assert router.get_model_account(d.selected_model) != "openai-codex"
    assert d.plan_budget_denied is False, d.reason


def test_astra_stays_excluded_from_automatic_routing(clean_state, governor_open):
    """gpt-6-astra is force_model-only; the governor must not open a door to it."""
    d = router.route_task(
        complexity_score=0.99,
        task_type="planning",
        prompt="rearchitect the budget governor from first principles",
        requires_tools=True,
    )
    assert d.selected_model != "gpt-6-astra", d.reason
