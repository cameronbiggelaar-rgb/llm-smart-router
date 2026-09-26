"""Batch B39a (RED) — the governor must record its class on every routed request.

The defect
----------
``_apply_plan_governor`` returned ``klass=""`` for any non-plan model, so
``budget_class`` was only ever written when a ChatGPT-plan model was actually
*selected*. With GPT rate-limited, no request selects one, so **all 195,139 rows
in router_logs carried ``budget_class = ''``** and the nightly digest reported
the whole table as ``ungoverned`` — which the operator read as "the governor
looks inert".

The governor was not inert. It was never asked. These tests pin the three states
apart:

* governor off            -> ``''``  (no policy ran — the rollback contract)
* armed, plan not reached -> the class, ``denied=False``
* armed, plan reached     -> the class, ``denied`` per policy
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

PLAN_MODELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def statuses():
    """Every model up by default, with the saved state restored afterwards."""
    saved = {m: router._MODEL_STATUSES.get(m) for m in router.MODEL_COST_ORDER}
    for m in router.MODEL_COST_ORDER:
        router._MODEL_STATUSES[m] = router.ModelStatus(model=m, available=True)
    yield
    for m, st in saved.items():
        if st is None:
            router._MODEL_STATUSES.pop(m, None)
        else:
            router._MODEL_STATUSES[m] = st


def _reading(used_pct=10.0, allowed=True):
    return gb.Reading(
        source="live",
        allowed=allowed,
        primary=gb.WindowReading(
            used_percent=used_pct, limit_window_seconds=18000, reset_after_seconds=165300
        ),
        secondary=gb.WindowReading(
            used_percent=used_pct, limit_window_seconds=604800, reset_after_seconds=165300
        ),
    )


@pytest.fixture
def armed(monkeypatch):
    """The governor armed with the shipped allowlist, account healthy.

    Healthy account means the allowlist is the only thing deciding, so a routine
    class is denied by policy and an architectural one is allowed.
    """
    cfg = gb.GovernorConfig(
        enabled=True,
        allowlist=[gb.CLASS_ARCHITECTURAL],
        hard_cap_percent=95.0,
    )
    monkeypatch.setattr(gb, "load_governor_config", lambda force=False: cfg)
    monkeypatch.setattr(gb, "fetch_reading", lambda *a, **kw: _reading())
    return cfg


@pytest.fixture
def disarmed(monkeypatch):
    cfg = gb.GovernorConfig(
        enabled=False,
        allowlist=[gb.CLASS_ARCHITECTURAL],
        hard_cap_percent=95.0,
    )
    monkeypatch.setattr(gb, "load_governor_config", lambda force=False: cfg)
    monkeypatch.setattr(gb, "fetch_reading", lambda *a, **kw: _reading())
    return cfg


def _darken_plan():
    """Take every ChatGPT-plan model offline — the live state until the reset.

    A far-future cooldown is required, not just ``available = False``:
    ``route_task`` opens with a ``probe_all_recovering()`` sweep, and
    ``probe_recovery`` treats a *satisfied* cooldown as proof of health and calls
    ``mark_available``. Darkening with the flag alone is silently undone, and the
    test then passes while exercising the wrong branch (the same pitfall
    documented in test_gpt_budget_tool_branch.py).
    """
    until = time.time() + 3600
    for m in PLAN_MODELS:
        st = router._MODEL_STATUSES[m]
        st.available = False
        st.rate_limited_until = until


# ── The live case: armed, plan dark ───────────────────────────────────────────

def test_routine_tool_request_records_its_class_while_the_plan_is_dark(statuses, armed):
    """The exact live configuration: every plan model rate-limited.

    The tool branch degrades to ``deepseek-v4-pro`` (tier 8, floor 13), so no plan
    model is ever in play. The class must still be recorded — that is the whole
    difference between "armed but idle" and "no governor".
    """
    _darken_plan()
    d = router.route_task(
        complexity_score=0.85, task_type="coding",
        prompt="refactor the parse loop", requires_tools=True,
    )
    assert not router._is_plan_model(d.selected_model), d.reason
    assert d.budget_class == gb.CLASS_ROUTINE, (
        f"armed governor must classify a routine request even when the plan is "
        f"dark; got {d.budget_class!r} ({d.reason})"
    )
    assert d.plan_budget_denied is False, "nothing was denied — no plan model was reached"


def test_architectural_tool_request_records_architectural_class_while_dark(statuses, armed):
    """An architectural ask served off-plan (because the plan is down) is still
    architectural work — and must not be recorded as routine."""
    _darken_plan()
    d = router.route_task(
        complexity_score=0.5, task_type="coding",
        prompt="rearchitect the storage layer and design the migration plan",
        requires_tools=True,
    )
    assert d.budget_class == gb.CLASS_ARCHITECTURAL, (
        f"architectural work must be classified architectural; got "
        f"{d.budget_class!r} ({d.reason})"
    )


def test_non_tool_request_records_its_class(statuses, armed):
    """The ordinary capability path records a class for non-plan picks too.

    Previously guarded by ``if _is_plan_model(selected)``, so a cheap non-plan
    pick wrote nothing.
    """
    d = router.route_task(
        complexity_score=0.2, task_type="coding",
        prompt="what does this function do", requires_tools=False,
    )
    assert not router._is_plan_model(d.selected_model), d.reason
    assert d.budget_class != "", (
        f"armed governor must classify a non-tool non-plan request; got "
        f"{d.budget_class!r} ({d.reason})"
    )


# ── The rollback contract must not move ───────────────────────────────────────

def test_governor_off_records_no_class(statuses, disarmed):
    """``enabled=false`` is the rollback lever: byte-identical to pre-B37.

    Recording a class with the governor off would fabricate a policy history for
    traffic that was never governed.
    """
    d = router.route_task(
        complexity_score=0.85, task_type="coding",
        prompt="refactor the parse loop", requires_tools=True,
    )
    assert d.budget_class == "", d.reason
    assert d.plan_budget_denied is False


def test_classifier_failure_records_no_class_rather_than_inventing_one(statuses, armed, monkeypatch):
    """A classifier bug must not over-report the protected lane.

    ``_plan_classify`` fails safe to ``architectural`` (never deny on a bug). That
    is right for *gating* and wrong for *reporting*: it would mark every
    unclassifiable request as the lane gpt is reserved for. The recording path
    must fail to ``""`` instead.
    """
    def _boom(*a, **kw):
        raise RuntimeError("classifier exploded")

    monkeypatch.setattr(router, "_plan_classify", _boom)
    d = router.route_task(
        complexity_score=0.85, task_type="coding",
        prompt="refactor the parse loop", requires_tools=True,
    )
    assert d.budget_class == "", f"a classifier failure must not invent a class: {d.budget_class!r}"


# ── §5.5: the recorded class must also contain the fallback chain ─────────────

def test_recorded_class_keeps_plan_models_out_of_a_routine_chain(statuses, armed):
    """Recording the class arms failover containment for ordinary requests too.

    The chain walks UP in tier, so a routine request served by a non-plan model
    would otherwise escalate straight into the plan — the leak re-entering
    through the back door. Previously the class was empty on this path, so
    ``klass=None`` left the chain unfiltered.
    """
    _darken_plan()
    # Low floor so the pick is a cheap non-plan model even with the plan up.
    for m in PLAN_MODELS:
        router._MODEL_STATUSES[m] = router.ModelStatus(model=m, available=True)
    d = router.route_task(
        complexity_score=0.2, task_type="coding",
        prompt="rename this variable", requires_tools=True,
    )
    assert not router._is_plan_model(d.selected_model), d.reason

    # Non-vacuity: prove this configuration WOULD leak without the class, so the
    # assertion below cannot pass just because the plan is dark.
    unfiltered = router._build_tool_fallback_chain(d.selected_model)
    assert any(router._is_plan_model(m) for m in unfiltered), (
        f"test is vacuous — the unfiltered chain contains no plan model: {unfiltered}"
    )

    for model in d.fallback_chain:
        assert not router._is_plan_model(model), (
            f"routine request escalated into the reserved plan via {model}"
        )


def test_chain_verdict_is_computed_once_not_per_model(statuses, armed, monkeypatch):
    """Hoisting guard: the per-model verdict call is a ~87 µs state read.

    ``_build_tool_fallback_chain`` called ``_plan_denied_for_class`` **inside**
    the loop over available models. Recording the class universally puts that on
    the hot path for every tool request: ~14 models x 87 µs adds ~1.2 ms to a
    measured 667 µs ``route_task``. The verdict must not scale with model count.
    """
    calls = {"n": 0}
    real = router._plan_denied_for_class

    def _counting(klass):
        calls["n"] += 1
        return real(klass)

    monkeypatch.setattr(router, "_plan_denied_for_class", _counting)
    available = router.get_available_models()
    assert len(available) > 5, "need several available models for this to mean anything"

    router._build_tool_fallback_chain("deepseek-v4-pro", klass=gb.CLASS_ROUTINE)

    assert calls["n"] <= 1, (
        f"verdict recomputed {calls['n']} times for {len(available)} available "
        f"models — hoist it out of the loop"
    )
