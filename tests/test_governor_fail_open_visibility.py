"""Batch B44a (RED) — the governor's silent fail-open.

Why this module exists
----------------------
An armed allowlist governor recorded `plan_budget_denied = 0` on **all 5,211**
GPT rows ever logged, while serving 4,392 of them as `budget_class='routine'`.
`plan_budget_denied = 1` appears on 132 rows, all on one day (2026-09-27), all
from a hard-cap window — never from the allowlist.

`_apply_plan_governor` has four exits and three of them record a class without a
verdict. The one that explains the data is the fail-open on an unreadable reading
(gpt_budget.py:572): `_read_usage_payload()` swallows every exception → `None` →
`Reading(source='unavailable')` → `VERDICT_OPEN` **before the allowlist is
consulted** → `(selected, klass, False, False)`. The plan model is served, the
class is stamped, nothing is logged.

Every other exit is excluded by the data: `enabled=false` would leave the class
empty (37,598 rows carry one); the classifier and exception exits both log a
warning (journal count for those strings is 0); `_is_plan_model` returns True for
the bare names the DB actually holds; and `binding_window is None` is itself only
reachable via `source='unavailable'`.

Two things are wrong and both are pinned here:

1. **Direction.** Under a *deny-list* "never starve the plan" is sound. Under an
   *allowlist* an unverifiable entitlement is not consent — failing open re-opens
   the leak at the exact moment the governor cannot notice. Routine must not ride
   a blind probe onto the plan.
2. **Visibility.** A blind window must be countable from the row data alone, and
   no row may claim `plan governor allowed class 'X'` when no verdict was reached.

The three-state model (denied / served / blind) is the whole point: B39's lesson
was that an unwritten column reported as a count is a false zero. A *silently*
written column is worse — it reads as a deliberate allow.

`enabled=false` must still short-circuit to `VERDICT_OPEN` with no class, so the
rollback contract stays byte-identical to pre-B37 behaviour.
"""

from __future__ import annotations

import sys
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
    """Every model up, so selection is decided by entitlement alone."""
    saved = {m: router._MODEL_STATUSES.get(m) for m in CODEX_MODELS + OLLAMA_MODELS}
    for m in CODEX_MODELS + OLLAMA_MODELS:
        router._MODEL_STATUSES[m] = router.ModelStatus(model=m, available=True)
    yield
    for m, st in saved.items():
        if st is None:
            router._MODEL_STATUSES.pop(m, None)
        else:
            router._MODEL_STATUSES[m] = st


def _live_reading(used_pct=0.0, allowed=True):
    return gb.Reading(
        source="live",
        allowed=allowed,
        primary=gb.WindowReading(
            used_percent=used_pct, limit_window_seconds=18000, reset_after_seconds=18000
        ),
        secondary=gb.WindowReading(
            used_percent=used_pct, limit_window_seconds=604800, reset_after_seconds=604800
        ),
    )


def _unreadable_reading():
    """Exactly what fetch_reading() returns when the probe fails."""
    return gb.Reading(source="unavailable", allowed=None, primary=None, secondary=None)


def _architectural_cfg():
    return gb.GovernorConfig(
        enabled=True,
        allowlist=[gb.CLASS_ARCHITECTURAL],
        routine_ceiling_percent=0,
        hard_cap_percent=95.0,
    )


@pytest.fixture
def governor_blind(monkeypatch):
    """Armed allowlist governor whose account probe is UNREADABLE."""
    monkeypatch.setattr(gb, "load_governor_config", lambda force=False: _architectural_cfg())
    monkeypatch.setattr(gb, "fetch_reading", lambda cfg=None, **kw: _unreadable_reading())


@pytest.fixture
def governor_open(monkeypatch):
    """Armed allowlist governor with a healthy, fully-readable account."""
    monkeypatch.setattr(gb, "load_governor_config", lambda force=False: _architectural_cfg())
    monkeypatch.setattr(gb, "fetch_reading", lambda cfg=None, **kw: _live_reading(0.0))


# ── 1. Policy level: direction of the fail-open ───────────────────────────────

def test_unverifiable_reading_is_a_distinct_verdict():
    """'Could not read the account' must not be spelled 'open'.

    Collapsing them is what made a blind governor indistinguishable from a
    permissive one in every row and every log line.
    """
    assert hasattr(gb, "VERDICT_UNVERIFIED"), (
        "no VERDICT_UNVERIFIED — an unreadable reading still reports as VERDICT_OPEN"
    )
    assert gb.VERDICT_UNVERIFIED != gb.VERDICT_OPEN


def test_unreadable_reading_denies_routine_by_the_allowlist():
    """Under an allowlist, an unverifiable entitlement is not consent.

    This is the leak: routine rode a blind probe straight onto the plan and was
    recorded as class='routine', denied=0 — 4,392 rows.
    """
    v = gb.evaluate_policy(_unreadable_reading(), _architectural_cfg(), klass=gb.CLASS_ROUTINE)
    assert v == gb.VERDICT_DENIED, (
        f"routine rode an unreadable reading onto the plan (verdict={v!r}); "
        f"expected {gb.VERDICT_DENIED!r}"
    )


def test_unreadable_reading_does_not_starve_the_architectural_lane():
    """The old invariant still holds where it was meant to: the plan is not starved.

    Architectural is a named class, so it stays entitled — but it is reported as
    UNVERIFIED, never as allowed.
    """
    v = gb.evaluate_policy(
        _unreadable_reading(), _architectural_cfg(), klass=gb.CLASS_ARCHITECTURAL
    )
    assert v == gb.VERDICT_UNVERIFIED, v


def test_rollback_lever_still_wins_over_an_unreadable_reading():
    """enabled=false is the rollback: nothing is governed, no class, verdict open."""
    cfg = gb.GovernorConfig(
        enabled=False, allowlist=[gb.CLASS_ARCHITECTURAL], hard_cap_percent=95.0
    )
    assert gb.evaluate_policy(_unreadable_reading(), cfg, klass=gb.CLASS_ROUTINE) == (
        gb.VERDICT_OPEN
    )


# ── 2. Routing level: the row that explains Oct 4–5 ───────────────────────────

def test_routine_does_not_ride_a_blind_probe_onto_the_plan(clean_state, governor_blind):
    """THE regression: the Oct-5 row shape must be impossible.

    Currently produces a gpt model with class='routine' and denied=0 — the exact
    signature of 4,392 logged rows.
    """
    d = router.route_task(
        complexity_score=0.85,
        task_type="coding",
        prompt="refactor the parse loop to use a deque",
        requires_tools=True,
    )
    plan = set(router.models_in_account(router._governed_account()))
    assert d.selected_model not in plan, (
        f"routine served from the plan on an unreadable reading: "
        f"model={d.selected_model!r} class={d.budget_class!r} denied={d.plan_budget_denied} "
        f"— reason={d.reason!r}"
    )
    assert d.selected_model == "deepseek-v4-pro", d.reason
    assert d.plan_budget_denied is True


def test_blind_window_is_flagged_on_the_decision(clean_state, governor_blind):
    """A blind window must be countable from the row, not inferable from absence.

    Without this the digest reports routine_denied=0 and reads as 'the governor
    looks inert' — B39's false zero again, one layer down.
    """
    d = router.route_task(
        complexity_score=0.9,
        task_type="planning",
        prompt="rearchitect the budget governor to be account-keyed",
        requires_tools=True,
    )
    assert getattr(d, "governor_fail_open", None) is True, (
        "RoutingDecision carries no governor_fail_open flag — a blind window is "
        "indistinguishable from an evaluated one"
    )


def test_architectural_still_served_when_blind(clean_state, governor_blind):
    """Do not over-block: the named lane keeps the plan while the probe is down."""
    d = router.route_task(
        complexity_score=0.9,
        task_type="planning",
        prompt="rearchitect the budget governor to be account-keyed",
        requires_tools=True,
    )
    assert d.selected_model == "gpt-5.6-sol", d.reason
    assert d.plan_budget_denied is False


# ── 3. The reason string must not assert a verdict that never happened ───────

def test_reason_never_claims_allowed_on_a_blind_probe(clean_state, governor_blind):
    """'plan governor allowed class X' on a blind window is a false statement."""
    d = router.route_task(
        complexity_score=0.9,
        task_type="planning",
        prompt="rearchitect the budget governor to be account-keyed",
        requires_tools=True,
    )
    assert "plan governor allowed class" not in d.reason, (
        f"row claims the allowlist was consulted when the probe was unreadable: {d.reason!r}"
    )
    assert "unverified" in d.reason.lower(), d.reason


def test_reason_never_claims_allowed_when_no_plan_model_was_in_play(clean_state, governor_open):
    """The short-circuit at router.py:1567 also appends 'allowed class'.

    A routine deepseek pick never consults the allowlist, so the row must say the
    class was *recorded*, not allowed.
    """
    d = router.route_task(
        complexity_score=0.3,
        task_type="coding",
        prompt="write a for loop",
        requires_tools=True,
    )
    if d.budget_class and d.selected_model not in set(
        router.models_in_account(router._governed_account())
    ):
        assert "plan governor allowed class" not in d.reason, (
            f"no-plan pick claims a verdict it never requested: {d.reason!r}"
        )


# ── 4. Chains must agree with the decision ────────────────────────────────────

def test_denied_classes_still_contain_the_fallback_chain(clean_state, governor_open):
    """A new verdict value must not silently flip chain containment."""
    d = router.route_task(
        complexity_score=0.85,
        task_type="coding",
        prompt="refactor the parse loop to use a deque",
        requires_tools=True,
    )
    assert d.plan_budget_denied is True
    plan = set(router.models_in_account(router._governed_account()))
    assert not (set(d.fallback_chain) & plan), (
        f"denied request still carries plan models in its escalation chain: {d.fallback_chain}"
    )
