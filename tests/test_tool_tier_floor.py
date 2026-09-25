"""Batch B38a (RED) — the tool-capable pick must honour a capability floor.

Why this module exists
----------------------
``_select_tool_capable_model()`` takes no arguments and picks the **highest-tier
available** tool-capable model, *ignoring* ``min_tier`` entirely. Its docstring
states this is deliberate quality policy ("prefer the MOST CAPABLE available …
so tool work keeps the best reasoning quality"). Measured consequence on 30d of
``router_logs``:

  * ``tool-bearing request — selected proven tool-capable model gpt-5.6-sol
    (tier 13)`` = **1,267 calls / 49.4 Mtok** — every one an ordinary tool
    request, because the branch has no complexity gate at all;
  * of 1,760 ``sol`` calls, 1,267 arrived via this branch, 489 via escalation,
    and only **3** via the genuine architectural tier-13 lane. A tier-13 model
    was reached by ordinary tool work **422×** more often than by the lane it
    exists for.

Every other selector in the module takes a floor: ``_select_model(min_tier)``
returns the *cheapest* model meeting it. This module makes the tool branch
consistent with that, which is the §9.4 fix.

Semantics pinned here (mirroring ``_select_model``):
  1. floor met by something  -> the CHEAPEST tool-capable model meeting it;
  2. nothing meets the floor -> degrade predictably to the most capable
     available (design line 131), never fail closed on a tier miss;
  3. floor 0 (default)       -> cheapest tool-capable. NOTE this deliberately
     CHANGES behaviour for un-floored callers: the pre-B38 answer was always the
     top model. Two tests that encoded the old behaviour were updated with it
     (``test_governor_disabled_matches_pre_b37_behaviour``,
     ``test_tool_routing_prefers_most_capable_tool_model``).

The architectural floor
-----------------------
The tool branch composes ``max(_estimate_min_tier(...), _architectural_floor(prompt))``.
The second term exists because ``_estimate_min_tier`` scans only the request's
own task_type, while the tier-13 architectural sub-type lives solely under
``planning`` — so a genuine architectural ask tagged ``coding`` would get a floor
of 11 and be demoted off ``gpt-5.6-sol``. ``_architectural_floor`` uses
``gpt_budget.classify_request`` (which scans every task type, per rev 4) to keep
routing and budget classification in agreement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import models  # noqa: E402
import router  # noqa: E402

# Tool-capable, in MODEL_COST_ORDER (cheap -> expensive), with tiers:
#   glm-5.3 6 | deepseek-v4-pro 8 | gpt-5.5 10 | gpt-5.6-luna 11
#   gpt-5.6-terra 12 | gpt-5.6-sol 13
TOOL_CAPABLE = ["glm-5.3", "deepseek-v4-pro", "gpt-5.5", "gpt-5.6-luna",
                "gpt-5.6-terra", "gpt-5.6-sol"]
ALL_STATUS = TOOL_CAPABLE + ["llama3.1:8b", "dolphin3", "gpt-6-astra"]


@pytest.fixture(autouse=True)
def ungovened_plan(monkeypatch):
    """Stub the plan-budget governor OFF for this module.

    These are CAPABILITY tests: they assert which model the tool branch picks for
    a given capability floor, not budget policy (``test_gpt_budget_*`` owns
    policy). The governor is ARMED in the shipped policy file, and the live
    account is currently closed, so without this stub an architectural request is
    correctly DENIED and reselected onto a non-plan model — which would make
    these tests fail for a reason that has nothing to do with routing floors.
    (Same convention as ``_ungoverned_plan`` in test_gpt56_gpt6_lanes.py,
    established when the governor was armed.)
    """
    import gpt_budget as gb

    cfg = gb.GovernorConfig(
        enabled=False,
        allowlist=[gb.CLASS_ARCHITECTURAL],
        hard_cap_percent=95.0,
    )
    monkeypatch.setattr(gb, "load_governor_config", lambda force=False: cfg)
    monkeypatch.setattr(gb, "fetch_reading", lambda cfg=None, **kw: _reading(0.0))
    return cfg


def _reading(used_pct: float = 0.0, allowed: bool = True):
    """A benign account reading (plenty of headroom), so the governor is quiet.

    Shape copied from the other governor tests — ``Reading`` wraps two
    ``WindowReading``s (primary 5h / secondary 7d), not scalar credit fields.
    """
    import gpt_budget as gb

    return gb.Reading(
        source="live",
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


@pytest.fixture
def all_tool_models_available():
    """Every tool-capable model available, so the test measures POLICY not limits."""
    saved = {m: router._MODEL_STATUSES.get(m) for m in ALL_STATUS}
    for m in ALL_STATUS:
        router._MODEL_STATUSES[m] = router.ModelStatus(model=m, available=True)
    yield
    for m, st in saved.items():
        if st is None:
            router._MODEL_STATUSES.pop(m, None)
        else:
            router._MODEL_STATUSES[m] = st


@pytest.fixture
def only_pro_available():
    """Only deepseek-v4-pro (tier 8) available — the 'floor unreachable' case."""
    saved = {m: router._MODEL_STATUSES.get(m) for m in ALL_STATUS}
    for m in ALL_STATUS:
        avail = m == "deepseek-v4-pro"
        router._MODEL_STATUSES[m] = router.ModelStatus(model=m, available=avail)
    yield
    for m, st in saved.items():
        if st is None:
            router._MODEL_STATUSES.pop(m, None)
        else:
            router._MODEL_STATUSES[m] = st


# ── 1. The floor is honoured: cheapest model meeting it ──────────────────────
#
# "Cheapest" is MODEL_COST_ORDER, which is PRICE-based, not tier-based — verified
# against the price book (effective 2026-09-13):
#   deepseek-v4-pro  tier 8   $0.66 / $1.98 per Mtok
#   glm-5.3          tier 6   $1.40 / $4.40
#   gpt-5.5          tier 10  $15   / $60
#   gpt-5.6-luna     tier 11  $16   / $64
#   gpt-5.6-terra    tier 12  $17   / $68
#   gpt-5.6-sol      tier 13  $18   / $72
# So pro (tier 8) precedes glm-5.3 (tier 6) in cost order: it is cheaper AND
# more capable. That ordering matters for every expectation below.

@pytest.mark.parametrize("floor,expected", [
    (0,  "deepseek-v4-pro"),  # cheapest tool-capable
    (1,  "deepseek-v4-pro"),
    (6,  "deepseek-v4-pro"),  # pro (8) satisfies floor 6 and is cheaper than glm-5.3
    (7,  "deepseek-v4-pro"),
    (8,  "deepseek-v4-pro"),  # exactly meets
    (9,  "gpt-5.5"),          # pro (8) too low
    (10, "gpt-5.5"),
    (11, "gpt-5.6-luna"),
    (12, "gpt-5.6-terra"),
    (13, "gpt-5.6-sol"),
])
def test_floor_selects_cheapest_model_meeting_it(all_tool_models_available, floor, expected):
    """The whole point: tier floor respected, and the CHEAPEST sufficient model wins.

    Before this fix the answer was 'gpt-5.6-sol' for every floor — including 0.
    """
    got = router._select_tool_capable_model(floor)
    assert got == expected, (
        f"floor {floor}: expected cheapest tool-capable meeting the floor "
        f"({expected}), got {got}"
    )


def test_floor_zero_is_cheapest_not_most_capable(all_tool_models_available):
    """Explicitly pins the reversal: floor 0 must NOT return the tier-13 model."""
    got = router._select_tool_capable_model(0)
    assert got != "gpt-5.6-sol", (
        "floor 0 still returned the most-capable tool model — the §9.4 defect"
    )
    assert got == "deepseek-v4-pro"


def test_no_floor_is_cheapest_tool_capable(all_tool_models_available):
    """Default (absent) floor must equal floor 0, so existing callers are unchanged."""
    assert router._select_tool_capable_model() == router._select_tool_capable_model(0)


# ── 2. Floor unreachable -> degrade predictably, never fail closed ───────────

def test_unreachable_floor_degrades_to_most_capable(only_pro_available):
    """Design line 131: degrade predictably. A tier miss is not a hard failure.

    Only pro (tier 8) is up and the request wants 13. Failing closed would throw
    away a perfectly serviceable tool-capable model; the design says degrade.
    """
    got = router._select_tool_capable_model(13)
    assert got == "deepseek-v4-pro", (
        "an unreachable floor must degrade to the most capable available "
        "tool model, not return '' (fail closed)"
    )


def test_no_tool_model_at_all_still_fails_closed():
    """A genuine capability gap (nothing tool-capable) must still fail closed."""
    saved = {m: router._MODEL_STATUSES.get(m) for m in ALL_STATUS}
    try:
        for m in ALL_STATUS:
            router._MODEL_STATUSES[m] = router.ModelStatus(model=m, available=False)
        got = router._select_tool_capable_model(5)
    finally:
        for m, st in saved.items():
            if st is None:
                router._MODEL_STATUSES.pop(m, None)
            else:
                router._MODEL_STATUSES[m] = st
    assert got == "", "no tool-capable model available must yield '' so the caller fails closed"


# ── 3. The floor must never select a NON-tool-capable model ─────────────────

def test_floor_never_returns_non_tool_capable_model(all_tool_models_available):
    """Tool capability is the binding gate; the floor only ranks WITHIN it.

    A high floor must not tempt the selector into a non-tool model (e.g. a
    hypothetical tier-14 chat model) — that would silently break tool calling,
    which is the one guarantee this function exists to provide.
    """
    for floor in range(0, 16):
        got = router._select_tool_capable_model(floor)
        assert got == "" or router._normalize_model_name(got) in router.TOOL_CAPABLE_MODELS, (
            f"floor {floor} returned non-tool-capable model {got!r}"
        )


def test_astra_excluded_even_at_high_floor(all_tool_models_available):
    """gpt-6-astra (tier 14) is EXCLUDED_FROM_AUTO_ROUTING and must never be picked."""
    for floor in (13, 14, 99):
        got = router._select_tool_capable_model(floor)
        assert "astra" not in got, f"floor {floor} returned the excluded model {got!r}"


# ── 4. Cost order is respected (cheapest-meeting is by MODEL_COST_ORDER) ─────

def test_cheapest_meeting_is_a_cheaper_model_not_a_higher_tier(all_tool_models_available):
    """A floor of 4 must return pro (tier 8), never glm-5.3 (tier 6).

    Guards against an implementation that sorts by tier and returns the first
    match above the floor while ignoring price — pro is both cheaper and more
    capable, so "cheapest meeting the floor" and "lowest tier meeting the floor"
    genuinely differ here. That difference is what this pins.
    """
    assert router._select_tool_capable_model(4) == "deepseek-v4-pro"
    assert router._select_tool_capable_model(6) == "deepseek-v4-pro"


def test_cost_order_is_price_based_and_pinned():
    """Sanity: MODEL_COST_ORDER is the basis for 'cheapest'. Pin the tool-capable
    slice AND the price facts that justify it, so a price-book change that
    reorders these models fails loudly here rather than silently re-routing."""
    order = [m for m in models.MODEL_COST_ORDER if m in TOOL_CAPABLE]
    assert order == ["deepseek-v4-pro", "glm-5.3", "gpt-5.5", "gpt-5.6-luna",
                     "gpt-5.6-terra", "gpt-5.6-sol"], (
        f"tool-capable cost order changed: {order}"
    )
    # pro must be cheaper than glm-5.3 for the ordering above to be coherent.
    import sqlite3

    conn = sqlite3.connect(SCRIPTS_DIR.parent / "data" / "router_logs.db")
    rows = dict(
        (m, (i, o))
        for m, i, o in conn.execute(
            """SELECT model, input_usd_per_1m, output_usd_per_1m FROM model_pricing
               WHERE model IN ('deepseek-v4-pro','glm-5.3')
                 AND effective_from = (SELECT MAX(effective_from) FROM model_pricing p2
                                       WHERE p2.model = model_pricing.model)"""
        )
    )
    conn.close()
    assert rows["deepseek-v4-pro"][0] < rows["glm-5.3"][0], (
        f"pro is no longer cheaper than glm-5.3 ({rows}) — the cost order and the "
        "floor expectations must be revisited together"
    )


# ── 5. _architectural_floor: the all-task-type scan ─────────────────────────

def test_architectural_floor_fires_for_architectural_prompts():
    """A genuinely architectural prompt raises the floor to 13."""
    assert router._architectural_floor(
        "rearchitect the storage layer and find failure states"
    ) == 13


def test_architectural_floor_silent_for_ordinary_prompts():
    """Ordinary tool work must NOT be raised to the architectural floor."""
    for prompt in ("write a for loop", "refactor the parse loop", "fix the bug",
                   "add a test", ""):
        assert router._architectural_floor(prompt) == 0, prompt


def test_architectural_floor_uses_the_all_task_type_classifier():
    """The reason _architectural_floor exists rather than a raw min_tier.

    ``_estimate_min_tier`` -> ``match_sub_type(task_type, prompt)`` scans only the
    REQUEST's task type, and the tier-13 sub-type lives solely under ``planning``.
    So the same architectural prompt tagged ``coding`` yields floor 6 from the
    estimator (the tier-13 keyword block is never consulted) — far short of sol.
    ``_architectural_floor`` uses the all-task-type classifier, so it still
    returns 13 and the request keeps sol.
    """
    import gpt_budget as gb

    prompt = "rearchitect the storage layer and find failure states"
    assert gb.classify_request(None, prompt, False) == gb.CLASS_ARCHITECTURAL

    estimator_floor = router._estimate_min_tier(
        complexity_score=0.5,
        task_type="coding",
        has_niche_references=False,
        has_format_constraint=False,
        instruction_count=1,
        is_subagent=False,
        parent_model=None,
        prompt=prompt,
    )
    arch_floor = router._architectural_floor(prompt)
    assert arch_floor == 13
    # Composed: the architectural floor must win over the task-type-scoped one.
    assert max(estimator_floor, arch_floor) == 13, (
        f"composition regressed: estimator={estimator_floor} arch={arch_floor}"
    )


def test_architectural_floor_is_governor_independent(all_tool_models_available, ungovened_plan):
    """The floor is a CAPABILITY notion, not an entitlement.

    It must apply whether or not the allowlist is armed, otherwise disarming the
    governor would silently demote architectural work off sol. The autouse fixture
    already stubs the governor OFF, so this asserts the floor still fires here.
    """
    assert router._architectural_floor(
        "rearchitect the storage layer and find failure states"
    ) == 13


def test_architectural_floor_fires_even_when_governor_armed(all_tool_models_available):
    """Armed governor must not change the CAPABILITY floor (only the entitlement).

    Pairs with the test above: OFF gives 13, and so must ON — the floor is about
    what the request needs, the allowlist is about what it is permitted.
    """
    import gpt_budget as gb

    orig_load, orig_fetch = gb.load_governor_config, gb.fetch_reading
    gb.load_governor_config = lambda force=False: gb.GovernorConfig(
        enabled=True, allowlist=[gb.CLASS_ARCHITECTURAL], hard_cap_percent=95.0
    )
    gb.fetch_reading = lambda cfg=None, **kw: _reading(10.0)
    try:
        assert router._architectural_floor(
            "rearchitect the storage layer and find failure states"
        ) == 13
    finally:
        gb.load_governor_config, gb.fetch_reading = orig_load, orig_fetch


def test_architectural_floor_fails_safe_on_classifier_error(all_tool_models_available):
    """A classifier explosion must yield 0, not invent a floor."""
    import gpt_budget as gb

    orig = gb.classify_request

    def boom(*a, **kw):
        raise RuntimeError("classifier exploded")

    gb.classify_request = boom
    try:
        assert router._architectural_floor("rearchitect everything") == 0
    finally:
        gb.classify_request = orig


# ── 6. End-to-end: the branch itself honours the floor ─────────────────────

def test_route_task_ordinary_tool_work_takes_a_cheap_tool_model():
    """The §9.4 leak, end-to-end through route_task.

    An ordinary coding request with tools must NOT land on the tier-13 plan
    model — it did before B38 (1,267 such calls in 30d).
    """
    saved = {m: router._MODEL_STATUSES.get(m) for m in ALL_STATUS}
    try:
        for m in ALL_STATUS:
            router._MODEL_STATUSES[m] = router.ModelStatus(model=m, available=True)
        d = router.route_task(
            complexity_score=0.1,
            task_type="coding",
            prompt="write a for loop",
            requires_tools=True,
        )
    finally:
        for m, st in saved.items():
            if st is None:
                router._MODEL_STATUSES.pop(m, None)
            else:
                router._MODEL_STATUSES[m] = st
    assert d.selected_model != "gpt-5.6-sol", d.reason
    assert "capability floor" in d.reason, d.reason


def test_route_task_architectural_tool_work_keeps_sol():
    """The counterweight: a genuine architectural tool request still gets sol.

    Without ``_architectural_floor`` in the composition this request would be
    demoted to a cheap estimator-picked model (floor 6 from the task-type-scoped
    estimator, which never consults the tier-13 block), under-serving the exact
    lane B38 exists to protect.
    """
    saved = {m: router._MODEL_STATUSES.get(m) for m in ALL_STATUS}
    try:
        for m in ALL_STATUS:
            router._MODEL_STATUSES[m] = router.ModelStatus(model=m, available=True)
        d = router.route_task(
            complexity_score=0.5,
            task_type="planning",
            prompt="rearchitect the storage layer and find failure states",
            requires_tools=True,
        )
    finally:
        for m, st in saved.items():
            if st is None:
                router._MODEL_STATUSES.pop(m, None)
            else:
                router._MODEL_STATUSES[m] = st
    assert d.selected_model == "gpt-5.6-sol", d.reason
