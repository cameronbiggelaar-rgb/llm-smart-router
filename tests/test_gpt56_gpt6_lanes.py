"""Tests for the gpt-5.6 tiers (11-13) and the gpt-6-astra lane (14).

Lane 1 (gpt-5.6): three agentic-coding models registered as the new top
routing tiers above gpt-5.5 (10). They are reached ONLY via escalation and
the explicit rethink/rearchitect trigger — routine work must stay on
ollama-cloud (the floor is NOT raised).

Lane 2 (gpt-6-astra): registered at tier 14 but NOT in the routing table and
EXCLUDED from automatic fallback/escalation chains — reachable ONLY via
force_model. This honours the user's directive to keep the $20/mo ChatGPT
capacity for the bigger problems and never let routine work drift onto the
strong models.

These tests are RED against the current code (gpt-5.6 / gpt-6 not registered,
clamp at 10) and go GREEN once the registry + clamp + exclusion land.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import models as m
from router import (
    route_task,
    mark_available,
    escalate_on_failure,
    _build_fallback_chain,
    _estimate_min_tier,
    MODEL_CAPABILITY_TIERS,
)

ALL_MODELS = [
    "llama3.1:8b",
    "dolphin3",
    "deepseek-v4-flash",
    "minimax-m2.7:cloud",
    "glm-5",
    "glm-5.1",
    "glm-5.3",
    "glm-5.2",
    "qwen3.5",
    "deepseek-v4-pro",
    "deepseek-v3.1:671b",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "gpt-6-astra",
]


@pytest.fixture(autouse=True)
def _all_models_available():
    for model in ALL_MODELS:
        mark_available(model)
    yield


# ── Registry ────────────────────────────────────────────────────────────────


def test_gpt56_models_registered_priced_and_tiered():
    """The three 5.6 models must be in the registry, priced, and tiered 11-13."""
    expected = {"gpt-5.6-luna": 11, "gpt-5.6-terra": 12, "gpt-5.6-sol": 13}
    for name, tier in expected.items():
        assert name in m.MODEL_REGISTRY, f"{name} missing from MODEL_REGISTRY"
        assert name in m.MODEL_CAPABILITY_TIERS, f"{name} missing from tiers"
        assert m.MODEL_CAPABILITY_TIERS[name] == tier, (
            f"{name} tier {m.MODEL_CAPABILITY_TIERS[name]} != {tier}"
        )
        assert name in {c.model for c in m.DEFAULT_MODEL_COSTS}, (
            f"{name} missing from DEFAULT_MODEL_COSTS"
        )


def test_gpt6_astra_registered_at_tier_14():
    """gpt-6-astra must be registered at tier 14 (above the 5.6 models)."""
    assert "gpt-6-astra" in m.MODEL_REGISTRY
    assert m.MODEL_CAPABILITY_TIERS["gpt-6-astra"] == 14


def test_gpt56_models_ordered_above_gpt55():
    """MODEL_COST_ORDER must place gpt-5.5 before the 5.6 models (cheapest first)."""
    order = m.MODEL_COST_ORDER
    assert order.index("gpt-5.5") < order.index("gpt-5.6-luna") < order.index("gpt-5.6-terra") < order.index("gpt-5.6-sol") < order.index("gpt-6-astra")


# ── Rethink/rearchitect trigger (Lane 1) ────────────────────────────────────


def test_rethink_rearchitect_routes_to_gpt56_sol():
    """A deliberate rethink/rearchitect planning ask must reach gpt-5.6-sol.

    Requires the clamp ceiling raised from 10 to 13 (3.4a) — otherwise the
    tier-13 floor is clamped to 10 and routes to gpt-5.5.
    """
    decision = route_task(
        complexity_score=0.5,
        task_type="planning",
        prompt="rethink the architecture and find failure states",
    )
    assert decision.selected_model == "gpt-5.6-sol", (
        f"rethink/rearchitect should route to gpt-5.6-sol, got {decision.selected_model}"
    )


def test_estimate_min_tier_reaches_13_for_rethink():
    """The clamp must allow the rethink floor (13) through, not cap at 10."""
    tier = _estimate_min_tier(
        complexity_score=0.5,
        task_type="planning",
        has_niche_references=False,
        has_format_constraint=False,
        instruction_count=0,
        is_subagent=False,
        parent_model="",
        prompt="rearchitect the system design",
    )
    assert tier == 13, f"rethink floor should be 13, got {tier}"


# ── Escalation chain (Lane 1) ────────────────────────────────────────────────


def test_escalation_from_gpt55_goes_to_gpt56_luna():
    """gpt-5.5 failure must escalate to gpt-5.6-luna (first tier > 10)."""
    decision = escalate_on_failure("gpt-5.5", error_type="rate_limit")
    assert decision.selected_model == "gpt-5.6-luna", (
        f"escalation from gpt-5.5 should hit gpt-5.6-luna, got {decision.selected_model}"
    )


def test_escalation_chain_luna_terra_sol():
    """Escalation must walk luna -> terra -> sol in order."""
    assert escalate_on_failure("gpt-5.6-luna", error_type="rate_limit").selected_model == "gpt-5.6-terra"
    assert escalate_on_failure("gpt-5.6-terra", error_type="rate_limit").selected_model == "gpt-5.6-sol"


def test_fallback_chain_from_gpt55_includes_56_models():
    """gpt-5.5's fallback chain must include the 5.6 models (but NOT gpt-6)."""
    chain = _build_fallback_chain("gpt-5.5")
    assert "gpt-5.6-luna" in chain
    assert "gpt-5.6-terra" in chain
    assert "gpt-5.6-sol" in chain


# ── Regression guards: floor NOT raised (Lane 1) ────────────────────────────


def test_routine_implement_still_routes_to_ollama_cloud():
    """Routine coding 'implement/build' must stay on ollama-cloud, NOT gpt-5.6."""
    decision = route_task(
        complexity_score=0.5,
        task_type="coding",
        prompt="implement a function to parse the config",
    )
    assert decision.selected_model not in {"gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"}, (
        f"routine implement should not hit gpt-5.6, got {decision.selected_model}"
    )
    assert decision.selected_model in {"glm-5.3", "glm-5.2", "qwen3.5", "deepseek-v4-pro"}, (
        f"routine implement should stay on ollama-cloud, got {decision.selected_model}"
    )


def test_routine_debug_crash_still_routes_to_ollama_cloud():
    """Routine debugging 'crash/race/deadlock' must stay on ollama-cloud, NOT gpt-5.6."""
    decision = route_task(
        complexity_score=0.5,
        task_type="debugging",
        prompt="fix the race condition causing the crash",
    )
    assert decision.selected_model not in {"gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"}, (
        f"routine debug should not hit gpt-5.6, got {decision.selected_model}"
    )
    assert decision.selected_model in {"qwen3.5", "deepseek-v4-pro", "glm-5.3", "glm-5.2"}, (
        f"routine debug should stay on ollama-cloud, got {decision.selected_model}"
    )


# ── gpt-6-astra lane (Lane 2) ────────────────────────────────────────────────


def test_gpt6_astra_never_auto_selected():
    """gpt-6-astra must NOT be auto-selected for any normal task type."""
    for kwargs in (
        dict(complexity_score=1.0, task_type="planning", prompt="architecture system design"),
        dict(complexity_score=1.0, task_type="coding", prompt="implement a complex feature"),
        dict(complexity_score=1.0, task_type="debugging", prompt="crash deadlock race"),
        dict(complexity_score=1.0, task_type="research", prompt="deep research analyze"),
    ):
        decision = route_task(**kwargs)
        assert decision.selected_model != "gpt-6-astra", (
            f"gpt-6-astra must never be auto-selected, got {decision.selected_model} for {kwargs}"
        )


def test_gpt6_astra_reachable_via_force_model():
    """gpt-6-astra must be reachable via force_model (the explicit lane)."""
    decision = route_task(force_model="gpt-6-astra")
    assert decision.selected_model == "gpt-6-astra"


def test_gpt6_astra_excluded_from_fallback_chain():
    """gpt-6-astra must NOT appear in any automatic fallback chain."""
    chain = _build_fallback_chain("gpt-5.5")
    assert "gpt-6-astra" not in chain, (
        f"gpt-6-astra must be excluded from fallback chains, got {chain}"
    )


def test_gpt6_astra_excluded_from_escalation():
    """Escalation from the top 5.6 model must NOT reach gpt-6-astra."""
    decision = escalate_on_failure("gpt-5.6-sol", error_type="rate_limit")
    assert decision.selected_model != "gpt-6-astra", (
        f"escalation must not reach gpt-6-astra, got {decision.selected_model}"
    )
