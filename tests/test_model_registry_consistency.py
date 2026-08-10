"""Consistency invariants for the model registry (single source of truth).

These tests assert that DEFAULT_MODEL_COSTS, MODEL_COST_ORDER and
MODEL_CAPABILITY_TIERS never drift apart. They are RED today because
deepseek-v3.1:671b is missing from MODEL_CAPABILITY_TIERS and dolphin3
is missing from all three tables.

GREEN target: a single MODEL_REGISTRY dict derives all three tables so
they are always consistent by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import models as m


def _cost_names() -> set[str]:
    return {c.model for c in m.DEFAULT_MODEL_COSTS}


def _order_names() -> set[str]:
    return set(m.MODEL_COST_ORDER)


def _tier_names() -> set[str]:
    return set(m.MODEL_CAPABILITY_TIERS.keys())


def test_cost_and_order_sets_are_identical() -> None:
    """Every priced model must appear in the cost-order escalation chain."""
    assert _cost_names() == _order_names()


def test_every_priced_model_has_a_capability_tier() -> None:
    """No model may be priced/ordered but lack a capability tier.

    Today deepseek-v3.1:671b is in DEFAULT_MODEL_COSTS and MODEL_COST_ORDER
    but missing from MODEL_CAPABILITY_TIERS — this test is RED.
    """
    missing = _cost_names() - _tier_names()
    assert missing == set(), f"priced models missing a capability tier: {sorted(missing)}"


def test_every_tiered_model_is_priced() -> None:
    """No capability tier may reference an unpriced/unknown model."""
    unknown = _tier_names() - _cost_names()
    assert unknown == set(), f"tiered models with no cost entry: {sorted(unknown)}"


def test_cost_order_is_monotonic_by_input_cost() -> None:
    """MODEL_COST_ORDER must be sorted by ascending input cost per 1M tokens."""
    by_name = {c.model: c for c in m.DEFAULT_MODEL_COSTS}
    inputs = [by_name[name].input_cost_per_1m for name in m.MODEL_COST_ORDER]
    assert inputs == sorted(inputs), (
        f"MODEL_COST_ORDER not sorted by input cost: {list(zip(m.MODEL_COST_ORDER, inputs))}"
    )


def test_local_and_private_models_registered() -> None:
    """llama3.1:8b and dolphin3 must be priced and tiered (they are reachable
    backends configured in Hermes config, currently dolphin3 is absent)."""
    for name in ("llama3.1:8b", "dolphin3"):
        assert name in _cost_names(), f"{name} missing from DEFAULT_MODEL_COSTS"
        assert name in _tier_names(), f"{name} missing from MODEL_CAPABILITY_TIERS"


def test_policy_sets_only_reference_registered_models() -> None:
    """Router policy constants (TOOL_CAPABLE_MODELS, PRIVATE_MODEL) must only
    reference models that exist in MODEL_REGISTRY. This prevents drift between
    policy code and the registry."""
    import router
    reg = set(m.MODEL_REGISTRY.keys())
    unregistered = (set(router.TOOL_CAPABLE_MODELS) | {router.PRIVATE_MODEL}) - reg
    assert unregistered == set(), (
        f"policy references unregistered models: {sorted(unregistered)}"
    )
