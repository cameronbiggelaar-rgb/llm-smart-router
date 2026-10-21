"""Model retirement — deepseek-v4-flash and qwen3.5 are gone.

Ollama retires `deepseek-v4-flash:0731` and `qwen3.5:397b` on 2026-09-25. A live
probe of `ollama.com/v1/models` (2026-09-19) confirmed both bare names in our
registry alias those retiring tags — there is no un-tagged `:latest` fallback.

These tests pin the post-retirement invariants so a future edit cannot silently
reintroduce a dead model:

* neither name appears in MODEL_REGISTRY;
* neither name is routable (MODEL_COST_ORDER / MODEL_CAPABILITY_TIERS);
* `deepseek-v4-flash` is gone from TOOL_CAPABLE_MODELS;
* neither name appears in the summariser ladder;
* the tier-7 normal-chat band now resolves to `minimax-m3` (the
  next-cheapest tier >= 7 after B46), not qwen3.5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import models as m  # noqa: E402
import router  # noqa: E402


RETIRED = ("deepseek-v4-flash", "qwen3.5")


def test_retired_models_absent_from_registry():
    for name in RETIRED:
        assert name not in m.MODEL_REGISTRY, f"{name} should be retired from the registry"


def test_retired_models_absent_from_cost_order():
    for name in RETIRED:
        assert name not in m.MODEL_COST_ORDER, f"{name} should not be routable by cost"


def test_retired_models_absent_from_capability_tiers():
    for name in RETIRED:
        assert name not in m.MODEL_CAPABILITY_TIERS, f"{name} should not carry a tier"


def test_deepseek_v4_flash_absent_from_tool_capable():
    assert "deepseek-v4-flash" not in router.TOOL_CAPABLE_MODELS


def test_retired_models_absent_from_summariser_ladder():
    for ctx in (0, 120_000, 400_000):
        ladder = router._session_compression_ladder(ctx)
        for name in RETIRED:
            assert name not in ladder, f"{name} still in ladder at {ctx} tokens"


def _cheapest_meeting(min_tier: int, available):
    for model in available:
        if router._normalize_model_name(model) in router.EXCLUDED_FROM_AUTO_ROUTING:
            continue
        if m.MODEL_CAPABILITY_TIERS.get(model, 0) >= min_tier:
            return model
    return ""


AUTO_ROUTABLE = [
    name
    for name in m.MODEL_COST_ORDER
    if name not in router.EXCLUDED_FROM_AUTO_ROUTING
]


def test_tier_7_band_resolves_to_minimax_m3():
    """The tier-7 band resolves to the cheapest frontier model at tier >= 7.

    After B46, minimax-m3 (tier 8, $0.60) undercuts deepseek-v4-pro (tier 8,
    $1.32 peak), so it is the next-cheapest model at tier >= 7. This is the
    deterministic outcome of the price-ascending cost-order loop, not a
    special case.
    """
    selected = _cheapest_meeting(7, AUTO_ROUTABLE)
    assert selected == "minimax-m3", (
        f"tier-7 routing selected {selected}, expected minimax-m3 after B46"
    )
