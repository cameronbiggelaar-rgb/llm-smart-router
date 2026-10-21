"""A price correction is a routing change, so the reordering is pinned.

`MODEL_COST_ORDER` is *derived* from `MODEL_REGISTRY` sorted by input price, and
`get_available_models()` returns that order filtered by availability. The auto
router picks the cheapest available model meeting a capability tier, so moving a
price moves the routing decision — silently, unless asserted.

The B18 correction (2026-09-13) moved five models and flipped exactly one tier
band. The B46 refresh (2026-10-22) repriced deepseek-v4-pro to its peak rate,
added minimax-m3 and minimax-m3.1-flash, retired minimax-m2.7:cloud, and flips
**four** bands (tier 4, 6, 7, 8). These tests make that consequence explicit so
the next price edit cannot be mistaken for bookkeeping.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import models as m  # noqa: E402
import router  # noqa: E402


def _cheapest_meeting(min_tier: int, available):
    """Mirror of `_select_model`'s core loop, over an injected model list."""
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


def test_cost_order_is_ascending_by_input_price():
    """The escalated chain must stay monotonic in price, or 'cheapest' is a lie."""
    prices = [m.MODEL_REGISTRY[n]["input"] for n in m.MODEL_COST_ORDER]
    assert prices == sorted(prices), (
        "MODEL_COST_ORDER is no longer ascending by input price — the router's "
        "cheapest-first assumption is broken"
    )


def test_corrected_models_sit_in_their_new_positions():
    """Pin the corrected ordering, so a price edit shows up as a routing diff."""
    order = m.MODEL_COST_ORDER

    # deepseek-v4-pro, repriced to its PEAK rate ($1.32), is still cheaper than
    # every glm ($1.40), so it precedes them all.
    for glm in ("glm-5.3", "glm-5.2"):
        assert order.index("deepseek-v4-pro") < order.index(glm), (
            f"deepseek-v4-pro ($1.32) should now precede {glm}"
        )
    # minimax-m3 ($0.60) undercuts deepseek-v4-pro at any hour.
    assert order.index("minimax-m3") < order.index("deepseek-v4-pro")


def test_tier_6_band_selects_the_cheapest_capable_model():
    """Tier 6 resolves to minimax-m3, the cheapest frontier-capable model.

    Before B46, deepseek-v4-pro (tier 8, $0.66) won this band. Its peak repricing
    ($1.32) plus the new minimax-m3 (tier 8, $0.60) moves the band to minimax-m3.
    """
    selected = _cheapest_meeting(6, AUTO_ROUTABLE)
    assert selected == "minimax-m3", (
        f"tier-6 routing selected {selected}, expected minimax-m3 — the "
        "B46 refresh changed this band; update the test only with a deliberate "
        "pricing decision"
    )


@pytest.mark.parametrize(
    "min_tier,expected",
    [
        (1, "llama3.1:8b"),
        (2, "dolphin3"),
        (3, "deepseek-v4.1-flash"),
        (4, "minimax-m3.1-flash"),
        (5, "glm-5.3-flash"),
        (7, "minimax-m3"),
        (8, "minimax-m3"),
        (9, "gpt-5.5"),
        (10, "gpt-5.5"),
    ],
)
def test_bands_select_as_expected_after_b46(min_tier, expected):
    """Pin every band's cheapest-capable model after the B46 refresh."""
    assert _cheapest_meeting(min_tier, AUTO_ROUTABLE) == expected


def test_price_correction_is_visible_as_a_routing_change():
    """The linkage itself: prices decide routing, so they are not bookkeeping."""
    assert "deepseek-v4-pro" in m.MODEL_COST_ORDER
    # deepseek-v4-pro ($1.32 peak) outranks the glm family ($1.40) by price.
    assert m.MODEL_COST_ORDER.index("deepseek-v4-pro") < m.MODEL_COST_ORDER.index("glm-5.3")
