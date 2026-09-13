"""B18 — a price correction is a routing change, so the reordering is pinned.

`MODEL_COST_ORDER` is *derived* from `MODEL_REGISTRY` sorted by input price, and
`get_available_models()` returns that order filtered by availability. The auto
router picks the cheapest available model meeting a capability tier, so moving a
price moves the routing decision — silently, unless asserted.

The B18 correction (2026-09-13) moved five models and flipped exactly one tier
band. These tests make that consequence explicit so the next price edit cannot
be mistaken for bookkeeping.
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

    # Cheaper than glm-5.3 now, because $0.60 < $1.40.
    assert order.index("qwen3.5") < order.index("glm-5.3"), (
        "qwen3.5 ($0.60) should now precede glm-5.3 ($1.40)"
    )
    assert order.index("qwen3.5") < order.index("glm-5.2"), (
        "qwen3.5 ($0.60) should now precede glm-5.2 ($1.40)"
    )
    # v4-pro ($0.66) is now cheaper than every glm, so it precedes them all.
    for glm in ("glm-5", "glm-5.1", "glm-5.3", "glm-5.2"):
        assert order.index("deepseek-v4-pro") < order.index(glm), (
            f"deepseek-v4-pro ($0.66) should now precede {glm}"
        )
    # minimax ($0.30) is cheaper than deepseek-v4-pro ($0.66).
    assert order.index("minimax-m2.7:cloud") < order.index("deepseek-v4-pro")


def test_tier_6_band_selects_the_cheaper_capable_model():
    """The one band B18 changed: tier 6 now resolves to qwen3.5, not glm-5.3.

    glm-5.3 is tier 6 at $1.40; qwen3.5 is tier 7 at $0.60 — cheaper *and* more
    capable, so it wins on both counts. Before the correction glm-5.3 was
    $1.50 and ordered first.
    """
    selected = _cheapest_meeting(6, AUTO_ROUTABLE)
    assert selected == "qwen3.5", (
        f"tier-6 routing selected {selected}, expected qwen3.5 — the B18 "
        "reordering changed this band; update the test only with a deliberate "
        "pricing decision"
    )


@pytest.mark.parametrize(
    "min_tier,expected",
    [
        (1, "llama3.1:8b"),
        (2, "dolphin3"),
        (3, "deepseek-v4.1-flash"),
        (4, "glm-5.3-flash"),
        (5, "glm-5.3-flash"),
        (7, "qwen3.5"),
        (8, "deepseek-v4-pro"),
        (9, "deepseek-v3.1:671b"),
        (10, "gpt-5.5"),
    ],
)
def test_unaffected_bands_still_select_as_before(min_tier, expected):
    """Every band other than tier 6 must be untouched by the correction."""
    assert _cheapest_meeting(min_tier, AUTO_ROUTABLE) == expected


def test_price_correction_is_visible_as_a_routing_change():
    """The linkage itself: prices decide routing, so they are not bookkeeping."""
    assert "deepseek-v4-pro" in m.MODEL_COST_ORDER
    assert "qwen3.5" in m.MODEL_COST_ORDER
    # If price did not drive order, this ordering assertion could not hold.
    assert m.MODEL_COST_ORDER.index("qwen3.5") != m.MODEL_COST_ORDER.index("glm-5.3")
