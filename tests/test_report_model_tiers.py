"""Tests that the routing report's tier/compute tables are derived from
MODEL_REGISTRY, so newly registered models (gpt-5.6-luna/terra/sol,
gpt-6-astra) are measured correctly instead of falling back to tier 0 /
1.0x. Guards against the report silently mis-measuring the very lanes it
is meant to monitor.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from models import MODEL_REGISTRY  # noqa: E402

# The report is a hyphenated script, so load it from its path.
_SPEC = importlib.util.spec_from_file_location(
    "check_routing_stats", SCRIPTS_DIR / "check-routing-stats.py"
)
crs = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(crs)


def test_tiers_derived_from_model_registry():
    # Every registered model must have a tier in the report's table.
    for name, cfg in MODEL_REGISTRY.items():
        assert crs.TIERS.get(name) == cfg["tier"], (
            f"TIERS missing or wrong for {name}: expected tier {cfg['tier']}"
        )


def test_compute_units_derived_from_model_registry():
    for name, cfg in MODEL_REGISTRY.items():
        assert crs.COMPUTE_UNITS.get(name) == cfg["ratio"], (
            f"COMPUTE_UNITS missing or wrong for {name}: expected {cfg['ratio']}"
        )


def test_new_lanes_are_measured():
    # The four new models must be present with their real tiers/ratios.
    assert crs.TIERS["gpt-5.6-luna"] == 11
    assert crs.TIERS["gpt-5.6-terra"] == 12
    assert crs.TIERS["gpt-5.6-sol"] == 13
    assert crs.TIERS["gpt-6-astra"] == 14
    assert crs.COMPUTE_UNITS["gpt-5.6-sol"] == 36.0
    assert crs.COMPUTE_UNITS["gpt-6-astra"] == 40.0


def test_no_hardcoded_duplication():
    # The report must not carry a separate hardcoded copy of the tables that
    # can drift from MODEL_REGISTRY. It should reference the registry.
    import inspect

    src = inspect.getsource(crs)
    # The dicts are built from MODEL_REGISTRY, not a literal list of models.
    assert "MODEL_REGISTRY" in src


def test_float_tiers_do_not_break_integer_formatting():
    # glm-5.2 has a fractional tier (6.5) in MODEL_REGISTRY. The model
    # distribution row prints tiers with a width-5 format; a float tier must
    # not crash the report (regression: "Unknown format code 'd' for float").
    import inspect

    src = inspect.getsource(crs)
    # TIERS is derived (no hardcoded copy); find the distribution print line.
    dist_line = next(l for l in src.splitlines() if "TIERS.get(model" in l and "print(" in l)
    # It must render floats: use :g-style, never integer-only :d on a tier.
    assert "get(model, 0):<5g" in dist_line or "get(model, 0):<6g" in dist_line, (
        f"model-distribution tier format must accept float tiers, got:\n{dist_line}"
    )
    # And the registry really does carry a float that would have crashed :d.
    assert crs.TIERS["glm-5.2"] == 6.5

