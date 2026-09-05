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
