"""Retire two dead Ollama references: glm-5 and deepseek-v3.1:671b.

Both are in MODEL_REGISTRY but absent from the live `ollama.com/v1/models`
catalog (20 models). Routing to either would fail-and-escalate. Neither has
been exercised in 7 days (both dormant), but they are latent escalation bugs —
the same category as the deepseek-v4-flash / qwen3.5 retirement.

  - `glm-5`        (tier 4)  — catalog has glm-5.1/5.2/5.3, no bare glm-5.
  - `deepseek-v3.1:671b` (tier 9) — rung 5 of the summariser ladder AND in cost
    order; not in the catalog at all (it was also the source of the historical
    codex-429 storm: selected as "most capable" fallback but not a discovered
    backend — see test_batch2_reachable_backend.py).

RED against the current registry; GREEN once both names are removed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import models as m
import router


def test_dead_models_absent_from_registry():
    assert "glm-5" not in m.MODEL_REGISTRY
    assert "deepseek-v3.1:671b" not in m.MODEL_REGISTRY


def test_dead_models_absent_from_cost_order():
    assert "glm-5" not in m.MODEL_COST_ORDER
    assert "deepseek-v3.1:671b" not in m.MODEL_COST_ORDER


def test_dead_models_absent_from_tiers():
    assert "glm-5" not in m.MODEL_CAPABILITY_TIERS
    assert "deepseek-v3.1:671b" not in m.MODEL_CAPABILITY_TIERS


def test_dead_model_absent_from_summariser_ladder():
    ladder = router._session_compression_ladder(0)
    assert "deepseek-v3.1:671b" not in ladder
    assert "glm-5" not in ladder


def test_live_catalog_names_all_registered_ollama_models_are_present():
    """Sanity anchor: the models we DO route to are still registered.

    Guards against over-deletion — removing the two dead names must not touch
    the live workhorses.
    """
    for name in (
        "deepseek-v4.1-flash",
        "glm-5.1",
        "glm-5.2",
        "glm-5.3",
        "glm-5.3-flash",
        "minimax-m2.7:cloud",
        "deepseek-v4-pro",
    ):
        assert name in m.MODEL_REGISTRY, f"{name} must remain registered"
