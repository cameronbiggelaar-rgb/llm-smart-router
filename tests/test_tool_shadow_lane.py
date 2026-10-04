"""Shadow lane scoping — tool-bearing traffic to a candidate model.

Extends traffic-splitting so an experiment can be scoped to "tool-bearing
requests whose incumbent is a specific model", not just a workload/tier. This is
what makes it possible to observe deepseek-v4.1-flash on real tool payloads
without firing on all normal_chat traffic.

Non-negotiables under test:
* match.requires_tools filters to tool-bearing requests only (True = requires tools)
* match.incumbent_model filters to one incumbent (empty = any)
* a non-tool normal_chat request never matches a tool-scoped experiment
* the new dimensions round-trip through build_experiment and experiment_to_row
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from traffic_split import (  # noqa: E402
    Experiment,
    build_experiment,
    experiment_to_row,
    pick_experiment,
)


def make(**kw):
    base = dict(name="n", enabled=True, model="deepseek-v4.1-flash", mode="shadow")
    base.update(kw)
    return Experiment(**base)


# ── construction ──────────────────────────────────────────────────────────────

def test_build_experiment_parses_tool_scope():
    e = build_experiment(
        {"name": "n", "model": "m", "mode": "shadow",
         "match": {"workload": ["normal_chat"], "requires_tools": True,
                   "incumbent_model": "deepseek-v4-pro"}}
    )
    assert e.match_requires_tools is True
    assert e.match_incumbent_model == "deepseek-v4-pro"


def test_build_experiment_defaults_tool_scope_off():
    e = build_experiment({"name": "n", "model": "m", "mode": "shadow",
                          "match": {"workload": ["normal_chat"]}})
    assert e.match_requires_tools is False
    assert e.match_incumbent_model == ""


def test_experiment_to_row_roundtrips_tool_scope():
    e = make(match_workload=("normal_chat",), match_requires_tools=True,
             match_incumbent_model="deepseek-v4-pro")
    row = experiment_to_row(e)
    import json
    mj = json.loads(row["match_json"])
    assert mj["requires_tools"] is True
    assert mj["incumbent_model"] == "deepseek-v4-pro"


# ── selection ─────────────────────────────────────────────────────────────────

def test_pick_matches_tool_bearing_incumbent():
    e = make(match_workload=("normal_chat",), match_requires_tools=True,
             match_incumbent_model="deepseek-v4-pro")
    assert pick_experiment([e], workload="normal_chat", requires_tools=True,
                           incumbent_model="deepseek-v4-pro") is e


def test_pick_rejects_non_tool_request():
    e = make(match_workload=("normal_chat",), match_requires_tools=True,
             match_incumbent_model="deepseek-v4-pro")
    assert pick_experiment([e], workload="normal_chat", requires_tools=False,
                           incumbent_model="deepseek-v4-pro") is None


def test_pick_rejects_wrong_incumbent():
    e = make(match_workload=("normal_chat",), match_requires_tools=True,
             match_incumbent_model="deepseek-v4-pro")
    assert pick_experiment([e], workload="normal_chat", requires_tools=True,
                           incumbent_model="glm-5.3") is None


def test_pick_rejects_non_matching_workload_even_when_tooled():
    """A tool-scoped experiment must not fire on a tooled non-target workload."""
    e = make(match_workload=("normal_chat",), match_requires_tools=True,
             match_incumbent_model="deepseek-v4-pro")
    assert pick_experiment([e], workload="session_compression", requires_tools=True,
                           incumbent_model="deepseek-v4-pro") is None


def test_unscoped_incumbent_matches_any():
    e = make(match_workload=("normal_chat",), match_requires_tools=True)
    assert pick_experiment([e], workload="normal_chat", requires_tools=True,
                           incumbent_model="whatever") is e
