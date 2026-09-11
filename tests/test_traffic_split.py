"""B4 — directing production traffic to a test model (shadow + percentage split).

The requested feature: config names a model, a match, and a flag/percentage, and
some traffic is sent to the candidate before committing to it.

Non-negotiables under test:
* deterministic, sticky bucketing (a session must not flap between arms)
* shadow mode never changes the production answer
* the split percentage is honoured within tolerance
* fail-closed: a bad config routes normally rather than half-applying
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
    load_experiments,
    pick_experiment,
    select_arm,
    split_ratio_observed,
)

YAML = """
experiments:
  - name: glm53flash-compression-shadow
    enabled: true
    model: glm-5.3-flash
    mode: shadow
    match:
      workload: [session_compression]
    notes: observe only

  - name: v41-to-glm53-split
    enabled: true
    model: glm-5.3-flash
    mode: split
    percent: 10
    match:
      workload: [session_compression]

  - name: disabled-experiment
    enabled: false
    model: glm-5.2
    mode: split
    percent: 50
    match:
      workload: [session_compression]
"""


@pytest.fixture()
def cfg(tmp_path):
    p = tmp_path / "experiments.yaml"
    p.write_text(YAML)
    return str(p)


# ── loading ───────────────────────────────────────────────────────────────────

def test_load_experiments_parses_all_fields(cfg):
    exps = load_experiments(cfg)
    assert len(exps) == 3
    e = exps[0]
    assert e.name == "glm53flash-compression-shadow"
    assert e.model == "glm-5.3-flash"
    assert e.mode == "shadow"
    assert e.enabled is True
    assert "session_compression" in e.match_workload


def test_load_experiments_missing_file_returns_empty(tmp_path):
    assert load_experiments(str(tmp_path / "nope.yaml")) == []


def test_load_experiments_malformed_returns_empty(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("experiments: [ this is : not : valid")
    assert load_experiments(str(p)) == []


def test_load_experiments_rejects_invalid_mode(tmp_path):
    """An unknown mode must not silently become a split."""
    p = tmp_path / "x.yaml"
    p.write_text("experiments:\n  - name: bad\n    model: m\n    mode: nonsense\n")
    exps = load_experiments(str(p))
    assert exps == []


def test_load_experiments_clamps_percent_to_0_100(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text(
        "experiments:\n  - name: over\n    model: m\n    mode: split\n    percent: 250\n"
    )
    assert load_experiments(str(p))[0].percent == 100.0


# ── selection ─────────────────────────────────────────────────────────────────

def test_pick_experiment_matches_workload(cfg):
    exps = load_experiments(cfg)
    picked = pick_experiment(exps, workload="session_compression")
    assert picked is not None
    # the shadow experiment is listed first and matches
    assert picked.name == "glm53flash-compression-shadow"


def test_pick_experiment_ignores_non_matching_workload(cfg):
    exps = load_experiments(cfg)
    assert pick_experiment(exps, workload="normal_chat") is None


def test_pick_experiment_skips_disabled(cfg):
    exps = [e for e in load_experiments(cfg) if e.name == "disabled-experiment"]
    assert pick_experiment(exps, workload="session_compression") is None


def test_pick_experiment_respects_min_tier(cfg):
    exps = load_experiments(cfg)
    e = Experiment(
        name="tiered", enabled=True, model="m", mode="split", percent=10,
        match_workload=("session_compression",), match_min_tier=5,
    )
    assert pick_experiment([e], workload="session_compression", tier=6) is not None
    assert pick_experiment([e], workload="session_compression", tier=3) is None


# ── bucketing ─────────────────────────────────────────────────────────────────

def test_select_arm_shadow_always_control():
    """Shadow is observe-only: production always serves control."""
    e = Experiment(name="s", enabled=True, model="m", mode="shadow", percent=50,
                   match_workload=("session_compression",))
    for i in range(50):
        assert select_arm(e, request_id=f"r{i}") == "control"


def test_select_arm_is_deterministic_and_sticky():
    """Same session always lands in the same arm — no mid-conversation flapping."""
    e = Experiment(name="s", enabled=True, model="m", mode="split", percent=50,
                   match_workload=("session_compression",))
    first = select_arm(e, request_id="r1", session_id="sess-abc")
    for i in range(20):
        assert select_arm(e, request_id=f"r{i}", session_id="sess-abc") == first


def test_select_arm_percent_zero_is_all_control():
    e = Experiment(name="s", enabled=True, model="m", mode="split", percent=0,
                   match_workload=("session_compression",))
    arms = {select_arm(e, session_id=f"s{i}") for i in range(100)}
    assert arms == {"control"}


def test_select_arm_percent_100_is_all_treatment():
    e = Experiment(name="s", enabled=True, model="m", mode="split", percent=100,
                   match_workload=("session_compression",))
    arms = {select_arm(e, session_id=f"s{i}") for i in range(100)}
    assert arms == {"treatment"}


def test_select_arm_hits_requested_percentage_within_tolerance():
    """10% split over 5000 sessions must land near 10% (+/- 2 points)."""
    e = Experiment(name="s", enabled=True, model="m", mode="split", percent=10,
                   match_workload=("session_compression",))
    observed = split_ratio_observed(e, n=5000)
    assert 8.0 <= observed <= 12.0, f"observed {observed}%"


def test_select_arm_50_percent_splits_evenly():
    e = Experiment(name="s", enabled=True, model="m", mode="split", percent=50,
                   match_workload=("session_compression",))
    observed = split_ratio_observed(e, n=5000)
    assert 48.0 <= observed <= 52.0, f"observed {observed}%"


def test_bucketing_uses_session_when_available():
    """Request ids differ but the session decides the arm."""
    e = Experiment(name="s", enabled=True, model="m", mode="split", percent=100,
                   match_workload=("session_compression",))
    assert select_arm(e, request_id="a", session_id="sess-x") == "treatment"


# ── construction ──────────────────────────────────────────────────────────────

def test_build_experiment_from_dict():
    e = build_experiment(
        {"name": "n", "model": "m", "mode": "split", "percent": 25,
         "match": {"workload": ["session_compression"], "min_tier": 4}}
    )
    assert e.name == "n"
    assert e.percent == 25.0
    assert e.match_workload == ("session_compression",)
    assert e.match_min_tier == 4


def test_build_experiment_requires_name_and_model():
    assert build_experiment({"mode": "split"}) is None
    assert build_experiment({"name": "only-name"}) is None
