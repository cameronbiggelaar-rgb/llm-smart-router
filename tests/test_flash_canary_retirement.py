"""B40 regression tests for retiring the served flash canary.

The promoted policy must reproduce the proven cohort in the router itself, so
traffic no longer depends on endpoint experiment metadata.
"""

from pathlib import Path
import sys

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import router  # noqa: E402


def test_flash_is_a_proven_tool_capable_model():
    assert "deepseek-v4.1-flash" in router.TOOL_CAPABLE_MODELS


def test_eligible_former_canary_cohort_promotes_pro_to_flash(monkeypatch):
    monkeypatch.setattr(router, "is_model_available", lambda model: True)

    assert router._promote_flash_tool_model(
        "deepseek-v4-pro",
        workload_type="normal_chat",
        task_type="coding",
        requires_tools=True,
    ) == "deepseek-v4.1-flash"


@pytest.mark.parametrize("task_type", ["debugging", "testing"])
def test_canary_exclusions_stay_on_pro(monkeypatch, task_type):
    monkeypatch.setattr(router, "is_model_available", lambda model: True)

    assert router._promote_flash_tool_model(
        "deepseek-v4-pro",
        workload_type="normal_chat",
        task_type=task_type,
        requires_tools=True,
    ) == "deepseek-v4-pro"


def test_higher_capability_baseline_is_not_demoted(monkeypatch):
    monkeypatch.setattr(router, "is_model_available", lambda model: True)

    assert router._promote_flash_tool_model(
        "gpt-5.6-sol",
        workload_type="normal_chat",
        task_type="planning",
        requires_tools=True,
    ) == "gpt-5.6-sol"


def test_flash_unavailable_leaves_pro_selected(monkeypatch):
    monkeypatch.setattr(
        router,
        "is_model_available",
        lambda model: router._normalize_model_name(model) != "deepseek-v4.1-flash",
    )

    assert router._promote_flash_tool_model(
        "deepseek-v4-pro",
        workload_type="normal_chat",
        task_type="coding",
        requires_tools=True,
    ) == "deepseek-v4-pro"


def test_flash_tool_fallback_starts_with_pro(monkeypatch):
    monkeypatch.setattr(
        router,
        "get_available_models",
        lambda: ["deepseek-v4.1-flash", "deepseek-v4-pro", "gpt-5.5"],
    )

    chain = router._build_tool_fallback_chain("deepseek-v4.1-flash")
    assert chain[0] == "deepseek-v4-pro"


def test_eligible_route_uses_flash_without_experiment_hook(monkeypatch):
    monkeypatch.setattr(router, "probe_all_recovering", lambda: [])
    monkeypatch.setattr(
        router,
        "get_available_models",
        lambda: ["deepseek-v4.1-flash", "deepseek-v4-pro", "gpt-5.5"],
    )
    monkeypatch.setattr(router, "is_model_available", lambda model: True)
    monkeypatch.setattr(
        router,
        "_apply_plan_governor",
        lambda selected, **kwargs: (selected, "", False, False, False),
    )

    decision = router.route_task(
        complexity_score=0.2,
        task_type="coding",
        prompt="update this small function",
        workload_type="normal_chat",
        requires_tools=True,
    )
    assert decision.selected_model == "deepseek-v4.1-flash", decision.reason
    assert decision.fallback_chain[0] == "deepseek-v4-pro"


def test_excluded_route_uses_pro_without_experiment_hook(monkeypatch):
    monkeypatch.setattr(router, "probe_all_recovering", lambda: [])
    monkeypatch.setattr(
        router,
        "get_available_models",
        lambda: ["deepseek-v4.1-flash", "deepseek-v4-pro", "gpt-5.5"],
    )
    monkeypatch.setattr(router, "is_model_available", lambda model: True)
    monkeypatch.setattr(
        router,
        "_apply_plan_governor",
        lambda selected, **kwargs: (selected, "", False, False, False),
    )

    decision = router.route_task(
        complexity_score=0.2,
        task_type="debugging",
        prompt="debug this small function",
        workload_type="normal_chat",
        requires_tools=True,
    )
    assert decision.selected_model == "deepseek-v4-pro", decision.reason


def test_served_canary_is_disabled_in_config():
    config = yaml.safe_load((SCRIPTS / "experiments.yaml").read_text())
    experiment = next(
        item for item in config["experiments"]
        if item["name"] == "v41flash-tool-canary"
    )
    assert experiment["enabled"] is False
