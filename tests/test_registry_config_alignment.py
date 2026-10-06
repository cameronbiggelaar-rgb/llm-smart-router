"""Registry ↔ config.yaml alignment — the two-list drift invariant.

Why this exists
---------------
`MODEL_REGISTRY` (scripts/models.py) is the router's single source of truth:
`MODEL_COST_ORDER` and `MODEL_CAPABILITY_TIERS` are derived from it, and
`route_task` selects from it. But being *registered* is not being *servable* —
the endpoint can only dispatch a model that `discover_backends()` finds in
Hermes' `~/.hermes/config.yaml` (`fallback_providers` entries, plus the two
hardcoded local Ollama models).

Those two lists are maintained by hand, in two different files, with no test
binding them. The measured consequence (2026-10-05→06): `glm-5.3-flash` was
registered at $0.15/$0.50 and **selected 42 times by the router**, but never
added to `config.yaml` — every selection silently fell back to
`glm-5.3:cloud` at $1.40/$4.40, a ~9x price multiplier on mid-tier traffic in
a router whose purpose is to save money.

The same drift class, both directions:
  - registry→config: a registered model with no backend (glm-5.3-flash,
    minimax-m2.7:cloud, glm-5.1) either silently upgrades on every call or
    fail-escalates if reachable models are exhausted.
  - config→registry: a backend nobody prices or tiers (the retired
    deepseek-v4-flash ghost, removed in B41) still shows up in /v1/models and
    can be force-model'd into an unpriced call.
  - stub entries: a fallback_providers entry with `provider:` but no `model:`
    is silently skipped by discover_backends — the residue B41 left when it
    deleted the model line but not the entry.

This suite binds the three invariants against the REAL config file. The
conftest isolates the suite from production by redirecting HOME, so the real
path is restored explicitly: patch `ep.HERMES_CONFIG_PATH` back to the
operator's real config, clear the discovery cache, and run. On a machine
without that file (CI, foreign checkout) the tests skip.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import models as m  # noqa: E402

# The operator's real config: repo is ~/.hermes/skills/llm-smart-router, so
# real HOME is five parents up from this file. Never derived from $HOME (the
# conftest redirects it for isolation).
_REAL_CONFIG = Path(__file__).resolve().parents[4] / ".hermes" / "config.yaml"

# Local models discover_backends() hardcodes (never in fallback_providers).
_HARDCODED_LOCALS = {"llama3.1:8b", "dolphin3"}

# Suffixes the endpoint bridges between registry names and backend keys
# (biggie_llm_endpoint.py ~4388).
_BACKEND_SUFFIXES = (":cloud", ":local", ":ollama")


def _load_config() -> dict:
    if not _REAL_CONFIG.exists():
        pytest.skip(f"real config not found at {_REAL_CONFIG}")
    with open(_REAL_CONFIG) as f:
        return yaml.safe_load(f) or {}


@pytest.fixture()
def endpoint(monkeypatch):
    """biggie_llm_endpoint with discovery pointed at the REAL config.

    conftest redirects HOME to a temp dir (production isolation), so
    load_hermes_config() would read nothing. Point the module's path constant
    back at the real file and drop both caches so discovery re-reads it.
    """
    if not _REAL_CONFIG.exists():
        pytest.skip(f"real config not found at {_REAL_CONFIG}")
    import biggie_llm_endpoint as ep

    monkeypatch.setattr(ep, "HERMES_CONFIG_PATH", _REAL_CONFIG)
    monkeypatch.setattr(ep, "_backends_cache", None)
    monkeypatch.setattr(ep, "_backends_cache_time", 0.0)
    # discover_backends() runs _check_codex_token_age() first; against the
    # real auth.json that emits a token-age warning on every full-suite run.
    # The credential-health signal is not what these tests pin.
    monkeypatch.setattr(ep, "_check_codex_token_age", lambda: None)
    yield ep
    importlib.reload(ep)


def _registry_resolves_to_backend(backend_keys: set) -> dict:
    """Map each registry model to the backend key it would dispatch to.

    Mirrors the endpoint's own resolution (biggie_llm_endpoint.py ~4382):
    exact key first, then the :cloud/:local/:ollama suffix bridge.
    """
    resolved = {}
    for model in m.MODEL_REGISTRY:
        if model in backend_keys:
            resolved[model] = model
            continue
        for suffix in _BACKEND_SUFFIXES:
            if f"{model}{suffix}" in backend_keys:
                resolved[model] = f"{model}{suffix}"
                break
    return resolved


class TestRegistryModelsAreServable:
    """Every registered model must resolve to a discovered backend."""

    def test_every_registry_model_resolves_to_a_backend(self, endpoint):
        backends = endpoint.discover_backends()
        resolved = _registry_resolves_to_backend(set(backends))
        unresolvable = sorted(set(m.MODEL_REGISTRY) - set(resolved))
        assert not unresolvable, (
            f"Registered but un-servable models: {unresolvable}. "
            "The router selects from MODEL_REGISTRY but discover_backends() "
            "cannot serve them — every selection silently falls back to a "
            "pricier model (the glm-5.3-flash leak) or escalates on failure. "
            "Add the model to config.yaml fallback_providers, or remove it "
            "from MODEL_REGISTRY."
        )

    def test_drift_leak_model_is_directly_servable(self, endpoint):
        """The exact 2026-10 leak: glm-5.3-flash must be a backend, not a
        silent-upgrade source."""
        backends = endpoint.discover_backends()
        assert "glm-5.3-flash" in backends or "glm-5.3-flash:cloud" in backends, (
            "glm-5.3-flash is registered at $0.15/$0.50 and selected by the "
            "router, but has no backend — it leaked ~9x cost on every call."
        )


class TestConfigBackendsAreRegistered:
    """Every non-local backend must be priced and tiered in MODEL_REGISTRY."""

    def test_every_cloud_backend_is_registered(self, endpoint):
        backends = endpoint.discover_backends()
        unregistered = []
        for name, info in backends.items():
            if info.get("provider") == "local":
                continue
            if name in m.MODEL_REGISTRY:
                continue
            stripped = name
            for suffix in _BACKEND_SUFFIXES:
                if stripped.endswith(suffix):
                    stripped = stripped[: -len(suffix)]
            if stripped not in m.MODEL_REGISTRY:
                unregistered.append(name)
        assert not unregistered, (
            f"Discovered backends absent from MODEL_REGISTRY: {unregistered}. "
            "They appear in /v1/models and accept force_model requests but "
            "have no price or tier — unpriced calls and the B41 ghost class."
        )


class TestFallbackProvidersHygiene:
    """fallback_providers entries must be well-formed."""

    def test_no_stub_entries_missing_model_key(self):
        config = _load_config()
        stubs = [
            i
            for i, entry in enumerate(config.get("fallback_providers", []))
            if isinstance(entry, dict) and not entry.get("model")
        ]
        assert not stubs, (
            f"fallback_providers entries at indices {stubs} have no 'model:' "
            "key — discover_backends() silently skips them (B41 removed the "
            "model line but left the entry)."
        )


class TestBiggieLlmModelListSync:
    """providers.biggie-llm.models must mirror the servable set."""

    def test_biggie_llm_models_match_discovered_set(self):
        config = _load_config()
        listed = set((config.get("providers", {}).get("biggie-llm", {}) or {}).get("models", {}) or {})
        expected = _HARDCODED_LOCALS | {
            entry["model"]
            for entry in config.get("fallback_providers", [])
            if isinstance(entry, dict) and entry.get("model")
        }
        missing = sorted(expected - listed)
        extra = sorted(listed - expected)
        assert not missing, (
            f"providers.biggie-llm.models is missing {missing}. Hermes CLI "
            "clients can only pick models advertised here — the two lists "
            "drift apart again (B41 kept them in sync by hand)."
        )
        assert not extra, (
            f"providers.biggie-llm.models lists {extra}, which no "
            "fallback_providers entry serves — stale entries serve as the "
            "ghost-backend class the B41 cleanup removed."
        )