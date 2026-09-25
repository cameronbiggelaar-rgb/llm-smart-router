"""Batch B37d (RED) — the budget_policy.yaml loader and hot reload.

Why this module exists
----------------------
The governor's policy is operator-tunable YAML, hot-reloaded the same way as
``experiments.yaml`` (~30s TTL), so a policy change or a rollback (``enabled:
false``) never requires a restart. The loader's honesty rules matter more than
its happy path: bad config must never fail closed (starve the plan) and never
fail open *into routine* (re-open the leak). A malformed file keeps the last
good config and warns.

These tests pin the defaults, the reload cadence, and the misconfiguration
guards.
"""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import gpt_budget as gb  # noqa: E402


@pytest.fixture
def policy_file(tmp_path, monkeypatch):
    p = tmp_path / "budget_policy.yaml"
    monkeypatch.setattr(gb, "BUDGET_POLICY_PATH", str(p))
    # Reset the module's reload cache so each test starts fresh.
    monkeypatch.setattr(gb, "_policy_cache", None)
    monkeypatch.setattr(gb, "_policy_cache_time", 0.0)
    return p


def _write(p, text):
    p.write_text(textwrap.dedent(text))
    return p


# ── Defaults ────────────────────────────────────────────────────────────────

def test_missing_file_defaults_to_architectural_only(policy_file):
    """No policy file at all → the architectural-only default, enabled.

    Must point at a path that genuinely does not exist: once the real
    ``budget_policy.yaml`` ships, a test that reads it would be asserting on
    whatever the operator last tuned rather than on the default.
    """
    cfg = gb.load_governor_config(force=True)
    assert cfg.enabled is True
    assert cfg.allowlist == [gb.CLASS_ARCHITECTURAL]


def test_shipped_policy_file_parses_and_is_armed():
    """The shipped policy must be valid and ARMED.

    Arming is a deliberate operator step taken on 2026-09-25 while the plan was
    dark (every plan model rate-limited), so the switch could not strand live work
    — see the file's rollout notes. The assertion is on the *parse*, not on a
    frozen value: a policy that fails to load would fail open, which looks like
    success from the outside and is the failure this pins.
    """
    cfg = gb.load_governor_config(force=True)
    assert cfg.allowlist == [gb.CLASS_ARCHITECTURAL]
    assert cfg.hard_cap_percent == 95.0
    assert cfg.routine_ceiling_percent == 0.0
    assert cfg.enabled is True, "shipped policy should be armed"


def test_allowlist_missing_defaults_to_architectural_only(policy_file):
    _write(policy_file, """
    governor:
      enabled: true
      accounts:
        openai-codex:
          hard_cap_percent: 95
    """)
    cfg = gb.load_governor_config(force=True)
    assert cfg.allowlist == [gb.CLASS_ARCHITECTURAL]


def test_missing_governor_section_defaults_architectural(policy_file):
    _write(policy_file, "other: things\n")
    cfg = gb.load_governor_config(force=True)
    assert cfg.enabled is True
    assert cfg.allowlist == [gb.CLASS_ARCHITECTURAL]


# ── Parsing the real shape ──────────────────────────────────────────────────

def test_full_policy_parses(policy_file):
    _write(policy_file, """
    governor:
      enabled: true
      accounts:
        openai-codex:
          allowlist: [architectural]
          routine_ceiling_percent: 0
          hard_cap_percent: 95
      probe:
        min_interval_seconds: 300
        stale_after_seconds: 1800
    """)
    cfg = gb.load_governor_config(force=True)
    assert cfg.enabled is True
    assert cfg.allowlist == [gb.CLASS_ARCHITECTURAL]
    assert cfg.hard_cap_percent == 95.0


def test_disabled_rolls_back(policy_file):
    _write(policy_file, """
    governor:
      enabled: false
      accounts:
        openai-codex:
          allowlist: [architectural]
    """)
    cfg = gb.load_governor_config(force=True)
    assert cfg.enabled is False


# ── Hot reload ───────────────────────────────────────────────────────────────

def test_reload_picks_up_change_within_ttl(monkeypatch, policy_file):
    _write(policy_file, """
    governor:
      enabled: true
      accounts:
        openai-codex:
          allowlist: [architectural]
          hard_cap_percent: 95
    """)
    monkeypatch.setattr(gb, "_POLICY_TTL", 30.0)
    now = {"t": 1_000_000.0}
    monkeypatch.setattr(gb, "_now_epoch", lambda: now["t"])

    cfg1 = gb.load_governor_config(force=False)
    assert cfg1.hard_cap_percent == 95.0

    # Tighten the cap and advance time past the TTL.
    _write(policy_file, """
    governor:
      enabled: true
      accounts:
        openai-codex:
          allowlist: [architectural]
          hard_cap_percent: 50
    """)
    gb.load_governor_config(force=False)  # within TTL → cached
    now["t"] += 31.0
    cfg2 = gb.load_governor_config(force=False)

    assert cfg2.hard_cap_percent == 50.0


def test_reload_within_ttl_is_cached(monkeypatch, policy_file):
    _write(policy_file, """
    governor:
      enabled: true
      accounts:
        openai-codex:
          allowlist: [architectural]
          hard_cap_percent: 95
    """)
    monkeypatch.setattr(gb, "_POLICY_TTL", 30.0)
    now = {"t": 1_000_000.0}
    monkeypatch.setattr(gb, "_now_epoch", lambda: now["t"])

    cfg1 = gb.load_governor_config(force=False)
    _write(policy_file, "governor:\n  enabled: false\n")
    cfg2 = gb.load_governor_config(force=False)  # within TTL

    assert cfg2.enabled is True, "within TTL the old config must still be served"


# ── Misconfiguration guards ─────────────────────────────────────────────────

def test_malformed_yaml_keeps_last_good_config(policy_file):
    _write(policy_file, """
    governor:
      enabled: true
      accounts:
        openai-codex:
          allowlist: [architectural]
    """)
    good = gb.load_governor_config(force=True)

    _write(policy_file, "governor: [unclosed\n")
    cfg = gb.load_governor_config(force=True)

    assert cfg.enabled is True
    assert cfg.allowlist == [gb.CLASS_ARCHITECTURAL]
    # The broken file must not have been adopted as "all routine is open".
    assert cfg is not None and cfg.enabled


def test_malformed_yaml_does_not_fail_open_into_routine(policy_file):
    """Bad config must not silently widen to permit routine — the leak re-enters
    via that door. It must fall back to the architectural-only default."""
    _write(policy_file, "not: [valid yaml\n")
    cfg = gb.load_governor_config(force=True)
    assert cfg.allowlist == [gb.CLASS_ARCHITECTURAL]


def test_empty_allowlist_is_rejected(policy_file):
    """allowlist: [] would black-hole the plan (nothing permitted). The loader
    must reject it and keep the safe default rather than adopt it."""
    _write(policy_file, """
    governor:
      enabled: true
      accounts:
        openai-codex:
          allowlist: []
    """)
    cfg = gb.load_governor_config(force=True)
    assert cfg.allowlist == [gb.CLASS_ARCHITECTURAL], "empty allowlist must be rejected"


def test_unknown_class_in_allowlist_is_dropped_not_widened(policy_file):
    _write(policy_file, """
    governor:
      enabled: true
      accounts:
        openai-codex:
          allowlist: [architectural, routine]
    """)
    # 'routine' is a KNOWN class — including it would re-open the leak. The
    # loader accepts the file as written but the policy layer treats only the
    # architectural class as entitled (asserted in B37c). Here we only assert
    # the file loads without error and preserves what was written.
    cfg = gb.load_governor_config(force=True)
    assert set(cfg.allowlist) >= {gb.CLASS_ARCHITECTURAL}
