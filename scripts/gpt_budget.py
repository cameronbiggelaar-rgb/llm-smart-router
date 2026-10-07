"""ChatGPT plan-budget governor — B37 (batch a: the authoritative reader).

Why this module exists
----------------------
The ChatGPT plan had no governor, so it boom-busted: Sep 20-21 burned a whole
7-day window in ~32h and then went dark for 5.66 days. Capacity is not
modellable — two cap events on the same plan differ 3.5x — so nothing here is
*estimated*. The account's own `/usage` verdict is read and treated as truth.

Honesty + safety rules (see tests/test_gpt_budget_reading.py):
  * the account's ``allowed=false`` is authoritative and outranks everything;
  * an unreadable probe must never starve the architectural lane (the lane is the
    point of the whole plan) — but under an ALLOWLIST an unverifiable entitlement
    is not consent either, so an unreadable reading denies every class the policy
    does not explicitly name and reports the named ones as ``unverified``. Failing
    open here re-opened the exact leak the allowlist exists to close, silently:
    measured 2026-10-04/05, routine work rode a blind probe onto the plan 1,083
    times with ``plan_budget_denied = 0``;
  * one probe per ``min_interval_seconds``, never per request — this sits on
    the hot routing path;
  * a remembered value is always labelled ``cache``, never ``live``;
  * past the staleness bound, old data is not evidence → the reading is
    ``unavailable``, which is a *distinct* verdict (``unverified``), never ``open``.

The fetch reuses Hermes' authoritative ``agent.account_usage.fetch_account_usage``
rather than re-implementing the HTTP (token refresh, the 401 retry, the
happy-eyeballs client, the account-id header all live there and would drift).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# The billing account the ChatGPT plan lives under. The governor keys on this
# account, not on individual model names, because a 429 on the account caps
# every model that draws on it.
GOVERNED_ACCOUNT = "openai-codex"

# Where the last reading is persisted (survives a restart → no cold probe).
# Tests monkeypatch this to a throwaway path before any call.
GOVERNOR_STATE_DB = str(
    Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "router_state.db"
)

# Path to Hermes' own package, for the authoritative usage fetcher.
_HERMES_AGENT_PATH = str(Path.home() / ".hermes" / "hermes-agent")


@dataclass(frozen=True)
class WindowReading:
    """One usage window from the account's /usage response."""

    used_percent: float
    limit_window_seconds: int
    reset_after_seconds: int
    reset_at: Optional[int] = None


@dataclass(frozen=True)
class Reading:
    """A decoded account-usage verdict.

    ``source`` is one of ``live``, ``cache``, ``unavailable``. Only ``live``
    reflects a fresh network read; ``cache`` is a remembered prior verdict;
    ``unavailable`` means the endpoint could not be read and the governor must
    fail open.
    """

    source: str
    allowed: Optional[bool]  # the account's own rate_limit.allowed
    primary: Optional[WindowReading]
    secondary: Optional[WindowReading]
    scoped: dict[str, WindowReading] = field(default_factory=dict)
    has_credits: bool = False
    banked_resets: int = 0
    fetched_at: float = 0.0

    def _window_is_open(self, w: Optional[WindowReading]) -> bool:
        if w is None:
            return True
        return w.used_percent < 100.0


@dataclass(frozen=True)
class ProbeConfig:
    """Probe cadence for the governor's single source of truth."""

    min_interval_seconds: float = 300.0
    stale_after_seconds: float = 1800.0


# ── Parsing ──────────────────────────────────────────────────────────────────

def _window(raw: Any) -> Optional[WindowReading]:
    if not isinstance(raw, dict):
        return None
    used = raw.get("used_percent")
    if used is None:
        return None
    try:
        used_pct = float(used)
        limit = int(raw.get("limit_window_seconds") or 0)
        reset_after = int(raw.get("reset_after_seconds") or 0)
        reset_at_raw = raw.get("reset_at")
        reset_at = int(reset_at_raw) if reset_at_raw is not None else None
    except (TypeError, ValueError):
        return None
    return WindowReading(
        used_percent=used_pct,
        limit_window_seconds=limit,
        reset_after_seconds=reset_after,
        reset_at=reset_at,
    )


def parse_reading(payload: Any, fetched_at: float) -> Reading:
    """Decode a raw /usage body into a Reading.

    Malformed input never raises: it degrades to ``unavailable`` so the routing
    path keeps running and fails open.
    """
    if not isinstance(payload, dict):
        return Reading(source="unavailable", allowed=None, primary=None, secondary=None)

    rate_limit = payload.get("rate_limit")
    if not isinstance(rate_limit, dict):
        return Reading(source="unavailable", allowed=None, primary=None, secondary=None)

    primary = _window(rate_limit.get("primary_window"))
    secondary = _window(rate_limit.get("secondary_window"))

    scoped: dict[str, WindowReading] = {}
    addl = payload.get("additional_rate_limits")
    if isinstance(addl, list):
        for entry in addl:
            if not isinstance(entry, dict):
                continue
            model = entry.get("model")
            rl = entry.get("rate_limit")
            w = _window((rl or {}).get("primary_window")) or _window((rl or {}).get("secondary_window"))
            if model and w is not None:
                scoped[str(model)] = w

    credits = payload.get("credits") or {}
    has_credits = bool(credits.get("has_credits"))

    reset_credits = payload.get("rate_limit_reset_credits") or {}
    banked = reset_credits.get("available_count")
    try:
        banked_resets = int(banked) if banked is not None else 0
    except (TypeError, ValueError):
        banked_resets = 0

    allowed_raw = rate_limit.get("allowed")
    allowed = bool(allowed_raw) if allowed_raw is not None else None

    source = "live" if (primary is not None or secondary is not None) else "unavailable"

    return Reading(
        source=source,
        allowed=allowed,
        primary=primary,
        secondary=secondary,
        scoped=scoped,
        has_credits=has_credits,
        banked_resets=banked_resets,
        fetched_at=fetched_at,
    )


def binding_window(reading: Reading) -> Optional[WindowReading]:
    """The window that actually caps the plan.

    We take the fullest of the two — a window at 100% is what makes the plan
    dark regardless of what the other says — and treat a scoped allowance on a
    governed model at 100% the same way.
    """
    candidates = [w for w in (reading.primary, reading.secondary) if w is not None]
    candidates += [w for w in reading.scoped.values() if w.used_percent >= 100.0]
    if not candidates:
        return None
    return max(candidates, key=lambda w: w.used_percent)


def is_open(reading: Reading) -> bool:
    """Whether the plan may serve a request.

    The account's own ``allowed=false`` is authoritative and closes the plan.
    An unreadable state (no windows, no verdict) fails OPEN.
    """
    if reading.allowed is False:
        return False
    w = binding_window(reading)
    if w is None:
        return True
    return w.used_percent < 100.0


# ── Fetch (with cache + staleness) ───────────────────────────────────────────

def _read_usage_payload() -> Optional[dict]:
    """Read the account's raw /usage body via Hermes' authoritative fetcher.

    Returns None on any failure (network, auth, malformed) — the caller then
    fails open. Reusing ``agent.account_usage`` means the request shape, the
    401-refresh retry, and the account-id header are not duplicated here.
    """
    try:
        import sys

        if _HERMES_AGENT_PATH not in sys.path:
            sys.path.insert(0, _HERMES_AGENT_PATH)
        from agent.account_usage import fetch_account_usage

        snapshot = fetch_account_usage(GOVERNED_ACCOUNT)
        if snapshot is None:
            return None
        return snapshot.raw if isinstance(snapshot.raw, dict) else None
    except Exception:
        return None


def _now_epoch() -> float:
    return time.time()


def fetch_reading(
    cfg: Optional[ProbeConfig] = None,
    *,
    use_cache: bool = True,
    now: Callable[[], float] = _now_epoch,
) -> Reading:
    """Fetch the current account verdict, honouring the probe cadence.

    One network probe per ``min_interval_seconds``; inside the interval the
    last persisted value is returned labelled ``cache``. Past
    ``stale_after_seconds`` a failed re-probe means the old value is no longer
    evidence → ``unavailable`` (fail open), never a stale ``cache``.
    """
    cfg = cfg or ProbeConfig()
    t = now()

    if use_cache:
        cached = load_persisted_reading()
        if cached is not None:
            age = t - cached.fetched_at
            if age < cfg.min_interval_seconds:
                return replace(cached, source="cache")

    try:
        payload = _read_usage_payload()
    except Exception:
        payload = None

    if payload is None:
        # Re-probe failed. A recent cache is still usable; a stale one is not.
        if use_cache:
            cached = load_persisted_reading()
            if cached is not None and (t - cached.fetched_at) < cfg.stale_after_seconds:
                return replace(cached, source="cache")
        return Reading(source="unavailable", allowed=None, primary=None, secondary=None)

    reading = parse_reading(payload, fetched_at=t)
    if reading.source == "live":
        persist_reading(reading)
    return reading


# ── Persistence (survives a restart) ─────────────────────────────────────────

def _state_conn() -> sqlite3.Connection:
    path = Path(GOVERNOR_STATE_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS governor_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    return conn


def _reading_to_dict(reading: Reading) -> dict:
    def wd(w: Optional[WindowReading]) -> Optional[dict]:
        if w is None:
            return None
        return {
            "used_percent": w.used_percent,
            "limit_window_seconds": w.limit_window_seconds,
            "reset_after_seconds": w.reset_after_seconds,
            "reset_at": w.reset_at,
        }

    return {
        "source": reading.source,
        "allowed": reading.allowed,
        "primary": wd(reading.primary),
        "secondary": wd(reading.secondary),
        "scoped": {k: wd(v) for k, v in reading.scoped.items()},
        "has_credits": reading.has_credits,
        "banked_resets": reading.banked_resets,
        "fetched_at": reading.fetched_at,
    }


def _reading_from_dict(d: dict) -> Reading:
    def wd(raw: Optional[dict]) -> Optional[WindowReading]:
        if not isinstance(raw, dict):
            return None
        return WindowReading(
            used_percent=float(raw.get("used_percent", 0.0)),
            limit_window_seconds=int(raw.get("limit_window_seconds", 0)),
            reset_after_seconds=int(raw.get("reset_after_seconds", 0)),
            reset_at=raw.get("reset_at"),
        )

    scoped: dict[str, WindowReading] = {}
    for k, v in (d.get("scoped") or {}).items():
        parsed = wd(v)
        if parsed is not None:
            scoped[k] = parsed

    return Reading(
        source=str(d.get("source", "cache")),
        allowed=d.get("allowed"),
        primary=wd(d.get("primary")),
        secondary=wd(d.get("secondary")),
        scoped=scoped,
        has_credits=bool(d.get("has_credits")),
        banked_resets=int(d.get("banked_resets", 0)),
        fetched_at=float(d.get("fetched_at", 0.0)),
    )


def persist_reading(reading: Reading) -> None:
    conn = _state_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO governor_state (key, value) VALUES (?, ?)",
            ("last_reading", json.dumps(_reading_to_dict(reading))),
        )
        conn.commit()
    finally:
        conn.close()


def load_persisted_reading() -> Optional[Reading]:
    conn = _state_conn()
    try:
        row = conn.execute(
            "SELECT value FROM governor_state WHERE key = ?", ("last_reading",)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    try:
        return _reading_from_dict(json.loads(row[0]))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# ── Request classification (B37b) ────────────────────────────────────────────

# The two canonical classes the allowlist distinguishes. "architectural" is the
# only class permitted onto the ChatGPT plan; everything else is routine.
CLASS_ARCHITECTURAL = "architectural"
CLASS_ROUTINE = "routine"

# Sub-types at or above this tier are the architectural lane. Only the
# tier-13 planning sub-type reaches it today (routing_table.yaml).
_ARCHITECTURAL_MIN_TIER = 13


# ── Lane classification (B37h) ───────────────────────────────────────────────
#
# One question — "which routing lane fired?" — with two consumers: the router
# (at enforcement time) and check-routing-stats.py (reporting, from the logged
# reason). Two implementations would drift, and a drifted report is worse than
# none: it would show a leak that is not there, or hide one that is.
#
# The names below are the reporting vocabulary already in use, and the
# precedence is load-bearing — changing it silently re-buckets history.
LANE_FORCE_MODEL = "force_model"
LANE_ESCALATION = "escalation"
LANE_RETHINK_REARCHITECT = "rethink_rearchitect"
LANE_NORMAL = "normal"


def classify_lane(reason: str, tier: int = 0, *, escalated: bool = False) -> str:
    """Classify which routing lane a logged request used.

    Precedence, highest first — matching the counts
    ``check-routing-stats.py`` has always reported:

    1. ``force_model`` — an explicit operator override.
    2. ``escalation`` — a retry after a failure (the ``escalated`` column is
       authoritative, so a terse reason still buckets correctly).
    3. ``rethink_rearchitect`` — the tier-13+ architectural trigger, the only
       normal-routing path to gpt-5.6-sol.
    4. ``normal`` — everything else.

    A **denial** is never the architectural lane, even though denial reasons
    quote the allowlist and therefore contain the word "architectural". Denials
    are recognised by their ``plan account denied`` marker and are explicitly
    excluded before the architectural check, so the operator's view of the
    architectural lane cannot be inflated by routine traffic that was refused
    the plan.
    """
    r = reason or ""
    if "forced model" in r:
        return LANE_FORCE_MODEL
    if escalated or "escalation" in r:
        return LANE_ESCALATION
    # A refusal is never architectural work. Checked before the tier test
    # because a denial reason quotes the allowlist wording.
    if "plan account denied" in r:
        return LANE_NORMAL
    if tier >= _ARCHITECTURAL_MIN_TIER:
        return LANE_RETHINK_REARCHITECT
    return LANE_NORMAL


# ── Stored-history classification (B37k) ─────────────────────────────────────
#
# Replay cannot see the prompt: `router_logs` stores `prompt_length`, not text.
# So calibration reads the logged reason instead. The tier the router NEEDED is
# recorded as "capability tier N needed", while the tier it CHOSE appears as
# "selected X (tier N)" — including in the tool branch, where a tier-13 model can
# be chosen for a request that needed tier ≤8. Reading the selected tier as the
# needed one would make replay claim the governor spares traffic it denies, which
# is the exact error that would hide an over-denial.
_TIER_NEEDED_RE = re.compile(r"tier\s+(\d+)\s+needed")


def classify_stored_request(reason: str, model: str = "", *, escalated: bool = False) -> str:
    """Classify a HISTORICAL row for calibration replay.

    Conservative by construction: an architectural verdict requires a positive
    signal (the ``force_model`` override, or an explicit "tier 13 needed"
    clause). Everything else is routine. Because the prompt is unavailable, this
    can only under-detect architectural work — never over-detect it — so the
    replay's denial count is an upper bound.
    """
    r = reason or ""
    if "forced model" in r:
        return CLASS_ARCHITECTURAL
    m = _TIER_NEEDED_RE.search(r)
    if m:
        try:
            if int(m.group(1)) >= _ARCHITECTURAL_MIN_TIER:
                return CLASS_ARCHITECTURAL
        except ValueError:  # pragma: no cover - regex guarantees digits
            pass
    return CLASS_ROUTINE


def classify_request(
    task_type: Optional[str],
    prompt: Optional[str],
    force_model: bool,
) -> str:
    """Classify a request as architectural or routine.

    Architectural iff:
      * ``force_model`` is set (the operator's unconditional escape hatch), OR
      * any task type's sub-type matches the prompt at tier >= 13.

    The scan walks EVERY task type's sub-types, not just ``task_type``. The
    prior detector keyed on ``task_type``, so an architectural prompt tagged
    ``debugging`` (which has no tier-13 sub-type) returned None — and
    ``debugging`` is ~30% of traffic. Reading the signal from the prompt text
    across the whole table is the fix that makes the allowlist safe.

    The lazy ``router`` import avoids a circular import: router.py will call
    back into this module at the enforcement points (B37e/f/g).
    """
    if force_model:
        return CLASS_ARCHITECTURAL

    if not prompt or not prompt.strip():
        return CLASS_ROUTINE

    try:
        from router import load_routing_table

        table = load_routing_table()
    except Exception:
        table = {}

    prompt_lower = prompt.lower()
    for task_config in table.values():
        if not isinstance(task_config, dict):
            continue
        for sub in task_config.get("sub_types", []):
            if not isinstance(sub, dict):
                continue
            if sub.get("tier", 0) < _ARCHITECTURAL_MIN_TIER:
                continue
            keywords = sub.get("keywords", []) or []
            exact_keywords = sub.get("exact_keywords", []) or []
            if any(kw in prompt_lower for kw in keywords):
                return CLASS_ARCHITECTURAL
            if any(kw in prompt_lower for kw in exact_keywords):
                return CLASS_ARCHITECTURAL

    return CLASS_ROUTINE


# ── Policy evaluation (B37c) ─────────────────────────────────────────────────

VERDICT_OPEN = "open"
VERDICT_CLOSED = "closed"
VERDICT_DENIED = "denied"
# The account could not be read, so entitlement is UNKNOWN. Under an allowlist
# this is not the same as "open": a class the policy does not name must not ride
# a blind probe onto the plan. Kept as its own value so observability can tell
# "the governor allowed this" apart from "the governor was blind".
VERDICT_UNVERIFIED = "unverified"


class GovernorConfig:
    """The plan-budget policy, as loaded from budget_policy.yaml.

    ``enabled`` is the master switch (false == today's behaviour, the rollback
    lever). ``allowlist`` is the ONLY entitlement: a class must be named here to
    reach the plan at all. ``hard_cap_percent`` is the reactive backstop for
    every class. ``routine_ceiling_percent`` is retained for explicitness (0 =
    routine gets none) but the allowlist — not this knob — is what governs.

    Legacy pacing keys (``slack``, ``final_stretch_seconds``, ``schedule``) are
    deliberately dropped: those concepts were deleted in design rev 4 and must
    have no effect if a stale hand-edited file still carries them.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        allowlist: Optional[list] = None,
        routine_ceiling_percent: float = 0.0,
        hard_cap_percent: float = 95.0,
        **legacy: Any,
    ) -> None:
        self.enabled = bool(enabled)
        self.allowlist = list(allowlist) if allowlist is not None else [CLASS_ARCHITECTURAL]
        self.routine_ceiling_percent = float(routine_ceiling_percent)
        self.hard_cap_percent = float(hard_cap_percent)
        # legacy keys are intentionally not stored.


def _known_classes(allowlist) -> set:
    return {c for c in allowlist if c in (CLASS_ARCHITECTURAL, CLASS_ROUTINE)}


def evaluate_policy(reading: Reading, cfg: GovernorConfig, *, klass: str) -> str:
    """Turn a reading + class into a verdict: open / closed / denied.

    Order of precedence (most authoritative first):
      1. ``enabled=false`` → open for every class (rollback == today).
      2. Unreadable reading → the allowlist still decides: a named class is
         ``unverified`` (served, but flagged), an unnamed class is ``denied``.
         An unverifiable entitlement is not consent under an allowlist.
      3. ``reading.allowed is False`` → closed (the account's own verdict).
      4. ``used >= hard_cap_percent`` → closed (reactive backstop, all classes).
      5. Allowlist: class in allowlist → open; otherwise → denied.

    ``denied`` (policy) is distinct from ``closed`` (capacity) so observability
    can tell *why* a request was refused.
    """
    if not cfg.enabled:
        return VERDICT_OPEN

    if reading.source == "unavailable":
        # The probe is blind. Never starve the plan for the classes the policy
        # explicitly names, but never let an unnamed class through on an
        # unverifiable entitlement either: that is the leak (routine rode a blind
        # probe onto the plan 1,083 times on 2026-10-04/05, denied=0) and it is
        # invisible, because nothing is logged and the class is still stamped.
        allowed = _known_classes(cfg.allowlist)
        return VERDICT_UNVERIFIED if klass in allowed else VERDICT_DENIED

    if reading.allowed is False:
        return VERDICT_CLOSED

    w = binding_window(reading)
    if w is None:
        # No readable window but no explicit close either → open (fail open).
        return VERDICT_OPEN

    if w.limit_window_seconds == 0:
        # Division guard: never throttle on a malformed window.
        allowed = _known_classes(cfg.allowlist)
        return VERDICT_OPEN if klass in allowed else VERDICT_DENIED

    if w.used_percent >= cfg.hard_cap_percent:
        return VERDICT_CLOSED

    allowed = _known_classes(cfg.allowlist)
    return VERDICT_OPEN if klass in allowed else VERDICT_DENIED


# ── Config loading (B37d) ────────────────────────────────────────────────────

# Path to the operator-tunable policy file. Tests monkeypatch this to a
# throwaway path before any call.
BUDGET_POLICY_PATH = str(
    Path(__file__).resolve().parent / "budget_policy.yaml"
)

# Re-read cadence, matching experiments.yaml's hot-reload idiom (~30s).
_POLICY_TTL = 30.0

_policy_cache: Optional[GovernorConfig] = None
_policy_cache_time: float = 0.0
_policy_load_error_logged: bool = False


def _default_config() -> GovernorConfig:
    return GovernorConfig(
        enabled=True,
        allowlist=[CLASS_ARCHITECTURAL],
        routine_ceiling_percent=0,
        hard_cap_percent=95.0,
    )


def _parse_policy(text: str) -> Optional[GovernorConfig]:
    """Parse the policy file text into a GovernorConfig, or None if invalid.

    Malformed YAML returns None (caller keeps the last good config). An
    ``allowlist: []`` is treated as a misconfiguration and rejected (None) so
    the plan can never be silently black-holed. Unknown class names are dropped,
    not used to widen entitlement.
    """
    if yaml is None:
        return None
    try:
        data = yaml.safe_load(text)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    governor = data.get("governor") or {}
    if not isinstance(governor, dict):
        return _default_config()

    enabled = bool(governor.get("enabled", True))

    accounts = governor.get("accounts") or {}
    account = accounts.get(GOVERNED_ACCOUNT) if isinstance(accounts, dict) else None
    if not isinstance(account, dict):
        account = {}

    allowlist = account.get("allowlist", [CLASS_ARCHITECTURAL])
    if not isinstance(allowlist, list):
        allowlist = [CLASS_ARCHITECTURAL]
    # Drop unknown class names; never let an unknown token widen entitlement.
    allowlist = [c for c in allowlist if c in (CLASS_ARCHITECTURAL, CLASS_ROUTINE)]
    if not allowlist:
        # Empty allowlist would black-hole the plan — reject the whole file.
        return None

    hard_cap_percent = float(account.get("hard_cap_percent", 95.0))
    routine_ceiling_percent = float(account.get("routine_ceiling_percent", 0.0))

    return GovernorConfig(
        enabled=enabled,
        allowlist=allowlist,
        routine_ceiling_percent=routine_ceiling_percent,
        hard_cap_percent=hard_cap_percent,
    )


def load_governor_config(force: bool = False) -> GovernorConfig:
    """Load the current policy, cached with a ~30s TTL for hot reload.

    On any parse failure (malformed YAML, empty allowlist) the last good config
    is retained and a warning is logged once — never fail closed (starve the
    plan) and never fail open into routine (re-open the leak).
    """
    global _policy_cache, _policy_cache_time, _policy_load_error_logged

    now = _now_epoch()
    if _policy_cache is not None and not force and (now - _policy_cache_time) < _POLICY_TTL:
        return _policy_cache

    try:
        text = Path(BUDGET_POLICY_PATH).read_text()
    except OSError:
        # No file yet → default (architectural-only, enabled).
        cfg = _default_config()
        _policy_cache = cfg
        _policy_cache_time = now
        return cfg

    parsed = _parse_policy(text)
    if parsed is None:
        if not _policy_load_error_logged:
            import logging

            logging.getLogger(__name__).warning(
                "budget_policy.yaml is invalid or has an empty allowlist — "
                "keeping the last good config (or the architectural-only default)"
            )
            _policy_load_error_logged = True
        # Keep the last good config; fall back to default if none yet.
        cfg = _policy_cache if _policy_cache is not None else _default_config()
    else:
        cfg = parsed
        _policy_load_error_logged = False

    _policy_cache = cfg
    _policy_cache_time = now
    return cfg
