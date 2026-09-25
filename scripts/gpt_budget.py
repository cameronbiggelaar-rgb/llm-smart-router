"""ChatGPT plan-budget governor — B37 (batch a: the authoritative reader).

Why this module exists
----------------------
The ChatGPT plan had no governor, so it boom-busted: Sep 20-21 burned a whole
7-day window in ~32h and then went dark for 5.66 days. Capacity is not
modellable — two cap events on the same plan differ 3.5x — so nothing here is
*estimated*. The account's own `/usage` verdict is read and treated as truth.

Honesty + safety rules (see tests/test_gpt_budget_reading.py):
  * the account's ``allowed=false`` is authoritative and outranks everything;
  * an unreadable probe FAILS OPEN — a broken endpoint must never starve the
    architectural lane (the lane is the point of the whole plan);
  * one probe per ``min_interval_seconds``, never per request — this sits on
    the hot routing path;
  * a remembered value is always labelled ``cache``, never ``live``;
  * past the staleness bound, old data is not evidence → fail open.

The fetch reuses Hermes' authoritative ``agent.account_usage.fetch_account_usage``
rather than re-implementing the HTTP (token refresh, the 401 retry, the
happy-eyeballs client, the account-id header all live there and would drift).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

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
