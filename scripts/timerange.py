"""Timestamp safety for router_logs analysis.

Why this exists
---------------
``router_logs.timestamp`` is written as ``datetime.now(timezone.utc).isoformat()``,
i.e. ``YYYY-MM-DDTHH:MM:SS.ffffff+00:00``. SQLite compares TEXT
lexicographically, so a window bound built with a SPACE separator
(``'2026-09-14 18:20'``) sorts BELOW every stored value -- because ``'T'``
(0x54) is greater than ``' '`` (0x20). The comparison silently matches the
ENTIRE table. No error is raised.

That exact mistake produced a false "6-hour window" of 7,205 rows (~24h of
data) and an incorrect "~$840/day compaction escalation" figure in the
2026-09-14/15 session. The measured truth was $0 after the fix.

A second, related trap: ``agent.log`` timestamps are local AEST (+10) with no
offset marker, while the DB is UTC. Comparing them directly is off by ten
hours and looks entirely plausible.

This module makes both mistakes fail loudly.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Tuple

# Matches the ISO form the router writes: date, 'T', time. Fraction/offset optional.
_ISO_T_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?")
_SPACE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}")


def assert_iso_bound(value: str) -> str:
    """Return ``value`` if it is a valid ISO-T bound; raise otherwise.

    Rejects the space-separated form specifically, because that is the one that
    fails silently (it sorts below every stored row rather than erroring).
    """
    if not isinstance(value, str):
        raise ValueError(f"timestamp bound must be a string, got {type(value).__name__}")
    if _SPACE_RE.match(value):
        raise ValueError(
            f"space-separated timestamp bound {value!r} would silently match the "
            "entire table (T > space). Use the ISO-T form, e.g. "
            f"{value.replace(' ', 'T')}"
        )
    if not _ISO_T_RE.match(value):
        raise ValueError(f"not an ISO-T timestamp bound: {value!r}")
    return value


def db_window(hours: int = 6, *, now: datetime | None = None) -> Tuple[str, str]:
    """Return ``(lo, hi)`` UTC bounds that compare correctly in SQLite.

    Both bounds use the ISO-T form so they sort correctly against stored rows.
    ``hi`` is inclusive-of-now plus a small margin so a row written mid-query
    is not missed.
    """
    if not isinstance(hours, int) or hours <= 0:
        raise ValueError("hours must be a positive int")
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    ref = ref.astimezone(timezone.utc)
    lo_dt = ref - timedelta(hours=hours)
    hi_dt = ref + timedelta(seconds=60)
    fmt = "%Y-%m-%dT%H:%M:%S"
    lo = assert_iso_bound(lo_dt.strftime(fmt))
    hi = assert_iso_bound(hi_dt.strftime(fmt))
    return lo, hi


def assert_same_clock(a: str, b: str) -> bool:
    """Raise unless ``a`` and ``b`` denote the same instant in the same clock.

    Use before comparing an ``agent.log`` timestamp (AEST, often offset-naive)
    with a DB timestamp (UTC). Catches both the ten-hour skew and the
    naive-vs-aware mismatch.
    """
    def _parse(value: str) -> datetime:
        if not isinstance(value, str):
            raise ValueError("timestamp must be a string")
        text = value.strip()
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"unparseable timestamp {value!r}") from exc

    da, db_ = _parse(a), _parse(b)
    if (da.tzinfo is None) != (db_.tzinfo is None):
        raise ValueError(
            "clock mismatch: one timestamp is timezone-aware and the other is "
            f"naive ({a!r} vs {b!r}). Logs are local AEST; the DB is UTC."
        )
    if da.tzinfo is not None:
        da, db_ = da.astimezone(timezone.utc), db_.astimezone(timezone.utc)
    if da != db_:
        raise ValueError(
            f"clock mismatch: {a!r} and {b!r} are different instants "
            f"({abs((da - db_).total_seconds()):.0f}s apart). If this is ~36000s, "
            "you are comparing AEST log lines with UTC DB rows."
        )
    return True
