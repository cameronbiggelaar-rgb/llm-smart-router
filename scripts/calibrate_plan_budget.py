#!/usr/bin/env python3
"""Calibration replay for the plan-budget governor (B37k).

Answers the §5.7 question before the policy is armed: replayed against recorded
history, how much of the plan's traffic would the allowlist have refused, and did
it ever refuse the architectural lane?

How classification works, and its honest limit
----------------------------------------------
`router_logs` stores `prompt_length`, not the prompt text, so the prompt-based
classifier the router uses at enforcement time cannot be replayed. This reads the
logged `routing_reason` instead: the `force_model` override, or an explicit
"capability tier N needed" clause at or above the architectural tier.

Replay can therefore only *under*-detect architectural requests, never
over-detect them. Its routine-denial count is an **upper bound**, and any
architectural denial it reports is a prompt to check the classifier rather than
proof of a policy failure. The limitation is printed with every report so the
numbers are never read as exact.

Read-only: this script never writes to the database.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpt_budget as gb  # noqa: E402

DEFAULT_DB = str(Path(__file__).resolve().parent.parent / "data" / "router_logs.db")

LIMITATION = (
    "Prompts are not stored (router_logs keeps prompt_length only), so replay "
    "classifies from the logged routing reason. It can under-detect architectural "
    "requests but never over-detect them: the routine-denial count is an UPPER "
    "bound, and an architectural denial here means check the classifier, not that "
    "the policy is wrong."
)


def _governed_models(conn: sqlite3.Connection) -> List[str]:
    """Registered models on the governed plan account."""
    try:
        import models

        return [m for m, cfg in models.MODEL_REGISTRY.items() if cfg.get("provider") == "openai-codex"]
    except Exception:  # pragma: no cover - registry is always importable here
        return []


def replay(
    conn: sqlite3.Connection,
    *,
    since: str,
    allowlist: Optional[Sequence[str]] = None,
) -> dict:
    """Replay recorded plan-account traffic through the policy.

    ``allowlist`` defaults to the shipped policy's allowlist. Returns per-day
    counts plus totals. Never raises on an empty or column-poor database.
    """
    if allowlist is None:
        allowlist = [gb.CLASS_ARCHITECTURAL]
    conn.row_factory = sqlite3.Row

    cols = {r[1] for r in conn.execute("PRAGMA table_info(router_logs)")}
    if not cols:
        return {"days": {}, "totals": _empty_totals(), "limitation": LIMITATION,
                "note": "router_logs table not present"}

    plan_models = _governed_models(conn)
    if not plan_models:
        return {"days": {}, "totals": _empty_totals(), "limitation": LIMITATION,
                "note": "no plan-account models in the registry"}

    placeholders = ",".join("?" * len(plan_models))
    escalated_col = "escalated" if "escalated" in cols else "0 AS escalated"
    reason_col = "routing_reason" if "routing_reason" in cols else "'' AS routing_reason"
    tok_cols = (
        "COALESCE(input_tokens, 0) AS input_tokens, COALESCE(output_tokens, 0) AS output_tokens"
        if "input_tokens" in cols
        else "0 AS input_tokens, 0 AS output_tokens"
    )

    rows = conn.execute(
        f"""
        SELECT timestamp, final_model, {reason_col}, {escalated_col}, {tok_cols}
        FROM router_logs
        WHERE timestamp >= ?
          AND COALESCE(final_model, '') IN ({placeholders})
        ORDER BY timestamp
        """,
        (since, *plan_models),
    ).fetchall()

    days: Dict[str, dict] = {}
    totals = _empty_totals()

    for r in rows:
        klass = gb.classify_stored_request(
            r["routing_reason"] or "",
            r["final_model"] or "",
            escalated=bool(r["escalated"]),
        )
        day = (r["timestamp"] or "")[:10]
        bucket = days.setdefault(day, _empty_day())
        tokens = int(r["input_tokens"] or 0) + int(r["output_tokens"] or 0)

        if klass in allowlist:
            bucket["architectural_served"] += 1
            bucket["architectural_tokens"] += tokens
            totals["architectural_served"] += 1
            totals["architectural_tokens"] += tokens
        else:
            bucket["routine_denied"] += 1
            bucket["routine_denied_tokens"] += tokens
            totals["routine_denied"] += 1
            totals["routine_denied_tokens"] += tokens
            if klass == gb.CLASS_ARCHITECTURAL:
                # Only reachable with a narrowed allowlist. Tracked separately
                # because starving the architectural lane is a different failure
                # from refusing routine work — it is the one §5.7 forbids.
                totals["architectural_denied"] += 1

    return {
        "days": days,
        "totals": totals,
        "limitation": LIMITATION,
        "architectural_lane_denied": totals["architectural_denied"] > 0,
        "window_days": len(days),
    }


def _empty_day() -> dict:
    return {
        "architectural_served": 0,
        "architectural_tokens": 0,
        "routine_denied": 0,
        "routine_denied_tokens": 0,
    }


def _empty_totals() -> dict:
    # `architectural_denied` is structurally impossible under the shipped
    # allowlist (architectural is IN it); it exists so an operator can replay a
    # narrower list and see the starvation it would cause.
    return {
        "architectural_served": 0,
        "architectural_tokens": 0,
        "routine_denied": 0,
        "routine_denied_tokens": 0,
        "architectural_denied": 0,
    }


def format_report(result: dict, days_back: int) -> str:
    if result.get("note"):
        return f"Plan-budget calibration: {result['note']}"

    totals = result["totals"]
    lines = [
        f"Plan-budget calibration — replay of the last {days_back}d of plan-account traffic",
        "",
        f"  {'day':<12}{'arch served':>13}{'routine denied':>16}{'denied Mtok':>14}",
    ]
    for day in sorted(result["days"]):
        d = result["days"][day]
        lines.append(
            f"  {day:<12}{d['architectural_served']:>13}{d['routine_denied']:>16}"
            f"{d['routine_denied_tokens'] / 1_000_000:>14.1f}"
        )
    lines += [
        "",
        f"  architectural served:   {totals['architectural_served']:>8,}",
        f"  routine denied:         {totals['routine_denied']:>8,}"
        f"  ({totals['routine_denied_tokens'] / 1_000_000:.1f} Mtok removed from the plan)",
    ]
    if result["architectural_lane_denied"]:
        lines += [
            "",
            "  WARNING: the architectural lane was denied during replay. The",
            "  detector's keyword coverage is too narrow for this policy (§5.7).",
        ]
    elif totals["architectural_served"] == 0:
        lines += [
            "",
            "  NOTE: no architectural traffic in this window — the lane is dormant,",
            "  so replay cannot confirm the detector protects real work.",
        ]
    lines += ["", f"  LIMITATION: {result['limitation']}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--since", default=None, help="explicit ISO start (overrides --days)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--allowlist", default=None,
                    help="comma-separated classes to allow (default: shipped policy)")
    args = ap.parse_args()

    since = args.since or (
        datetime.now(timezone.utc) - timedelta(days=args.days)
    ).isoformat()

    allowlist = None
    if args.allowlist:
        allowlist = [c.strip() for c in args.allowlist.split(",") if c.strip()]

    conn = sqlite3.connect(args.db)
    try:
        result = replay(conn, since=since, allowlist=allowlist)
    finally:
        conn.close()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(format_report(result, args.days))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
