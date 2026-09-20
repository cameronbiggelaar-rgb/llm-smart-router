#!/usr/bin/env python3
"""Tool-choice agreement report — flash (shadow) vs pro (incumbent).

The shadow lane proves flash *can* emit tool calls. This report answers the
question that actually gates cutover: does flash pick the SAME tool (and
argument shape) as pro would, for the same request?

Pairs every shadow observation (is_shadow=1, candidate flash) with its
incumbent pro row (is_shadow=0, final_model=deepseek-v4-pro) by request_id, and
scores:
  * tool agreement — do the two rows name the same tools (set equality)?
  * arg-signature agreement — same tool, same canonical args (hash)?
  * empty-content rate on the shadow side — the historical failure mode.

Usage:
  python3 report_tool_agreement.py            # human report
  python3 report_tool_agreement.py --json     # machine-readable
  python3 report_tool_agreement.py --since 24h   # trailing window (default 24h)

Exit code 0 = report produced. Non-zero = could not produce a report (no pairs,
bad DB, etc.).
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROUTER_LOGS = str(
    Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "router_logs.db"
)


def _db_path() -> str:
    import os

    return os.environ.get("BIGGIE_ROUTER_DB", ROUTER_LOGS)


def _pairs(conn, since_hours: float) -> list[dict]:
    """Return shadow rows paired with their incumbent pro row by request_id."""
    since_iso = None
    if since_hours:
        from datetime import datetime, timedelta, timezone

        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        since_iso = cutoff.isoformat()

    where = "WHERE is_shadow = 1"
    params: list = []
    if since_iso:
        where += " AND timestamp >= ?"
        params.append(since_iso)

    shadow_rows = conn.execute(
        f"""
        SELECT request_id, tool_names, tool_args_sig, finish_reason,
               saw_tool_calls, model_used, cost_usd
        FROM router_logs {where}
        ORDER BY timestamp
        """,
        params,
    ).fetchall()

    pairs = []
    for s in shadow_rows:
        rid = s["request_id"]
        inc = conn.execute(
            """
            SELECT tool_names, tool_args_sig, finish_reason, saw_tool_calls,
                   final_model
            FROM router_logs
            WHERE is_shadow = 0 AND request_id = ? AND error_type != 'streaming_in_progress'
            ORDER BY timestamp DESC LIMIT 1
            """,
            (rid,),
        ).fetchone()
        if inc is None:
            # Incumbent row not yet written (shadow can complete first) or
            # rolled up — skip; it cannot be scored either way.
            continue
        pairs.append({"shadow": s, "incumbent": inc})
    return pairs


def _tool_set(names: str) -> set[str]:
    if not names:
        return set()
    return {t.strip() for t in names.split(",") if t.strip()}


def _score(pairs: list[dict]) -> dict:
    total = len(pairs)
    both_have_tools = 0
    agree = 0
    arg_agree = 0
    shadow_empty_content = 0
    shadow_errors = 0
    disagree_examples: list[dict] = []

    for p in pairs:
        s, i = p["shadow"], p["incumbent"]
        s_names = _tool_set(s["tool_names"])
        i_names = _tool_set(i["tool_names"])

        if s["finish_reason"] not in ("tool_calls", "stop", ""):
            shadow_errors += 1
        if s["finish_reason"] == "" and not s["saw_tool_calls"]:
            shadow_empty_content += 1

        if not s_names or not i_names:
            # At least one side emitted no tool call — not a scorable tool pair.
            continue
        both_have_tools += 1

        if s_names == i_names:
            agree += 1
            if (s["tool_args_sig"] or "") == (i["tool_args_sig"] or ""):
                arg_agree += 1
        elif len(disagree_examples) < 10:
            disagree_examples.append(
                {
                    "request_id": s["request_id"],
                    "shadow_tools": sorted(s_names),
                    "incumbent_tools": sorted(i_names),
                }
            )

    return {
        "total_pairs": total,
        "both_have_tools": both_have_tools,
        "tool_agreement": agree,
        "tool_agreement_rate": (agree / both_have_tools) if both_have_tools else None,
        "arg_agreement": arg_agree,
        "arg_agreement_rate": (arg_agree / both_have_tools) if both_have_tools else None,
        "shadow_empty_content": shadow_empty_content,
        "shadow_empty_content_rate": (shadow_empty_content / total) if total else None,
        "shadow_errors": shadow_errors,
        "disagree_examples": disagree_examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--since", default="24h", help="trailing window, e.g. 24h, 7d, 0 for all")
    args = ap.parse_args()

    since_hours = 0.0
    if args.since and args.since != "0":
        unit = args.since[-1]
        try:
            n = float(args.since[:-1])
        except ValueError:
            print(f"bad --since value: {args.since}", file=sys.stderr)
            return 2
        since_hours = n * {"h": 1, "d": 24, "w": 168}.get(unit, 1)

    try:
        conn = sqlite3.connect(_db_path())
        conn.row_factory = sqlite3.Row
    except Exception as e:
        print(f"cannot open DB: {e}", file=sys.stderr)
        return 2

    pairs = _pairs(conn, since_hours)
    score = _score(pairs)

    if score["total_pairs"] == 0:
        print("no shadow/incumbent pairs in window — shadow may not be running yet", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(score, indent=2))
        return 0

    rate = score["tool_agreement_rate"]
    arg_rate = score["arg_agreement_rate"]
    empty = score["shadow_empty_content_rate"]
    print("Tool-choice agreement — flash (shadow) vs pro (incumbent)")
    print("=" * 60)
    print(f"pairs scored:            {score['total_pairs']}")
    print(f"both emitted tool calls: {score['both_have_tools']}")
    print(f"tool set agreement:      {score['tool_agreement']} "
          f"({rate*100:.1f}% of scorable)" if rate is not None else "tool set agreement: n/a")
    if arg_rate is not None:
        print(f"arg-signature agreement: {score['arg_agreement']} ({arg_rate*100:.1f}%)")
        print("  (strict equality — expect near 0% across different models; it is a")
        print("   degeneration detector, NOT a cutover gate. Tool-SET agreement is.)")
    print(f"shadow empty-content:    {score['shadow_empty_content']} "
          f"({empty*100:.1f}%)" if empty is not None else "shadow empty-content: n/a")
    print(f"shadow error/other:      {score['shadow_errors']}")

    if score["disagree_examples"]:
        print("\nDisagreement examples (shadow vs incumbent tool sets):")
        for ex in score["disagree_examples"]:
            print(f"  {ex['request_id']}: shadow={ex['shadow_tools']} vs pro={ex['incumbent_tools']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
