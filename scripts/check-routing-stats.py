#!/usr/bin/env python3
"""Biggie LLM Endpoint — Routing Effectiveness Report

Run daily to monitor routing effectiveness. Shows:
  - Sessions since switch
  - Model distribution with routing table tiers
  - Estimated compute savings vs gpt-5.5
  - Task type distribution
  - Correction rate
  - Routing table sub-type match counts

Usage:
  python3 check-routing-stats.py              # full report
  python3 check-routing-stats.py --brief      # one-line summary
  python3 check-routing-stats.py --json       # JSON output for dashboards
"""

import sqlite3
import sys
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

STATE_DB = str(Path.home() / ".hermes" / "state.db")
ROUTER_LOGS = str(Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "router_logs.db")

# Switch timestamp: 2026-07-31 14:31 AEST = 04:31 UTC
SWITCH_TS = datetime(2026, 7, 31, 4, 31, 0, tzinfo=timezone.utc).timestamp()

# Compute units (relative to deepseek-v4-flash = 1.0)
COMPUTE_UNITS = {
    "llama3.1:8b": 0.0,
    "qwen3:14b": 0.0,
    "deepseek-v4-flash": 1.0,
    "minimax-m2.7:cloud": 2.0,
    "glm-5": 2.0,
    "glm-5.1": 2.5,
    "glm-5.2:cloud": 3.0,
    "deepseek-v4-pro": 4.0,
    "deepseek-v3.1:671b": 10.0,
    "gpt-5.5": 30.0,
}

# Tier reference
TIERS = {
    "llama3.1:8b": 1,
    "qwen3:14b": 2,
    "deepseek-v4-flash": 3,
    "minimax-m2.7:cloud": 4,
    "glm-5": 4,
    "glm-5.1": 5,
    "glm-5.2:cloud": 6,
    "deepseek-v4-pro": 7,
    "deepseek-v3.1:671b": 8,
    "gpt-5.5": 10,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt_ts(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")

def fmt_num(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:,}"
    return str(n)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    brief = "--brief" in sys.argv
    as_json = "--json" in sys.argv

    conn = sqlite3.connect(STATE_DB)
    conn.row_factory = sqlite3.Row

    # ── Sessions since switch ──────────────────────────────────────────────
    sessions = conn.execute("""
        SELECT id, model, parent_session_id, started_at, message_count, tool_call_count
        FROM sessions WHERE started_at > ?
        ORDER BY started_at
    """, (SWITCH_TS,)).fetchall()

    sids = [s["id"] for s in sessions]
    total_sessions = len(sessions)
    main_sessions = sum(1 for s in sessions if not s["parent_session_id"])
    sub_sessions = total_sessions - main_sessions

    # ── Model usage ────────────────────────────────────────────────────────
    usage = []
    if sids:
        placeholders = ",".join("?" * len(sids))
        usage = conn.execute(f"""
            SELECT model, billing_provider, SUM(api_call_count) as calls,
                   SUM(input_tokens) as total_in, SUM(output_tokens) as total_out
            FROM session_model_usage WHERE session_id IN ({placeholders})
            GROUP BY model, billing_provider
            ORDER BY calls DESC
        """, sids).fetchall()

    total_calls = sum(r["calls"] for r in usage)
    total_in = sum(r["total_in"] for r in usage)
    total_out = sum(r["total_out"] for r in usage)

    # ── Compute savings ────────────────────────────────────────────────────
    actual_units = sum(r["calls"] * COMPUTE_UNITS.get(r["model"], 1.0) for r in usage)
    gpt55_units = total_calls * COMPUTE_UNITS.get("gpt-5.5", 30.0)
    savings_pct = ((gpt55_units - actual_units) / gpt55_units * 100) if gpt55_units > 0 else 0

    # ── Task type distribution (from router_logs.db) ────────────────────────
    task_types = {}
    corrections = 0
    routing_misses = 0
    try:
        rconn = sqlite3.connect(ROUTER_LOGS)
        rconn.row_factory = sqlite3.Row
        rows = rconn.execute("""
            SELECT task_type, correction_detected, cheaper_model_would_work
            FROM router_logs
        """).fetchall()
        for r in rows:
            tt = r["task_type"] or "unknown"
            task_types[tt] = task_types.get(tt, 0) + 1
            if r["correction_detected"]:
                corrections += 1
            if r["cheaper_model_would_work"]:
                routing_misses += 1
        rconn.close()
    except Exception:
        pass

    # ── Routing table sub-type match count ──────────────────────────────────
    # We can't track this from state.db directly — the endpoint logs it
    # to its own logger. We'll add tracking in a future iteration.

    # ── Report ─────────────────────────────────────────────────────────────
    if as_json:
        report = {
            "period": {
                "since": fmt_ts(SWITCH_TS),
                "now": fmt_ts(datetime.now(tz=timezone.utc).timestamp()),
            },
            "sessions": {
                "total": total_sessions,
                "main": main_sessions,
                "subagent": sub_sessions,
            },
            "usage": {
                "total_calls": total_calls,
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
            },
            "compute": {
                "actual_units": round(actual_units, 2),
                "gpt55_units": round(gpt55_units, 2),
                "savings_pct": round(savings_pct, 1),
            },
            "models": [
                {
                    "model": r["model"],
                    "provider": r["billing_provider"],
                    "calls": r["calls"],
                    "input_tokens": r["total_in"],
                    "output_tokens": r["total_out"],
                    "tier": TIERS.get(r["model"], 0),
                    "compute_units": COMPUTE_UNITS.get(r["model"], 1.0),
                }
                for r in usage
            ],
            "task_types": task_types,
            "corrections": corrections,
            "routing_misses": routing_misses,
        }
        print(json.dumps(report, indent=2))
        return

    if brief:
        print(f"📊 Biggie Router: {total_sessions} sessions, {total_calls} calls, "
              f"{savings_pct:.0f}% compute saved vs gpt-5.5")
        return

    # ── Full report ────────────────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     Biggie LLM Endpoint — Routing Effectiveness Report         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Period:     {fmt_ts(SWITCH_TS)} — now")
    print(f"  Sessions:   {total_sessions} total ({main_sessions} main, {sub_sessions} subagent)")
    print(f"  API calls:  {fmt_num(total_calls)} total ({fmt_num(total_in)} in / {fmt_num(total_out)} out tokens)")
    print()

    # ── Model distribution ────────────────────────────────────────────────
    print("  ── Model Distribution ──")
    print(f"  {'Model':30s} {'Tier':5s} {'Calls':8s} {'%':6s} {'Compute':8s}")
    print(f"  {'─'*30} {'─'*5} {'─'*8} {'─'*6} {'─'*8}")
    for r in usage:
        pct = r["calls"] / total_calls * 100 if total_calls > 0 else 0
        cu = COMPUTE_UNITS.get(r["model"], 1.0)
        print(f"  {r['model']:30s} {TIERS.get(r['model'], 0):<5d} {r['calls']:8d} {pct:5.1f}% {cu:7.1f}x")
    print()

    # ── Compute savings ────────────────────────────────────────────────────
    print(f"  ── Compute Savings ──")
    print(f"  Actual compute units:  {actual_units:>10.1f}")
    print(f"  If all gpt-5.5:       {gpt55_units:>10.1f}")
    print(f"  Savings:              {savings_pct:>9.1f}%")
    print(f"  Equivalent gpt-5.5 calls saved: {int(gpt55_units / 30.0 - total_calls):>6d}")
    print()

    # ── Task type distribution ─────────────────────────────────────────────
    if task_types:
        print(f"  ── Task Type Distribution (from router logs) ──")
        for tt, count in sorted(task_types.items(), key=lambda x: -x[1]):
            pct = count / sum(task_types.values()) * 100
            print(f"  {tt:20s} {count:5d} ({pct:5.1f}%)")
        print()

    # ── Quality signals ────────────────────────────────────────────────────
    print(f"  ── Quality Signals ──")
    print(f"  Corrections detected:  {corrections}")
    if total_calls > 0:
        print(f"  Correction rate:       {corrections / total_calls * 100:.2f}%")
    print(f"  Routing misses:        {routing_misses} (cheaper model would work)")
    print()

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"  ── Summary ──")
    if savings_pct > 50:
        print(f"  🟢 Excellent — {savings_pct:.0f}% compute saved vs gpt-5.5")
    elif savings_pct > 20:
        print(f"  🟡 Good — {savings_pct:.0f}% compute saved vs gpt-5.5")
    else:
        print(f"  🔴 Low savings ({savings_pct:.0f}%) — check routing table")
    print()

    conn.close()


if __name__ == "__main__":
    main()
