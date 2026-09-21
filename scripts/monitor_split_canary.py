#!/usr/bin/env python3
"""Split-canary health monitor — flash (treatment) vs pro (control).

Watches the served split canary ``v41flash-tool-canary`` for early signs of
regression, so a bad flash rollout is caught within hours instead of waiting
for the next nightly digest.

Signals (each compared against the control arm in the same window):
  * volume — treatment vs control request counts (should track the 10% split)
  * escalation rate — flash misses bounced to pro (treatment escalated=1)
  * hard-failure rate — error_type present (malformed/empty/degeneration/error)
  * cost per completed request — the whole point of the canary

Exit 0 = report produced (or no canary rows yet, empty stdout for the
cron watchdog). Exit 2 = cannot open DB.

Usage:
  python3 monitor_split_canary.py            # human report
  python3 monitor_split_canary.py --json     # machine-readable
  python3 monitor_split_canary.py --since 6h # trailing window (default 6h)
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

ROUTER_LOGS = str(
    Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "router_logs.db"
)
CANARY = "v41flash-tool-canary"


def _db_path() -> str:
    return os.environ.get("BIGGIE_ROUTER_DB", ROUTER_LOGS)


def _since_iso(since: str):
    from datetime import datetime, timedelta, timezone

    n = float(since[:-1]) if len(since) > 1 else 0.0
    unit = {"h": 1, "d": 24, "w": 168}.get(since[-1], 1)
    return (datetime.now(timezone.utc) - timedelta(hours=n * unit)).isoformat()


def _collect(conn, since: str) -> dict:
    since_iso = _since_iso(since)

    def arm_stats(arm: str) -> dict:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n,
                   COALESCE(SUM(escalated), 0) AS esc,
                   COALESCE(SUM(CASE WHEN error_type != '' THEN 1 ELSE 0 END), 0) AS hard_fail,
                   COALESCE(SUM(cost_usd), 0.0) AS cost,
                   COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0) AS ok
            FROM router_logs
            WHERE experiment = ? AND experiment_arm = ?
              AND is_shadow = 0 AND timestamp >= ?
            """,
            (CANARY, arm, since_iso),
        ).fetchone()
        n = row["n"] or 0
        cost = row["cost"] or 0.0
        ok = row["ok"] or 0
        return {
            "arm": arm,
            "requests": n,
            "escalated": row["esc"] or 0,
            "hard_failures": row["hard_fail"] or 0,
            "cost_usd": cost,
            "completed": ok,
            "escalation_rate": (row["esc"] / n) if n else None,
            "hard_failure_rate": (row["hard_fail"] / n) if n else None,
            "cost_per_completed": (cost / ok) if ok else None,
        }

    return {
        "treatment": arm_stats("treatment"),
        "control": arm_stats("control"),
        "window": since,
    }


def _fmt_pct(v) -> str:
    return f"{v*100:.1f}%" if v is not None else "n/a"


def _fmt_usd(v) -> str:
    return f"${v:.4f}" if v is not None else "n/a"


def _report(stats: dict) -> str:
    t, c = stats["treatment"], stats["control"]
    lines = [
        f"Split canary — {CANARY} (window {stats['window']})",
        "=" * 62,
    ]
    total = t["requests"] + c["requests"]
    lines.append(f"requests (total):       {total}  treatment={t['requests']} control={c['requests']}")
    if total:
        lines.append(f"treatment share:        {t['requests']/total*100:.1f}%  (target 10%)")
    lines.append("")
    lines.append(f"  escalation rate:  flash {_fmt_pct(t['escalation_rate'])}  "
                 f"(control {_fmt_pct(c['escalation_rate'])})")
    lines.append(f"  hard-failure rate:flash {_fmt_pct(t['hard_failure_rate'])}  "
                 f"(control {_fmt_pct(c['hard_failure_rate'])})")
    lines.append(f"  cost/completed:    flash {_fmt_usd(t['cost_per_completed'])}  "
                 f"(control {_fmt_usd(c['cost_per_completed'])})")
    lines.append("")
    lines.append(f"  flash escalations: {t['escalated']} / {t['requests']} reqs")
    lines.append(f"  flash hard fails:  {t['hard_failures']} / {t['requests']} reqs")

    # Guard rails (only flag when there is enough volume to be meaningful).
    if t["requests"] >= 30:
        if t["escalation_rate"] is not None and t["escalation_rate"] > 0.10:
            lines.append("\n⚠ escalation rate >10% on flash — consider rolling back "
                         "(flip enabled:false)")
        if t["hard_failure_rate"] is not None and t["hard_failure_rate"] > 0.05:
            lines.append("\n⚠ hard-failure rate >5% on flash — consider rolling back")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--since", default="6h", help="trailing window, e.g. 6h, 24h, 7d")
    args = ap.parse_args()

    try:
        conn = sqlite3.connect(_db_path())
        conn.row_factory = sqlite3.Row
    except Exception as e:
        print(f"cannot open DB: {e}", file=sys.stderr)
        return 2

    stats = _collect(conn, args.since)

    if stats["treatment"]["requests"] + stats["control"]["requests"] == 0:
        # Watchdog contract: nothing to report -> exit 0 with empty stdout.
        # (A non-zero exit would raise an error alert on every quiet tick.)
        return 0

    if args.json:
        print(json.dumps(stats, indent=2))
        return 0

    print(_report(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
