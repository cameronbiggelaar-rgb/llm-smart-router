#!/usr/bin/env python3
"""Split-canary health monitor — flash health, with or without a live control.

Watches the served canary ``v41flash-tool-canary`` for early signs of
regression, so a bad flash rollout is caught within hours instead of waiting
for the next nightly digest.

Two eras, selected automatically by whether control rows exist in the window:

  * **split** — flash (treatment) and pro (control) both carry traffic, so the
    two arms are compared directly.
  * **flash_only** — the canary has been flipped to percent=100 (B36d), so all
    eligible traffic is flash and there is no live control. Flash is instead
    compared against a **pre-flip pro baseline**: what this same eligible
    cohort cost and failed like while it ran on pro.

Signals:
  * volume — requests in the window
  * escalation rate — flash misses bounced to pro (treatment escalated=1)
  * hard-failure rate — error_type present (malformed/empty/degeneration/error)
  * cost per completed request — the whole point of the canary
  * baseline delta — flash vs the pre-flip pro cohort baseline

Guardrails are both absolute (escalation > 10%, hard-failure > 5%) and
delta-based (more than +5pp escalation / +2pp hard-failure over baseline),
evaluated only when both the window and the baseline carry enough volume to be
meaningful.

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
BASELINE_MODEL = "deepseek-v4-pro"

# The eligible cohort the canary matches on. The baseline must describe THIS
# population on pro, not pro's traffic at large, or the comparison is between
# two different workloads and the delta means nothing.
ELIGIBLE_EXCLUDED_TASK_TYPES = ("debugging", "testing")

# Guardrail thresholds.
ABSOLUTE_ESCALATION_MAX = 0.10
ABSOLUTE_HARD_FAILURE_MAX = 0.05
DELTA_ESCALATION_MAX = 0.05    # +5pp over baseline
DELTA_HARD_FAILURE_MAX = 0.02  # +2pp over baseline
MIN_VOLUME_FOR_FLAGS = 30      # below this, rates are too noisy to judge


def _db_path() -> str:
    return os.environ.get("BIGGIE_ROUTER_DB", ROUTER_LOGS)


def _since_iso(since: str):
    from datetime import datetime, timedelta, timezone

    n = float(since[:-1]) if len(since) > 1 else 0.0
    unit = {"h": 1, "d": 24, "w": 168}.get(since[-1], 1)
    return (datetime.now(timezone.utc) - timedelta(hours=n * unit)).isoformat()


def _arm_stats(conn, arm: str, since_iso: str) -> dict:
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


def _baseline_stats(conn, since_iso: str) -> dict:
    """Pro behaviour for the SAME eligible cohort, before the flip.

    Excludes canary rows entirely (they are flash after B36d) and restricts to
    the cohort predicates the canary matches on, so the baseline describes
    "what this slice cost and failed like while it ran on pro".
    """
    placeholders = ",".join("?" * len(ELIGIBLE_EXCLUDED_TASK_TYPES))
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS n,
               COALESCE(SUM(escalated), 0) AS esc,
               COALESCE(SUM(CASE WHEN error_type != '' THEN 1 ELSE 0 END), 0) AS hard_fail,
               COALESCE(SUM(cost_usd), 0.0) AS cost,
               COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0) AS ok
        FROM router_logs
        WHERE is_shadow = 0
          AND timestamp >= ?
          AND (experiment IS NULL OR experiment = '' OR experiment != ?)
          AND (final_model = ? OR model_used = ?)
          AND requires_tools = 1
          AND workload_type = 'normal_chat'
          AND task_type NOT IN ({placeholders})
        """,
        (since_iso, CANARY, BASELINE_MODEL, BASELINE_MODEL,
         *ELIGIBLE_EXCLUDED_TASK_TYPES),
    ).fetchone()
    n = row["n"] or 0
    cost = row["cost"] or 0.0
    ok = row["ok"] or 0
    return {
        "requests": n,
        "escalated": row["esc"] or 0,
        "hard_failures": row["hard_fail"] or 0,
        "cost_usd": cost,
        "completed": ok,
        "escalation_rate": (row["esc"] / n) if n else None,
        "hard_failure_rate": (row["hard_fail"] / n) if n else None,
        "cost_per_completed": (cost / ok) if ok else None,
    }


def _collect(conn, since: str, baseline_since: str = "7d") -> dict:
    since_iso = _since_iso(since)
    treatment = _arm_stats(conn, "treatment", since_iso)
    control = _arm_stats(conn, "control", since_iso)
    baseline = _baseline_stats(conn, _since_iso(baseline_since))
    return {
        "treatment": treatment,
        "control": control,
        "baseline": baseline,
        "arm_mode": "split" if control["requests"] else "flash_only",
        "window": since,
        "baseline_window": baseline_since,
    }


def _fmt_pct(v) -> str:
    return f"{v*100:.1f}%" if v is not None else "n/a"


def _fmt_usd(v) -> str:
    return f"${v:.4f}" if v is not None else "n/a"


def _guardrail_flags(stats: dict) -> list:
    """Return human-readable guardrail warnings for this window."""
    t = stats["treatment"]
    base = stats.get("baseline") or {}
    flags = []

    if t["requests"] < MIN_VOLUME_FOR_FLAGS:
        return flags

    if t["escalation_rate"] is not None and t["escalation_rate"] > ABSOLUTE_ESCALATION_MAX:
        flags.append(
            f"escalation rate {_fmt_pct(t['escalation_rate'])} >10% on flash — "
            "consider rolling back (percent 100 -> 0)"
        )
    if t["hard_failure_rate"] is not None and t["hard_failure_rate"] > ABSOLUTE_HARD_FAILURE_MAX:
        flags.append(
            f"hard-failure rate {_fmt_pct(t['hard_failure_rate'])} >5% on flash — "
            "consider rolling back"
        )

    # Delta vs the pre-flip pro baseline. Only meaningful when the baseline has
    # real volume; a 3-request baseline has no authority to condemn flash.
    if base.get("requests", 0) >= MIN_VOLUME_FOR_FLAGS:
        b_esc, t_esc = base.get("escalation_rate"), t["escalation_rate"]
        if b_esc is not None and t_esc is not None and (t_esc - b_esc) > DELTA_ESCALATION_MAX:
            flags.append(
                f"escalation {_fmt_pct(t_esc)} vs baseline {_fmt_pct(b_esc)} "
                f"(+{(t_esc-b_esc)*100:.1f}pp) — above baseline tolerance"
            )
        b_hf, t_hf = base.get("hard_failure_rate"), t["hard_failure_rate"]
        if b_hf is not None and t_hf is not None and (t_hf - b_hf) > DELTA_HARD_FAILURE_MAX:
            flags.append(
                f"hard-failure {_fmt_pct(t_hf)} vs baseline {_fmt_pct(b_hf)} "
                f"(+{(t_hf-b_hf)*100:.1f}pp) — above baseline tolerance"
            )
    return flags


def _report(stats: dict) -> str:
    t, c = stats["treatment"], stats["control"]
    base = stats.get("baseline") or {}
    mode = stats.get("arm_mode", "split")

    lines = [
        f"Split canary — {CANARY} (window {stats['window']}, mode {mode})",
        "=" * 62,
    ]

    if mode == "split":
        total = t["requests"] + c["requests"]
        lines.append(
            f"requests (total):       {total}  "
            f"treatment={t['requests']} control={c['requests']}"
        )
        if total:
            lines.append(f"treatment share:        {t['requests']/total*100:.1f}%")
        lines.append("")
        lines.append(f"  escalation rate:  flash {_fmt_pct(t['escalation_rate'])}  "
                     f"(control {_fmt_pct(c['escalation_rate'])})")
        lines.append(f"  hard-failure rate:flash {_fmt_pct(t['hard_failure_rate'])}  "
                     f"(control {_fmt_pct(c['hard_failure_rate'])})")
        lines.append(f"  cost/completed:    flash {_fmt_usd(t['cost_per_completed'])}  "
                     f"(control {_fmt_usd(c['cost_per_completed'])})")
    else:
        lines.append(
            f"flash requests:         {t['requests']}  "
            "(no live control — canary is 100%)"
        )
        lines.append("")
        lines.append(f"  escalation rate:  flash {_fmt_pct(t['escalation_rate'])}")
        lines.append(f"  hard-failure rate:flash {_fmt_pct(t['hard_failure_rate'])}")
        lines.append(f"  cost/completed:    flash {_fmt_usd(t['cost_per_completed'])}")

    lines.append("")
    lines.append(f"  flash escalations: {t['escalated']} / {t['requests']} reqs")
    lines.append(f"  flash hard fails:  {t['hard_failures']} / {t['requests']} reqs")

    if base.get("requests"):
        lines.append("")
        lines.append(f"  baseline (pro, eligible cohort, {stats['baseline_window']}):")
        lines.append(
            f"    n={base['requests']}  escalation {_fmt_pct(base['escalation_rate'])}"
            f"  hard-failure {_fmt_pct(base['hard_failure_rate'])}"
            f"  cost/completed {_fmt_usd(base['cost_per_completed'])}"
        )
        if (t["requests"] >= MIN_VOLUME_FOR_FLAGS
                and t["cost_per_completed"] and base["cost_per_completed"]):
            delta = (1 - t["cost_per_completed"] / base["cost_per_completed"]) * 100
            lines.append(f"    flash is {delta:.1f}% cheaper per completed task")

    for flag in _guardrail_flags(stats):
        lines.append(f"\n⚠ {flag}")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--since", default="6h", help="trailing window, e.g. 6h, 24h, 7d")
    ap.add_argument("--baseline-since", default="7d",
                    help="lookback for the pre-flip pro baseline (default 7d)")
    args = ap.parse_args()

    try:
        conn = sqlite3.connect(_db_path())
        conn.row_factory = sqlite3.Row
    except Exception as e:
        print(f"cannot open DB: {e}", file=sys.stderr)
        return 2

    stats = _collect(conn, args.since, args.baseline_since)

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
