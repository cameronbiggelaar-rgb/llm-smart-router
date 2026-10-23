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
  python3 check-routing-stats.py              # full report (read-only vs snapshot)
  python3 check-routing-stats.py --brief      # one-line summary
  python3 check-routing-stats.py --json       # JSON output for dashboards
  python3 check-routing-stats.py --save-snapshot   # persist the cumulative baseline
                                                    # (nightly cron only; omitting it makes
                                                    # ad-hoc runs read-only vs the baseline)
"""

import sqlite3
import sys
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

# ── Config ────────────────────────────────────────────────────────────────────

STATE_DB = str(Path.home() / ".hermes" / "state.db")
ROUTER_LOGS = str(Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "router_logs.db")
SNAPSHOT_FILE = str(Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "routing_stats_snapshot.json")

# Switch timestamp: 2026-07-31 14:31 AEST = 04:31 UTC
SWITCH_DT = datetime(2026, 7, 31, 4, 31, 0, tzinfo=timezone.utc)
SWITCH_TS = SWITCH_DT.timestamp()
SWITCH_ISO = SWITCH_DT.isoformat()

# Tier reference — derived from MODEL_REGISTRY (single source of truth) so
# newly registered models (gpt-5.6-luna/terra/sol, gpt-6-astra) are measured
# correctly instead of silently falling back to tier 0 / 1.0x.
from models import MODEL_REGISTRY  # noqa: E402

# Lane classification is shared with the router (B37h) so the report and the
# routing decision can never disagree about which lane fired.
from gpt_budget import classify_lane as _classify_lane  # noqa: E402

TIERS = {name: cfg["tier"] for name, cfg in MODEL_REGISTRY.items()}

# Compute units (relative to deepseek-v4-flash = 1.0) — derived from
# MODEL_REGISTRY ratios.
COMPUTE_UNITS = {name: cfg["ratio"] for name, cfg in MODEL_REGISTRY.items()}

# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt_ts(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")

def fmt_num(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{int(n):,}"
    return str(n)


def _load_snapshot() -> Optional[dict]:
    """Return the last persisted cumulative snapshot, or None on first run."""
    try:
        with open(SNAPSHOT_FILE) as fh:
            return json.load(fh)
    except (FileNotFoundError, ValueError):
        return None


def _save_snapshot(snap: dict) -> None:
    """Persist the cumulative snapshot (atomic: unique tmp then rename)."""
    Path(SNAPSHOT_FILE).parent.mkdir(parents=True, exist_ok=True)
    # A per-process tmp path so two concurrent runs can never interleave writes
    # into a shared temp file and move a half-written snapshot into place.
    tmp = f"{SNAPSHOT_FILE}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(snap, fh, indent=2)
    os.replace(tmp, SNAPSHOT_FILE)


def _delta(cur: float, prev):
    """Day-over-day delta helper: None when there's no prior baseline."""
    if prev is None:
        return None
    return cur - prev

# ── Main ──────────────────────────────────────────────────────────────────────

def _plan_budget_summary(conn: sqlite3.Connection, period_start_iso: str) -> dict:
    """Summarise plan-governor activity for the digest (B37j).

    Never raises. The nightly report is the operator's only routine view of the
    system, so a missing column, an unreadable probe, or a governor bug must
    degrade to a note — a traceback here would turn an information problem into
    a monitoring outage that reads as "everything is broken".

    ``available: False`` (rather than zeros) when the columns are absent, because
    "no denials" and "no data" are opposite facts and the first must never be
    reported when the second is true.
    """
    summary: dict = {
        "available": False,
        "window_used_percent": None,
        "routine_denied": None,
        "architectural_served": None,
        "ungoverned": None,
        "governed": None,
        "blind": None,
        "armed": None,
        "note": "",
    }
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(router_logs)")}
    except sqlite3.Error as exc:
        summary["note"] = f"router_logs unreadable: {exc}"
        return summary

    if "budget_class" not in cols:
        summary["note"] = "budget_class column not present yet (migration pending)"
        _attach_probe(summary)
        return summary

    try:
        rows = conn.execute(
            """
            SELECT COALESCE(NULLIF(budget_class, ''), 'ungoverned') AS klass,
                   COALESCE(plan_budget_denied, 0) AS denied,
                   COUNT(*) AS calls,
                   COALESCE(SUM(governor_fail_open), 0) AS blind
            FROM router_logs
            WHERE timestamp >= ?
            GROUP BY klass, denied
            """,
            (period_start_iso,),
        ).fetchall()
    except sqlite3.Error as exc:
        summary["note"] = f"plan-budget query failed: {exc}"
        _attach_probe(summary)
        return summary

    summary["available"] = True
    summary["routine_denied"] = 0
    summary["architectural_served"] = 0
    summary["ungoverned"] = 0
    summary["governed"] = 0
    summary["blind"] = 0
    for klass, denied, calls, blind in rows:
        if klass == "ungoverned":
            summary["ungoverned"] += calls
        elif denied:
            if klass == "routine":
                summary["routine_denied"] += calls
        elif klass == "architectural":
            summary["architectural_served"] += calls
        if klass != "ungoverned":
            summary["governed"] += calls
        summary["blind"] += blind
    _attach_probe(summary)
    return summary


def _attach_probe(summary: dict) -> None:
    """Best-effort: current window used % and whether the governor is armed.

    ``armed`` distinguishes "armed but idle" (governor on, class recorded, yet no
    plan model selected so no denial) from "off / pre-governor" (empty class on
    every row) — the exact ambiguity behind the recurring "governor looks inert"
    flag. An unreadable probe leaves the fields None.
    """
    try:
        import gpt_budget
    except Exception:
        summary["window_used_percent"] = None
        return

    try:
        cfg = gpt_budget.load_governor_config()
        summary["armed"] = bool(getattr(cfg, "enabled", False))
    except Exception:
        summary["armed"] = None
    try:
        reading = gpt_budget.fetch_reading()
        w = gpt_budget.binding_window(reading)
        summary["window_used_percent"] = round(w.used_percent, 1) if w else None
    except Exception:
        summary["window_used_percent"] = None


def main():
    brief = "--brief" in sys.argv
    as_json = "--json" in sys.argv
    today = "--today" in sys.argv
    # Snapshot persistence is opt-in so ad-hoc/verification runs can read the
    # day-over-day baseline but never overwrite or race-corrupt it. Only the
    # nightly cron passes --save-snapshot.
    save_snapshot = "--save-snapshot" in sys.argv

    if today:
        period_start_dt = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        period_start_dt = SWITCH_DT
    period_start_ts = period_start_dt.timestamp()
    period_start_iso = period_start_dt.astimezone(timezone.utc).isoformat()

    conn = sqlite3.connect(STATE_DB)
    conn.row_factory = sqlite3.Row

    # ── Sessions with activity since switch ────────────────────────────────
    # Use last_seen from model_usage to catch sessions that started before
    # the switch but continued after (e.g. long-running subagent chains)
    usage = conn.execute("""
        SELECT u.session_id, u.model, u.billing_provider, u.api_call_count,
               u.input_tokens, u.output_tokens, u.first_seen, u.last_seen,
               s.started_at, s.parent_session_id
        FROM session_model_usage u
        JOIN sessions s ON s.id = u.session_id
        WHERE u.last_seen > ?
        ORDER BY u.last_seen
    """, (period_start_ts,)).fetchall()

    # Deduplicate by session_id — one row per session. Only the distinct count
    # and parent_session_id are consumed below, and parent_session_id is
    # session-level (constant across a session's rows), so whether the kept row
    # is the earliest or latest is immaterial.
    seen_sessions = set()
    session_list = []
    for r in usage:
        if r["session_id"] not in seen_sessions:
            seen_sessions.add(r["session_id"])
            session_list.append(r)

    def canonical_model(name):
        """Collapse provider suffix aliases used by Hermes state.db."""
        aliases = {
            "deepseek-v4-flash:cloud": "deepseek-v4-flash",
            "glm-5.2:cloud": "glm-5.2",
            "glm-5.3:cloud": "glm-5.3",
        }
        return aliases.get(name, name)

    total_sessions = len(session_list)
    main_sessions = sum(1 for s in session_list if not s["parent_session_id"])
    sub_sessions = total_sessions - main_sessions

    # Aggregate model usage across all records
    model_agg = {}
    providers_by_model = {}
    for r in usage:
        model = canonical_model(r["model"])
        if model not in model_agg:
            model_agg[model] = {"calls": 0, "total_in": 0, "total_out": 0}
            providers_by_model[model] = set()
        providers_by_model[model].add(r["billing_provider"] or "")
        model_agg[model]["calls"] += r["api_call_count"]
        model_agg[model]["total_in"] += r["input_tokens"]
        model_agg[model]["total_out"] += r["output_tokens"]

    total_calls = sum(v["calls"] for v in model_agg.values())
    total_in = sum(v["total_in"] for v in model_agg.values())
    total_out = sum(v["total_out"] for v in model_agg.values())

    # ── Compute savings ────────────────────────────────────────────────────
    actual_units = sum(data["calls"] * COMPUTE_UNITS.get(model, 1.0) for model, data in model_agg.items())
    gpt55_units = total_calls * COMPUTE_UNITS.get("gpt-5.5", 30.0)
    savings_pct = ((gpt55_units - actual_units) / gpt55_units * 100) if gpt55_units > 0 else 0

    # ── Router endpoint observations (direct calls to Biggie endpoint) ───────
    # Hermes session_model_usage does not tell us whether a call was routed via
    # Biggie or made directly to a backend provider. The endpoint log does: rows
    # with requested_model populated were observed at /v1/chat/completions.
    direct_observed = None
    router_observed = None
    router_completion_rows = 0
    router_start_rows = 0
    router_abandoned_streams = 0
    router_selected = {}
    router_requested = {}
    router_task_types = {}
    router_lanes = {"rethink_rearchitect": 0, "escalation": 0, "force_model": 0, "normal": 0}
    # Initialised here: the router_logs block below is wrapped in try/except, so
    # an unreadable DB leaves this at its "no data" default rather than raising
    # UnboundLocalError at report-assembly time.
    plan_budget = {
        "available": False,
        "window_used_percent": None,
        "routine_denied": None,
        "architectural_served": None,
        "ungoverned": None,
        "note": "router_logs unreadable",
    }
    corrections = 0
    routing_misses = 0
    completion_success = 0
    completion_failures = 0
    failure_by_type: Dict[str, int] = {}
    try:
        rconn = sqlite3.connect(ROUTER_LOGS)
        rconn.row_factory = sqlite3.Row

        # Prefer completion/failure rows for counting routed requests. Streaming
        # start rows are deliberately excluded to avoid double-counting.
        observed_rows = rconn.execute("""
            SELECT *
            FROM router_logs
            WHERE timestamp >= ?
              AND COALESCE(requested_model, '') != ''
              AND COALESCE(error_type, '') != 'streaming_in_progress'
        """, (period_start_iso,)).fetchall()

        start_rows = rconn.execute("""
            SELECT request_id
            FROM router_logs
            WHERE timestamp >= ?
              AND streaming = 1
              AND COALESCE(error_type, '') = 'streaming_in_progress'
              AND COALESCE(request_id, '') != ''
        """, (period_start_iso,)).fetchall()
        router_start_rows = len(start_rows)
        completed_ids = {r["request_id"] for r in observed_rows if r["request_id"]}
        router_abandoned_streams = sum(1 for r in start_rows if r["request_id"] not in completed_ids)

        direct_observed = 0
        router_observed = 0
        for r in observed_rows:
            requested = r["requested_model"] or "unknown"
            selected = canonical_model(r["final_model"] or r["model_used"] or "unknown")
            router_requested[requested] = router_requested.get(requested, 0) + 1
            router_selected[selected] = router_selected.get(selected, 0) + 1
            if requested in ("biggie-router", "biggie-llm"):
                router_observed += 1
            else:
                direct_observed += 1

            tt = r["task_type"] or r["workload_type"] or "unknown"
            router_task_types[tt] = router_task_types.get(tt, 0) + 1
            # Classify which routing lane fired, from the logged routing_reason.
            # Delegated to the shared helper (B37h) so routing and reporting can
            # never disagree about what a lane is — a drifted report would show
            # a leak that is not there, or hide one that is.
            reason = (r["routing_reason"] if "routing_reason" in r.keys() else "") or ""
            model = canonical_model(r["final_model"] or r["model_used"] or "unknown")
            router_lanes[_classify_lane(
                reason=reason,
                tier=TIERS.get(model, 0),
                escalated=bool(r["escalated"]),
            )] += 1
            if r["success"]:
                completion_success += 1
            else:
                completion_failures += 1
                et = r["error_type"] or "unknown"
                failure_by_type[et] = failure_by_type.get(et, 0) + 1
            if (r["user_corrected"] if "user_corrected" in r.keys() else 0):
                corrections += 1
            if (r["cheaper_model_would_work"] if "cheaper_model_would_work" in r.keys() else 0):
                routing_misses += 1

        router_completion_rows = len(observed_rows)
        rconn.close()
    except Exception:
        pass

    # Plan-budget digest (B37j). Deliberately OUTSIDE the block above and in its
    # own guarded read: that block is a broad `except Exception: pass`, so any
    # unrelated failure (a column this query does not use) would otherwise
    # silently erase the plan-budget line — the one line that reports whether the
    # governor is working. Independent read, independent failure mode.
    try:
        _pbconn = sqlite3.connect(ROUTER_LOGS)
        try:
            plan_budget = _plan_budget_summary(_pbconn, period_start_iso)
        finally:
            _pbconn.close()
    except Exception as exc:
        plan_budget = dict(plan_budget)
        plan_budget["note"] = f"router_logs unreadable: {exc}"

    # ── Day-over-day snapshot ─────────────────────────────────────────────
    # Persist cumulative counters each run so the digest can report genuine
    # day-over-day deltas instead of reading "first run" forever (the period
    # backdates to the 2026-07-31 switch, so every run looked like the first).
    prev = _load_snapshot()
    snapshot = {
        "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        "total_calls": total_calls,
        "total_in": total_in,
        "total_out": total_out,
        "actual_units": actual_units,
        "gpt55_units": gpt55_units,
        "savings_pct": savings_pct,
        "router_completion_rows": router_completion_rows,
        "router_observed": router_observed,
        "lanes": router_lanes,
        "completion_success": completion_success,
        "completion_failures": completion_failures,
        "corrections": corrections,
        "routing_misses": routing_misses,
    }
    if save_snapshot:
        _save_snapshot(snapshot)
    dod = {
        "has_baseline": prev is not None,
        "calls": _delta(total_calls, prev.get("total_calls") if prev else None),
        "completion_rows": _delta(router_completion_rows, prev.get("router_completion_rows") if prev else None),
        "completion_success": _delta(completion_success, prev.get("completion_success") if prev else None),
        "completion_failures": _delta(completion_failures, prev.get("completion_failures") if prev else None),
        "corrections": _delta(corrections, prev.get("corrections") if prev else None),
        "routing_misses": _delta(routing_misses, prev.get("routing_misses") if prev else None),
        "lanes": {
            lane: _delta(router_lanes.get(lane, 0), (prev.get("lanes") or {}).get(lane) if prev else None)
            for lane in router_lanes
        },
    }

    # The v41flash-tool-canary is retired (B40): its treatment was promoted into
    # ordinary router policy (`_promote_flash_tool_model`). No canary block is
    # emitted any more — the experiment stays frozen in router_logs history only.

    # Keep the legacy name for the existing report section, but now it is scoped
    # to endpoint-observed completion/failure rows since the switch.
    task_types = router_task_types

    # ── Routing table sub-type match count ──────────────────────────────────
    # We can't track this from state.db directly — the endpoint logs it
    # to its own logger. We'll add tracking in a future iteration.

    # ── Report ─────────────────────────────────────────────────────────────
    if as_json:
        report = {
            "period": {
                "since": fmt_ts(period_start_ts),
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
            "router_endpoint": {
                "observed_requests": router_completion_rows,
                "routed_requests": router_observed,
                "direct_backend_requests": direct_observed,
                "stream_start_rows": router_start_rows,
                "abandoned_streams": router_abandoned_streams,
                "requested_models": router_requested,
                "selected_models": router_selected,
                "lanes": router_lanes,
                "plan_budget": plan_budget,
                "note": (
                    "Direct-vs-routed stats are available only for requests that hit "
                    "the Biggie endpoint. Hermes state.db does not label provider calls "
                    "as direct or routed."
                ),
            },
            "models": [
                {
                    "model": model,
                    "providers": sorted(providers_by_model.get(model, set())),
                    "calls": data["calls"],
                    "input_tokens": data["total_in"],
                    "output_tokens": data["total_out"],
                    "tier": TIERS.get(model, 0),
                    "compute_units": COMPUTE_UNITS.get(model, 1.0),
                }
                for model, data in sorted(model_agg.items(), key=lambda x: -x[1]["calls"])
            ],
            "task_types": task_types,
            "corrections": corrections,
            "routing_misses": routing_misses,
            "completion": {
                "success": completion_success,
                "failures": completion_failures,
                "success_rate_pct": round(
                    completion_success / (completion_success + completion_failures) * 100, 2
                ) if (completion_success + completion_failures) else None,
                "failure_by_type": failure_by_type,
            },
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
    print(f"  Period: {fmt_ts(period_start_ts)} — now")
    print(f"  Sessions:   {total_sessions} total ({main_sessions} main, {sub_sessions} subagent)")
    print(f"  API calls:  {fmt_num(total_calls)} total ({fmt_num(total_in)} in / {fmt_num(total_out)} out tokens)")
    print()

    # ── Model distribution ────────────────────────────────────────────────
    print("  ── Model Distribution ──")
    print(f"  {'Model':30s} {'Tier':5s} {'Calls':8s} {'%':6s} {'Compute':8s}")
    print(f"  {'─'*30} {'─'*5} {'─'*8} {'─'*6} {'─'*8}")
    for model, data in sorted(model_agg.items(), key=lambda x: -x[1]["calls"]):
        pct = data["calls"] / total_calls * 100 if total_calls > 0 else 0
        cu = COMPUTE_UNITS.get(model, 1.0)
        print(f"  {model:30s} {TIERS.get(model, 0):<5g} {data['calls']:8d} {pct:5.1f}% {cu:7.1f}x")
    print()

    # ── Router endpoint observability ──────────────────────────────────────
    print("  ── Router Endpoint Observability ──")
    if direct_observed is None:
        print("  Not available — router_logs.db could not be read.")
    else:
        direct_count = direct_observed or 0
        routed_count = router_observed or 0
        direct_pct = direct_count / router_completion_rows * 100 if router_completion_rows else 0
        routed_pct = routed_count / router_completion_rows * 100 if router_completion_rows else 0
        print(f"  Endpoint-observed requests: {fmt_num(router_completion_rows)}")
        print(f"  Routed via Biggie:          {fmt_num(routed_count)} ({routed_pct:5.1f}%)")
        print(f"  Direct backend requested:   {fmt_num(direct_count)} ({direct_pct:5.1f}%)")
        print(f"  Stream start rows:          {fmt_num(router_start_rows)}")
        print(f"  Abandoned/in-flight streams:{fmt_num(router_abandoned_streams):>8s}")
        if router_selected:
            print("  Selected models observed:")
            for model, count in sorted(router_selected.items(), key=lambda x: -x[1])[:8]:
                pct = count / router_completion_rows * 100 if router_completion_rows else 0
                print(f"    {model:28s} {count:8d} ({pct:5.1f}%)")
        print(" Note: Hermes state.db records aggregate model usage, but does not label")
        print(" calls as direct vs routed. Direct-call stats here only cover calls")
        print(" that passed through the Biggie endpoint and populated requested_model.")
        print()
        print(" Routing Lanes (why a strong model was chosen) ")
        for lane, count in router_lanes.items():
            label = {
                "rethink_rearchitect": "rethink/rearchitect trigger",
                "escalation": "failure escalation",
                "force_model": "force_model override",
                "normal": "normal routing",
            }[lane]
            print(f" {label:28s} {count:6d}")
        print()

        # ── Plan budget (B37j) ─────────────────────────────────────────────
        # Rendered even when unavailable: "no data" and "zero denials" are
        # opposite facts, and silence would read as the reassuring one.
        print(" Plan Budget (ChatGPT plan — architectural lane only) ")
        if not plan_budget.get("available"):
            print(f"  Not available — {plan_budget.get('note') or 'no data'}")
        else:
            used = plan_budget.get("window_used_percent")
            armed = plan_budget.get("armed")
            blind = plan_budget.get("blind", 0) or 0
            print(f"  Governor:                   {'ARMED' if armed else 'off' if armed is False else 'unknown'}")
            print(f"  Window used:                {used if used is not None else 'n/a'}%")
            if blind:
                print(f"  Routine denied:               {plan_budget.get('routine_denied', 0):6d}  (leak NOT provably closed — {blind} blind)")
            else:
                print(f"  Routine denied (leak closed): {plan_budget.get('routine_denied', 0):6d}")
            print(f"  Architectural calls served:   {plan_budget.get('architectural_served', 0):6d}")
            print(f"  Governed (class recorded):    {plan_budget.get('governed', 0):6d}")
            print(f"  Ungoverned (pre-governor):    {plan_budget.get('ungoverned', 0):6d}")
            if blind:
                print(f"  Blind (governor unreadable):  {blind:6d}  — cannot confirm these were denied")
            if armed and plan_budget.get("governed", 0) == 0 and plan_budget.get("ungoverned", 0) == 0:
                print("  (armed, no governed traffic in window — idle, not inert)")
        print()

    # ── Compute savings ────────────────────────────────────────────────────
    print(f"  ── Compute Savings ──")
    print(f"  Actual compute units:  {actual_units:>10.1f}")
    print(f"  If all gpt-5.5:       {gpt55_units:>10.1f}")
    print(f"  Savings:              {savings_pct:>9.1f}%")
    print(f"  Equivalent gpt-5.5 calls saved: {int((gpt55_units - actual_units) / COMPUTE_UNITS.get('gpt-5.5', 30.0)):>6d}")
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
    completed = completion_success + completion_failures
    if completed:
        rate = completion_success / completed * 100
        print(f"  Completion success rate: {rate:.2f}%  ({completion_success} ok / {completion_failures} failed, in-flight streams excluded)")
        if failure_by_type:
            print("  Failures by type:")
            for et, cnt in sorted(failure_by_type.items(), key=lambda x: -x[1]):
                print(f"    {et:34s} {cnt:6d}")
    print(f"  Corrections detected:  {corrections}")
    if router_completion_rows > 0:
        print(f"  Correction rate:       {corrections / router_completion_rows * 100:.2f}%")
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

    # ── Day-over-day delta ─────────────────────────────────────────────────
    print(f"  ── Day-over-day (since last run) ──")
    if not dod["has_baseline"]:
        print("  (first run — no prior baseline; deltas from tomorrow)")
    else:
        def _signed(v):
            return "n/a" if v is None else f"{v:+,d}"
        print(f"  API calls:        {_signed(dod['calls'])}")
        print(f"  Endpoint rows:    {_signed(dod['completion_rows'])}")
        print(f"  Completion ok:    {_signed(dod['completion_success'])}")
        print(f"  Completion fail:  {_signed(dod['completion_failures'])}")
        print(f"  Corrections:      {_signed(dod['corrections'])}")
        print(f"  Routing misses:   {_signed(dod['routing_misses'])}")
        lane_label = {
            "rethink_rearchitect": "rethink/rearchitect",
            "escalation": "failure escalation",
            "force_model": "force_model override",
            "normal": "normal routing",
        }
        for lane, d in dod["lanes"].items():
            print(f"  Lane {lane_label.get(lane, lane):20s} {_signed(d)}")
    if not save_snapshot:
        print("  (manual run — snapshot not persisted; pass --save-snapshot to update the baseline)")
    print()

    conn.close()


if __name__ == "__main__":
    main()
