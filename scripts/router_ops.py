#!/usr/bin/env python3
"""Router maintenance CLI — roll up, report, propose, purge.

Run daily by the systemd timer, or by hand:

    router_ops.py rollup            # roll up recent days into findings
    router_ops.py report            # unit economics + routing audit
    router_ops.py propose           # propose cheaper routing (never applies)
    router_ops.py purge --dry-run   # show what retention would delete
    router_ops.py purge --yes       # delete raw rows beyond retention
    router_ops.py audit             # logging/auditing health check
    router_ops.py maintain          # rollup + purge + report (the daily job)

Retention safety: ``purge`` REFUSES to delete any day that has no committed
rollup, because dropping raw rows before they are summarised is irreversible
data loss.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimiser import format_proposal, propose, write_candidate  # noqa: E402
from rollup import (  # noqa: E402
    DEFAULT_KEEP_DAYS,
    findings,
    migrate,
    purge_raw,
    rollup_day,
    vacuum_if_needed,
)
from allowance import (
    Snapshot,
    SnapshotStore,
    burn_report,
    format_report,
)
from unit_economics import backfill_costs, seed_prices, unit_cost  # noqa: E402

DEFAULT_DB = Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "router_logs.db"
CANDIDATE_PATH = Path(__file__).resolve().parent / "candidate_routing.yaml"


def _conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    migrate(conn)
    return conn


def cmd_rollup(conn, args) -> int:
    """Roll up the last N days, including today (late rows are picked up by
    re-running: rollup_day is delete-then-insert for that day)."""
    print(f"Rolling up {args.days} day(s) ending {date.today().isoformat()}:")
    for i in range(args.days - 1, -1, -1):
        day = (date.today() - timedelta(days=i)).isoformat()
        n = rollup_day(conn, day)
        print(f"  {day} rolled up ({n} model/workload rows)")
    return 0


def cmd_report(conn, args) -> int:
    since = (date.today() - timedelta(days=args.days)).isoformat()
    rows = unit_cost(conn, since=since)
    if args.json:
        print(json.dumps([r.__dict__ for r in rows], indent=2, default=str))
        return 0

    print(f"Unit cost — since {since}")
    print(f"{'model':26s} {'call_type':20s} {'calls':>8s} {'$ total':>10s} {'$/call':>10s} "
          f"{'quality':>8s}")
    print("-" * 88)
    for r in rows:
        q = f"{r.quality_avg:.3f}" if r.quality_n else "  n/a"
        print(
            f"{r.model[:26]:26s} {r.call_type[:20]:20s} {r.calls:8d} "
            f"{r.cost_usd:10.2f} {r.cost_per_call:10.5f} {q:>8s}"
        )
    total = sum(r.cost_usd for r in rows)
    calls = sum(r.calls for r in rows)
    print("-" * 88)
    print(f"{'TOTAL':26s} {'':20s} {calls:8d} {total:10.2f}")
    return 0


def cmd_findings(conn, args) -> int:
    rows = findings(conn, days=args.days)
    if args.json:
        print(json.dumps([r.__dict__ for r in rows], indent=2, default=str))
        return 0
    print(f"Findings — last {args.days} day(s)  ({len(rows)} rows)")
    for r in rows[:120]:
        print(
            f"  {r.day}  {r.workload_type[:22]:22s} {r.model[:24]:24s} "
            f"calls={r.calls:6d} ${r.cost_usd:8.3f} "
            f"p95={r.latency_p95 or 0:7.1f}s esc={r.escalated_calls:4d} "
            f"q={r.quality_avg if r.quality_avg is not None else 'n/a'}"
        )
    if len(rows) > 120:
        print(f"  ... {len(rows) - 120} more")
    return 0


def cmd_propose(conn, args) -> int:
    p = propose(conn, days=args.days)
    print(format_proposal(p))
    write_candidate(p, str(CANDIDATE_PATH))
    print(f"\nWrote {CANDIDATE_PATH} (applied: false — a human promotes it)")
    return 0


def cmd_purge(conn, args) -> int:
    res = purge_raw(
        conn,
        keep_days=args.retention,
        dry_run=not args.yes,
        batch=args.batch,
    )
    verb = "Would delete" if not args.yes else "Deleted"
    print(
        f"{verb} {res.deleted} of {res.candidates} candidate raw row(s) older than "
        f"{args.retention} day(s) (cutoff {res.cutoff})"
    )
    if res.refused:
        print(f"REFUSED: {res.reason}")
    if args.yes:
        vac = vacuum_if_needed(conn)
        print(f"VACUUM: {'ran' if vac else 'not needed'}")
    return 0


def cmd_audit(conn, args) -> int:
    """Logging/audit health: is the log capturing what the optimiser needs?"""
    since = (date.today() - timedelta(days=args.days)).isoformat()
    total = conn.execute(
        "SELECT COUNT(*) FROM router_logs WHERE timestamp >= ? AND " + BILLABLE_ROW_SQL, (since,)
    ).fetchone()[0]
    unknown = conn.execute(
        "SELECT COUNT(*) FROM router_logs WHERE timestamp >= ? AND cost_unknown = 1 "
        "AND " + BILLABLE_ROW_SQL,
        (since,),
    ).fetchone()[0]
    priced = conn.execute(
        "SELECT COUNT(*) FROM router_logs WHERE timestamp >= ? AND cost_unknown = 0 "
        "AND " + BILLABLE_ROW_SQL,
        (since,),
    ).fetchone()[0]
    q = conn.execute(
        "SELECT COUNT(*) FROM router_logs WHERE timestamp >= ? AND quality_score IS NOT NULL "
        "AND " + BILLABLE_ROW_SQL,
        (since,),
    ).fetchone()[0]
    exp = conn.execute(
        "SELECT COUNT(*) FROM router_logs WHERE timestamp >= ? AND experiment != '' "
        "AND " + BILLABLE_ROW_SQL,
        (since,),
    ).fetchone()[0]
    no_cost = conn.execute(
        "SELECT COUNT(*) FROM router_logs WHERE timestamp >= ? AND cost_unknown = 0 "
        "AND cost_usd = 0 AND " + BILLABLE_ROW_SQL,
        (since,),
    ).fetchone()[0]
    print(f"Logging audit — since {since}")
    print(f"  rows                 {total}")
    print(f"  priced (known cost)  {priced}")
    print(f"  unpriced (flagged)   {unknown}")
    print(f"  quality measured     {q}")
    print(f"  experiment-tagged    {exp}")
    print(f"  priced-but-$0 rows   {no_cost}   (free local models; expected to be >0)")
    if total and unknown / total > 0.05:
        print(
            f"  WARNING: {100.0 * unknown / total:.1f}% of calls have no price on record. "
            "Add them to MODEL_REGISTRY or the spend figure understates cost."
        )
    return 0


def cmd_backfill(conn, args) -> int:
    """Reconstruct cost for rows logged before cost capture existed."""
    seed_prices(conn)
    res = backfill_costs(conn, dry_run=not args.yes)
    if not args.yes:
        print(
            f"DRY RUN: would reconstruct {res['would_update']} row(s) "
            f"({res['skipped_unpriced']} model(s) have no price on record and stay unknown)"
        )
        print("Re-run with --yes to apply.")
        return 0
    print(
        f"Reconstructed {res['updated']} of {res['rows_total']} row(s) as "
        f"{res['pricing_version']}"
    )
    print(
        f"  {res['skipped_unpriced']} model(s) unpriced -> left as cost_unknown=1 "
        f"(never invent a price)"
    )
    print(
        f"  {res['rows_missing_output_tokens']} row(s) have no output tokens "
        f"(streaming); their output cost is understated"
    )
    print(f"  unmatched after: {res.get('unpriced_after', 0)}")
    print()
    print("NOTE: these are reconstructed figures. Filter pricing_version LIKE 'backfill%'")
    print("      to separate them from measured costs in any report.")
    return 0


def cmd_maintain(conn, args) -> int:
    """The daily job: roll up, prune, report, propose. Never applies changes."""
    rc = cmd_rollup(conn, argparse.Namespace(days=args.days))
    if rc:
        return rc
    print()
    rc = cmd_purge(
        conn,
        argparse.Namespace(retention=args.retention, yes=True, batch=args.batch),
    )
    if rc:
        return rc
    print()
    rc = cmd_report(conn, argparse.Namespace(days=args.days, json=False))
    if rc:
        return rc
    print()
    return cmd_propose(conn, argparse.Namespace(days=args.days))


def _ledger_burn_inputs(conn: sqlite3.Connection, days: int):
    """Average input size and billed token burn for the window.

    A streaming request writes two rows (start marker + completion), so
    AVG/SUM must run over billable rows only or every figure is inflated.
    """
    since = (date.today() - timedelta(days=days)).isoformat()
    row = conn.execute(
        """
        SELECT AVG(input_tokens) AS avg_in,
               SUM(input_tokens) AS total_in,
               COUNT(DISTINCT request_id) AS reqs
        FROM router_logs
        WHERE substr(timestamp, 1, 10) >= substr(?, 1, 10)
          AND input_tokens > 0
          AND COALESCE(cost_unknown, 0) = 0
          AND COALESCE(error_type, '') != 'streaming_in_progress'
        """,
        (since,),
    ).fetchone()
    return since, float(row[0] or 0.0), float(row[1] or 0.0), int(row[2] or 0)


def cmd_allowance(conn, args) -> int:
    store = SnapshotStore(Path(args.store) if args.store else None)

    if args.record:
        if args.spend is None or args.requests is None:
            print("--record needs --spend and --requests (the dashboard figures)")
            return 2
        snap = Snapshot(
            at=args.at or datetime.now().strftime("%Y-%m-%dT%H:%M"),
            week_to_date_usd=args.spend,
            credit_billed_requests=args.requests,
            balance_usd=args.balance,
        )
        store.append(snap)
        print(
            f"Recorded dashboard snapshot at {snap.at}: "
            f"${snap.week_to_date_usd} over {snap.credit_billed_requests} "
            f"credit-billed requests"
        )
        return 0

    since, avg_in, total_in, reqs = _ledger_burn_inputs(conn, args.days)
    report = burn_report(
        snapshots=store.load(),
        ledger_avg_input_tokens=avg_in,
        period_tokens=total_in,
        allowance_usd=args.allowance,
        ledger_requests=reqs,
    )
    if args.json:
        print(json.dumps(
            {"since": since, "avg_input_tokens": avg_in, "requests": reqs,
             **report.__dict__}, indent=2, default=str))
        return 0
    print(f"Window: since {since} ({reqs} requests, avg {avg_in:,.0f} input tokens)")
    print(format_report(report))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, days=3):
        p.add_argument("--days", type=int, default=days)
        p.add_argument("--json", action="store_true")

    common(sub.add_parser("rollup", help="roll up recent days into findings"))
    common(sub.add_parser("report", help="unit economics report"), days=7)
    common(sub.add_parser("findings", help="show durable daily findings"), days=7)
    common(sub.add_parser("propose", help="propose cheaper routing (never applies)"), days=14)
    common(sub.add_parser("audit", help="logging/audit health check"), days=7)

    bf = sub.add_parser("backfill", help="reconstruct cost for pre-instrumentation rows")
    bf.add_argument("--yes", action="store_true", help="apply (default is dry-run)")

    pu = sub.add_parser("purge", help="delete raw rows beyond retention")
    pu.add_argument("--retention", type=int, default=DEFAULT_KEEP_DAYS)
    pu.add_argument("--batch", type=int, default=50000)
    pu.add_argument("--yes", action="store_true", help="actually delete (default is dry-run)")

    al = sub.add_parser("allowance", help="allowance-burn early warning vs the vendor dashboard")
    al.add_argument("--days", type=int, default=7)
    al.add_argument("--json", action="store_true")
    al.add_argument("--allowance", type=float, default=None,
    help="plan included usage in USD (default BIGGIE_ALLOWANCE_USD or 100)")
    al.add_argument("--store", default=None, help="snapshot store path")
    al.add_argument("--record", action="store_true", help="record a dashboard reading")
    al.add_argument("--spend", type=float, default=None, help="dashboard week-to-date USD")
    al.add_argument("--requests", type=int, default=None, help="dashboard credit-billed requests")
    al.add_argument("--balance", type=float, default=None, help="dashboard balance USD")
    al.add_argument("--at", default=None, help="snapshot timestamp (AEST)")


    mt = sub.add_parser("maintain", help="daily job: rollup + purge + report + propose")
    mt.add_argument("--days", type=int, default=3)
    mt.add_argument("--retention", type=int, default=DEFAULT_KEEP_DAYS)
    mt.add_argument("--batch", type=int, default=50000)

    args = ap.parse_args()
    conn = _conn(args.db)
    return {
        "rollup": cmd_rollup,
        "report": cmd_report,
        "findings": cmd_findings,
        "propose": cmd_propose,
        "purge": cmd_purge,
        "audit": cmd_audit,
        "backfill": cmd_backfill,
        "maintain": cmd_maintain,
        "allowance": cmd_allowance,
    }[args.cmd](conn, args)


if __name__ == "__main__":
    raise SystemExit(main())
