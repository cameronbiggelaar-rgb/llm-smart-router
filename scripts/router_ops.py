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
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quality_probe import format_metrics as format_quality_metrics  # noqa: E402
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


ENDPOINT_HEALTH_URL = "http://127.0.0.1:8080/health"


def _fetch_health(timeout: float = 5.0) -> dict | None:
    """Read the endpoint's /health, or None if it cannot be read.

    Never raises: this is called by a daily job whose entire purpose is to catch
    problems, so an exception here would be a monitoring outage. An unreachable
    endpoint is a LIVENESS problem (reported by its own check), not a staleness
    one — conflating them would cry wolf on every restart.
    """
    try:
        resp = httpx.get(ENDPOINT_HEALTH_URL, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:  # noqa: BLE001 - unreachable is a valid outcome, not a crash
        return None


def _repo_head_sha() -> str | None:
    """Current HEAD of this checkout, or None if undeterminable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:  # noqa: BLE001 - fail open
        pass
    return None


def _endpoint_source_has_stamp() -> bool:
    """Does the deployed endpoint source carry the build-stamp feature?

    If the file on disk knows how to stamp itself, then a RUNNING endpoint that
    reports no `build` block is definitionally running a pre-stamp build — which
    is exactly the deploy-lag case this whole check exists to catch. Without this
    the detector would answer UNKNOWN (honest but useless) during the very
    situation it was built for.
    """
    try:
        source = (Path(__file__).resolve().parent / "biggie_llm_endpoint.py").read_text()
        return "_BUILD_STAMP" in source
    except OSError:
        return False


def evaluate_deploy(health: dict | None, repo_sha: str | None) -> dict:
    """Decide OK / STALE / UNKNOWN from a /health payload and the repo's HEAD.

    'unknown' means 'could not determine', which is NOT the same as 'different'.
    Treating it as a mismatch would flag every non-checkout host forever.
    """
    if health is None:
        return {
            "status": "UNKNOWN",
            "reason": "endpoint /health unreachable",
            "running_sha": None,
            "repo_sha": repo_sha,
            "changed_files": [],
        }
    build = health.get("build") or {}
    changed = list(build.get("changed_files") or [])

    # No build block at all, but the source on disk has the feature: the running
    # process predates build stamping. That is not "unknown" — it is proof of a
    # stale deployment, and it is the single most likely staleness there is.
    if not build and _endpoint_source_has_stamp():
        return {
            "status": "STALE",
            "reason": "running endpoint reports no build stamp, but the deployed "
                      "source has one — it is serving pre-stamp code",
            "running_sha": None,
            "repo_sha": repo_sha,
            "changed_files": [],
        }

    running = build.get("git_sha")
    if running in (None, "", "unknown") or repo_sha in (None, "", "unknown"):
        return {
            "status": "UNKNOWN",
            "reason": "could not determine a git sha",
            "running_sha": running,
            "repo_sha": repo_sha,
            "changed_files": changed,
        }

    # The endpoint's own mtime verdict catches uncommitted edits, which a sha
    # comparison alone cannot see. Either signal is sufficient to be stale.
    if running != repo_sha or build.get("stale"):
        return {
            "status": "STALE",
            "reason": "running build is not the current source",
            "running_sha": running,
            "repo_sha": repo_sha,
            "changed_files": changed,
        }
    return {
        "status": "OK",
        "reason": "running build matches HEAD",
        "running_sha": running,
        "repo_sha": repo_sha,
        "changed_files": [],
    }


def cmd_deploy_check(conn, args) -> int:
    """Is the running endpoint serving the current source?

    Detection only — never restarts anything. An automatic restart on a detected
    mismatch could mask a crash-loop (restart -> stale -> restart), which is a
    worse failure than a stale build.
    """
    verdict = evaluate_deploy(_fetch_health(), _repo_head_sha())
    if getattr(args, "json", False):
        print(json.dumps(verdict, indent=2))
    else:
        print("Deploy check — is the endpoint serving the current source?")
        for key in ("status", "reason", "running_sha", "repo_sha"):
            print(f"  {key}: {verdict[key]}")
        if verdict["changed_files"]:
            print("  changed files:")
            for path in verdict["changed_files"]:
                print(f"    {path}")
        if verdict["status"] == "STALE":
            print(
                "  ACTION: restart the endpoint to load the current code —\n"
                "    sudo systemctl restart biggie-llm-endpoint"
            )
    return 1 if verdict["status"] == "STALE" else 0


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
    # Staleness is reported but NEVER fatal here: turning it into a failed daily
    # job would convert an information problem into a monitoring outage, and the
    # rollup/purge/report work above would stop running.
    print()
    try:
        if cmd_deploy_check(conn, argparse.Namespace(json=False)):
            print(
                "WARNING: a stale build is serving traffic. This does not fail the "
                "job; restart the endpoint to load the current code."
            )
    except Exception as exc:  # noqa: BLE001 - never break the daily job
        print(f"WARNING: deploy check could not complete: {type(exc).__name__}: {exc}")
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


def cmd_quality(conn, args) -> int:
    """Show sampled compression-quality measurements."""
    print(format_quality_metrics(conn))
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

    dc = sub.add_parser("deploy-check", help="is the endpoint serving the current source?")
    dc.add_argument("--json", action="store_true")

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


    q = sub.add_parser("quality", help="sampled compression-quality measurements")
    q.add_argument("--json", action="store_true")

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
        "quality": cmd_quality,
        "allowance": cmd_allowance,
        "deploy-check": cmd_deploy_check,
    }[args.cmd](conn, args)


if __name__ == "__main__":
    raise SystemExit(main())
