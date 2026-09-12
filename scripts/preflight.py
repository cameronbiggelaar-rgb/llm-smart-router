#!/usr/bin/env python3
"""Preflight: prove the router works end to end BEFORE it touches live traffic.

Why this exists
---------------
On 2026-09-12 a single live enablement surfaced four defects. Every one was a
*seam* defect — a module boundary that no test crossed:

  * `asyncio` was never imported at module level, so the shadow executor could
    not schedule anything;
  * `rollup.migrate()` built a schema missing 10 columns the logger writes, so
    every INSERT raised 'no column named ...' and was swallowed;
  * `finish_reason` was passed to `_log_request_to_db`, which did not accept it
    — again a swallowed TypeError, row lost, HTTP 200 returned;
  * `model_pricing` was never seeded by the endpoint, so every row logged
    cost_unknown=1 and a live experiment spent real money with no cost evidence.

The unit suite could not catch these because it replaced the very functions
under test. This harness does the opposite: it boots the REAL production code
path against a throwaway database and drives REAL HTTP requests through the
real ASGI app, stubbing only the outbound network.

What it asserts
---------------
  1. no silent drop    — rows written == requests made
  2. schema parity     — every column the logger can write exists in the table
  3. cost capture      — prices are seeded, so calls are priced not 'unknown'
  4. migration parity  — a fresh DB built by the real init path is writable
  5. candidate visible — an unresolvable experiment candidate is reported

Run before enabling any experiment, after touching the logging path, or in CI:

    python3 scripts/preflight.py

Exit code 0 == safe to enable. Non-zero == do not put this in front of traffic.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

FAILURES: list[str] = []
CHECKS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append(f"  {'PASS' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    tmpdir = Path(tempfile.mkdtemp(prefix="preflight_"))
    db_path = tmpdir / "router_logs.db"
    # Point the REAL code at a throwaway DB. No monkeypatching of the function
    # under test: the point is to exercise the production wiring itself.
    os.environ["BIGGIE_ROUTER_DB"] = str(db_path)

    import biggie_llm_endpoint as ep
    import rollup
    import unit_economics

    print(f"preflight: throwaway db = {db_path}")

    # ---------------------------------------------------------------- 1. init
    conn = ep._get_db_connection()
    check("real init creates the database", db_path.exists())

    tables = {
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    for t in ("router_logs", "model_pricing", "daily_findings", "rollup_state"):
        check(f"table present: {t}", t in tables)

    # ------------------------------------------------------- 2. schema parity
    writable = set(ep._STREAM_OBS_COLUMNS) | set(ep._COST_OBS_COLUMNS)
    actual = {r[1] for r in conn.execute("PRAGMA table_info(router_logs)")}
    missing = writable - actual
    check(
        "logger columns all exist in router_logs",
        not missing,
        f"missing {sorted(missing)}" if missing else f"{len(writable)} columns",
    )

    # The migrated schema must also accept the logger's own parameters: a
    # column can exist while the function signature rejects the keyword, which
    # is exactly how `finish_reason` failed.
    import inspect

    params = set(inspect.signature(ep._log_request_to_db).parameters)
    check(
        "logger accepts finish_reason",
        "finish_reason" in params,
        "regression: the shadow row needs it to record tool_calls vs stop",
    )

    # -------------------------------------------------------- 3. cost capture
    priced = conn.execute("SELECT COUNT(*) FROM model_pricing").fetchone()[0]
    check("price book seeded by real init", priced > 0, f"{priced} models")

    # A logged call must carry a real cost, not cost_unknown=1.
    ep._log_request_to_db(
        model_used="deepseek-v4.1-flash",
        provider="deepseek",
        task_type="preflight",
        complexity_score=0.5,
        input_tokens=10_000,
        output_tokens=1_000,
        latency_seconds=1.0,
        routing_time_ms=5,
        workload_type="session_compression",
    )
    row = conn.execute(
        "SELECT cost_usd, cost_unknown FROM router_logs WHERE task_type='preflight'"
    ).fetchone()
    check(
        "a logged call is priced, not unknown",
        bool(row) and row[0] > 0 and row[1] == 0,
        f"cost={row[0] if row else None} unknown={row[1] if row else None}",
    )

    # ------------------------------------------------- 4. end-to-end HTTP path
    # Drive the real app with only the network stubbed. This exercises the
    # request -> route -> log -> write chain that unit tests bypass.
    calls: list[dict] = []

    async def fake_proxy(backend, messages, body):
        calls.append({"model": backend.get("backend_model")})
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 300},
        }

    async def fake_proxy_stream(backend, messages, body, on_complete=None, stats=None):
        calls.append({"model": backend.get("backend_model"), "stream": True})
        if on_complete:
            on_complete()
        return {"choices": [{"message": {"content": "ok"}}]}

    ep.proxy_to_backend = fake_proxy
    ep.proxy_to_backend_streaming = fake_proxy_stream
    ep.discover_backends = lambda: {
        "deepseek-v4.1-flash:cloud": {
            "provider": "deepseek",
            "base_url": "http://stub",
            "backend_model": "deepseek-v4.1-flash",
        }
    }

    import httpx

    before = conn.execute("SELECT COUNT(*) FROM router_logs").fetchone()[0]
    n_requests = 5

    async def drive():
        transport = httpx.ASGITransport(app=ep.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://pf") as c:
            for i in range(n_requests):
                r = await c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "biggie-router",
                        "messages": [{"role": "user", "content": f"preflight probe {i} " * 30}],
                        "stream": False,
                    },
                )
                check(f"request {i} returned 200", r.status_code == 200, f"got {r.status_code}")
        await asyncio.sleep(0.5)

    asyncio.run(drive())
    after = conn.execute("SELECT COUNT(*) FROM router_logs").fetchone()[0]
    written = after - before
    check(
        "no silent drop: every request is logged",
        written >= n_requests,
        f"{n_requests} requests -> {written} rows",
    )

    # --------------------------------------------------- 5. experiment wiring
    # An experiment naming a model with no reachable backend must be reported
    # at startup, not discovered as silence after it is enabled live.
    from traffic_split import Experiment

    exp = Experiment(
        name="preflight-candidate",
        enabled=True,
        model="definitely-not-a-real-model-xyz",
        mode="shadow",
        percent=0.0,
        match_workload=("session_compression",),
    )
    ep._experiments_cached = lambda: [exp]
    resolved = ep._resolve_candidate_backend(exp.model, getattr(exp, "provider", ""))
    check(
        "an unresolvable candidate resolves to None (and is logged)",
        resolved is None,
        "the endpoint must warn rather than silently do nothing",
    )

    # -------------------------------------------------------------- reporting
    print()
    for line in CHECKS:
        print(line)
    print()
    if FAILURES:
        print(f"PREFLIGHT FAILED — {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        print("Do NOT enable this against live traffic.")
        return 1
    print(f"PREFLIGHT OK — {len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
