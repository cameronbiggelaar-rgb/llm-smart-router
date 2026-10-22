"""Pin every numeric value the nightly digest emits.

Why this module exists
----------------------
``check-routing-stats.py`` is the operator's only routine view of router
effectiveness. Its numbers must be correct — a wrong "savings %" or a silently
zeroed lane count reads as health when the router is leaking. The always-zero
"equivalent gpt-5.5 calls saved" bug (fixed 2026-10-23) slipped through because
no test pinned the value. This module pins every *remaining* calculation so a
regression shows up as a red test, not a misread digest.

Covered here (the other report test modules cover lanes / tiers / success rate /
plan-budget already):
  - compute savings: ``actual_units``, ``gpt55_units``, ``savings_pct``,
    including the negative-savings and zero-calls edge cases
  - session counting (total / main / subagent)
  - model aggregation + ``canonical_model`` alias collapse (incl. retired-model
    fallback to default 1.0x ratio)
  - direct-vs-routed split + abandoned-stream counting
  - task-type distribution
  - corrections / routing_misses, including the missing-column guard that a
    silent ``IndexError`` used to break (loop aborted, stats zeroed)
  - day-over-day snapshot deltas (first run vs. second run)
  - ``fmt_num`` / ``_delta`` helpers
"""

from __future__ import annotations

import importlib.util
import io
import json
import sqlite3
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location(
    "check_routing_stats", SCRIPTS_DIR / "check-routing-stats.py"
)
crs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crs)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts() -> float:
    return datetime.now(timezone.utc).timestamp()


# ── Harness ──────────────────────────────────────────────────────────────────

# Full router_logs schema mirroring production, with the fit-signal columns that
# the report reads via `.keys()` guards (routing_reason, user_corrected,
# cheaper_model_would_work) plus the governor columns.
_FIT_COLS = (
    "routing_reason TEXT DEFAULT '', "
    "user_corrected INTEGER DEFAULT 0, "
    "cheaper_model_would_work INTEGER DEFAULT 0, "
)
_GOV_COLS = (
    "budget_class TEXT NOT NULL DEFAULT '', "
    "plan_budget_denied INTEGER NOT NULL DEFAULT 0, "
    "governor_fail_open INTEGER NOT NULL DEFAULT 0"
)


def _state_db(path: Path, usage_rows: list[dict]) -> str:
    """Build a minimal state.db: sessions + session_model_usage."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, started_at REAL, parent_session_id TEXT
        );
        CREATE TABLE session_model_usage (
            session_id TEXT, model TEXT, billing_provider TEXT,
            api_call_count INTEGER, input_tokens INTEGER, output_tokens INTEGER,
            first_seen REAL, last_seen REAL
        );
        """
    )
    for r in usage_rows:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, parent_session_id) VALUES (?, ?)",
            (r["session_id"], r.get("parent")),
        )
        conn.execute(
            """INSERT INTO session_model_usage
               (session_id, model, billing_provider, api_call_count,
                input_tokens, output_tokens, last_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                r["session_id"],
                r["model"],
                r.get("provider", ""),
                r["calls"],
                r.get("in_tok", r["calls"] * 100),
                r.get("out_tok", r["calls"] * 50),
                r.get("last_seen", _ts()),
            ),
        )
    conn.commit()
    conn.close()
    return str(path)


def _router_db(path: Path, rows: list[dict], *, fit_cols: bool = True) -> str:
    """Build a router_logs.db. ``fit_cols=False`` omits user_corrected and
    cheaper_model_would_work to exercise the missing-column guard."""
    conn = sqlite3.connect(str(path))
    fit = _FIT_COLS if fit_cols else ""
    conn.execute(
        f"""
        CREATE TABLE router_logs (
            timestamp TEXT NOT NULL, session_id TEXT, model_used TEXT, provider TEXT,
            task_type TEXT, success INTEGER DEFAULT 1, escalated INTEGER DEFAULT 0,
            error_type TEXT, request_id TEXT, requested_model TEXT,
            streaming INTEGER DEFAULT 0, workload_type TEXT DEFAULT '',
            requires_tools INTEGER DEFAULT 0, input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0, final_model TEXT DEFAULT '',
            {fit}
            {_GOV_COLS}
        )
        """
    )
    schema_cols = {
        r[1] for r in conn.execute("PRAGMA table_info(router_logs)")
    }
    for r in rows:
        keys = [k for k in r if k in schema_cols]
        conn.execute(
            f"INSERT INTO router_logs ({', '.join(keys)}) VALUES ({','.join('?' * len(keys))})",
            [r[k] for k in keys],
        )
    conn.commit()
    conn.close()
    return str(path)


def _run(state_db: str, router_db: str, tmp_path: Path, *, json_out: bool = True,
         snapshot: str | None = None):
    """Run main() against the given DBs, isolating the snapshot file."""
    crs.STATE_DB = state_db
    crs.ROUTER_LOGS = router_db
    crs.SNAPSHOT_FILE = str(snapshot or (tmp_path / "snapshot.json"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        sys.argv = ["check-routing-stats.py"] + (["--json"] if json_out else [])
        crs.main()
    sys.argv = ["check-routing-stats.py"]
    return json.loads(buf.getvalue()) if json_out else buf.getvalue()


# Row helpers
def _u(session_id: str, model: str, calls: int, *, parent=None, in_tok=None,
       out_tok=None) -> dict:
    r = {"session_id": session_id, "model": model, "calls": calls, "parent": parent}
    if in_tok is not None:
        r["in_tok"] = in_tok
    if out_tok is not None:
        r["out_tok"] = out_tok
    return r


def _r(requested: str, final: str = "", **kw) -> dict:
    row = {
        "timestamp": _now(), "session_id": "s", "model_used": final or requested,
        "provider": "ollama-cloud", "task_type": "other", "success": 1,
        "escalated": 0, "error_type": "", "request_id": "", "requested_model": requested,
        "streaming": 0, "workload_type": "", "requires_tools": 0,
        "input_tokens": 0, "output_tokens": 0, "final_model": final,
        "routing_reason": "", "user_corrected": 0, "cheaper_model_would_work": 0,
        "budget_class": "", "plan_budget_denied": 0, "governor_fail_open": 0,
    }
    row.update(kw)
    return row


# ── Helpers ──────────────────────────────────────────────────────────────────

def test_fmt_num_formats_millions_thousands_plain():
    assert crs.fmt_num(2_500_000) == "2.5M"
    assert crs.fmt_num(1_000_000) == "1.0M"
    assert crs.fmt_num(1_000_000_000) == "1000.0M"
    assert crs.fmt_num(98_063) == "98,063"
    assert crs.fmt_num(1_000) == "1,000"
    assert crs.fmt_num(999) == "999"
    assert crs.fmt_num(0) == "0"


def test_delta_returns_none_without_baseline():
    assert crs._delta(10, None) is None
    assert crs._delta(10.5, 4.0) == 6.5
    assert crs._delta(5, 8) == -3


# ── Compute savings ──────────────────────────────────────────────────────────

def test_compute_savings_math(tmp_path):
    # 100 × deepseek-v4-pro (4.0x) + 50 × deepseek-v4.1-flash (1.0x)
    # actual = 400 + 50 = 450; gpt55 = 150 * 30 = 4500; savings = 90.0%
    sdb = _state_db(tmp_path / "state.db", [
        _u("s1", "deepseek-v4-pro", 100),
        _u("s1", "deepseek-v4.1-flash", 50),
    ])
    rdb = _router_db(tmp_path / "router.db", [])
    out = _run(sdb, rdb, tmp_path)

    assert out["compute"]["actual_units"] == 450.0
    assert out["compute"]["gpt55_units"] == 4500.0
    assert out["compute"]["savings_pct"] == 90.0
    assert out["usage"]["total_calls"] == 150


def test_compute_savings_negative_when_model_costlier_than_baseline(tmp_path):
    # gpt-6-astra ratio 40.0 > gpt-5.5's 30.0 → negative savings, no crash.
    sdb = _state_db(tmp_path / "state.db", [_u("s1", "gpt-6-astra", 100)])
    rdb = _router_db(tmp_path / "router.db", [])
    out = _run(sdb, rdb, tmp_path)

    assert out["compute"]["actual_units"] == 4000.0
    assert out["compute"]["gpt55_units"] == 3000.0
    assert out["compute"]["savings_pct"] == -33.3


def test_zero_calls_savings_pct_zero_no_divzero(tmp_path):
    sdb = _state_db(tmp_path / "state.db", [])
    rdb = _router_db(tmp_path / "router.db", [])
    out = _run(sdb, rdb, tmp_path)

    assert out["compute"] == {"actual_units": 0.0, "gpt55_units": 0.0, "savings_pct": 0.0}
    assert out["usage"]["total_calls"] == 0
    assert out["sessions"]["total"] == 0


# ── Sessions ─────────────────────────────────────────────────────────────────

def test_session_counts_main_vs_subagent(tmp_path):
    sdb = _state_db(tmp_path / "state.db", [
        _u("s1", "deepseek-v4-pro", 10),
        _u("s2", "glm-5.3", 5),
        _u("s3", "deepseek-v4.1-flash", 3, parent="s1"),
    ])
    rdb = _router_db(tmp_path / "router.db", [])
    out = _run(sdb, rdb, tmp_path)

    assert out["sessions"] == {"total": 3, "main": 2, "subagent": 1}


# ── Model aggregation + canonical aliases ───────────────────────────────────

def test_canonical_model_collapses_cloud_aliases_and_aggregates(tmp_path):
    # glm-5.2:cloud and glm-5.2 collapse to one model with summed calls.
    # deepseek-v4-flash:cloud collapses to deepseek-v4-flash, which is retired →
    # falls back to default tier 0 / ratio 1.0 (never a crash).
    sdb = _state_db(tmp_path / "state.db", [
        _u("s1", "glm-5.2:cloud", 5),
        _u("s1", "glm-5.2", 3),
        _u("s2", "deepseek-v4-flash:cloud", 2),
    ])
    rdb = _router_db(tmp_path / "router.db", [])
    out = _run(sdb, rdb, tmp_path)

    by_model = {m["model"]: m for m in out["models"]}
    assert by_model["glm-5.2"]["calls"] == 8
    assert by_model["glm-5.2"]["tier"] == 6.5
    assert by_model["glm-5.2"]["compute_units"] == 3.0
    # Retired model still counted, default ratio keeps actual_units sane.
    assert by_model["deepseek-v4-flash"]["calls"] == 2
    assert by_model["deepseek-v4-flash"]["tier"] == 0
    assert by_model["deepseek-v4-flash"]["compute_units"] == 1.0


# ── Direct vs routed ────────────────────────────────────────────────────────

def test_direct_vs_routed_split(tmp_path):
    sdb = _state_db(tmp_path / "state.db", [])
    rdb = _router_db(tmp_path / "router.db", [
        _r("biggie-llm", "minimax-m3"),
        _r("biggie-router", "glm-5.3"),
        _r("deepseek-v4-pro", "deepseek-v4-pro"),
    ])
    out = _run(sdb, rdb, tmp_path)

    ep = out["router_endpoint"]
    assert ep["observed_requests"] == 3
    assert ep["routed_requests"] == 2
    assert ep["direct_backend_requests"] == 1
    assert ep["selected_models"] == {"minimax-m3": 1, "glm-5.3": 1, "deepseek-v4-pro": 1}


def test_abandoned_streams_counted(tmp_path):
    sdb = _state_db(tmp_path / "state.db", [])
    rdb = _router_db(tmp_path / "router.db", [
        # completion for req1 → req1 is "completed"
        _r("biggie-llm", "minimax-m3", request_id="req1"),
        # streaming start for req1 → completed, NOT abandoned
        _r("biggie-llm", "", request_id="req1", streaming=1,
           error_type="streaming_in_progress"),
        # streaming start for req2 → never completed → abandoned
        _r("biggie-llm", "", request_id="req2", streaming=1,
           error_type="streaming_in_progress"),
    ])
    out = _run(sdb, rdb, tmp_path)

    ep = out["router_endpoint"]
    assert ep["observed_requests"] == 1  # only the completion row
    assert ep["stream_start_rows"] == 2
    assert ep["abandoned_streams"] == 1


# ── Task types ───────────────────────────────────────────────────────────────

def test_task_type_distribution(tmp_path):
    sdb = _state_db(tmp_path / "state.db", [])
    rdb = _router_db(tmp_path / "router.db", [
        _r("biggie-llm", "minimax-m3", task_type="coding"),
        _r("biggie-llm", "minimax-m3", task_type="coding"),
        _r("biggie-llm", "minimax-m3", task_type="summarization"),
    ])
    out = _run(sdb, rdb, tmp_path)

    assert out["task_types"] == {"coding": 2, "summarization": 1}


# ── Corrections / routing misses (incl. missing-column guard) ───────────────

def test_corrections_and_routing_misses_counted(tmp_path):
    sdb = _state_db(tmp_path / "state.db", [])
    rdb = _router_db(tmp_path / "router.db", [
        _r("biggie-llm", "minimax-m3", user_corrected=1, cheaper_model_would_work=1),
        _r("biggie-llm", "minimax-m3"),
    ])
    out = _run(sdb, rdb, tmp_path)

    assert out["corrections"] == 1
    assert out["routing_misses"] == 1


def test_missing_fit_columns_do_not_zero_out_other_stats(tmp_path):
    """Regression: a schema missing user_corrected / cheaper_model_would_work
    used to raise IndexError inside the per-row loop, which the broad
    ``except Exception: pass`` swallowed — aborting the loop and zeroing
    completion_success, failures, lanes and selected models. The guard (line 348)
    keeps the loop running; corrections/routing_misses fall back to 0."""
    sdb = _state_db(tmp_path / "state.db", [])
    rdb = _router_db(tmp_path / "router.db", [
        _r("biggie-llm", "minimax-m3", success=1),
        _r("biggie-llm", "minimax-m3", success=0, error_type="timeout"),
    ], fit_cols=False)
    out = _run(sdb, rdb, tmp_path)

    # The guard must not abort the loop: both rows still counted.
    assert out["completion"]["success"] == 1
    assert out["completion"]["failures"] == 1
    assert out["completion"]["failure_by_type"] == {"timeout": 1}
    assert out["router_endpoint"]["observed_requests"] == 2
    # And the guarded fields degrade to 0, not "no data".
    assert out["corrections"] == 0
    assert out["routing_misses"] == 0


# ── Day-over-day snapshot ────────────────────────────────────────────────────

def test_correction_rate_uses_endpoint_population_not_total_calls(tmp_path):
    """Regression: the correction rate denominator used to be ``total_calls``
    (all API calls from state.db) while ``corrections`` is counted only over
    endpoint observations (router_logs.db) — a cross-DB population mismatch that
    silently diluted the rate. It must divide by ``router_completion_rows``."""
    sdb = _state_db(tmp_path / "state.db", [_u("s1", "deepseek-v4-pro", 100)])
    rdb = _router_db(tmp_path / "router.db", [
        _r("biggie-llm", "minimax-m3", user_corrected=1),
    ] + [
        _r("biggie-llm", "minimax-m3") for _ in range(9)
    ])
    out = _run(sdb, rdb, tmp_path, json_out=False)

    # 1 correction over 10 endpoint completions = 10.00% (not 1/100 = 1.00%).
    assert "Correction rate:       10.00%" in out


def test_snapshot_first_run_has_no_baseline(tmp_path):
    sdb = _state_db(tmp_path / "state.db", [_u("s1", "deepseek-v4-pro", 10)])
    rdb = _router_db(tmp_path / "router.db", [])
    out = _run(sdb, rdb, tmp_path, json_out=False)

    assert "first run — no prior baseline" in out
    assert "Day-over-day" in out


def test_snapshot_second_run_reports_deltas(tmp_path):
    snap = str(tmp_path / "snap.json")
    # First run: 10 calls, no baseline.
    sdb1 = _state_db(tmp_path / "s1.db", [_u("s1", "deepseek-v4-pro", 10)])
    rdb1 = _router_db(tmp_path / "r1.db", [])
    out1 = _run(sdb1, rdb1, tmp_path, json_out=False, snapshot=snap)
    assert "first run — no prior baseline" in out1

    # Second run: 15 calls → +5 delta.
    sdb2 = _state_db(tmp_path / "s2.db", [_u("s1", "deepseek-v4-pro", 15)])
    rdb2 = _router_db(tmp_path / "r2.db", [])
    out2 = _run(sdb2, rdb2, tmp_path, json_out=False, snapshot=snap)

    assert "first run — no prior baseline" not in out2
    assert "API calls:        +5" in out2
    # The persisted snapshot now carries the newer cumulative total.
    with open(snap) as fh:
        assert json.load(fh)["total_calls"] == 15
