"""B11 — the production DB path must be overridable, and must default identically.

Motivation (a real defect, not a hypothetical): the path was hardcoded, so an
exploratory probe could not redirect it and silently wrote stub rows into the
PRODUCTION database. The same hardcoding forced every test to replace
``_get_db_connection`` wholesale — which is precisely why the real
initialisation path (schema creation, price seeding) was never executed by any
test, and why two live defects survived a 266-test suite.

These tests pin the contract:
  1. with no override, the path is exactly what production uses today;
  2. an override is honoured;
  3. the real logger writes to the override and leaves production untouched.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402
import rollup  # noqa: E402

# The exact path production must keep using when nothing overrides it.
PRODUCTION_DB = Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "router_logs.db"


@pytest.fixture(autouse=True)
def _reset_conn(monkeypatch):
    """Each test starts with no cached connection."""
    monkeypatch.setattr(ep, "_sqlite_conn", None, raising=False)
    monkeypatch.setattr(ep, "_sqlite_lock", None, raising=False)
    yield
    monkeypatch.setattr(ep, "_sqlite_conn", None, raising=False)
    monkeypatch.setattr(ep, "_sqlite_lock", None, raising=False)


def test_default_path_is_the_live_production_path(monkeypatch):
    """No override -> byte-identical to the historical hardcoded path."""
    monkeypatch.delenv("BIGGIE_ROUTER_DB", raising=False)
    assert ep._router_db_path() == PRODUCTION_DB


def test_override_is_honoured(monkeypatch, tmp_path):
    target = tmp_path / "throwaway.db"
    monkeypatch.setenv("BIGGIE_ROUTER_DB", str(target))
    assert ep._router_db_path() == target


def test_real_logger_writes_to_override_and_not_production(monkeypatch, tmp_path):
    """The whole point: drive the REAL path without touching production.

    This is the test that was impossible before the path was injectable, and it
    is the shape of test that would have caught the probe leaking stub rows.
    """
    target = tmp_path / "throwaway.db"
    monkeypatch.setenv("BIGGIE_ROUTER_DB", str(target))

    before = None
    if PRODUCTION_DB.exists():
        conn = sqlite3.connect(f"file:{PRODUCTION_DB}?mode=ro", uri=True)
        before = conn.execute("SELECT COUNT(*) FROM router_logs").fetchone()[0]
        conn.close()

    # Real initialisation + real write, no fakes.
    conn = ep._get_db_connection()
    assert Path(target).exists(), "the override path must be the one initialised"
    ep._log_request_to_db(
        model_used="deepseek-v4.1-flash",
        provider="deepseek",
        task_type="test",
        complexity_score=0.5,
        input_tokens=1000,
        output_tokens=250,
        latency_seconds=1.5,
        routing_time_ms=10,
        workload_type="session_compression",
    )
    n = conn.execute("SELECT COUNT(*) FROM router_logs").fetchone()[0]
    assert n == 1, f"row must land in the override DB, found {n}"

    if before is not None:
        conn2 = sqlite3.connect(f"file:{PRODUCTION_DB}?mode=ro", uri=True)
        after = conn2.execute("SELECT COUNT(*) FROM router_logs").fetchone()[0]
        conn2.close()
        assert after == before, "a test must never write to the production database"


def test_init_on_a_fresh_db_seeds_prices(monkeypatch, tmp_path):
    """Real init must seed the price book.

    Live defect: model_pricing was empty on production, so every row logged
    cost_unknown=1 forever and a live experiment produced real spend with no
    cost evidence. Only driving the real init path reveals this.
    """
    target = tmp_path / "fresh.db"
    monkeypatch.setenv("BIGGIE_ROUTER_DB", str(target))
    conn = ep._get_db_connection()
    n = conn.execute("SELECT COUNT(*) FROM model_pricing").fetchone()[0]
    assert n > 0, "real init must seed model_pricing, or cost capture is fiction"


def test_real_init_and_write_agree_on_schema(monkeypatch, tmp_path):
    """A row must actually be accepted by the table the real path builds.

    Live defect: the migrated schema was missing 10 columns the logger writes,
    so every INSERT raised 'no column named ...' and was swallowed — requests
    returned 200 while all observability was silently lost.
    """
    target = tmp_path / "schema.db"
    monkeypatch.setenv("BIGGIE_ROUTER_DB", str(target))
    conn = ep._get_db_connection()

    writable = set(ep._STREAM_OBS_COLUMNS) | set(ep._COST_OBS_COLUMNS)
    actual = {r[1] for r in conn.execute("PRAGMA table_info(router_logs)")}
    missing = writable - actual
    assert not missing, f"logger can write columns the table lacks: {sorted(missing)}"

    # And prove it with a real write of every observability field we can set.
    ep._log_request_to_db(
        model_used="glm-5.3",
        provider="ollama-cloud",
        task_type="test",
        complexity_score=0.9,
        input_tokens=5000,
        output_tokens=900,
        latency_seconds=3.0,
        routing_time_ms=12,
        workload_type="session_compression",
        finish_reason="stop",
        saw_tool_calls=False,
        quality_score=0.91,
        quality_method="fact_coverage_v1",
        experiment="shadowtest",
        is_shadow=False,
    )
    row = conn.execute(
        "SELECT finish_reason, quality_score, experiment FROM router_logs"
    ).fetchone()
    assert row == ("stop", 0.91, "shadowtest")
