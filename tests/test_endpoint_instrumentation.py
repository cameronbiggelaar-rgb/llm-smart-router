"""B7 — endpoint wiring: cost capture, quality hook, and the experiment hook.

These test the pure helpers the endpoint calls, so they run without a live
server and without touching the production router_logs database.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402
from rollup import migrate  # noqa: E402
from traffic_split import Experiment  # noqa: E402


@pytest.fixture()
def conn(tmp_path):
    c = sqlite3.connect(str(tmp_path / "router_logs.db"))
    migrate(c)
    return c


# ── cost capture ──────────────────────────────────────────────────────────────

def test_cost_fields_prices_a_known_model(conn):
    cost, unknown = ep.compute_cost_fields("glm-5.3", 1_000_000, 0, conn=conn)
    assert cost == pytest.approx(1.50)
    assert unknown == 0


def test_cost_fields_prices_input_and_output_separately(conn):
    # 1M in @ $1.50 + 1M out @ $4.50 = $6.00
    cost, unknown = ep.compute_cost_fields("glm-5.3", 1_000_000, 1_000_000, conn=conn)
    assert cost == pytest.approx(6.00)
    assert unknown == 0


def test_cost_fields_flags_unknown_model_rather_than_reporting_zero(conn):
    """An unpriced model must be flagged, not silently costed at $0.

    Silently costing an unknown model as free is how a spend number becomes
    fiction — the flag is what keeps the accounting honest.
    """
    cost, unknown = ep.compute_cost_fields("brand-new-model", 1_000_000, 1_000_000, conn=conn)
    assert cost == 0.0
    assert unknown == 1


def test_cost_fields_free_local_model_is_known_zero(conn):
    """A genuinely free local model is $0 and NOT flagged unknown."""
    cost, unknown = ep.compute_cost_fields("llama3.1:8b", 1_000_000, 1_000_000, conn=conn)
    assert cost == 0.0
    assert unknown == 0


def test_cost_fields_strips_provider_suffix(conn):
    a, _ = ep.compute_cost_fields("glm-5.3:cloud", 1_000_000, 0, conn=conn)
    b, _ = ep.compute_cost_fields("glm-5.3", 1_000_000, 0, conn=conn)
    assert a == b


def test_cost_fields_no_tokens_is_zero(conn):
    cost, unknown = ep.compute_cost_fields("glm-5.3", 0, 0, conn=conn)
    assert cost == 0.0
    assert unknown == 0


def test_cost_fields_never_raises_on_bad_input(conn):
    """Logging must never break a request — cost capture is best-effort."""
    cost, unknown = ep.compute_cost_fields("glm-5.3", None, None, conn=conn)
    assert isinstance(cost, float)
    assert unknown in (0, 1)


# ── experiment hook ───────────────────────────────────────────────────────────

def _exp(**kw):
    base = dict(
        name="t", enabled=True, model="candidate", mode="split",
        percent=100, match_workload=("session_compression",),
    )
    base.update(kw)
    return Experiment(**base)


def test_experiment_hook_disabled_means_no_change(conn):
    e = _exp(enabled=False)
    served, obs = ep.apply_experiment(
        "incumbent", e, workload_type="session_compression", request_id="r1"
    )
    assert served == "incumbent"
    assert obs["is_shadow"] is False
    assert obs["experiment_arm"] == ""


def test_experiment_hook_split_serves_candidate(conn):
    served, obs = ep.apply_experiment(
        "incumbent", _exp(mode="split", percent=100),
        workload_type="session_compression", request_id="r1",
    )
    assert served == "candidate"
    assert obs["experiment"] == "t"
    assert obs["experiment_arm"] == "treatment"
    assert obs["is_shadow"] is False


def test_experiment_hook_shadow_keeps_incumbent_visible(conn):
    """Shadow must NOT change what the user gets — only what we observe.

    ``is_shadow`` must stay False on THIS row: it describes the incumbent's
    real production call. The candidate's own call is logged separately by
    ``run_shadow_experiment``. Stamping it here moved genuine production calls
    (and their spend) into the shadow bucket.
    """
    served, obs = ep.apply_experiment(
        "incumbent", _exp(mode="shadow", percent=0),
        workload_type="session_compression", request_id="r1",
    )
    assert served == "incumbent"
    assert obs["experiment_arm"] == "shadow"
    assert obs["is_shadow"] is False
    assert obs["shadow_model"] == "candidate"


def test_experiment_hook_shadow_records_candidate_for_observation(conn):
    served, obs = ep.apply_experiment(
        "incumbent", _exp(mode="shadow", percent=0),
        workload_type="session_compression", request_id="r1",
    )
    assert obs["shadow_model"] == "candidate"
    assert served == "incumbent"


def test_experiment_hook_wrong_workload_is_untouched(conn):
    served, obs = ep.apply_experiment(
        "incumbent", _exp(mode="split", percent=100),
        workload_type="normal_chat", request_id="r1",
    )
    assert served == "incumbent"
    assert obs["experiment"] == ""


def test_experiment_hook_none_experiment_is_noop(conn):
    served, obs = ep.apply_experiment(
        "incumbent", None, workload_type="normal_chat", request_id="r1"
    )
    assert served == "incumbent"
    assert obs["is_shadow"] is False


def test_experiment_hook_is_deterministic_per_request_id(conn):
    """Same request id -> same arm, so retries never flip treatment."""
    a, _ = ep.apply_experiment(
        "inc", _exp(mode="split", percent=50), workload_type="session_compression", request_id="fixed"
    )
    b, _ = ep.apply_experiment(
        "inc", _exp(mode="split", percent=50), workload_type="session_compression", request_id="fixed"
    )
    assert a == b


# ── logging persists the new fields ───────────────────────────────────────────

def test_log_request_to_db_persists_cost_and_quality(conn, monkeypatch):
    monkeypatch.setattr(ep, "_get_db_connection", lambda: conn)
    ep._ensure_log_columns(conn)
    ep._log_request_to_db(
        model_used="glm-5.3",
        provider="ollama-cloud",
        task_type="session_compression",
        complexity_score=0.5,
        input_tokens=1000,
        output_tokens=100,
        latency_seconds=1.0,
        routing_time_ms=5,
        cost_usd=0.0123,
        cost_unknown=0,
        quality_score=0.875,
        quality_method="fact_coverage_v1",
        experiment="exp-1",
        experiment_arm="shadow",
        is_shadow=True,
        pricing_version="2026-09-12",
    )
    row = conn.execute(
        "SELECT cost_usd, cost_unknown, quality_score, quality_method, "
        "experiment, experiment_arm, is_shadow, pricing_version FROM router_logs"
    ).fetchone()
    assert row[0] == pytest.approx(0.0123)
    assert row[1] == 0
    assert row[2] == pytest.approx(0.875)
    assert row[3] == "fact_coverage_v1"
    assert row[4] == "exp-1"
    assert row[5] == "shadow"
    assert row[6] == 1
    assert row[7] == "2026-09-12"


def test_log_request_to_db_defaults_are_safe(conn, monkeypatch):
    """Omitting the new fields must still insert cleanly (back-compat)."""
    monkeypatch.setattr(ep, "_get_db_connection", lambda: conn)
    ep._ensure_log_columns(conn)
    ep._log_request_to_db(
        model_used="deepseek-v4-flash",
        provider="ollama-cloud",
        task_type="qa",
        complexity_score=0.1,
        input_tokens=10,
        output_tokens=5,
        latency_seconds=0.2,
        routing_time_ms=1,
    )
    row = conn.execute(
        "SELECT cost_unknown, quality_score, is_shadow FROM router_logs"
    ).fetchone()
    assert row[0] == 1          # unknown cost by default, never a fake 0
    assert row[1] is None       # unmeasured, not zero
    assert row[2] == 0


def test_ensure_log_columns_adds_every_new_column(conn):
    """The endpoint's own migration must add all rollup columns."""
    conn.execute("DROP TABLE router_logs")
    conn.execute(
        "CREATE TABLE router_logs (id INTEGER PRIMARY KEY, timestamp TEXT, "
        "model_used TEXT, input_tokens INTEGER, output_tokens INTEGER)"
    )
    conn.commit()
    ep._ensure_log_columns(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(router_logs)").fetchall()}
    for c in ("cost_usd", "cost_unknown", "quality_score", "quality_method",
              "experiment", "experiment_arm", "is_shadow", "pricing_version"):
        assert c in cols, f"missing {c}"
