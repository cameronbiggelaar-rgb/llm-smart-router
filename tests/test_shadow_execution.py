"""B10 — the shadow path must actually run, and never pollute production numbers.

A shadow experiment that only stamps rows ``is_shadow=1`` without ever calling
the candidate is worse than no experiment: 19k rows would claim observational
coverage of a model that was never exercised, and the optimiser would be fed
phantom evidence. So the executor is tested for real behaviour:

  1. it genuinely issues a second call to the candidate;
  2. the candidate's output is DISCARDED (what the user gets is unchanged);
  3. its cost is recorded as a separate row, not folded into the incumbent's;
  4. it is bounded/cheap-by-default so an experiment cannot stall production;
  5. a candidate failure never breaks the user's request.

Plus rollup: shadow calls cost real money and must be reported separately, or
the production spend figure is overstated.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402
import rollup  # noqa: E402
from traffic_split import Experiment  # noqa: E402


@pytest.fixture()
def conn(tmp_path):
    c = sqlite3.connect(str(tmp_path / "router_logs.db"))
    rollup.migrate(c)
    from unit_economics import seed_prices

    seed_prices(c)
    return c


def _exp(**kw):
    base = dict(
        name="shadowtest",
        enabled=True,
        model="glm-5.3-flash",
        mode="shadow",
        percent=0.0,
        match_workload=("session_compression",),
    )
    base.update(kw)
    return Experiment(**base)


# ── the executor actually calls the candidate ─────────────────────────────────

def test_run_shadow_actually_calls_the_candidate(monkeypatch):
    """The whole point: a real second call. A no-op must fail this test."""
    calls = []

    async def fake_proxy(backend, messages, body):
        calls.append({"backend": backend, "body": dict(body)})
        return {
            "choices": [{"message": {"role": "assistant", "content": "candidate answer"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
        }

    monkeypatch.setattr(ep, "proxy_to_backend", fake_proxy)

    backend = {"provider": "ollama-cloud", "base_url": "http://x", "backend_model": "glm-5.3-flash"}
    msgs = [{"role": "user", "content": "summarise this"}]
    rows = []

    asyncio.run(
        ep.run_shadow_experiment(
            _exp(),
            backend=backend,
            messages=msgs,
            request_body={"stream": True, "max_tokens": 4096},
            request_id="req1",
            session_id="sess1",
            incumbent_model="deepseek-v4.1-flash:cloud",
            on_row=rows.append,
        )
    )

    assert len(calls) == 1, "candidate was never called — shadow would be fiction"
    assert calls[0]["backend"] is backend
    assert len(rows) == 1
    assert rows[0]["model_used"] == "glm-5.3-flash"
    assert rows[0]["is_shadow"] is True


def test_shadow_call_is_forced_non_streaming(monkeypatch):
    """We cannot consume a stream we intend to discard, so shadow must not ask
    for one — otherwise the call hangs holding the payload."""
    seen = {}

    async def fake_proxy(backend, messages, body):
        seen["stream"] = body.get("stream")
        return {"choices": [{"message": {"content": "x"}}], "usage": {}}

    monkeypatch.setattr(ep, "proxy_to_backend", fake_proxy)
    asyncio.run(
        ep.run_shadow_experiment(
            _exp(),
            backend={"provider": "p", "base_url": "u", "backend_model": "m"},
            messages=[{"role": "user", "content": "hi"}],
            request_body={"stream": True},
            request_id="r",
            session_id="s",
            incumbent_model="m",
            on_row=lambda r: None,
        )
    )
    assert seen["stream"] is False, "shadow must request non-streaming"


def test_shadow_output_is_discarded(monkeypatch):
    """Nothing from the candidate may leak into the servable result."""
    async def fake_proxy(backend, messages, body):
        return {"choices": [{"message": {"content": "CANDIDATE TEXT"}}], "usage": {}}

    monkeypatch.setattr(ep, "proxy_to_backend", fake_proxy)
    got = asyncio.run(
        ep.run_shadow_experiment(
            _exp(),
            backend={"provider": "p", "base_url": "u", "backend_model": "m"},
            messages=[{"role": "user", "content": "hi"}],
            request_body={},
            request_id="r",
            session_id="s",
            incumbent_model="m",
            on_row=lambda r: None,
        )
    )
    assert got is None, "shadow must return nothing the caller could serve"


def test_shadow_failure_never_raises(monkeypatch):
    """A broken candidate must not break the user's request."""
    async def boom(backend, messages, body):
        raise RuntimeError("candidate exploded")

    monkeypatch.setattr(ep, "proxy_to_backend", boom)
    rows = []
    got = asyncio.run(
        ep.run_shadow_experiment(
            _exp(),
            backend={"provider": "p", "base_url": "u", "backend_model": "m"},
            messages=[{"role": "user", "content": "hi"}],
            request_body={},
            request_id="r",
            session_id="s",
            incumbent_model="m",
            on_row=rows.append,
        )
    )
    assert got is None
    assert len(rows) == 1
    assert rows[0]["success"] is False, "failure should be recorded as failure"


def test_shadow_is_disabled_when_experiment_disabled(monkeypatch):
    """An enabled=False experiment must not spend a cent."""
    called = []

    async def fake_proxy(backend, messages, body):
        called.append(1)
        return {"choices": [{"message": {"content": "x"}}], "usage": {}}

    monkeypatch.setattr(ep, "proxy_to_backend", fake_proxy)
    asyncio.run(
        ep.run_shadow_experiment(
            _exp(enabled=False),
            backend={"provider": "p", "base_url": "u", "backend_model": "m"},
            messages=[{"role": "user", "content": "hi"}],
            request_body={},
            request_id="r",
            session_id="s",
            incumbent_model="m",
            on_row=lambda r: None,
        )
    )
    assert called == [], "disabled experiment still called the candidate"


def test_shadow_row_prices_the_candidate_not_the_incumbent(monkeypatch, conn):
    """Cost must be attributed to the model that actually ran."""
    async def fake_proxy(backend, messages, body):
        return {
            "choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
        }

    monkeypatch.setattr(ep, "proxy_to_backend", fake_proxy)
    rows = []
    asyncio.run(
        ep.run_shadow_experiment(
            _exp(),
            backend={"provider": "p", "base_url": "u", "backend_model": "m"},
            messages=[{"role": "user", "content": "hi"}],
            request_body={},
            request_id="r",
            session_id="s",
            incumbent_model="deepseek-v4.1-flash:cloud",
            on_row=rows.append,
            conn=conn,
        )
    )
    # glm-5.3-flash input is $0.15/1M — not the incumbent's price.
    assert rows[0]["cost_usd"] == pytest.approx(0.15)
    assert rows[0]["model_used"] == "glm-5.3-flash"


# ── rollup must not fold shadow spend into production spend ───────────────────

def test_rollup_separates_shadow_from_production(conn):
    """Shadow calls are real money but not production traffic. If they are
    summed together the reported production spend is wrong."""
    cols = (
        "timestamp, model_used, provider, task_type, complexity_score, "
        "input_tokens, output_tokens, latency_seconds, success, workload_type, "
        "cost_usd, cost_unknown, is_shadow"
    )
    conn.execute(
        f"INSERT INTO router_logs ({cols}) VALUES "
        "('2026-09-01T10:00:00', 'deepseek-v4.1-flash:cloud', 'p', 'summarise', 1.0,"
        " 1000, 100, 1.0, 1, 'session_compression', 0.010, 0, 0),"
        "('2026-09-01T10:00:01', 'glm-5.3-flash', 'p', 'summarise', 1.0,"
        " 1000, 100, 2.0, 1, 'session_compression', 0.999, 0, 1)"
    )
    conn.commit()

    rollup.rollup_day(conn, "2026-09-01")

    prod = conn.execute(
        "SELECT model, calls, cost_usd FROM daily_findings "
        "WHERE day='2026-09-01' AND model='deepseek-v4.1-flash:cloud'"
    ).fetchone()
    shadow = conn.execute(
        "SELECT model, calls, cost_usd FROM daily_findings "
        "WHERE day='2026-09-01' AND model='glm-5.3-flash'"
    ).fetchone()

    assert prod is not None and shadow is not None
    assert prod[1] == 1 and prod[2] == pytest.approx(0.010)
    # The shadow row is retained (its cost is real) but labelled.
    assert shadow[1] == 1 and shadow[2] == pytest.approx(0.999)
    labelled = conn.execute(
        "SELECT is_shadow FROM daily_findings WHERE day='2026-09-01' "
        "AND model='glm-5.3-flash'"
    ).fetchone()
    assert labelled is not None
    # is_shadow lives on the finding so a reader can exclude it.
    assert labelled[0] == 1


def test_rollup_production_spend_excludes_shadow(conn):
    """A total-spend query scoped to production must not include shadow cost."""
    cols = (
        "timestamp, model_used, provider, task_type, complexity_score, "
        "input_tokens, output_tokens, latency_seconds, success, workload_type, "
        "cost_usd, cost_unknown, is_shadow"
    )
    conn.execute(
        f"INSERT INTO router_logs ({cols}) VALUES "
        "('2026-09-02T10:00:00', 'm', 'p', 't', 1.0, 1000, 100, 1.0, 1, 'normal_chat', 5.0, 0, 0),"
        "('2026-09-02T10:00:01', 'm2', 'p', 't', 1.0, 1000, 100, 1.0, 1, 'normal_chat', 7.0, 0, 1)"
    )
    conn.commit()
    rollup.rollup_day(conn, "2026-09-02")

    prod_total = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM daily_findings "
        "WHERE day='2026-09-02' AND is_shadow = 0"
    ).fetchone()[0]
    assert prod_total == pytest.approx(5.0), "shadow spend leaked into production total"


# ── a candidate need not be in production routing ─────────────────────────────

def test_shadow_can_reach_a_candidate_that_is_not_routable(monkeypatch):
    """The candidate under test must NOT have to be in the production fallback
    chain.

    Registering the candidate there to make shadow work would risk the very
    thing shadow exists to avoid: it could start serving real traffic. So an
    experiment may declare its own provider, and shadow resolves the backend
    from that instead of from the routing table.
    """
    monkeypatch.setattr(ep, "discover_backends", lambda: {
        "deepseek-v4.1-flash:cloud": {"provider": "ollama-cloud", "base_url": "http://x", "backend_model": "deepseek-v4.1-flash"},
    })
    monkeypatch.setattr(ep, "load_hermes_config", lambda: {
        "providers": {"ollama-cloud": {"base_url": "https://ollama.com/v1", "api_key_env": "OLLAMA_API_KEY"}},
    })
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

    calls = []

    async def fake_proxy(backend, messages, body):
        calls.append(backend)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr(ep, "proxy_to_backend", fake_proxy)

    # The candidate is nowhere in discover_backends().
    assert "glm-5.3-flash" not in ep.discover_backends()

    exp = _exp(provider="ollama-cloud")
    rows = []
    asyncio.run(
        ep.run_shadow_experiment(
            exp,
            messages=[{"role": "user", "content": "hi"}],
            request_body={},
            request_id="r",
            session_id="s",
            incumbent_model="deepseek-v4.1-flash:cloud",
            on_row=rows.append,
        )
    )

    assert len(calls) == 1, "shadow could not reach an unroutable candidate"
    assert calls[0]["backend_model"] == "glm-5.3-flash"
    assert rows[0]["success"] is True


def test_experiment_declares_optional_provider():
    """provider is optional and parsed without breaking existing configs."""
    from traffic_split import build_experiment

    e = build_experiment({"name": "n", "model": "m", "mode": "shadow"})
    assert e.provider == ""
    e2 = build_experiment({"name": "n", "model": "m", "mode": "shadow", "provider": "ollama-cloud"})
    assert e2.provider == "ollama-cloud"


# ── pricing must be seeded, or cost evidence never accrues ───────────────────


def test_endpoint_connection_seeds_model_pricing():
    """A connection created by the endpoint must be able to price a call.

    Otherwise every row logs cost_unknown=1 forever — including shadow rows,
    which exist precisely to produce cost evidence about a candidate.
    """
    import biggie_llm_endpoint as ep

    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "seed.db")
    c = sqlite3.connect(db)
    rollup.migrate(c)
    c.close()

    saved = (ep._sqlite_conn, ep._sqlite_lock)
    try:
        ep._sqlite_conn = None
        ep._sqlite_lock = None
        real_connect = sqlite3.connect
        sqlite3.connect = lambda *a, **k: real_connect(db)
        try:
            conn2 = ep._get_db_connection()
        finally:
            sqlite3.connect = real_connect

        priced = conn2.execute("SELECT COUNT(*) FROM model_pricing").fetchone()[0]
        assert priced > 0, "endpoint must seed model_pricing on first connection"
    finally:
        ep._sqlite_conn, ep._sqlite_lock = saved


# ── migrated must mean writable ───────────────────────────────────────────────

# ── four bugs found against live traffic ─────────────────────────────────────


def test_shadow_does_not_stamp_the_production_row(monkeypatch):
    """The incumbent row is NOT a shadow observation.

    Live bug: apply_experiment set is_shadow=True for shadow mode, and the
    endpoint copies those fields onto the row it logs for the incumbent's real
    call. A genuine 3.07s production call got stamped is_shadow=1, which both
    inflates the shadow bucket and removes real spend from production totals.
    """
    exp = _exp()
    served, obs = ep.apply_experiment(
        "deepseek-v4.1-flash:cloud", exp,
        workload_type="session_compression", request_id="r1",
    )
    assert served == "deepseek-v4.1-flash:cloud"
    assert obs["experiment_arm"] == "shadow"
    assert obs["is_shadow"] is False, "the incumbent's row is not a shadow call"


def test_shadow_row_keeps_the_real_workload_so_it_can_be_compared(monkeypatch):
    """Shadow evidence must land in the same workload bucket as the incumbent,
    otherwise incumbent-vs-candidate comparison is impossible. Separation is by
    is_shadow, not by inventing a 'shadow' workload."""

    async def fake_proxy(backend, messages, body):
        return {"choices": [{"message": {"content": "summary"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    monkeypatch.setattr(ep, "proxy_to_backend", fake_proxy)
    rows = []
    asyncio.run(ep.run_shadow_experiment(
        _exp(), backend={"provider": "ollama-cloud", "base_url": "http://x"},
        messages=[{"role": "user", "content": "long text"}],
        request_body={"stream": False}, request_id="r1",
        incumbent_model="deepseek-v4.1-flash:cloud",
        workload_type="session_compression", on_row=rows.append,
    ))
    assert rows[0]["workload_type"] == "session_compression"
    assert rows[0]["is_shadow"] is True


def test_shadow_tool_call_is_a_response_not_a_failure(monkeypatch):
    """A candidate that answers with tool_calls has responded.

    Live bug: real compression payloads carry tools, so glm-5.3-flash answered
    finish_reason='tool_calls' with empty content, and the executor recorded
    success=0 / error='shadow_empty'. That is wrong twice over: it calls a
    well-formed response a failure, and it hides WHY there is no scoreable
    summary.
    """

    async def fake_proxy(backend, messages, body):
        return {"choices": [{"message": {"tool_calls": [{"id": "c1"}]}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}}

    monkeypatch.setattr(ep, "proxy_to_backend", fake_proxy)
    rows = []
    asyncio.run(ep.run_shadow_experiment(
        _exp(), backend={"provider": "ollama-cloud", "base_url": "http://x"},
        messages=[{"role": "user", "content": "long text"}],
        request_body={"stream": False}, request_id="r1", on_row=rows.append,
    ))
    row = rows[0]
    assert row["success"] is True, "a tool_calls response is a response"
    assert row["saw_tool_calls"] is True
    assert row["quality_score"] is None, "no summary means nothing to score"


def test_unit_cost_excludes_shadow_spend_by_default():
    """Shadow calls cost real money but are not production spend. A report that
    adds them together overstates what the router actually spends serving.
    """
    tmp = tempfile.mkdtemp()
    c = sqlite3.connect(os.path.join(tmp, "uc.db"))
    rollup.migrate(c)
    for shadow in (0, 1):
        c.execute(
            "INSERT INTO router_logs (timestamp, model_used, workload_type, is_shadow, "
            "cost_usd, cost_unknown, input_tokens, output_tokens, latency_seconds, success) "
            "VALUES ('2026-09-01T00:00:00', 'm', 'session_compression', ?, 1.0, 0, 10, 5, 1.0, 1)",
            (shadow,),
        )
    c.commit()

    import unit_economics

    prod = unit_economics.unit_cost(c, since="2026-09-01")
    assert sum(r.cost_usd for r in prod) == 1.0, "production must exclude shadow spend"
    both = unit_economics.unit_cost(c, since="2026-09-01", include_shadow=True)
    assert sum(r.cost_usd for r in both) == 2.0


def test_optimiser_ignores_unpromoted_shadow_evidence():
    """Ranking production routing must not treat an un-promoted candidate's
    shadow rows as if they were serving production traffic."""
    tmp = tempfile.mkdtemp()
    c = sqlite3.connect(os.path.join(tmp, "opt.db"))
    rollup.migrate(c)
    # A model that only ever ran as a shadow candidate, and one that serves.
    for model, shadow in (("serving", 0), ("cand", 1)):
        c.execute(
            "INSERT INTO daily_findings (day, workload_type, model, call_type, calls, "
            "cost_usd, quality_avg, quality_n, is_shadow) VALUES "
            "('2026-09-01','session_compression',?,'session_compression',100,1.0,0.9,100,?)",
            (model, shadow),
        )
    c.commit()

    import optimiser

    default = optimiser.rank_models(c, "session_compression", days=30)
    assert [r.model for r in default] == ["serving"], "shadow-only model must not rank"
    both = optimiser.rank_models(c, "session_compression", days=30, include_shadow=True)
    assert sorted(r.model for r in both) == ["cand", "serving"]


def test_migrated_db_accepts_the_endpoints_insert(conn, monkeypatch):
    """A DB that has only ever been through rollup.migrate() must accept the
    endpoint's own INSERT.

    Production hid this: the endpoint self-heals missing columns at first write,
    so on the live host "migrated" and "writable" happened to agree. A freshly
    migrated DB disagreed and every logged request failed with
    "table router_logs has no column named ..." while the request itself
    returned 200 — a silent loss of exactly the observability this build exists
    to provide. Pin the two definitions together.
    """
    import inspect

    src = inspect.getsource(ep._log_request_to_db)
    inserted = {c.strip() for c in src.split("INSERT INTO router_logs (", 1)[1].split(")", 1)[0].split(",")}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(router_logs)").fetchall()}
    missing = sorted(c for c in inserted if c and c not in cols)
    assert not missing, f"migrate() produced a schema the endpoint cannot write: {missing}"

    # Prove it by actually writing a shadow row through the real logger.
    monkeypatch.setattr(ep, "_get_db_connection", lambda: conn)
    monkeypatch.setattr(ep, "_get_sqlite_lock", lambda: __import__("threading").Lock())
    ep._log_request_to_db(
        model_used="glm-5.3-flash",
        provider="p",
        task_type="shadow",
        complexity_score=0.0,
        input_tokens=10,
        output_tokens=5,
        latency_seconds=0.1,
        routing_time_ms=0,
        is_shadow=True,
        experiment="shadowtest",
        experiment_arm="shadow",
    )
    written = conn.execute("SELECT COUNT(*) FROM router_logs WHERE is_shadow = 1").fetchone()[0]
    assert written == 1, "the endpoint could not write a row into a migrated DB"
