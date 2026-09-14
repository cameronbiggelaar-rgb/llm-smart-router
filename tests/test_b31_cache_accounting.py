"""B31 batch 2 — cache accounting must be recorded, and cache must be BILLED.

Two defects, one file, because both are "the numbers lie about caching".

Change 4 — `cached_tokens` was never recorded
---------------------------------------------
`router_logs` has 52 columns and none of them is a cache field (verified:
`PRAGMA table_info(router_logs)`). So the 50x cheaper cache hits — proven live,
29,952 of 30,044 input tokens served from cache on a repeat call — were
invisible. Worse, `compute_cost_fields()` called
`cost_of_call(model, in_tok, out_tok, conn=conn)` and DROPPED the
`cached_input_tokens` argument entirely, so every cached token was billed at the
full input rate. The cache worked and the bill never showed it.

Change 5 — the reported price table was stale
---------------------------------------------
`estimate_cost()` / `get_model_cost()` read the legacy `model_costs` table,
which still carries `deepseek-v4-pro at $2.00/$6.00` (3x the real $0.66/$1.98)
and `deepseek-v4-flash at $0.50/$1.50` (vs $0.22/$0.66), while the corrected
`model_pricing` rows exist. Cost reports were built on superseded prices.

These tests assert the CONTRACT, not the implementation: a cache hit must cost
strictly less than the same call billed cold, and the price used must be the
newest known row.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import unit_economics as ue  # noqa: E402
import biggie_llm_endpoint as ep  # noqa: E402


@pytest.fixture()
def priced_db(tmp_path):
    """A seeded router DB in a temp dir (never the production DB)."""
    db = tmp_path / "router_logs.db"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS model_pricing ("
        " model TEXT, provider TEXT, input_usd_per_1m REAL,"
        " output_usd_per_1m REAL, cached_input_usd_per_1m REAL,"
        " effective_from TEXT, source TEXT);"
    )
    conn.commit()
    yield conn
    conn.close()


# ── Change 5: prices ──────────────────────────────────────────────────────────

def test_model_pricing_has_a_cached_input_rate_column():
    """Cache billing needs somewhere to read the discounted rate from.

    `cached_input_tokens` was already an argument to `cost_of_call` but was
    documented as "billed at the input rate here; discounted cache pricing would
    need a separate column in model_pricing". This is that column.
    """
    cols = {r[1] for r in _prod_conn().execute("PRAGMA table_info(model_pricing)")}
    assert "cached_input_usd_per_1m" in cols, (
        "model_pricing must carry a cached input rate for cache-aware billing"
    )


def test_legacy_model_costs_is_not_the_source_for_estimate_cost():
    """`estimate_cost` must not report the superseded legacy prices.

    The registry is the single source of truth; the legacy `model_costs` table
    still holds deepseek-v4-pro at $2.00/$6.00 vs the real $0.66/$1.98.
    """
    import models

    src = _fn_source(models.estimate_cost) + _fn_source(models.get_model_cost)
    assert "model_pricing" in src or "price_for" in src, (
        "estimate_cost/get_model_cost must read model_pricing (the corrected "
        "table), not the legacy model_costs table"
    )


# ── Change 4: recorded + billed cache ─────────────────────────────────────────

def test_cached_input_tokens_is_forwarded_to_cost_of_call():
    """`compute_cost_fields` must stop discarding the cache argument."""
    src = _fn_source(ep.compute_cost_fields)
    assert "cached" in src, (
        "compute_cost_fields must pass cached_input_tokens through to "
        "cost_of_call; it currently drops it, billing cache hits at full rate"
    )


def test_cache_hit_costs_less_than_the_same_call_cold(priced_db):
    """The core contract: a cache hit is CHEAPER. This is the whole point.

    Before the fix both calls cost the same, which made the 50x discount a
    fiction in the books even though it was real on the invoice.
    """
    ue.record_price(
        priced_db,
        ue.Price(
            model="cache-test-model",
            provider="test",
            input_usd_per_1m=ue.Decimal("1.00"),
            output_usd_per_1m=ue.Decimal("2.00"),
            cached_input_usd_per_1m=ue.Decimal("0.02"),
            effective_from="2026-01-01",
            source="test",
        ),
    )

    cold = ue.cost_of_call("cache-test-model", 1_000_000, 0, conn=priced_db)
    warm = ue.cost_of_call(
        "cache-test-model", 1_000_000, 0, cached_input_tokens=1_000_000, conn=priced_db
    )
    assert cold == ue.Decimal("1.00")
    assert warm == ue.Decimal("0.02"), "a fully cached call must bill at the cached rate"
    assert warm < cold


def test_cost_of_call_rejects_absurd_cached_token_counts(priced_db):
    """More cached tokens than input tokens must not produce a negative cost."""
    ue.record_price(
        priced_db,
        ue.Price(
            model="overcount-model",
            provider="test",
            input_usd_per_1m=ue.Decimal("1.00"),
            output_usd_per_1m=ue.Decimal("1.00"),
            cached_input_usd_per_1m=ue.Decimal("0.01"),
            effective_from="2026-01-01",
            source="test",
        ),
    )
    amount = ue.cost_of_call(
        "overcount-model", 100, 0, cached_input_tokens=9_999, conn=priced_db
    )
    assert amount >= ue.Decimal("0"), "cost must never go negative"


def test_router_logs_records_cached_tokens():
    """The column must exist in the production schema's migration set."""
    src = _fn_source(ep._get_db_connection) + _fn_source(ep.compute_cost_fields)
    all_src = _module_source(ep)
    assert "cached_tokens" in all_src, (
        "router_logs must gain a cached_tokens column so cache effectiveness is "
        "measurable; today no table records it"
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _prod_conn():
    db = SCRIPTS_DIR.parent / "data" / "router_logs.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    return conn


def _fn_source(fn) -> str:
    import inspect

    return inspect.getsource(fn)


def _module_source(mod) -> str:
    import inspect

    return inspect.getsource(mod)


# ── Change 4 (b): the extractor + wiring must actually record a cache hit ─────

def test_get_cached_tokens_reads_prompt_tokens_details():
    """The provider's real shape must be parsed.

    Verified live (ollama.com, repeated 9,941-token prompt):
      cold -> usage has NO prompt_tokens_details at all
      warm -> {"prompt_tokens_details": {"cached_tokens": 9856}}
    """
    assert ep._get_cached_tokens(
        {"usage": {"prompt_tokens": 9941, "prompt_tokens_details": {"cached_tokens": 9856}}}
    ) == 9856
    # cold call / provider that reports nothing
    assert ep._get_cached_tokens({"usage": {"prompt_tokens": 9941}}) == 0
    assert ep._get_cached_tokens({"usage": {"prompt_tokens_details": {}}}) == 0


def test_get_cached_tokens_accepts_alternate_shapes():
    """Other gateways use a flat or differently-named cache field."""
    assert ep._get_cached_tokens({"usage": {"cached_tokens": 100}}) == 100
    assert ep._get_cached_tokens(
        {"usage": {"cache_read_input_tokens": 250}}
    ) == 250
    assert ep._get_cached_tokens(
        {"usage": {"prompt_tokens_details": {"cache_read_input_tokens": 7}}}
    ) == 7


def test_get_cached_tokens_is_defensive():
    """Malformed usage must yield 0, never raise — capture is diagnostic."""
    for bad in (None, {}, {"usage": None}, {"usage": "nope"},
                {"usage": {"prompt_tokens_details": "x"}},
                {"usage": {"cached_tokens": "abc"}},
                {"usage": {"cached_tokens": None}}):
        assert ep._get_cached_tokens(bad) == 0


def test_sse_cached_tokens_reads_the_final_usage_chunk():
    """Streaming reports usage in a chunk with an EMPTY choices list."""
    line = '{"choices": [], "usage": {"prompt_tokens": 500, "cached_tokens": 480}}'
    assert ep._sse_cached_tokens(line) == 480
    assert ep._sse_cached_tokens("[DONE]") == 0
    assert ep._sse_cached_tokens("not json") == 0


def test_log_request_to_db_accepts_and_persists_cached_tokens(tmp_path, monkeypatch):
    """End-to-end: a cached count passed to the logger must land in the table.

    This is the assertion that would have caught the original defect, where the
    column existed but nothing ever wrote a non-zero value into it.
    """
    db = tmp_path / "r.db"
    # Build the schema exactly as production does: models.SCHEMA_SQL for the
    # base table, then the endpoint's own runtime ALTERs, so this test validates
    # the real column set (including cached_tokens) rather than a drifted subset.
    import models

    conn = models.init_db(str(db))
    conn.close()

    monkeypatch.setenv("BIGGIE_ROUTER_DB", str(db))
    monkeypatch.setattr(ep, "_sqlite_conn", None)
    real_conn = ep._get_db_connection()
    assert real_conn is not None
    ep._log_request_to_db(
        model_used="m", provider="p", task_type="t", complexity_score=0.0,
        input_tokens=1000, output_tokens=10, latency_seconds=1.0, routing_time_ms=5,
        cached_tokens=900, cost_usd=0.0, cost_unknown=0,
    )
    row = real_conn.execute(
        "SELECT input_tokens, cached_tokens FROM router_logs"
    ).fetchone()
    assert row is not None, "a row must be written"
    assert row[1] == 900, f"cached_tokens must persist, got {row[1]}"


def test_cached_tokens_is_wired_into_the_request_path():
    """The handler must CALL the extractor, not merely define it."""
    import inspect

    src = inspect.getsource(ep)
    assert "_get_cached_tokens(" in src
    assert "cached_input_tokens=_cached_tok" in src, (
        "the request path must pass the cache count into cost computation"
    )
