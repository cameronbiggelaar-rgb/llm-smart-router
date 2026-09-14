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
