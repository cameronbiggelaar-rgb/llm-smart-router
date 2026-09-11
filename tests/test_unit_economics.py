"""B2 — unit cost accounting for models, call types, volumes and quality.

Assertions use hand-computed SPECIFIC numbers (not `> 0`) so a broken cost
formula cannot pass.
"""

from __future__ import annotations

import sqlite3
import sys
from decimal import Decimal
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from rollup import migrate  # noqa: E402
from unit_economics import (  # noqa: E402
    Price,
    cost_of_call,
    price_for,
    record_price,
    seed_prices,
    unit_cost,
)


@pytest.fixture()
def db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "router_logs.db"))
    migrate(conn)
    return conn


# ── pricing resolution ────────────────────────────────────────────────────────

def test_cost_of_call_exact_arithmetic(db):
    """100k input @ $1.50/1M + 20k output @ $4.50/1M = $0.24."""
    record_price(db, Price("glm-5.3", "ollama-cloud", Decimal("1.50"), Decimal("4.50"),
                           "2026-09-01", "list"))
    got = cost_of_call("glm-5.3", 100_000, 20_000, conn=db)
    assert got == Decimal("0.15") + Decimal("0.09")  # 0.24
    assert got == Decimal("0.240")


def test_cost_of_call_zero_tokens_is_zero(db):
    record_price(db, Price("glm-5.3", "ollama-cloud", Decimal("1.50"), Decimal("4.50"),
                           "2026-09-01", "list"))
    assert cost_of_call("glm-5.3", 0, 0, conn=db) == Decimal("0")


def test_cost_of_call_unknown_model_is_none(db):
    """Unknown pricing must be distinguishable from free."""
    assert cost_of_call("not-a-model", 1000, 1000, conn=db) is None


def test_free_local_model_is_zero_not_none(db):
    """A registered $0 local model is genuinely free — not 'unknown'."""
    record_price(db, Price("llama3.1:8b", "local", Decimal("0"), Decimal("0"),
                           "2026-09-01", "local"))
    got = cost_of_call("llama3.1:8b", 500_000, 500_000, conn=db)
    assert got == Decimal("0")
    assert got is not None


def test_provider_suffix_is_stripped_for_lookup(db):
    """Live logs use 'deepseek-v4.1-flash:cloud' — pricing is per base model."""
    record_price(db, Price("deepseek-v4.1-flash", "ollama-cloud", Decimal("0.15"),
                           Decimal("0.60"), "2026-09-01", "list"))
    assert cost_of_call("deepseek-v4.1-flash:cloud", 1_000_000, 0, conn=db) == Decimal("0.15")


def test_price_versioning_picks_price_in_force(db):
    """A call logged in Sept prices at Sept rates even after an Oct change."""
    record_price(db, Price("m", "p", Decimal("1.00"), Decimal("2.00"), "2026-09-01", "list"))
    record_price(db, Price("m", "p", Decimal("10.00"), Decimal("20.00"), "2026-10-01", "list"))
    assert cost_of_call("m", 1_000_000, 0, at="2026-09-15", conn=db) == Decimal("1.00")
    assert cost_of_call("m", 1_000_000, 0, at="2026-10-15", conn=db) == Decimal("10.00")
    # before any known price -> unknown, never silently the newest price
    assert cost_of_call("m", 1_000_000, 0, at="2026-08-01", conn=db) is None


def test_price_for_returns_none_for_unpriced(db):
    assert price_for("never-registered", conn=db) is None


# ── seeding from the registry ─────────────────────────────────────────────────

def test_seed_prices_registers_known_models(db):
    n = seed_prices(db)
    assert n > 0
    # glm-5.3-flash is registered at 0.15/0.50 in MODEL_REGISTRY
    assert cost_of_call("glm-5.3-flash", 1_000_000, 1_000_000, conn=db) == Decimal("0.65")


def test_seed_prices_is_idempotent(db):
    first = seed_prices(db)
    second = seed_prices(db)
    assert first > 0
    assert second == 0


# ── unit cost rollup ──────────────────────────────────────────────────────────

def _log(conn, model, workload, in_tok, out_tok, cost, calls=1, quality=None, success=1):
    for _ in range(calls):
        conn.execute(
            "INSERT INTO router_logs (timestamp, session_id, model_used, workload_type, "
            "input_tokens, output_tokens, cost_usd, cost_unknown, success, quality_score) "
            "VALUES ('2026-09-01T10:00:00+00:00','s',?,?,?,?,?,0,?,?)",
            (model, workload, in_tok, out_tok, cost, success, quality),
        )
    conn.commit()


def test_unit_cost_aggregates_per_model_and_call_type(db):
    _log(db, "glm-5.3", "session_compression", 1000, 100, 0.5, calls=3)
    _log(db, "glm-5.3", "normal_chat", 200, 50, 0.1, calls=2)
    rows = unit_cost(db, since="2026-09-01", until="2026-09-02")
    by = {(r.model, r.call_type): r for r in rows}
    comp = by[("glm-5.3", "session_compression")]
    assert comp.calls == 3
    assert comp.input_tokens == 3000
    assert comp.output_tokens == 300
    assert comp.cost_usd == pytest.approx(1.5)          # 3 * 0.5
    assert comp.cost_per_call == pytest.approx(0.5)     # 1.5 / 3
    # per 1k input tokens: 1.5 / (3000/1000) = 0.5
    assert comp.cost_per_1k_in == pytest.approx(0.5)


def test_unit_cost_cost_per_quality_point(db):
    """cost_per_quality_point = cost_per_call / quality_avg."""
    _log(db, "glm-5.3", "session_compression", 1000, 100, 0.4, calls=2, quality=0.8)
    rows = unit_cost(db, since="2026-09-01", until="2026-09-02")
    r = [x for x in rows if x.model == "glm-5.3"][0]
    assert r.calls == 2
    assert r.quality_n == 2
    assert r.quality_avg == pytest.approx(0.8)
    assert r.cost_per_call == pytest.approx(0.4)
    assert r.cost_per_quality_point == pytest.approx(0.5)   # 0.4 / 0.8


def test_unit_cost_ignores_unmeasured_quality(db):
    """NULL quality means 'not measured' and must not drag the average to 0."""
    _log(db, "a-model", "normal_chat", 100, 10, 0.2, calls=1, quality=None)
    _log(db, "a-model", "normal_chat", 100, 10, 0.2, calls=1, quality=1.0)
    rows = [r for r in unit_cost(db, since="2026-09-01", until="2026-09-02") if r.model == "a-model"]
    r = rows[0]
    assert r.calls == 2
    assert r.quality_n == 1                 # only one measured call
    assert r.quality_avg == pytest.approx(1.0)  # the NULL one excluded


def test_unit_cost_reports_unaccounted_calls(db):
    """Calls with unknown pricing are surfaced, not silently free."""
    db.execute(
        "INSERT INTO router_logs (timestamp, session_id, model_used, workload_type, "
        "input_tokens, output_tokens, cost_usd, cost_unknown) "
        "VALUES ('2026-09-01T10:00:00+00:00','s','mystery-model','normal_chat',100,10,0,1)"
    )
    db.commit()
    rows = [r for r in unit_cost(db, since="2026-09-01", until="2026-09-02")
            if r.model == "mystery-model"]
    assert rows[0].cost_unknown_calls == 1
    assert rows[0].cost_usd == 0.0


def test_unit_cost_respects_window(db):
    _log(db, "m1", "normal_chat", 100, 10, 1.0, calls=1)
    db.execute(
        "INSERT INTO router_logs (timestamp, session_id, model_used, workload_type, "
        "input_tokens, output_tokens, cost_usd) "
        "VALUES ('2026-09-05T10:00:00+00:00','s','m2','normal_chat',100,10,99.0)"
    )
    db.commit()
    rows = unit_cost(db, since="2026-09-01", until="2026-09-02")
    models = {r.model for r in rows}
    assert "m1" in models
    assert "m2" not in models          # outside the window


def test_unit_cost_empty_window_returns_empty(db):
    assert unit_cost(db, since="2030-01-01", until="2030-01-02") == []


def test_cost_is_decimal_not_float(db):
    """Money must not be a float — 3 x $0.10 must be exactly $0.30."""
    record_price(db, Price("m", "p", Decimal("0.10"), Decimal("0"), "2026-09-01", "list"))
    total = cost_of_call("m", 1_000_000, 0, conn=db) * 3
    assert total == Decimal("0.30")
    assert isinstance(total, Decimal)
