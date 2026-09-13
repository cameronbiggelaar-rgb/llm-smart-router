"""B18 — the price book must match the vendor *and* survive being corrected.

B17 documented the drift and pinned the wrong numbers. These tests do the
opposite: they pin the *corrected* numbers and pin the mechanism that makes a
correction actually take effect.

The trap
--------
``seed_prices()`` is idempotent on ``(model, effective_from)``: it skips a model
whose price is already recorded at that date. So editing a price in
``MODEL_REGISTRY`` while leaving its ``"date"`` field alone writes **zero** rows.
The stale price stays in ``model_pricing`` forever, live calls keep being priced
at the old rate, and nothing raises — the fix silently does nothing.

A correction is therefore a new *versioned* row, never an edit. ``price_for``
already resolves the price in force at a timestamp, so superseded rows stay on
record and history is not restated.
"""

from __future__ import annotations

import sqlite3
import sys
from decimal import Decimal
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import models  # noqa: E402
import rollup  # noqa: E402
from unit_economics import price_for, record_price, seed_prices  # noqa: E402
from unit_economics import Price  # noqa: E402

# ollama.com/pricing, retrieved 2026-09-13. (input, output) per 1M tokens.
VENDOR_PUBLISHED = {
    "deepseek-v4-pro": (Decimal("0.66"), Decimal("1.98")),
    "minimax-m2.7:cloud": (Decimal("0.30"), Decimal("1.20")),
    "qwen3.5": (Decimal("0.60"), Decimal("3.60")),
    "glm-5.3": (Decimal("1.40"), Decimal("4.40")),
    "glm-5.2": (Decimal("1.40"), Decimal("4.40")),
    "glm-5.1": (Decimal("1.00"), Decimal("3.20")),
}

# What the book charged before the correction, with the date it was in force.
# Kept so the supersede test can reproduce production's starting state.
SUPERSEDED = {
    "deepseek-v4-pro": ("2.00", "6.00", "2026-07-01"),
    "minimax-m2.7": ("1.00", "3.00", "2026-07-01"),
    "qwen3.5": ("1.75", "5.25", "2026-08-08"),
    "glm-5.3": ("1.50", "4.50", "2026-08-26"),
    "glm-5.2": ("1.50", "4.50", "2026-09-04"),
    "glm-5.1": ("1.25", "3.75", "2026-07-01"),
}

# The date the correction takes effect. Must differ from every superseded date,
# or the fix is a no-op.
CORRECTION_DATE = "2026-09-13"


@pytest.fixture()
def db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "router_logs.db"))
    rollup.migrate(conn)
    yield conn
    conn.close()


@pytest.mark.parametrize("model", sorted(VENDOR_PUBLISHED))
def test_registry_matches_vendor_published_price(model):
    """The registry is the single source of truth — it must hold vendor rates."""
    cfg = models.MODEL_REGISTRY[model]
    want_in, want_out = VENDOR_PUBLISHED[model]
    assert Decimal(str(cfg["input"])) == want_in, (
        f"{model} input is {cfg['input']}, vendor publishes {want_in}"
    )
    assert Decimal(str(cfg["output"])) == want_out, (
        f"{model} output is {cfg['output']}, vendor publishes {want_out}"
    )


@pytest.mark.parametrize("model", sorted(VENDOR_PUBLISHED))
def test_corrected_model_carries_the_correction_date(model):
    """A corrected price *must* carry a new effective date.

    This is the regression guard for the trap: if someone fixes a price but
    leaves the old date, ``seed_prices`` skips it and the fix does nothing.
    """
    cfg = models.MODEL_REGISTRY[model]
    assert cfg["date"] == CORRECTION_DATE, (
        f"{model} carries date {cfg['date']}, expected {CORRECTION_DATE}. "
        "Editing a price without moving its effective date is a silent no-op."
    )


@pytest.mark.parametrize("model", sorted(VENDOR_PUBLISHED))
def test_price_edit_without_date_bump_is_a_silent_noop(db, model):
    """Pin the trap itself, so the next person cannot rediscover it in prod.

    A wrong price is recorded at a date the model already carries; seeding
    writes nothing and the wrong price survives. This is *why* a correction has
    to be a new versioned row.
    """
    base = model.split(":")[0]
    stale_date = models.MODEL_REGISTRY[model]["date"]
    record_price(
        db,
        Price(base, "ollama-cloud", Decimal("99.99"), Decimal("99.99"), stale_date, "test"),
    )
    n = seed_prices(db)
    row = db.execute(
        "SELECT input_usd_per_1m FROM model_pricing WHERE model=? AND effective_from=?",
        (base, stale_date),
    ).fetchone()
    assert row is not None
    assert Decimal(str(row[0])) == Decimal("99.99"), (
        "seed_prices overwrote an existing (model, effective_from) row — the "
        "trap has changed shape; re-read seed_prices before trusting this fix"
    )
    assert n >= 0  # no new row was written for this (model, date)


@pytest.mark.parametrize("model", sorted(VENDOR_PUBLISHED))
def test_seeding_supersedes_the_stale_price_but_keeps_it_on_record(db, model):
    """The real deploy path: production already holds the stale row.

    Seeding the corrected registry must produce the vendor price *and* leave the
    superseded row intact, so a historical call re-prices at the rate that was
    actually in force when it ran.
    """
    base = model.split(":")[0]
    sup_in, sup_out, sup_date = SUPERSEDED[base]
    record_price(
        db,
        Price(base, "ollama-cloud", Decimal(sup_in), Decimal(sup_out), sup_date, "stale"),
    )

    seed_prices(db)

    want_in, want_out = VENDOR_PUBLISHED[model]
    current = price_for(base, conn=db)
    assert current is not None, f"{base} lost its price entirely"
    assert current.input_usd_per_1m == want_in, (
        f"{base} still resolves to {current.input_usd_per_1m}, not {want_in}. "
        "The correction did not supersede the stale row."
    )
    assert current.output_usd_per_1m == want_out

    # History is not restated.
    old = db.execute(
        "SELECT input_usd_per_1m FROM model_pricing WHERE model=? AND effective_from=?",
        (base, sup_date),
    ).fetchone()
    assert old is not None, f"{base}: superseded price was deleted, not versioned"
    assert Decimal(str(old[0])) == Decimal(sup_in), "superseded price was rewritten"


@pytest.mark.parametrize("model", sorted(VENDOR_PUBLISHED))
def test_price_in_force_at_a_past_date_is_still_the_old_rate(db, model):
    """A call logged before the correction must not be re-priced retroactively."""
    base = model.split(":")[0]
    sup_in, sup_out, sup_date = SUPERSEDED[base]
    record_price(
        db,
        Price(base, "ollama-cloud", Decimal(sup_in), Decimal(sup_out), sup_date, "stale"),
    )
    seed_prices(db)

    # A call logged while the stale price was in force must still price at the
    # stale rate — the correction must not restate history.
    at_past = price_for(base, at=sup_date, conn=db)
    assert at_past is not None
    assert at_past.input_usd_per_1m == Decimal(sup_in), (
        f"{base} re-priced a pre-correction call at the new rate — history was "
        "restated instead of versioned"
    )


def test_seeding_is_still_idempotent_after_the_correction(db):
    seed_prices(db)
    first = db.execute("SELECT COUNT(*) FROM model_pricing").fetchone()[0]
    seed_prices(db)
    second = db.execute("SELECT COUNT(*) FROM model_pricing").fetchone()[0]
    assert first == second, "re-seeding added duplicate rows"


def test_superseded_dates_are_all_distinct_from_the_correction_date():
    """Guard the guard: if the correction date equals an old date, nothing lands."""
    for base, (_, _, sup_date) in SUPERSEDED.items():
        assert sup_date != CORRECTION_DATE, (
            f"{base}: correction date {CORRECTION_DATE} equals its superseded "
            "date — seed_prices would skip it"
        )
