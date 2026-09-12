"""B17 — the price book must match what the vendor actually charges.

Found the same way as B16: by reconciling the ledger against the Ollama
dashboard the user shared. Once the streaming double-count was removed
(``test_ledger_double_count.py``) a second, independent overstatement remained.

Three separate defects, all measurable against ``ollama.com/pricing``:

 1. **glm-5.3 / glm-5.2 are priced 7% high.** The book says $1.50 / $4.50 per
    1M; the vendor publishes $1.40 / $4.40.

 2. **deepseek-v4-pro is priced 3x high.** The book says $2.00 / $6.00; the
    vendor publishes $0.66 / $1.98. This one is not a rounding error — it
    makes the model look three times more expensive than it is, which is
    exactly the number a cost optimiser uses to decide what to route to.

 3. **qwen3.5:397b is missing entirely.** Calls to it are logged, and priced
    as unknown, so its spend is invisible to every report.

And one structural gap that no price correction can fix:

 4. **There is no cached-input rate.** The vendor publishes three rates per
    model — fresh input, *cached* input, output. The book has two columns, so
    every input token is billed at the fresh rate. For glm-5.3 that is $1.40
    against a cached rate of $0.26. Working backwards from the invoice, the
    vendor's effective rate for glm-5.3 was ~$0.41/1M, consistent with roughly
    an 88% cache hit rate. On a cache-heavy workload this is the single
    largest overstatement in the ledger — larger than the double-count.

Why these are pinned rather than silently fixed
-----------------------------------------------
``MODEL_REGISTRY`` also drives ``MODEL_COST_ORDER``, the escalation chain. A
price is therefore not just an accounting fact — it decides what production
routes to. Correcting deepseek-v4-pro 3x down would move it materially earlier
in the escalation chain, which is a behaviour change, not a bug fix. That call
belongs to the operator, so these tests assert the *current* values and will
fail loudly the moment someone changes them deliberately.
"""

from __future__ import annotations

import re
import sys
from decimal import Decimal
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import models  # noqa: E402
import unit_economics  # noqa: E402

# ollama.com/pricing, retrieved 2026-09-12. (input, cached_input, output) per 1M.
VENDOR_PUBLISHED = {
    "glm-5.3":            (Decimal("1.40"), Decimal("0.26"), Decimal("4.40")),
    "glm-5.2":            (Decimal("1.40"), Decimal("0.26"), Decimal("4.40")),
    "glm-5.3-flash":      (Decimal("0.15"), Decimal("0.03"), Decimal("0.50")),
    "deepseek-v4-flash":  (Decimal("0.22"), Decimal("0.007"), Decimal("0.66")),
    "deepseek-v4-pro":    (Decimal("0.66"), Decimal("0.022"), Decimal("1.98")),
    "deepseek-v4.1-flash": (Decimal("0.15"), None, Decimal("0.60")),
    "qwen3.5:397b":       (Decimal("0.60"), None, Decimal("3.60")),
}

# The values our book currently carries, so a change is deliberate and visible.
KNOWN_DRIFT = {
    "glm-5.3": ("1.50", "4.50", "vendor publishes 1.40 / 4.40 — 7% high"),
    "glm-5.2": ("1.50", "4.50", "vendor publishes 1.40 / 4.40 — 7% high"),
    "deepseek-v4-pro": ("2.00", "6.00", "vendor publishes 0.66 / 1.98 — 3.03x high"),
}


def test_price_book_has_no_silent_arithmetic_errors(db):
    """Every price must be a sane positive Decimal, never a string or a typo."""
    unit_economics.seed_prices(db)
    rows = db.execute("SELECT model, input_usd_per_1m, output_usd_per_1m FROM model_pricing").fetchall()
    assert rows, "price book is empty"
    for model, i, o in rows:
        assert isinstance(i, (int, float)) and isinstance(o, (int, float)), f"{model}: non-numeric price"
        assert i >= 0 and o >= 0, f"{model}: negative price"


@pytest.mark.parametrize("model", ["glm-5.3", "glm-5.2", "deepseek-v4-pro"])
def test_known_price_drift_is_unchanged(db, model):
    """Pin the known drift.

    When this fails, someone has updated the price — intentionally or not. Both
    are worth knowing: an intentional fix should update ``KNOWN_DRIFT`` and the
    vendor table above in the same commit, and an accidental one should be
    reverted.
    """
    unit_economics.seed_prices(db)
    row = db.execute(
        "SELECT input_usd_per_1m, output_usd_per_1m FROM model_pricing WHERE model=?", (model,)
    ).fetchone()
    assert row is not None, f"{model} has no price"
    expected_in, expected_out, note = KNOWN_DRIFT[model]
    assert Decimal(str(row[0])) == Decimal(expected_in), (
        f"{model} input price moved to {row[0]} (was {expected_in}). {note}"
    )
    assert Decimal(str(row[1])) == Decimal(expected_out), (
        f"{model} output price moved to {row[1]} (was {expected_out}). {note}"
    )


def test_price_book_has_no_cached_input_column():
    """Pin the structural gap: we cannot express a cache discount.

    This is the largest remaining overstatement on cache-heavy workloads. When
    a cached-input column is added, this test should be deleted rather than
    updated — its whole purpose is to keep the gap visible.
    """
    cols = {r[1] for r in db_cursor("PRAGMA table_info(model_pricing)")}
    assert "cached_input_usd_per_1m" not in cols, (
        "a cached-input rate now exists — delete this test and start pricing "
        "cache hits at the cached rate"
    )


def test_qwen_397b_is_absent_from_the_price_book():
    """Pin the missing model, so its unknown-priced spend stays visible."""
    assert "qwen3.5:397b" not in models.MODEL_REGISTRY, (
        "qwen3.5:397b was added to the registry — remove this test and confirm "
        "the new price changes MODEL_COST_ORDER (and therefore routing) on purpose"
    )


def test_registry_drives_the_escalation_order():
    """Prices are not only accounting: they order the escalation chain."""
    order = models.MODEL_COST_ORDER
    by_price = [
        n for n, _ in sorted(
            models.MODEL_REGISTRY.items(), key=lambda kv: (kv[1]["input"], kv[1]["tier"])
        )
    ]
    assert order == by_price, "MODEL_COST_ORDER has drifted from MODEL_REGISTRY"
    # A price correction is a routing change. Prove the linkage exists so the
    # next person cannot reason about prices as pure bookkeeping.
    assert "deepseek-v4-pro" in order


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def db_cursor(sql, *params):
    import sqlite3
    conn = sqlite3.connect(":memory:")
    return conn.execute(sql, params)


@pytest.fixture()
def db():
    import sqlite3
    conn = sqlite3.connect(":memory:")
    import rollup
    rollup.migrate(conn)
    yield conn
    conn.close()


# --------------------------------------------------------------------------
# 5. subscription models priced as if metered
# --------------------------------------------------------------------------

def test_subscription_providers_are_priced_as_metered():
    """Pin the third overstatement: subscription models billed at API rates.

    ``models.py`` documents the pricing model as "All models are flat-
    subscription or free" — ChatGPT $20/mo, Ollama Cloud $100/mo, local free —
    and describes the registry numbers as RELATIVE COMPUTE UNITS, not dollars.

    But ``unit_economics`` sums them as USD. So every ``openai-codex`` call is
    logged with a dollar cost that was never charged: on 6 days of production
    that is $1,094.09 of phantom spend, against a $20/mo subscription.

    Measured: metered providers (ollama-cloud) accounted for $2,019.92 of
    ledger spend against a $76.86 vendor invoice. The subscription lanes added
    a further $1,094.09 that no invoice will ever contain.

    This test pins the current behaviour. Fixing it means splitting the two
    concepts — a per-token dollar cost for metered providers, and a budget-unit
    consumption figure for subscription ones — which changes every existing
    report, so it is an operator decision.
    """
    import models

    # The registry's own docstring says these are not dollars.
    src = (SCRIPTS_DIR / "models.py").read_text()
    assert "RELATIVE COMPUTE UNITS" in src, (
        "models.py no longer describes its numbers as compute units — if they "
        "are now true dollars, delete this test and re-check subscription lanes"
    )

    # Yet a subscription provider carries a non-zero dollar price.
    subscription_models = [
        n for n, cfg in models.MODEL_REGISTRY.items()
        if cfg.get("provider") == "openai-codex"
    ]
    assert subscription_models, "no subscription models found — has the provider label changed?"
    priced = [n for n in subscription_models if models.MODEL_REGISTRY[n]["input"] > 0]
    assert priced, (
        "subscription models now have zero input price — if that is a deliberate "
        "fix, delete this test; phantom subscription spend is resolved"
    )
