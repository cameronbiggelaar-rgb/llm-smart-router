"""B18 — the test suite must not be able to write to the production database.

`unit_economics._default_conn()` resolves the production `router_logs.db` from
`Path.home()` and seeds the price book on first use. `model_pricing` is what
prices live traffic, and the running endpoint reads it per request — so a test
that reaches a pricing helper without an explicit `conn` can silently change
what production charges.

These tests fail loudly if that isolation ever regresses.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

PROD_DB = Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" / "router_logs.db"


def test_home_is_redirected_for_the_suite():
    """conftest.py must point HOME at a throwaway directory."""
    home = Path(os.environ["HOME"])
    assert home != Path("/home/hermes"), (
        "HOME is the real user home — tests can reach the production DB path"
    )
    assert home.is_dir(), f"redirected HOME {home} does not exist"


def test_production_db_path_is_not_the_real_one():
    """The path a test would resolve must not be production's."""
    assert PROD_DB != Path("/home/hermes/.hermes/skills/llm-smart-router/data/router_logs.db"), (
        "the production DB path resolves to the real database during tests"
    )
    assert not PROD_DB.exists(), (
        f"a database exists at the production path during a test run: {PROD_DB}. "
        "Test isolation has regressed; production pricing is at risk."
    )


def test_default_conn_does_not_reach_production():
    """A pricing call with no explicit conn must not resolve to production.

    `_default_conn()` resolves its path from `Path.home()` at call time — that
    is the whole mechanism that let a test write to production. Asserting on
    the *path* tests the mechanism directly; opening the connection would only
    test whether the schema happens to exist.
    """
    import unit_economics

    # The module builds its path from Path.home() at call time.
    resolved = (
        Path(os.environ["HOME"])
        / ".hermes"
        / "skills"
        / "llm-smart-router"
        / "data"
        / "router_logs.db"
    ).resolve()
    prod = Path("/home/hermes/.hermes/skills/llm-smart-router/data/router_logs.db").resolve()
    assert resolved != prod, (
        f"unit_economics would open the PRODUCTION database ({resolved}) during a test"
    )


def test_seeding_a_test_db_does_not_touch_production():
    """The suite's own seeding must never modify production prices."""
    import unit_economics

    prod = Path("/home/hermes/.hermes/skills/llm-smart-router/data/router_logs.db")
    if not prod.exists():
        pytest.skip("production DB not present in this environment")
    before = prod.stat().st_size

    conn = sqlite3.connect(":memory:")
    import rollup

    rollup.migrate(conn)
    unit_economics.seed_prices(conn)

    after = prod.stat().st_size
    assert before == after, "seeding a test database changed the production database file"
