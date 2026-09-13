"""Test isolation guard.

Why this exists
---------------
`scripts/unit_economics.py::_default_conn()` resolves the production database
via `Path.home() / ".hermes" / "skills" / "llm-smart-router" / "data" /
"router_logs.db"` and *seeds the price book* on first use. Any test that
reaches a pricing helper without passing an explicit `conn` therefore opens the
real production database and writes to it.

That matters because `model_pricing` is what prices live traffic: a test run
could silently change the rates the running endpoint charges, and the running
endpoint reads `model_pricing` per request. A test mutating production pricing
is exactly the class of defect the price-book work exists to prevent, so the
suite must not be able to do it.

Redirecting `HOME` is sufficient and is the least invasive fix: every one of
those paths is derived from `Path.home()` at call time, so pointing `HOME` at a
temporary directory for the duration of the test session makes the production
path unreachable without patching any module internals.

`tests/test_prod_db_isolation.py` asserts this guard actually holds.
"""

from __future__ import annotations

import os
import tempfile

# Must happen at import time — before test modules import the router/endpoint
# modules — so no module-level path resolution can capture the real home first.
_TEST_HOME = tempfile.mkdtemp(prefix="hermes-test-home-")
os.environ["HOME"] = _TEST_HOME
