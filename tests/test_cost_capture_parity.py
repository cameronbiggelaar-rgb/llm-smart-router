"""B12 — cost capture must be impossible to forget at a call site.

Real defect, found by ``scripts/preflight.py`` the first time it ran: prices
were seeded and the price book was correct, yet a logged call still landed with
``cost_unknown=1``. The cause was not pricing — it was that
``_log_request_to_db`` *defaults* ``cost_unknown=1`` and never computes the cost
itself. The caller is expected to pass it.

Four of the five real call sites in the endpoint do not pass cost fields. On
production that meant 952 recent ``session_compression`` calls — 68.4M tokens,
the highest-volume workload in the system — were logged cost-blind. The
endpoint was healthy and the money was invisible.

That is a footgun, not a mistake: the safe behaviour must be the default one.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import biggie_llm_endpoint as ep  # noqa: E402


@pytest.fixture()
def fresh_db(monkeypatch, tmp_path):
    monkeypatch.setenv("BIGGIE_ROUTER_DB", str(tmp_path / "cost.db"))
    monkeypatch.setattr(ep, "_sqlite_conn", None, raising=False)
    conn = ep._get_db_connection()
    yield conn
    monkeypatch.setattr(ep, "_sqlite_conn", None, raising=False)


def _basic(**over):
    kw = dict(
        model_used="deepseek-v4.1-flash",
        provider="deepseek",
        task_type="test",
        complexity_score=0.5,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        latency_seconds=1.0,
        routing_time_ms=5,
        workload_type="session_compression",
    )
    kw.update(over)
    return kw


def test_logger_computes_cost_when_caller_omits_it(fresh_db):
    """The default path must be priced. Omitting cost must not blind the ledger."""
    ep._log_request_to_db(**_basic())
    row = fresh_db.execute(
        "SELECT cost_usd, cost_unknown FROM router_logs WHERE task_type='test'"
    ).fetchone()
    assert row is not None
    cost_usd, cost_unknown = row
    assert cost_unknown == 0, "an unpriced-by-omission call must not be 'unknown'"
    # deepseek-v4.1-flash: $0.15 in / $0.60 out per 1M -> 0.75 for 1M/1M
    assert abs(cost_usd - 0.75) < 0.01, f"expected ~0.75, got {cost_usd}"


def test_provider_suffixed_model_is_priced(fresh_db):
    """':cloud' suffixed ids are what production actually logs.

    A suffix mismatch on the price lookup would silently un-price the busiest
    workload, so pin it explicitly.
    """
    ep._log_request_to_db(**_basic(model_used="deepseek-v4.1-flash:cloud"))
    cost_usd, cost_unknown = fresh_db.execute(
        "SELECT cost_usd, cost_unknown FROM router_logs WHERE task_type='test'"
    ).fetchone()
    assert cost_unknown == 0
    assert cost_usd > 0


def test_explicit_cost_still_wins(fresh_db):
    """A caller that knows the true cost (e.g. a shadow candidate) is respected."""
    ep._log_request_to_db(**_basic(cost_usd=1.23, cost_unknown=0, model_used="glm-5.3-flash"))
    cost_usd, _ = fresh_db.execute(
        "SELECT cost_usd, cost_unknown FROM router_logs WHERE task_type='test'"
    ).fetchone()
    assert abs(cost_usd - 1.23) < 1e-9, "explicit cost must not be overwritten"


def test_genuinely_unpriced_model_is_still_flagged_unknown(fresh_db):
    """An unknown model must remain honestly 'unknown', not silently $0."""
    ep._log_request_to_db(**_basic(model_used="no-such-model-anywhere-zzz"))
    cost_usd, cost_unknown = fresh_db.execute(
        "SELECT cost_usd, cost_unknown FROM router_logs WHERE task_type='test'"
    ).fetchone()
    assert cost_unknown == 1, "unknowable cost must be flagged, not guessed"
    assert cost_usd == 0.0


def test_every_call_site_in_the_endpoint_is_cost_safe():
    """Static check: no call site may pass a cost *partially*.

    The original form of this check flagged call sites that omitted the cost
    fields. That is no longer a defect: omission is now the safe default, since
    the logger computes the cost itself. Keeping the old assertion would have
    meant asserting a solved problem.

    The residual footgun is the partial pass — supplying ``cost_usd`` without
    ``cost_unknown`` (or vice versa). Because ``cost_unknown is None`` means
    "compute it", a lone ``cost_usd`` is now silently *discarded* and replaced
    by the computed value. That is a new way to lose an explicitly-known cost,
    so it is what must be pinned.
    """
    import ast

    src = (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()
    tree = ast.parse(src)
    partial = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_log_request_to_db":
            kws = {k.arg for k in node.keywords}
            has_usd = "cost_usd" in kws
            has_unknown = "cost_unknown" in kws
            if has_usd != has_unknown:
                partial.append((node.lineno, "cost_usd" if has_usd else "cost_unknown"))
    assert not partial, (
        "call sites pass cost fields partially, so a known cost is silently "
        f"discarded and recomputed: {partial}. Pass both cost_usd and "
        "cost_unknown, or neither (the logger then computes it)."
    )


def test_omitting_cost_is_the_safe_default_not_a_footgun():
    """The invariant the fix establishes: forgetting a cost cannot blind the ledger.

    Guards against a regression to ``cost_unknown: int = 1``, which is how 952
    production compression calls were logged cost-blind.
    """
    import inspect

    sig = inspect.signature(ep._log_request_to_db)
    default = sig.parameters["cost_unknown"].default
    assert default is None, (
        "cost_unknown must default to None so the logger computes the cost; "
        f"found default={default!r}, which silently marks omitted costs unknown"
    )

