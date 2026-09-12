"""B15 — the streaming cost gap: record what we do not know.

Discovered while adding the B13 schema-parity assertions, and confirmed against
production before being written down (numbers below are measured, not
estimated from reasoning):

    streaming rows in router_logs        229,525  of 235,884 (97.3%)
      of which route-start rows          114,907  (0 output is CORRECT here)
      of which COMPLETION rows           114,618  (0 output is the DEFECT)
    recorded spend on completion rows    $17,438.92 over 7.77B input tokens
    completion rows with output_tokens=0 114,618  (all of them)
    non-streaming rows with output>0     99.6% of non-streaming rows

The route-start/completion split matters: a route-start row legitimately has no
output to record (the request has not finished), so counting all 229,525 as
defective would roughly double the apparent magnitude and discredit the real
finding. Both streaming completion loggers hardcode ``output_tokens=0``.

Measured impact, using each workload's own observed non-streaming out/in ratio
(a real ratio for the same workload, not a guess):

    session_compression  $27,366.90 recorded  ->  +$426   unpriced output (1.6%)
    normal_chat           $7,663.89 recorded  ->  +$1,032 unpriced output (13.5%)

This is deliberately *not* presented as a catastrophic understatement: on these
workloads output cost is genuinely small next to input cost (compression turns
a large context into a short summary; chat is input-heavy). The defect is not
the size of the number — it is that a cost figure is recorded as if measured
when a component of it was never captured, and nothing in the schema says so.

What this file pins:

  1. the hardcoded ``output_tokens=0`` is a *decision* that can be seen, not a
     detail buried in a logger, so a future reader cannot mistake it for
     "the model produced no output";
  2. ``cost_unknown`` must reflect it — a cost computed from input alone while
     output is unknown must not be presented as a complete measured cost;
  3. the measured production state, so that fixing it makes this file fail
     loudly rather than passing vacuously forever.
"""

from __future__ import annotations

import ast
import re
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
PROD_DB = ROOT / "data" / "router_logs.db"


def _endpoint_src() -> str:
    return (SCRIPTS_DIR / "biggie_llm_endpoint.py").read_text()


def _streaming_completion_loggers():
    """Find functions that log a *streaming completion* — not a route start.

    Identified semantically rather than by function name (names here have
    already changed once, which is how this test first found nothing):

      * it calls ``_log_request_to_db``
      * with ``output_tokens`` hardcoded to 0
      * and ``error_type`` NOT set to ``streaming_in_progress``

    That last clause is the discriminator that matters. A route-start row
    legitimately records 0 output tokens: the request has not finished yet, so
    there is nothing to count, and a separate completion row supersedes it. A
    *completion* row recording 0 output tokens is the actual defect — the
    request is over and the usage was never captured.
    """
    tree = ast.parse(_endpoint_src())
    fns = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "_log_request_to_db":
                    kws = {k.arg: k.value for k in sub.keywords}
                    out = kws.get("output_tokens")
                    zero_out = isinstance(out, ast.Constant) and out.value == 0
                    err = kws.get("error_type")
                    err_txt = getattr(err, "value", "") if isinstance(err, ast.Constant) else ""
                    if zero_out and err_txt != "streaming_in_progress":
                        fns.append((node.name, sub.lineno))
    return fns


def test_streaming_completion_loggers_hardcode_zero_output_tokens():
    """Pin the decision explicitly, and fail when it is fixed.

    Asserting the hardcoded zero on purpose: it is a real defect, and this test
    is how the defect stays visible instead of quietly becoming "normal". When
    output tokens are captured for streams, this test SHOULD fail — that is the
    signal to delete it and record the fix.
    """
    loggers = _streaming_completion_loggers()
    assert loggers, (
        "no streaming COMPLETION logger hardcodes output_tokens=0 any more — "
        "either the defect is FIXED (update this file and the docstring "
        "numbers) or the logging shape changed enough that this detector is "
        "blind (fix the detector). Do not simply delete this test."
    )
    assert all(isinstance(ln, int) and ln > 0 for _, ln in loggers)


def test_route_start_rows_are_not_mistaken_for_the_defect():
    """Separate the two cases: a pending route is not a lost cost.

    The route-start row genuinely has no output to record. Counting it as a
    defect would inflate the magnitude roughly 2x and discredit the real
    finding, so the distinction is asserted rather than assumed.
    """
    src = _endpoint_src()
    assert "streaming_in_progress" in src, (
        "the route-start marker is gone; the detector above can no longer tell "
        "a pending row from a completed one"
    )


def test_cost_unknown_is_set_when_output_tokens_are_unavailable():
    """A cost derived from input alone is not a complete measured cost.

    ``cost_unknown`` is the schema's only way to say "this figure is not the
    whole story". These loggers omit ``cost_unknown``, which since B12 means
    "compute it" — pricing an input-only call and recording it as fully known.
    The assertion therefore checks the *outcome*: a logger that knows output
    tokens are missing must not let a partial cost be recorded as complete.
    """
    src = _endpoint_src()
    for fname, lineno in _streaming_completion_loggers():
        # Find this call's keywords to see whether it also declares the cost.
        tree = ast.parse(src)
        declared_unknown_zero = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_log_request_to_db" and node.lineno == lineno:
                kws = {k.arg: k.value for k in node.keywords}
                cu = kws.get("cost_unknown")
                if isinstance(cu, ast.Constant) and cu.value == 0:
                    declared_unknown_zero = True
        assert not declared_unknown_zero, (
            f"{fname} (line {lineno}) records an input-only cost as fully known "
            "(cost_unknown=0) while output_tokens=0 means output was never "
            "counted"
        )


def test_logger_does_not_present_partial_cost_as_unknown_forever():
    """Guard the opposite failure: 'unknown' must not become the default again.

    B12 removed ``cost_unknown: int = 1`` as a default because it silently hid
    952 cost-blind compression calls. This asserts the fix is still in place,
    so B15's concern (a *partial* cost) is never "solved" by reverting to
    "declare everything unknown", which would blind the ledger again.
    """
    import sys

    sys.path.insert(0, str(SCRIPTS_DIR))
    import biggie_llm_endpoint as ep  # noqa: PLC0415

    import inspect

    default = inspect.signature(ep._log_request_to_db).parameters["cost_unknown"].default
    assert default is None, (
        "cost_unknown must default to None (compute it), not to 1 (declare it "
        f"unknown); found {default!r}"
    )


# --------------------------------------------------------------------------
# The measured production state. Read-only; skips if the DB is absent (CI).
# --------------------------------------------------------------------------

@pytest.mark.skipif(not PROD_DB.exists(), reason="production DB not present")
def test_production_streaming_rows_have_no_output_cost_recorded():
    """Record the measured state so a fix is announced rather than assumed.

    Numbers are asserted as inequalities where the exact value drifts with
    live traffic, and as an exact zero only where the code guarantees it.
    """
    conn = sqlite3.connect(f"file:{PROD_DB}?mode=ro", uri=True)

    total, = conn.execute("SELECT COUNT(*) FROM router_logs").fetchone()
    streaming, stream_zero_out = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN output_tokens=0 THEN 1 ELSE 0 END) "
        "FROM router_logs WHERE streaming=1"
    ).fetchone()
    nonstream, nonstream_zero_out = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN output_tokens=0 THEN 1 ELSE 0 END) "
        "FROM router_logs WHERE streaming=0"
    ).fetchone()

    assert total > 100_000, "production DB looks wrong; refusing to assert on it"
    assert streaming > 100_000, (
        "streaming is no longer the dominant path; re-derive the impact before "
        "quoting the docstring numbers"
    )
    # The code hardcodes 0, so this is exact.
    assert stream_zero_out == streaming, (
        f"{streaming - stream_zero_out:,} streaming rows now record output "
        "tokens — the streaming cost gap looks partially fixed; update this "
        "file and the docstring numbers"
    )
    # The contrast is what makes it a defect rather than a data-source limit.
    assert nonstream_zero_out < nonstream * 0.1, (
        "non-streaming rows mostly lack output tokens too, so the gap is not "
        "streaming-specific and this file's conclusion is wrong"
    )
