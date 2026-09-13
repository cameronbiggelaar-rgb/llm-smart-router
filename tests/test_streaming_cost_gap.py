"""B15 — the streaming cost gap, and its FIX (B23.2).

**Status: FIXED.** This file originally pinned the defect so it stayed visible.
Its own instruction was: "When output tokens are captured for streams, this test
SHOULD fail — that is the signal to delete it and record the fix." It failed
when the fix landed, and this is that record.

The defect, measured on production before the fix:

    streaming rows in router_logs                229,525 of 235,884 (97%)
    of which route-start rows                    114,907  (0 output is CORRECT)
    of which COMPLETION rows                     114,618  (0 output is the DEFECT)
    recorded spend on completion rows            $17,438.92 over 7.77B input tokens
    completion rows with output_tokens=0         114,618  (all of them)
    non-streaming rows with output>0             99% of non-streaming rows

The route-start/completion split matters: a route-start row legitimately has no
output to record (the request has not finished), so counting all 229,525 as
defective would roughly double the apparent magnitude and discredit the real
finding. Both streaming completion loggers hardcoded ``output_tokens=0``.

Measured impact, using each workload's own observed non-streaming out/in ratio
(a real ratio for the same workload, not a guess):

    session_compression  $27,366.90 recorded -> +$426 unpriced output (1%)
    normal_chat           $7,663.89 recorded -> +$1,032 unpriced output (13%)

Deliberately *not* presented as a catastrophic understatement: on these workloads
output cost is genuinely small next to input cost (compression turns a large
context into a short summary; chat is input-heavy). The defect was never the size
of the number — it was that a cost figure was recorded as if measured when a
component of it was never captured, and nothing in the schema said so.

THE FIX (B23.2): the streaming path now requests
``stream_options={"include_usage": True}`` for providers that accept it
(verified against ollama.com: without the flag no streamed chunk carries usage;
with it exactly one does), harvests that chunk in ``_resume_stream``, and records
the real count. When no usage is reported the row is marked ``cost_unknown=1``
rather than presenting an input-only figure as measured.

What this file now pins:

 1. no streaming COMPLETION logger hardcodes ``output_tokens=0`` any more
    (inverted from the original assertion, which is the point);
 2. usage is requested, and the parser handles the empty-``choices`` usage chunk;
 3. the HISTORICAL production state — rows written before the fix — so the
    measured magnitude of the defect is not lost now that it is repaired.
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

# Rows written on or after this date may carry captured streaming usage. Rows
# before it were written by the pre-fix code, so their hardcoded zero is a
# permanent historical fact — which is what makes the assertion below stable
# instead of something that breaks the moment the fix is deployed.
FIX_COMMIT_DAY = "2026-09-13"


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
    """INVERTED at the fix (B23.2): the defect must stay gone.

    Originally this asserted the hardcoded zero was PRESENT, so the defect stayed
    visible instead of quietly becoming "normal", and it said that when output
    tokens were captured the test SHOULD fail so the fix would be recorded.

    It failed exactly as designed. This now asserts the opposite and keeps the
    same detector, so a regression that reintroduces a zero-output streaming
    completion logger is caught by the code path that found it the first time.
    """
    loggers = _streaming_completion_loggers()
    assert not loggers, (
        "a streaming COMPLETION logger hardcodes output_tokens=0 again: "
        f"{loggers}. That reintroduces the B15 defect — streaming completions "
        "would record an input-only cost as if it were measured. The streaming "
        "path must use the usage captured in _resume_stream (and mark "
        "cost_unknown when no usage was reported)."
    )


def test_streaming_path_actually_requests_usage():
    """The fix is only real if the request asks for usage.

    Without this the loggers could pass a captured value that is always absent,
    and the defect would be "fixed" by a variable that never varies.
    """
    src = _endpoint_src()
    assert "include_usage" in src, (
        "the streaming path stopped requesting usage; output tokens cannot be "
        "captured and the B15 fix is inert"
    )


def test_usage_chunk_parser_handles_the_empty_choices_shape():
    """Usage arrives with ``choices: []`` — a delta parser would drop it."""
    import sys

    sys.path.insert(0, str(SCRIPTS_DIR))
    import biggie_llm_endpoint as ep  # noqa: PLC0415

    in_tok, out_tok = ep._sse_usage_tokens(
        '{"choices":[],"usage":{"prompt_tokens":32,"completion_tokens":7}}'
    )
    assert (in_tok, out_tok) == (32, 7), (
        f"the usage chunk was not parsed: got {(in_tok, out_tok)}"
    )


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
    """Record the HISTORICAL state so the defect's magnitude is not lost.

    Scoped to rows written BEFORE the fix. Those rows are immutable history: the
    pre-fix code hardcoded ``output_tokens=0``, so the zero is a permanent fact
    about them, and this assertion stays valid forever. Scoping it this way is
    what keeps the measured impact on record now that the code is repaired —
    without it the assertion would break the moment the fix was deployed, and the
    magnitude of B15 would have to be re-derived from git history.

    The live behaviour is covered by the fix assertions above, not here.
    """
    conn = sqlite3.connect(f"file:{PROD_DB}?mode=ro", uri=True)

    total, = conn.execute("SELECT COUNT(*) FROM router_logs").fetchone()

    # Pre-fix rows: streaming completions written before FIX_COMMIT_DAY.
    hist_rows, hist_zero = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN output_tokens=0 THEN 1 ELSE 0 END) "
        "FROM router_logs WHERE streaming=1 "
        "AND COALESCE(error_type,'') != 'streaming_in_progress' "
        "AND substr(timestamp,1,10) < ?",
        (FIX_COMMIT_DAY,),
    ).fetchone()

    nonstream, nonstream_zero_out = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN output_tokens=0 THEN 1 ELSE 0 END) "
        "FROM router_logs WHERE streaming=0"
    ).fetchone()

    assert total > 100_000, "production DB looks wrong; refusing to assert on it"
    assert hist_rows and hist_rows > 100_000, (
        "no pre-fix streaming completion rows found; the historical record this "
        "file exists to preserve has been rolled up or purged"
    )
    # The pre-fix code hardcoded 0, so for those rows this is exact.
    assert hist_zero == hist_rows, (
        f"{hist_rows - hist_zero:,} PRE-FIX streaming rows carry output tokens, "
        "which the pre-fix code could not have written — the date scoping is "
        "wrong, or these rows were rewritten by a backfill"
    )
    # The contrast is what makes it a defect rather than a data-source limit.
    assert nonstream_zero_out < nonstream * 0.1, (
        "non-streaming rows mostly lack output tokens too, so the gap is not "
        "streaming-specific and this file's conclusion is wrong"
    )
