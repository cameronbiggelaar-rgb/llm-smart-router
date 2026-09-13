# Plan — writer cleanup: one row per streaming request, and real output tokens

Status: **COMPLETE (code)** — both batches landed, 411 tests pass, preflight 16/16,
non-vacuity proven, and an end-to-end run against the real backend confirmed the
fix on live traffic. **Not yet live in production**: the running service (PID 927)
predates these edits, and restarting it is a production action awaiting explicit
go-ahead.

## Outcome (measured, not inferred)

End-to-end proof (`/tmp/e2e_writer.py`) — a second endpoint instance on port 8099
with its OWN temp DB, a real streaming request to the real backend:

    rows written for one streaming request        1   (was 2)
    surviving row  out=36 in=19 cost=$0.000185 cost_unknown=0
    expected glm-5.3 (19/1e6*1.40)+(36/1e6*4.40) = $0.000185  ✓ exact

The same call priced input-only — the old behaviour — is **$0.000027**, so the
pre-fix ledger understated this call by **6.9x**. On output-heavy workloads that
is the difference the B15 docstring called small (1% on compression) and larger
on chat (13%).

A vacuous test was found and removed by the non-vacuity proof: the first version
of the B23.2 gate test asserted `"include_usage" in src`, which still passed with
the provider gate disabled. Replaced with tests that drive the real preflight and
inspect the request body it builds. `/tmp/b23_proof.py` reverts each fix in a
throwaway copy and requires the matching test to fail — both now do.

## Why the writer, and not the data

The ledger held **242,964 rows / 64 days** at audit time. Two properties of the
*writer* made that number misleading and the data hard to trust:

1. **Every streaming request wrote two rows.** A start marker
   (`error_type='streaming_in_progress'`) and a completion row, sharing one
   `request_id`. Measured: **118,326 marker rows = 48% of the table**, and 99.x%
   of them had a completed twin. The marker is deliberate crash-recovery design
   (abandonment detection), so it must stay *visible* — but it is a placeholder,
   not a second event.
2. **Streaming completions hardcoded `output_tokens=0`.** Measured: **114,618
   completion rows recorded 0 output tokens** while carrying $17,438.92 of
   input-only spend. `cost_unknown` was not set, so an input-only figure was
   presented as a complete measured cost. This was the B15 defect, pinned (not
   fixed) by `tests/test_streaming_cost_gap.py`.

Both are fixed *at the source*, so the noise stops accumulating. Wiping history
would leave both generators running.

### The fix is viable, and that is measured, not assumed

`/tmp/probe_usage.py` probed ollama.com directly:

- stream **without** `stream_options`: 0 lines carrying non-null `usage`
- stream **with** `stream_options={"include_usage": true}`: **1 line carries
  `usage: {prompt_tokens, completion_tokens, ...}`** in a final chunk with
  `choices: []`

So output tokens are *obtainable* on the streaming path. The defect is that we
never ask. (Guard: only request it for providers known to accept it, and never
let a rejected `stream_options` break a request.)

## Batch 1 — delete-on-completion (one row per streaming request)

**Chosen over UPDATE-in-place** because it is strictly smaller and safer: the
surviving row is byte-identical to today's completion row, so every existing
consumer (rollup, reports, optimiser, unit economics) sees exactly what it saw
before — there is simply one row instead of two.

Rule: in `_log_request_to_db`, when the row being written is **not** a marker and
`request_id` is non-empty, delete that request's marker in the same transaction,
then insert as normal.

Properties this preserves:
- **Abandonment visibility** — a marker with no completion is never deleted, so
  `_find_abandoned_streams` keeps working. Measured: **289 such requests exist**.
- **`BILLABLE_ROW_SQL` stays canonical** — the exclusion filter becomes
  belt-and-braces (a safety net for legacy rows) rather than the only defence.
- **Escalated streams stay honest** — the failed model is preserved in
  `routing_reason` ("escalation from X (error_type)"), and `escalated=1`.
  Measured: 747 request_ids span >1 model.

Guard: an empty `request_id` must never delete anything — 170 billable rows share
an empty `request_id`, and a naive `DELETE ... WHERE request_id = ''` would wipe
every unpaired marker at once.

Tests (own file, `tests/test_streaming_marker_dedup.py`): marker+completion
leaves one row; the survivor is the billed completion; an unpaired marker
survives; an empty-`request_id` completion deletes nothing; the canonical sum is
one billable event.

## Batch 2 — capture real output tokens on the streaming path

Ask the upstream for usage, capture it from the final SSE chunk, and record it.
When usage is genuinely unavailable (degraded/abandoned stream), record
`cost_unknown=1` so a partial cost is never presented as measured — which is what
B15 asked for and never got.

Touches: `_preflight_openai_stream` (request `stream_options`, provider-gated),
`_resume_stream` (capture the usage chunk into `stats`), the two streaming
completion loggers (pass captured output tokens + honest `cost_unknown`),
`_wrap_non_streaming` / `_non_streaming_fallback_to_sse` (usage is available on
the non-streaming result).

Risk control: a backend that rejects `stream_options` must not break streaming.
Request it only for a known-good provider set and fall back cleanly if absent.

Tests (own file, `tests/test_streaming_usage_capture.py`): usage-bearing stream
records real output tokens; a stream without usage records `cost_unknown=1`
rather than a confident partial; the provider gate excludes providers not in the
set; an existing `stream_options` from the caller is not clobbered.

Then update `tests/test_streaming_cost_gap.py` — it deliberately *pins the
defect* and says so: "When output tokens are captured for streams, this test
SHOULD fail — that is the signal to delete it and record the fix."

## Sequencing and authority

- TDD per batch: RED → GREEN → commit, each batch its own test file. **Done.**
- **No prod write.** Both batches are code changes only. **Done.**
- **Restart is a production action** — the running service (PID 927) predates
  these edits. Making them live requires an explicit go-ahead; the plan stops
  short of bouncing `biggie-llm-endpoint.service`. **Still pending.**
- Existing tests must still pass. `rollup`/report semantics are unchanged: the
  marker exclusion remains canonical for historical rows. **411 pass.**

## What is NOT done, deliberately

- **The service has not been restarted.** Production is still writing two rows
  per streaming request and still recording `output_tokens=0`. The fix is inert
  until then.
- **Historical rows are untouched.** ~118k marker rows and ~114k zero-output
  completion rows remain in the ledger. They are now correct *as history* — the
  pre-fix code genuinely wrote that — and `BILLABLE_ROW_SQL` still excludes the
  markers. No backfill was attempted: rewriting them would fabricate output
  counts that were never captured.
- **The 09-13 date in `test_streaming_cost_gap.py` (`FIX_COMMIT_DAY`)** is the
  deployment boundary for assertions about pre-fix history. If the restart lands
  on a later day, that constant must move with it — otherwise the pre-fix scope
  would include rows written by the fixed code.
