# Plan — streaming shadow arm (measure flash's true streamed cost + cache rate)

## Problem

The `v41flash-tool-shadow` shadow forces `stream=False` (`run_shadow_experiment`,
line 954). That suppression is invisible in the tool-agreement metric but it
**inverts the cost evidence**, which is the *second* gate on cutover:

- Pro tool-bearing requests stream and run ~94% prompt-cache hits (avg 59,184
  cached of 62,866 input tokens).
- The shadow's forced non-stream means flash reports **0 cached tokens** on
  every row, so its logged `cost_usd` ($0.0121/req) is **68% higher** than
  pro's ($0.0072/req) — the opposite of the real economics.

A non-streamed flash call does not exercise the provider's prompt cache, so the
shadow cannot answer "is flash actually cheaper once it streams and caches like
pro?" — and that is the exact number the cost side of the cutover depends on.

Second-order win: the stream-mode sensitivity probe (`probe_stream_mode_sensitivity.py`)
already suggested flash picks `read_file` *more* (19 vs 15) and `terminal` *less*
(5 vs 9) when streamed. A streaming shadow therefore also re-tests whether the
~56–57% tool-agreement figure was itself depressed by forced non-streaming.

## Goal

Add a streaming shadow arm so a shadow experiment can call the candidate with
`stream=True` + `include_usage`, consume the stream, and record a shadow row
carrying **measured** `cached_tokens` (and thus a true cost) alongside the
existing tool-name/signature capture.

## Design

### 1. `Experiment.stream` flag (opt-in, default False)

`traffic_split.Experiment` gains `stream: bool = False`, parsed from
`match`-sibling key `stream:` in YAML (top-level, not inside `match`). Default
False preserves every existing experiment byte-for-byte. `experiments.yaml`
`v41flash-tool-shadow` gets `stream: true`.

### 2. Streaming shadow call path

New `run_shadow_experiment` branch (or a sibling helper `_shadow_stream_call`)
that, when `experiment.stream` is set:

- builds the candidate body with `stream=True` and
  `stream_options={"include_usage": True}` (mirrors `_preflight_openai_stream`'s
  body construction — same param forwarding, tools, tool_choice);
- opens the httpx stream directly, iterates SSE lines to completion, and
  accumulates: content text, tool-call deltas (reuse `_accumulate_tool_calls` /
  `_finalize_tool_calls`), usage input/output (reuse `_sse_usage_tokens`),
  cached tokens (reuse `_sse_cached_tokens`), finish_reason (reuse
  `_sse_finish_reason`);
- records a shadow row with `streaming=True`, `cached_tokens=captured`, and
  cost computed via `compute_cost_fields(..., cached_input_tokens=captured)`.

The stream is fully consumed and discarded — the user's response is unchanged,
preserving the shadow safety contract. Concurrency/timeout still bounded by the
existing semaphore and `_SHADOW_TIMEOUT_SECONDS`.

### 3. Why not reuse `proxy_to_backend_streaming`

`proxy_to_backend_streaming` returns a `StreamingResponse` and is designed to be
awaited inside a FastAPI request with a client to push SSE to. Shadow runs in a
bare `asyncio.create_task` with no downstream client, so there is nothing to
serve. The shadow needs the *raw consumed stats*, not a response object — hence
a dedicated consume-to-completion helper that reuses the existing SSE parsers.

## Success criteria

A streaming shadow produces rows where, for the candidate:
- `streaming=1`, and
- `cached_tokens` is non-zero on repeated-prefix sessions (proving the cache is
  exercised), and
- `cost_usd` reflects the discounted cached-input rate.

Then re-run `report_tool_agreement.py` and compare the streamed shadow's
tool-agreement rate and cost vs the non-streamed shadow. Two outcomes matter:
1. does streaming raise flash's tool-agreement (toward a cutover threshold)?
2. what is flash's *true* streamed cost / cache-hit rate vs pro?

## Batch plan (TDD, RED→GREEN→commit per batch)

1. **Batch A — `Experiment.stream` flag.** `traffic_split.py` dataclass + parser
   + `experiment_to_row`/`match_json`. Tests: `tests/test_traffic_split.py`
   (default False, parses `stream: true`, serialises to match_json).
2. **Batch B — streaming shadow call.** `_shadow_stream_call` helper + wiring in
   `run_shadow_experiment` to branch on `experiment.stream`. Tests:
   `tests/test_shadow_execution.py` (streaming shadow consumes stream, records
   `streaming=1` + `cached_tokens`, discards output, prices with cached discount,
   failure never raises).
3. **Batch C — enable on `v41flash-tool-shadow`** + end-to-end verify live
   shadow rows carry `streaming=1` and non-zero cached tokens.

## Non-goals

- No traffic movement (shadow stays observe-only; `stream` flag does not affect
  `mode: split`).
- No change to `report_tool_agreement.py` scoring (it already joins by
  `request_id` and reads `cached_tokens`/`cost_usd`).
- No change to the non-stream shadow default — the flag is strictly opt-in.
