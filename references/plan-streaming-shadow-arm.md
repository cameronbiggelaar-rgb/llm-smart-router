# Plan — streaming shadow arm (measure flash's true streamed cost + cache rate)

## Problem

The `v41flash-tool-shadow` shadow forces `stream=False` (`run_shadow_experiment`,
`body["stream"] = False`). Non-streaming suppresses the provider's prompt cache,
so the shadow candidate reports **0 cached tokens on every call**. That makes the
shadow's cost evidence *inverted*: flash logs `$0.0121/req` vs pro `$0.0072/req`,
purely because pro's streamed requests are ~94% cache hits while the shadow's
non-streamed calls get none. The shadow is measuring "flash with caching
disabled", which is not how flash would actually run in production.

The question "is flash cheaper in production?" is therefore currently
**unanswerable** from the shadow data. Streaming the shadow arm fixes the
measurement: it lets the candidate hit the prompt cache (so cached_tokens is
recorded and cost is computed at the real discounted rate), and it also
re-tests whether flash's tool choice improves when streamed (the probe hinted
it does: read_file 19 vs 15, terminal 5 vs 9).

## Scope

Add an opt-in `stream: true` flag to an experiment (the `Experiment` dataclass
and `experiments.yaml` loader already support it after Batch A). When set on a
shadow experiment, the candidate call streams with `stream_options.include_usage`
and the observation records `streaming=1`, real `cached_tokens`, and a cost
computed at the cached rate. Default stays non-streaming — zero change to the
existing arm.

## Design

1. **`Experiment.stream: bool = False`** (done, Batch A). Plumbed through
   `load_experiment` (yaml) and serialised into `match_json` only when True.

2. **Streaming shadow call.** New helper `_shadow_stream_call(backend, messages,
   body)` that mirrors the OpenAI-compatible streaming path:
   - `stream=True`, `stream_options.include_usage=True` when provider is in
     `STREAM_USAGE_CAPABLE_PROVIDERS` (ollama-cloud).
   - forward sampling params (temperature/top_p/max_tokens/stop/penalties),
     `tools`, `tool_choice` — same as `_preflight_openai_stream`.
   - consume the full SSE stream (aiter_lines) accumulating content text,
     tool-call deltas (via `_accumulate_tool_calls`/`_finalize_tool_calls`),
     usage tokens + cached tokens (`_sse_usage_tokens`, `_sse_cached_tokens`),
     finish_reason (`_sse_finish_reason`).
   - return a result dict shaped exactly like `proxy_to_backend`'s non-streaming
     return (`choices[0].message` with assembled `tool_calls`, `usage` with
     `prompt_tokens`/`completion_tokens` and a `cached_tokens` key). This lets
     the existing row-building code stay unchanged.

3. **`run_shadow_experiment` branch.** When `experiment.stream` is True, call
   `_shadow_stream_call` under the same semaphore + `wait_for` timeout, then:
   - `streaming=True` (instead of hardcoded False),
   - `cached_tokens=cached` in the row,
   - `compute_cost_fields(..., cached_input_tokens=cached)` so the cached rate
     applies.
   Quality scoring and tool-name extraction are identical (the normalized result
   carries `message.tool_calls`).

## Non-goals

- No change to the production routing path, `TOOL_CAPABLE_MODELS`, or the
  incumbent. This is observe-only, same as the existing shadow.
- No live traffic is moved. The streaming arm still discards its output.

## Verification

- RED: new tests in `tests/test_shadow_execution.py` for the streamed path
  (calls with stream=True + include_usage, records cached_tokens, prices at
  cached rate, streaming=1). GREEN after implementation.
- Full `tests/` still green (751 passed / 1 pre-existing).
- Endpoint restart to the new code, then confirm a streamed shadow row appears
  with non-zero cached_tokens and a cost lower than the non-streamed arm.

## Config to flip after deploy

In `experiments.yaml`, the shadow experiment gains `stream: true`. The existing
`v41flash-tool-shadow` arm stays non-streamed until the new arm proves out; the
two arms are distinguishable by `streaming` and `cached_tokens` in router_logs.
