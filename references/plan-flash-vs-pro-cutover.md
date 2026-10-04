# Plan — flash vs pro cutover evidence (tool-choice agreement)

## Problem

The shadow lane proves flash *can* emit tool calls (9/9 probe, ~99% `finish_reason=tool_calls` in live shadow), but it cannot yet answer the question that actually gates cutover:

> **Does flash pick the *same* tool (and equivalent args) that pro picks, on the same request?**

`router_logs` stores only `saw_tool_calls` (bool) and `finish_reason`. There is no column recording *which* tool was called. So a valid-JSON flash tool call to the *wrong* tool is invisible in the current metrics — and that is precisely the failure that would make a cutover degrade real tool-bearing work (file edits, searches, terminal).

## Goal

Add tool-name (and normalized-args) capture to both the incumbent (pro) path and the shadow (flash) path, join them per-request, and produce a daily agreement report that yields a single go/no-go number: **tool-name agreement rate** (and optionally arg-signature agreement).

## Design

### 1. New columns (two)

Add to `router_logs` (via `_COST_OBS_COLUMNS` in `biggie_llm_endpoint.py`, mirrored in `rollup.NEW_LOG_COLUMNS`):

- `tool_names` TEXT NOT NULL DEFAULT '' — comma-joined, ordered, deduped tool names actually called by the model this turn.
- `tool_args_sig` TEXT NOT NULL DEFAULT '' — a short canonical signature of the args (see below), empty when no tool call. Optional; captured only to enable arg-level agreement later. Kept bounded (hash) so we never store prompt-like data.

`tool_names` is the primary signal. `tool_args_sig` is secondary (arg-level agreement), deliberately non-authoritative — valid-but-different args can both be correct.

### 2. Capture on the incumbent (pro) path

Pro tool-bearing requests stream (`want_stream`). Tool names arrive as OpenAI-style deltas:
`delta.tool_calls[i].function.name` (first delta for index `i`) and `.arguments` (fragmented across deltas).

Add a bounded accumulator in the streaming path (mirror `_StreamChannelStats` / `_update_stream_channel_stats`) that:
- keys on `delta.tool_calls[i].index`,
- records `function.name` when present,
- appends `function.arguments` fragments per index.

Expose the ordered names list through `stats["tool_names"]`, then pass it into the completion logger (`_log_stream_complete` and `_log_escalated_stream_complete`) and the non-streaming fallback path.

For the non-streaming incumbent path (line 4447 etc.), extract names directly from `result` via the existing `_normalize_tool_call`.

### 3. Capture on the shadow (flash) path

Shadow forces `stream=False` (line 947), so `msg.get("tool_calls")` is a full list. Extract ordered names via `_normalize_tool_call` in `run_shadow_experiment` and add `tool_names` / `tool_args_sig` to the shadow row dict.

### 4. Join key

Shadow rows already carry the same `request_id` as the incumbent (line 1035). Join on `request_id` where `is_shadow=1` (flash) vs `is_shadow=0` (pro, `final_model=deepseek-v4-pro`) and same `workload_type=normal_chat`.

### 5. Report script

`scripts/report_tool_agreement.py`:
- pulls matched (pro, flash) pairs over a window,
- computes: tool-name set agreement (Jaccard), exact-name-list match rate, arg-signature match rate (when both non-empty),
- buckets by task_type,
- emits the single go/no-go line: name-agreement ≥ threshold AND empty-content rate ≤ 1% AND no degeneration.

## Success criteria (go/no-go)

Cutover is justified when, over a **≥2-day** window:
- flash tool-name **agreement vs pro ≥ 95%** (same set of tools, in order), AND
- flash `empty_content` escalation rate **≤ 1%**, AND
- flash degeneration/loop events **= 0**.

Cost is already proven (~$0.02 vs ~$0.04+/call); quality is the open gate.

## Batch plan (TDD, RED→GREEN→commit per batch)

1. **Batch A — schema + capture helpers.** New `tool_names`/`tool_args_sig` columns + `_normalize_tool_call` reuse + a streaming tool-name accumulator. Tests: `tests/test_tool_names_capture.py` (column migration, streaming delta accumulation, non-streaming extraction).
2. **Batch B — wire into loggers.** Pass `tool_names`/`tool_args_sig` through all `_log_request_to_db` call sites (incumbent stream/non-stream, escalated stream/non-stream, shadow). Tests: extend capture tests to assert row round-trip.
3. **Batch C — report script.** `report_tool_agreement.py` + tests over a synthetic fixture DB (paired rows) asserting agreement math.

## Non-goals

- No traffic movement (shadow stays observe-only until go/no-go returns green).
- No arg-level *semantic* correctness scoring (that needs a judge; out of scope — name agreement + empty-content + degeneration are the defensible gates).
- No change to `traffic_split.py` scoping (already correct).
