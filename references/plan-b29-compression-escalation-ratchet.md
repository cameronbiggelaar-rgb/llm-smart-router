# Plan — Fix session-compression routing ratchet (B29)

## Problem (measured, 2026-09-13 11:00 → 2026-09-14 00:00 UTC)

Compression is ~85% of router traffic and ~98% of ChatGPT-plan consumption.
Most of it is maintenance summarisation that should never touch the ChatGPT plan.

Root cause chain:

1. A native Hermes session-compression request **carries the parent session's
   `tools` schema** even though the summariser never calls tools.
2. `biggie_llm_endpoint.py` sets `requires_tools = bool(body.get("tools"))`
   → `True` for compression.
3. `route_task` routes compression correctly to the summariser ladder
   (deepseek-v4.1-flash) — the tool gate is bypassed at routing time (good).
4. flash intermittently returns **tool-call-only, no content** for compression.
5. The endpoint raises 502 → `escalate_on_failure(failed_model, requires_tools=True)`.
6. In `escalate_on_failure`, `requires_tools=True` hits the "most-capable tool
   model" branch → **gpt-5.6-sol** (ChatGPT plan).

Measured: 203 codex compression escalations in 8h, every one from
`deepseek-v4.1-flash:cloud (empty_content)`; $216 imputed. gpt-5.5 at 751
consecutive failures (circuit open), gpt-5.6-sol/terra/luna rate-limited —
the Plus plan cap, driven in large part by maintenance traffic.

## Fixes (4)

### Fix 1 — compression-specific escalation (break the ratchet)

**Target:** `scripts/router.py`.

- Extract the curated ladder construction in `_select_session_compression_model`
  into `_session_compression_ladder(context_tokens) -> List[str]` (the hardcoded
  `preferred` list + flash-family ceiling + `MODEL_CONTEXT_CEILING` filters).
  Keep `_select_session_compression_model` behaviour identical.
- Add `_next_session_compression_rung(failed_model, context_tokens) -> str`:
  return the first available ladder rung *after* the failed model; if the failed
  model is last (or not on the ladder), return the first available rung that is
  not the failed model. Return `""` only when the ladder is genuinely exhausted.
- Add `workload_type: str = ""` and `context_tokens: int = 0` params to
  `escalate_on_failure` (backward-compatible defaults). After the rate-limit /
  quality-failure bookkeeping and limp-home handling, and BEFORE the
  `requires_tools` branch, short-circuit when `workload_type == "session_compression"`:
  pick `_next_session_compression_rung`; if none, return `all_exhausted` decision.
  Compression escalation must never reach the generic tier-climb or the
  "most-capable tool model" branch.

### Fix 2 — strip `tools` from compression requests

**Target:** `scripts/biggie_llm_endpoint.py` (`/v1/chat/completions` handler,
~L3155–3167).

After `detect_workload_type`, when `workload_type == "session_compression"`:
- `body.pop("tools", None)` and `body.pop("tool_choice", None)` so the summariser
  is not offered tools (which provokes tool-call-only responses).
- force `requires_tools = False`.

### Fix 3 — tool-call-only compression retries a rung, not a promotion

**Target:** `scripts/biggie_llm_endpoint.py` escalation call (~L3614).

- Pass `workload_type=workload_type` and `context_tokens=features.get("context_tokens", 0)`
  into `escalate_on_failure` so the Fix-1 branch handles compression.
- The existing 502 for tool-call-only compression (L3018) already flows into
  `escalate_on_failure`; with Fix 1 + Fix 2 it retries the next summariser rung.

### Fix 4 — reserve the ChatGPT plan from every compression path

**Target:** `scripts/router.py`.

- Add module constant
  `CHATGPT_PLAN_RESERVED_FROM_COMPRESSION = {"gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}`.
- `_build_session_compression_fallback_chain`: exclude this set (replace the
  single `m != "gpt-5.5"` check).
- `_select_session_compression_model` fail-open loop: exclude this set (replace
  the `model == "gpt-5.5"` check), in addition to `EXCLUDED_FROM_AUTO_ROUTING`.

## Tests (TDD, one file per batch)

1. `tests/test_b29_compression_escalation.py` — Fix 1:
   - `escalate_on_failure(workload_type="session_compression", ...)` after a
     flash empty-content failure returns the next ladder rung (glm-5.3), never
     a gpt-5.x model.
   - exhausting the ladder returns `all_exhausted`.
   - a non-compression call is unchanged (still honours `requires_tools`).
2. `tests/test_b29_compression_tools_strip.py` — Fix 2:
   - a compression request whose body carries `tools` yields `requires_tools=False`
     and the tools are stripped before downstream; a normal request with `tools`
     keeps `requires_tools=True`.
3. `tests/test_b29_compression_capacity_reserve.py` — Fix 4:
   - the fail-open loop and the compression fallback chain never return any
     gpt-5.x model, even when such a model is the cheapest tier-3+ available.

## Verification

- Full suite green (currently 495 passed at `1a9be63`).
- `verify` via focused scripts: compression escalation stays on the ladder;
  gpt-5.x unreachable via compression fail-open / fallback chain.
- **No live restart** — service still runs pre-B28 code (PID 316924); restart
  needs explicit go-ahead and is out of scope for this batch.

## Out of scope

- Restarting `biggie-llm-endpoint.service` to load B28 + B29.
- Changing limp-home behaviour for compression.
- Any change to tool-capable routing for non-compression workloads.
