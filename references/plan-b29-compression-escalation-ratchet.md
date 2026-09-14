# Plan — Fix session-compression routing ratchet (B29)

**Status: implemented and committed (`b95f6c2`, `9310faa`, `45b9784`, `35f772f`).
Not live — service still runs pre-B28 code pending an explicit restart go-ahead.**

## Problem (measured, 2026-09-13 11:00 → 2026-09-14 00:00 UTC)

- 1,646 session-compression calls of 1,927 total rows — **~85% of router traffic**.
- ChatGPT plan (`openai-codex`): 524 calls, of which **515 compression / 9 real work**
  — **~98% of plan consumption was maintenance summarisation**.
- 203 codex compression escalations in 8h, **every one** from
  `deepseek-v4.1-flash:cloud (empty_content)`; ~$216 imputed (list price, not billed
  — all lanes are flat-subscription).
- Capacity signal: gpt-5.5 at 751 consecutive failures (circuit open),
  gpt-5.6-sol/terra/luna rate-limited — the Plus plan cap.

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

Steps 5→6 were the ratchet: an *offered* tool schema (which the summariser cannot
use) provoked a tool-call-only reply, which was then treated as evidence that a
tool-capable model was needed — promoting maintenance onto the scarcest resource.

## Fixes (4)

### Fix 1 — compression-specific escalation (break the ratchet)

**Target:** `scripts/router.py`.

- Extracted the curated ladder construction into
  `_session_compression_ladder(context_tokens) -> List[str]`. This is now the
  single source of truth for "which models may summarise", shared by initial
  selection *and* escalation, so a policy change applies to retries too.
- Added `_next_session_compression_rung(failed_model, context_tokens)`: the first
  available rung after the failed model, else any available non-reserved rung,
  `""` when genuinely exhausted.
- Added `workload_type: str = ""` / `context_tokens: int = 0` params to
  `escalate_on_failure` (backward-compatible defaults). The compression branch
  runs **before** the `requires_tools` branch, so a tool-flagged compaction still
  escalates on the ladder.
- Ladder exhaustion fails **closed** (`all_exhausted`, empty selection) rather
  than promoting onto the plan.

### Fix 2 — strip `tools` from compression requests

**Target:** `scripts/biggie_llm_endpoint.py`.

`_normalize_compression_request(body, workload_type)` strips `tools` and
`tool_choice` for compression and records `_requires_tools = False`; the handler
re-derives `requires_tools` from the normalised body. Non-compression workloads
keep their schemas unchanged. `proxy_to_backend` builds its upstream body from
the same mutated dict, so the strip reaches the wire (pinned by test).

### Fix 3 — tool-call-only compression retries a rung, not a promotion

**Target:** `scripts/biggie_llm_endpoint.py` escalation call site.

The call sites now pass `workload_type=workload_type` and
`context_tokens=features.get("context_tokens", 0)`, so the Fix-1 branch handles
compaction failures instead of the generic tool/tier climb.

### Fix 4 — reserve the ChatGPT plan from every compression path

**Target:** `scripts/router.py`.

- `CHATGPT_PLAN_RESERVED_FROM_COMPRESSION` name set.
- **Deviation from plan (stronger):** reservation now follows the *provider*
  (`COMPRESSION_RESERVED_PROVIDERS = {"openai-codex"}`), because the hand-listed
  names **missed `gpt-6-astra`**, which sits on the same plan and was eligible to
  summarise. The provider rule means a new plan model is excluded automatically.
- **Addition:** `COMPRESSION_RESERVED_PREFIXES = ("gpt-5", "gpt-6")` covers
  unregistered aliases reachable via `force_model`, where neither the name list
  nor the provider mapping can see them.
- `_build_session_compression_fallback_chain` and the fail-open loop both use
  `_is_reserved_from_compression` (replacing the single `gpt-5.5` checks).
- **Addition:** implemented `_validate_curated_ladder()`, which the ladder
  docstring had been claiming existed. It raises at import time if a rung is
  unregistered, reserved, or the ladder is empty.

## Tests (TDD, one file per batch)

1. `tests/test_b29_compression_escalation.py` (5) — Fix 1.
2. `tests/test_b29_compression_tools_strip.py` (5) — Fix 2, behavioural.
3. `tests/test_b29_ladder_invariants.py` (14) — Fix 4 + ladder invariants.
4. `tests/test_b29_wiring_contracts.py` (6) — static contracts parsed from
   source (dropped kwarg / pre-strip flag read / unwired normalise are invisible
   to behavioural tests), **mutation-verified**.

## Verification

- Full suite: **525 passed**.
- Defect reproduced and closed: the same synthetic tool-flagged compaction
  request escalates to `gpt-5.6-sol` on the pre-B29 call signature and to
  `deepseek-v4-flash` on the wired one.
- Exhaustion path: every curated rung down → still a summariser, never the plan;
  total outage → `""` + `all_exhausted=True`.

## Out of scope

- Restarting `biggie-llm-endpoint.service` to load B28 + B29 (needs explicit
  go-ahead; the service is live and serving traffic).
- Changing limp-home behaviour for compression.
- Any change to tool-capable routing for non-compression workloads.
