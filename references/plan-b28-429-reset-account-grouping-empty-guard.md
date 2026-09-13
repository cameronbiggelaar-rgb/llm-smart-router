# B28 — Parse 429 reset times, group by billing account, finish_reason-aware empty guard

## Goal

Stop the router from burning a whole model family against an account-wide quota
cap, and stop the empty-content guard from false-alarming on legit
`finish_reason:"length"` responses.

## Evidence

- 2026-09-13 outage: `openai-codex` (ChatGPT Plus) returned 429
  `usage_limit_reached`, `plan_type:"plus"`, `resets_at` epoch ≈ 02:27 AEST. The
  router escalated gpt-5.5 → 5.6-luna/terra/sol (all the *same* account) and
  drove gpt-5.5 to 75 consecutive failures, opening an 1800s circuit breaker
  and a 3600s rate-limit per rung. `ollama-cloud` returned a *different* 429
  (`usage credits auto reload monthly max reached`) at the same time — two
  independent billing systems.
- `mark_rate_limited` (router.py:254) uses pure exponential backoff
  `60s * 2^(failures-1)`, cap 3600s. It never reads the upstream reset time.
- `_handle_backend_status` (biggie_llm_endpoint.py:176) accepts `error_text` but
  all 7 call sites call it without it — the 429 body (with `resets_at`) is read
  at the call site and discarded.
- `_response_has_empty_content` (biggie_llm_endpoint.py:1286) looks only at
  `content`/`tool_calls`, ignores `finish_reason`. A `finish_reason:"length"`
  empty response (reasoning tokens ate a tiny `max_tokens` budget) is escalated
  as a degenerate 502. The shadow/eval path already special-cases `length`
  (line 553 `shadow_no_content_length`); the live escalation path does not.

## Scope (this batch)

1. **Parse the 429 reset time.** Add `retry_after_s` / `resets_at_s` params to
   `mark_rate_limited`. When the upstream supplies an explicit reset time, set
   `rate_limited_until` to it (clamped to sane bounds); exponential backoff only
   as fallback. Thread `error_text`/`Retry-After`/body through every
   `_handle_backend_status` call site so the reset time actually reaches the
   marker.
2. **Account/credential grouping.** Each model already carries `provider`
   (models.py). Add a billing-group notion so a 429 with `plan_type` /
   `usage credits` marks the whole group rate-limited; `escalate_on_failure`
   skips to a different group instead of cycling within it.
3. **`finish_reason`-aware empty guard.** `_response_has_empty_content` returns
   False when `finish_reason == "length"` (budget too small, not degenerate),
   mirroring the shadow path.

## Out of scope (deferred)

- In-memory/DB drift fix (`--recover` vs live `_MODEL_STATUSES`) → separate
  batch; needs an admin endpoint or SIGHUP reload.
- Health-vs-routing observability metric.

## Process

- TDD batch-by-batch: RED → GREEN → commit per batch, each batch its own test
  file.
- Focused tests, then full suite, then preflight before restart.
- No live restart until all three are green and the user is told.
