# Plan — Compression truncation: floor, escalation, restart, log line

Status: APPROVED (all four + anti-over-escalation fence). TDD: RED tests before patches.

## Root cause (already diagnosed)

`Context compression summary was truncated (finish_reason=length)` is the endpoint's
own output floor, not a backend/connection fault. 474/2,370 all-time compression calls
end `finish_reason=length`; 421/440 October hygiene failures are this. The floor scales
16,384→32,768 across 0→300,000 input tokens, granting only ~17–18K at real compression
sizes (~17–30K), and `deepseek-v4.1-flash` (99.6% of compression calls) burns a large
hidden reasoning trace before the summary — so the visible summary runs out of budget.

The endpoint also deliberately does NOT escalate on `length` (`_empty_content_reason`
treats it as "a caller budget problem rather than a backend defect"), but the caller here
IS the endpoint's own floor — so the truncated stream is handed to Hermes, which aborts.

## The four fixes

### Fix 1 — Raise the output floor (primary prevention)

- `SESSION_COMPRESSION_MIN_MAX_TOKENS` default `16384` → `24576`.
- `SESSION_COMPRESSION_MAX_MAX_TOKENS` default `32768` → `49152`.
- This is a ceiling, not reserved spend: 1,882 calls already finish early (avg 12,975 out).
- If a systemd drop-in overrides these via env, update it too (path TBD, see Fix 3).

### Fix 2 — Escalate on `finish_reason=length` for `session_compression` only

Escalate a truncated compression to the next summariser rung instead of surfacing a
user-visible failure. Two surfaces:

- **Non-streaming** (incl. `local`/`openai-codex` branch and the non-streaming fallback):
  after the empty-content/malformed/degeneration checks, add a truncation check that
  raises `HTTPException(502, "...truncated...")`.
- **Streaming** (the actual path — 2,369/2,379 compression calls stream): `finish_reason`
  arrives on the final chunk, so it must be known *before* the stream is committed. Extend
  `_preflight_openai_stream` so that, for `session_compression`, after first content it
  drains to `[DONE]` tracking `finish_reason`; set `pf.status = "truncated"` when it is
  `length` (the buffer then already holds the full stream). `proxy_to_backend_streaming`
  escalates on `truncated` exactly as it does on `empty`. `_resume_stream` replays the
  now-complete buffer (its existing `[DONE]` early-return already handles a fully-buffered
  preflight); preserve `finish_reason`/usage stats by stashing them on `pf` during the
  drain and copying into `stats` before that early return.

### Fix 3 — `Restart=always` on the `biggie-llm-endpoint` unit

`Restart=on-failure` does not recover a clean SIGTERM (the 13:50 outage). Change to
`Restart=always` so a stray stop can't leave the endpoint dead. Requires locating the unit
file (blocked earlier; resolve with `sudo systemctl show -p FragmentPath`).

### Fix 4 — Fix the misleading floor log line

`biggie_llm_endpoint.py:4523` logs `SESSION_COMPRESSION_MIN_MAX_TOKENS` (the constant)
instead of the computed floor actually applied to `body["max_tokens"]`. Log `body["max_tokens"]`.

## The fence (user requirement)

Escalation must never land compression on a pro/plan model. This is **already enforced**
by the existing summariser ladder and must be pinned, not re-invented:

- `_session_compression_ladder` = `[deepseek-v4.1-flash, glm-5.3, glm-5.2, deepseek-v4-pro]`
  (plan models excluded via `_is_reserved_from_compression`).
- `escalate_on_failure(workload_type="session_compression")` → `_next_session_compression_rung`
  advances **one rung**, and fails closed (`""`) on exhaustion rather than promoting.
- New work: (a) add `"truncation"` to the no-breaker-trip `error_type` tuple in
  `router.py:1900` so a truncation escalation does not open the circuit (it's an
  output-quality/budget issue, like `empty_content`); (b) map the endpoint's
  `"...truncated..."` detail to `fail_type = "truncation"`.

## Test files

- `test_batch9_compression_output_floor_scale.py` — assert new floor calibration
  (`MIN >= 24576`, floor at ~24.6K context > 19,314 = max observed truncated output).
- `test_batch10_streaming_finish_reason.py` — streaming truncation → preflight status
  `truncated` (or the new drain behaviour).
- `test_b29_ladder_invariants.py` — truncation escalation stays on the ladder (never pro).
- New `test_compression_truncation_escalation.py` — `_empty_content_reason`/truncation
  helper returns `length`; non-streaming truncation raises; `fail_type` mapping.
- `test_b28_finish_reason_guard.py` — existing finish_reason guard; extend if it already
  covers the `_sse_finish_reason` path.

## Order (TDD batches)

1. RED: floor calibration test + truncation-helper test.
2. GREEN: Fix 1 (constants) + `_response_is_truncated` helper.
3. RED: streaming preflight-drain + escalation-fence tests.
4. GREEN: Fix 2 (preflight drain, `truncated` status, endpoint `fail_type` map, router
   `"truncation"` tuple).
5. Fix 3 (unit file) + Fix 4 (log line) — no new logic to unit-test; verify via journal.

Each batch: RED → GREEN → commit. No merge/deploy without explicit sign-off.
