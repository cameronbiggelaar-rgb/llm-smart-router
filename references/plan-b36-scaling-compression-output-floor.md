# Plan — Scale the session-compression output floor by input size

**Date:** 2026-09-18
**Scope:** `scripts/biggie_llm_endpoint.py` only. No config, no Hermes-core edits.

## Background

A session in the main DM thread (`20260916_070848_88d7c827`) hit the compression
threshold (~335k tokens vs 96k) and every auto-compression attempt failed,
arming a 30s cooldown ("compression is currently blocked (cooldown:30)").

Root cause chain, reconstructed from the endpoint source + router logs:

1. Hermes's compressor sends native `session_compression` requests with **no
   `max_tokens` cap** — deliberately, so the summariser has room to reason.
2. The endpoint's `apply_session_compression_output_floor` injects
   `max_tokens=16384` whenever workload is detected as `session_compression`,
   **including when the caller sent no cap**.
3. `deepseek-v4.1-flash` (the compression lane model) emits a hidden reasoning
   trace before visible content. A 16,384 budget that is meant to cover
   *reasoning + summary* runs out for very large inputs (the middle window alone
   was ~677k tokens), so the stream ends with `finish_reason=length` and an
   incomplete/absent summary.
4. Hermes treats a truncated summary as unsafe → `summary_generation_aborted` →
   session not rotated → next turn re-triggers.

## Why the floor exists (must not be removed)

The floor's own docstring (lines 185–188): it exists because **low caller
budgets caused reasoning-only empty completions even at small context**. The
skill (llm-smart-router) records that the floor is *load-bearing against
empty-content*: "since the output-budget floor deployed there were 0 empty
events above 150K … failures were output-budget induced, not size induced."

So "don't inject when the caller sent no cap" is **wrong** — it would revert the
empty-content fix. The correct move is to make the floor **scale with input
size** so a genuinely large session gets a budget the summariser can actually
finish under.

## Evidence

- 158 rows since 2026-09-01 with `output_tokens == 16384` exactly (the
  truncation fingerprint). Distribution by input bucket: **67 @ 0–50k, 89 @
  50–100k, 3 @ ≥100k** — i.e. the truncation is *not* driven by huge inputs;
  it is reasoning-trace budget exhaustion even at modest context.
- Model mix of the truncated rows: `deepseek-v4.1-flash:cloud` 141,
  `deepseek-v4-flash:cloud` 13, `deepseek-v4-flash` 5 — all the DeepSeek
  reasoning-trace family.
- `finish_reason` is stored as `''` in **all 159** truncated rows and **all
  2,344** session_compression rows since Sep 15 — the endpoint does not persist
  `finish_reason` at all, so truncation vs natural-stop is currently
  unobservable in the ledger. This is a measurement gap to close in the same
  batch.

## Design

Replace the fixed `16384` with a **bounded linear scale** over input tokens,
and make the scale configurable via env. Keep the existing "explicit caller cap
above the floor is preserved" contract intact.

- `SESSION_COMPRESSION_MIN_MAX_TOKENS` = floor at low input (default 16384,
  unchanged).
- New `SESSION_COMPRESSION_MAX_MAX_TOKENS` = ceiling at high input (default
  32768; env `BIGGIE_SESSION_COMPRESSION_MAX_MAX_TOKENS`).
- New `SESSION_COMPRESSION_INPUT_FOR_MAX` = input size (tokens) at which the
  floor reaches the ceiling (default 300000; env
  `BIGGIE_SESSION_COMPRESSION_INPUT_FOR_MAX`).

Floor is computed as:

    floor = clamp(MIN + (MAX - MIN) * (input_tokens / INPUT_FOR_MAX),
                  MIN, MAX)

Input size is taken from the preflight context-token estimate (the
`features["context_tokens"]` already computed upstream) when available;
otherwise the body's message token estimate; otherwise MIN.

Preserve the existing rule: if the caller already sent an explicit
`max_tokens >= computed_floor`, do not touch it.

Additionally: persist `finish_reason` into the router ledger so truncation
becomes observable (small, non-behavioural addition to the logging row).

## Out of scope

- Gateway / Hermes-core `context_compressor.py` behaviour (truncation-abort) —
  owned by the other thread / core.
- The `prune:nothing_eligible` / lock-contention failure modes — separate.
- Anything in `hei` repo.

## Test strategy (TDD, batch 9)

New `tests/test_batch9_compression_output_floor_scale.py`:

1. **Floor at MIN for low input** — input 0 → floor == 16384.
2. **Floor scales up for mid input** — input 150k → floor strictly between MIN
   and MAX.
3. **Floor == MAX at/above INPUT_FOR_MAX** — input ≥ 300k → floor == 32768.
4. **Explicit caller cap preserved** — body `max_tokens=200000` → unchanged.
5. **Explicit caller cap below floor still raised** — body `max_tokens=2048` →
   raised to computed floor.
6. **Non-session_compression untouched** — returns False, body unchanged.
7. **Env overrides** — set the three env vars, assert the scale uses them.
8. **Non-vacuity probe** — remove the scaling logic, assert the mid-input test
   fails (proves the test actually exercises the scale).

Run via `python3 -m pytest` (skill repo convention; suite lives in `tests/`).
