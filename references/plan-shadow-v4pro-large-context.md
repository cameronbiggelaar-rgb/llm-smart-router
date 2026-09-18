# Plan — Shadow head-to-head: is v4-pro a safe large-context summariser?

Status: **RUN COMPLETE — clean data. One confound remains (prompt had no
instruction); a single real-task confirmation run settles the collapse decision.**

## Why this exists

Path 2 (shadow test before collapsing the ladder) was chosen because the
production data is too polluted to conclude. That instinct was right twice over:
both the original premise AND the first "corrected" figure turned out to be
artifacts.

## Correction 1 — the original premise was wrong

Claim: "v4-pro strictly dominates the glm rungs — cheaper ($0.66 vs $1.40),
higher tier (8 vs 6/6.5), proven to 439K context."

**The 439K is the max context v4-pro was *handed*, not proof it *succeeds*
there.** So this claim was unfounded.

## Correction 2 — the "v4-pro only 25% at >=150K" figure is ALSO an artifact

That number came from `/tmp/investigate_glm10.py`:
`deepseek-v4-pro, 262 calls, 66 saw_content` for `session_compression` at
>=150K, last 5 days. Drilling in:

* **All 262 rows are from 2026-09-14** — the `empty_content` storm day, not a
  representative window.
* Of those 262: **196 are `streaming_degraded_to_non_streaming` with
  `saw_tool_calls=1`** — a *transport* degradation, and they did emit tool calls,
  so `saw_content=0` does not mean "produced nothing".
* The 46 `empty_content` rows have **`saw_content=1`**.

So "25% saw content" measures a Sep-14 transport incident plus a `saw_content`
logging quirk, **not** v4-pro's ability to serve 150K+.

## Correction 3 — v1 of the harness was invalid (3 defects)

v1 (`scripts/shadow_v4pro_large_context.py`, first revision) reported all three
models as "empty" at 150K. Every bit of that was harness error:

1. **`max_tokens=256`, reused from `flash_ceiling_headtohead_v2.py`.** These
   models emit *reasoning* tokens before `content`. 256 was spent entirely on
   reasoning, so `content` came back empty with `finish_reason=length`. Scoring
   that as an empty stream is exactly the misread this test exists to avoid.
   **Fix:** `MAX_TOKENS = 16384`, matching what production injects for
   `session_compression`. Also record `reasoning_len` and score
   `reasoning_only` separately from `empty_stream`.
2. **Seed exceeded the target.** The v2 harness's default seed session
   (`20260612_202206_c89a7e`) is now empty, and the replacement
   (`20260911_065143_8e021a05`) is ~200K tokens — larger than the 150K target,
   so filler was **0 chars** and the "150K" test never reached its band.
   **Fix:** cap the seed at 40K chars, trimming from the oldest end.
3. **No account-cap detection.** A transient account-wide 429 turned every call
   into a 503, which v1 would have scored as a model failure. **Fix:** a
   preflight call; abort loudly rather than score 503s.

Validated after fixes at 30K (cheap): v4.1-flash `finish=stop content=1480`,
glm-5.2 `finish=stop content=1061`, plus 13,116 / 3,651 reasoning tokens.

## Run 1 (09:33) — flash + glm clean; v4-pro circuit-held

First live run of the corrected harness:

* `v4.1-flash`: 5/6 served content (1 reasoning-only at 200K); **0 empty**.
* `glm-5.2`: 6/6 served content; **0 empty**.
* `v4-pro`: **all 6 returned instant 503** (`all_models_exhausted`).

The v4-pro failures were NOT model results. At 09:18:18 the circuit breaker
opened for `deepseek-v4-pro` (1800s hold) after 3 consecutive failures — all
caused by the earlier credit-cap 429s. Every probe inside that window got an
instant breaker-503 while flash/glm (unheld) sailed through. Confirmed via
`router_state.db` `model_state` (`circuit_open_until`, `consecutive_failures=3`,
`last_error=rate_limit`). Re-ran v4-pro alone after the breaker closed at
09:48.

## Run 2 (09:50) — v4-pro alone, breaker clear

| size | run | result |
|---|---|---|
| 150K | r1 | `fin=stop content=729` (17.6s) |
| 150K | r2 | `fin=length content=0 reason=33695` (80.3s) — reasoning runaway |
| 200K | r1 | `fin=stop content=344` (11.6s) |
| 200K | r2 | `fin=stop content=475` (8.6s) |
| 250K | r1 | `fin=length content=0 reason=33458` (71.7s) — reasoning runaway |
| 250K | r2 | `fin=stop content=333` (9.3s) |

**v4-pro: 4/6 served content, 0 empty-stream, 0 errors, 0 escalated.**

## Clean result — the empty-stream failure is gone

Under clean conditions (cap cleared, breaker closed, 16,384 budget), **zero
models empty-streamed at any size**:

| model | served content | reasoning-only | empty-stream | errors |
|---|---|---|---|---|
| v4.1-flash | 5/6 | 1 | **0** | 0 |
| glm-5.2 | 6/6 | 0 | **0** | 0 |
| v4-pro | 4/6 | 2 | **0** | 0 |

This formally retires both prior claims: "v4-pro 25% at >=150K" (storm-day
artifact) and "proven to 439K" (now actually evidenced: v4-pro served real
content at 150K, 200K, 250K).

## The one confound — and the one remaining question

v4-pro served a usable summary 4/6 times but on **2/6** runs it emitted ~33K
reasoning tokens and hit the output cap before writing any content
(`finish=length`, 80s). glm-5.2 never does this; flash did once. Same *outcome*
as empty-stream (no summary), different cause (output budget vs reasoning
mismatch).

**Confound:** the synthetic payload carries NO instruction — the models literally
reply "I don't see a question or task." A reasoning model burning 33K tokens to
conclude "there's no task" is plausibly an artifact of the confusing no-task
prompt, not behaviour a real "summarise this session" call would trigger.

**Remaining run (cheap, one-shot):** same 3 sizes, v4-pro + glm-5.2 (control),
with a real *"summarise this conversation"* instruction. If v4-pro holds under a
real task → collapse both glm rungs. If it still runs away → keep glm-5.2 as
large-context insurance, drop only dormant glm-5.3.

## Decision rule

* v4-pro holds a usable summary under a real instruction → collapse ladder to
  `v4.1-flash → v4-pro` (drop glm-5.3 + glm-5.2), TDD batch, sign-off, restart.
* v4-pro still runs away → **keep glm-5.2**, drop only glm-5.3 (capped at 150K
  anyway, dormant since Sep 15).
* Roughly parity but cheaper → prefer v4-pro on cost, but require real quality
  parity, not just cost.

## Independent finding worth acting on regardless

`glm-5.3` bare is at **3.2%** served at >=150K (62 calls, 2 with content) vs
`glm-5.3:cloud` at 100% (13/13). The bare tag looks like the same
alias/registry trap seen with `deepseek-v4-flash`, and it is already capped at
150K by `MODEL_CONTEXT_CEILING`. Worth checking on its own — separate from the
v4-pro question.
