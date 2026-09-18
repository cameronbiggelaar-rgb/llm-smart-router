# Plan — Shadow head-to-head: is v4-pro a safe large-context summariser?

Status: **harness built + validated; live run BLOCKED on an Ollama account credit cap.**

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
logging quirk, **not** v4-pro's ability to serve 150K+. Neither an endorsement
nor a condemnation. The band remains genuinely unmeasured.

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
This is what proved defect 1 — the same two models scored "empty" under a
256-token budget and healthy under a 16,384 budget.

## The live blocker (current)

`ollama-cloud` returned an explicit account-level cap:

> `usage credits auto reload monthly max reached, increase your monthly max or
> add usage credits at https://ollama.com/settings`

Consequences observed (2026-09-19 ~09:19 AEST):

* All **11** ollama-cloud models held simultaneously; v4-pro held longest
  (**240s** vs 60s for the rest) because it carries the most production load.
* Production fell into **limp-home** (routing to the local model), self-exited
  after 66s, then re-entered intermittently.
* A minimal 1-token direct call still returns **HTTP 200**, so this is a
  threshold/contention cap, not a hard block.

Running the 150K/200K/250K probe under this condition is counterproductive:
probe traffic competes with production for a capped allowance, and any 503s are
indistinguishable from model failure — the exact contamination that ruined v1.

## What the run does once unblocked

Three-way forced-model head-to-head at 150K / 200K / 250K:

* `deepseek-v4.1-flash:cloud` — incumbent workhorse ($0.15 in)
* `glm-5.2:cloud`             — current large-context insurance ($1.40 in)
* `deepseek-v4-pro`           — proposed replacement ($0.66 in)

Per run: `used_model`, `served_content`, `reasoning_only`, `empty_stream`,
`finish`, `prompt_tokens`, `latency_s`. `X-Compression-Level: off` so the router
cannot pre-collapse the payload.

Estimated probe cost: ~**$3** total (18 large-context calls). Cost is not the
blocker; the capped allowance is.

## Decision rule

* v4-pro holds **>=81%** served at >=150K → collapse ladder to
  `v4.1-flash → v4-pro` (drop glm-5.3 + glm-5.2), TDD batch, sign-off, restart.
* v4-pro falls meaningfully below glm-5.2 → **keep glm-5.2**, drop only
  glm-5.3 (capped at 150K anyway).
* Roughly parity but cheaper → prefer v4-pro on cost, but require real quality
  parity, not just cost.

## Independent finding worth acting on regardless

`glm-5.3` bare is at **3.2%** served at >=150K (62 calls, 2 with content) vs
`glm-5.3:cloud` at 100% (13/13). The bare tag looks like the same
alias/registry trap seen with `deepseek-v4-flash`, and it is already capped at
150K by `MODEL_CONTEXT_CEILING`. Worth checking on its own — separate from the
v4-pro question.
