# Plan — retire two dead Ollama references + re-examine glm summariser rungs

**Date:** 2026-09-18
**Trigger:** ladder/catalog optimality review.

## Findings (grounded in live catalog + 7d router DB)

### #1 — Two dead references (real bugs, clean fix)

Both are in `MODEL_REGISTRY` but **absent** from the live `ollama.com/v1/models`
catalog (20 models). Routing to either would fail-and-escalate, but neither has
been exercised in 7 days so both are dormant:

- `glm-5` (tier 4) — catalog has `glm-5.1/5.2/5.3` but no bare `glm-5`.
- `deepseek-v3.1:671b` (tier 9) — rung 5 of the summariser ladder AND in cost
  order; not in the catalog at all.

### #2 — glm-5.3/glm-5.2 rungs (the "cost anomaly" was spike-inflated)

Re-examination of the 7d aggregate that first flagged this:

- `glm-5.3:cloud` compression spend **$229** = $223 on **2026-09-11 alone** — the
  documented `deepseek-v4.1-flash:cloud` empty_content cascade (already recovered).
  Steady-state: 33 calls 09-14, 1 call 09-15, **0 since**.
- **Last 2 days:** v4.1-flash:cloud = 896 calls $4.07; glm-5.3/5.2 ≈ $0.59 total
  (mostly the retiring flash). glm is negligible in steady state.
- `deepseek-v4.1-flash:cloud` already serves **2,262 ≥150K contexts / 7d** with
  ~85% success — the "flash unsafe on large contexts" belief is empirically dead;
  flash only escalates on empty_content (~15%).
- glm-5.3/5.2 traffic is **escalation-only** (empty_content from flash) or rare
  native selection when flash is unavailable. They earn their rung as the
  fail-closed summariser fallback — removing them would force empty summaries or
  burn ChatGPT plan capacity.

**Conclusion #2: no code change.** The rungs are correct. The real lever (flash
handling large contexts) is already pulled. A further optimisation (shadow-spike
whether `deepseek-v4-pro` @ $0.66 can replace glm-5.3 @ $1.40 as the
empty_content escalation target) is a <$1/day prize and needs a quality spike,
not a blind edit.

## Change set (#1 only)

1. `scripts/models.py` — remove `glm-5` and `deepseek-v3.1:671b` from
   `MODEL_REGISTRY`.
2. `scripts/router.py` — remove `deepseek-v3.1:671b` from the summariser ladder.
3. Update pinned tests that reference the two names.

## Routing consequence

- tier 4: was `glm-5` → now `glm-5.1` ($1.00, tier 5) as cheapest meeting ≥4…
  actually `minimax-m2.7:cloud` ($0.30, tier 4) already sits below glm-5 and wins
  tier 4 — so glm-5 was never selected. No behaviour change.
- tier 9: was `deepseek-v3.1:671b` → now `gpt-5.5` (tier 10). Only reachable via
  explicit tier-9 demand; rare.
