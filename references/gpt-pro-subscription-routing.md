# GPT Pro tier routing — architecture for Plus-only / Pro-only / both

## The one fact that shapes everything

**Plus and Pro are not two providers — they are two subscription tiers on the**
**same ChatGPT account.** Same endpoint (`chatgpt.com/backend-api/codex`), same
OAuth pool, different *usage budget* (Pro ≈ 5x Plus on Codex) and *model access*
(Pro unlocks frontier models a Plus plan won't serve).

Consequence: **you cannot run Plus and Pro concurrently on one account.** If the
goal is literal "best of both at the same time," that requires **two ChatGPT
accounts** ($20 + $200 = $220/mo when both active). If the goal is cheap on/off
peak capacity, **one account with a runtime tier toggle** is the answer — buying
Pro resets/raises an exhausted Plus account's budget, which is exactly the
current `exhausted` state the router is in right now.

Key operator lever: flipping Plus→Pro on an exhausted account is effectively a
**budget reset + a raise**, because OpenAI grants the new entitlement's limits on
next call. No code change needed — the same 3 credentials keep working.

## Design: an entitlement abstraction

Treat each `(provider, entitlement)` pair as its own **entitlement domain** with
its own credential sub-pool and its own rate-limit/circuit state. This one
abstraction degenerates to all three target configurations:

| Config | What it maps to | Cost while active |
|---|---|---|
| Plus-only | one domain, tier=`plus` | $20 |
| Pro-only | one domain, tier=`pro` | $200 |
| Plus **and** Pro (true concurrency) | two domains: `openai-codex-plus`, `openai-codex-pro` | $220 |

### 1. Entitlement on the openai-codex credential pool (auth.json)

Give each openai-codex pool an `entitlement` field (`plus`|`pro`), defaulting to
`plus`. Current auth.json has one pool of 3 creds → becomes `entitlement: plus`.
A two-account setup adds a second pool `openai-codex-pro` with `entitlement: pro`
(its own 1–3 tokens from the second account's device-code login).

### 2. Required entitlement per model (models.py MODEL_REGISTRY)

Add `"min_entitlement"` to the openai-codex entries:

```python
"gpt-5.5":       {..., "tier": 10, "min_entitlement": "plus"},
"gpt-5.6-luna":  {..., "tier": 11, "min_entitlement": "plus"},
"gpt-5.6-terra": {..., "tier": 12, "min_entitlement": "plus"},
"gpt-5.6-sol":   {..., "tier": 13, "min_entitlement": "plus"},
"gpt-6-astra":   {..., "tier": 14, "min_entitlement": "pro"},  # Pro-only
```

Rule enforced by the router: a ChatGPT model is routable **only if the account
entitlement ≥ `min_entitlement`**. So under Plus, astra stays force-only (as
today). Under Pro, the operator may allow astra into auto-routing (drop it from
`EXCLUDED_FROM_AUTO_ROUTING`, raise the clamp 13→14) — that's the "use Pro's
best" lever, kept as an explicit operator decision, not automatic.

### 3. Runtime subscription state (the on/off switch)

Persist a small state record (same mechanism as model circuit-breaker state):
`{account: "openai-codex", entitlement: "plus", changed_at}`. Expose:

- `POST /admin/subscription {"tier":"pro"}` and `"plus"` — flips the whole pool's
  entitlement at runtime. No deploy, no restart, no re-auth.
- `GET /admin/subscription` — current tier + which models it unlocks.
- The flip also **resets rate-limit/circuit state for the pool** (mark all models
  available), because a tier change is a budget change — stale "exhausted" state
  must not carry over and wrongly block a freshly-upgraded account.

### 4. Routing integration

- `_select_model(min_tier)`: skip a ChatGPT model unless entitlement is met; among
  those, cheapest capable as today.
- `_build_fallback_chain` / `_select_tool_capable_model` / `escalate_on_failure`:
  when walking for an openai-codex model, require `min_entitlement` satisfaction.
  For two-account concurrency, merge both domains into the chain so escalation
  walks Plus-sol → Pro-sol → Pro-astra rather than failing closed.
- Ollama-cloud branch, tool-capability gate, clamp, limp-home, private mode:
  **unchanged** — they are orthogonal to entitlement.

## Recommended path

1. **Registry**: add `min_entitlement` to MODEL_REGISTRY (mechanical; update the
   derived-cost tests that assert the top model name — see SKILL pitfalls).
2. **Runtime state + admin endpoints**: `set_subscription_tier` (persisted),
   `GET/POST /admin/subscription`, tier-flip resets pool model-availability.
3. **Router gate**: thread entitlement check through `_select_model`,
   `_build_fallback_chain`, `escalate_on_failure`, `_select_tool_capable_model`.
4. **Tests**: plus-tier routes 5.6-sol but not astra; pro-tier unlocks astra; a
   pro-requiring model with plus entitlement is skipped in the chain; tier-flip
   resets availability. Full suite green.
5. **Operate**: keep `entitlement=plus` normally. At expected peak, or the moment
   5.6-sol/5.5 start 429ing, `POST /admin/subscription {"tier":"pro"}` for the
   budget reset+raise; flip back when the peak passes.

Two-account concurrency only if you actually need simultaneous throughput at both
tiers — the extra $200/mo is rarely worth it for peak handling, which the single
toggle already solves.
