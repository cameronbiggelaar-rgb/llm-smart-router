# Price-book correction — B18

Scoped plan. Written before execution per repo convention (`references/plan-*.md`).

## Problem

`MODEL_REGISTRY` in `scripts/models.py` is the single source of truth for prices,
and `unit_economics.seed_prices()` copies it into `model_pricing`. Six registered
models carry rates that disagree with `ollama.com/pricing` (re-verified
2026-09-13). Because `MODEL_COST_ORDER` is *derived* from this registry, the
error is not bookkeeping-only: it orders the escalation chain and therefore
decides what production routes to.

## The trap that makes a naive fix a no-op

`seed_prices()` is idempotent on `(model, effective_from)` — it does
`SELECT 1 ... WHERE model=? AND effective_from=?` and `continue`s on a hit.
Editing a price in the registry while leaving its `"date"` field alone writes
**zero** rows: the stale price stays in `model_pricing` forever, and a live call
keeps being priced at the old rate with no error raised anywhere.

**Rule: price and date move together.** A correction is a new *versioned* row,
not an edit. `price_for(at=...)` already resolves the price in force at a
timestamp, so history is not restated.

## Corrections (vendor-verified, ollama.com/pricing)

| Model | Book (in/out) | Vendor (in/out) | Drift |
|---|---|---|---|
| deepseek-v4-pro | 2.00 / 6.00 | 0.66 / 1.98 | 3.03x high |
| minimax-m2.7 | 1.00 / 3.00 | 0.30 / 1.20 | 3.33x high |
| qwen3.5 | 1.75 / 5.25 | 0.60 / 3.60 | 2.92x high |
| glm-5.3 | 1.50 / 4.50 | 1.40 / 4.40 | 1.07x high |
| glm-5.2 | 1.50 / 4.50 | 1.40 / 4.40 | 1.07x high |
| glm-5.1 | 1.25 / 3.75 | 1.00 / 3.20 | 1.25x high |

New effective date: **2026-09-13**. Superseded rows are retained.

Left alone deliberately:
- `deepseek-v4.1-flash` (0.15/0.60) — not on the public pricing table; already
  treated as vendor-quoted/unverified.
- `glm-5`, `deepseek-v3.1:671b` — not published by the vendor; nothing to
  verify against.
- `gpt-5.*` / `gpt-6-astra` — openai-codex subscription lane. Priced-as-metered
  is a separate pinned defect (phantom spend), an operator decision, not a
  price correction.
- `ratio` values — a different concept (relative compute units), unchanged.

## Measured blast radius

Correcting the book reorders `MODEL_COST_ORDER`:

```
glm-5     6 -> 8      qwen3.5  10 -> 6
glm-5.1   7 -> 9      v4-pro   11 -> 7
glm-5.3   8 -> 10
glm-5.2   9 -> 11
```

Selection flips in exactly **1 of 14** tier bands:

- `min_tier=6`: `glm-5.3` (tier 6, $1.50) → `qwen3.5` (tier 7, $0.60)

That band carries real traffic (2,311 compression calls in the sampled window).
Every other band is unchanged.

## Batches

1. **tests/test_price_book_versioning.py** — the trap, dated versioning,
   corrected values. RED before the registry change.
2. **registry correction** + `KNOWN_DRIFT` updated to corrected truth.
3. **tests/test_price_book_routing.py** — pin the reordered chain and the single
   tier-6 flip.
4. **references/ollama-unit-economics.md** — replace the "book is stale" warning
   with the corrected table.

## Result (executed 2026-09-13)

| Batch | Status |
|---|---|
| 1 versioning tests | done — 32 tests, RED (18 failed) before the registry change |
| 2 registry correction | done — 6 models corrected, all at new date `2026-09-13` |
| 3 routing pins | done — 13 tests, incl. the single tier-6 flip |
| 4 reference table | done — warning replaced with the corrected table |

Suite: **360 passed** (baseline at `e675d88` was 308 passed, 1 skipped).
Preflight: **16/16 OK**.

### Defect found while verifying (pre-existing, fixed here)

`tests/test_shadow_execution.py` imports `biggie_llm_endpoint`, which reaches
`unit_economics._default_conn()` → `Path.home()/.hermes/skills/llm-smart-router/
data/router_logs.db` — the **production** database — and seeds the price book
into it. Every test run therefore wrote to production pricing.

Attribution: reproduces at `e675d88` with the pre-change registry (18 rows
seeded under a redirected HOME), so it is not introduced by this change. Fixed
by `tests/conftest.py` redirecting `HOME` for the session, guarded by
`tests/test_prod_db_isolation.py` (4 tests) so it cannot regress.

Effect on production: the live DB already carried corrected `2026-09-13` rows
before this commit — written by the leak, since the leak seeds whatever registry
the working tree holds. The correction landed in production this way rather than
through a deliberate deploy. Superseded rows were retained, so pre-correction
history still re-prices correctly (`price_for(at='2026-09-10')` → old rate).

### Not yet deployed

The running endpoint (PID 927, started 11:39) imported `MODEL_COST_ORDER` at
startup, so its **in-memory routing order is still the pre-correction one**.
Pricing is live-correct (reads `model_pricing` per request); routing is not.
Restarting the service is required for the tier-6 flip to take effect — flagged,
not performed, since it interrupts the inference path.

## Out of scope

- No cached-input rate column (structural, pinned by a test that must be deleted
  rather than updated when it lands).
- Subscription-vs-metered split.
- Backfilling the 5,276 `cost_unknown=1` rows in the live DB — a production
  write, demonstrated on a copy only.
