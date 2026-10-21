# Plan B46 — Model-Mix Refresh (price book vs live Ollama Cloud)

Status: APPROVED by user ("Yes do it and make sure we catch all references to all
models as we do the work"). This is a **price-book change**, not a config hot-reload:
it edits `MODEL_REGISTRY`, which feeds `MODEL_COST_ORDER` and
`MODEL_CAPABILITY_TIERS` and therefore the escalation ladder. Plan doc → RED tests →
registry/config edits → GREEN, per the refactor convention.

## 1. Problem

`MODEL_REGISTRY` (scripts/models.py) is dated 2026-09-13 and has drifted from the
live Ollama Cloud catalog. Consequences measured 2026-10-22:

- **`deepseek-v4-pro` carries ~90% of traffic** (13,726 / 15,211 calls in 7 days) at
  a price registered as off-peak ($0.66/$1.98) while the vendor also bills a peak rate
  ($1.32/$3.96, 12:00–18:00 UTC weekdays). Peak-window calls cost ~2× the assumed
  figure, and the ledger understates spend.
- **`minimax-m3` ($0.60/$2.40)** and **`minimax-m3.1-flash` ($0.15/$0.60)** exist in
  the live catalog but are unregistered — the router cannot use cheaper/comparable
  alternatives.
- **`minimax-m2.7:cloud`** is registered but has served **0 calls in 7 days**, is
  superseded by m3/m3.1, and its $0.30/$1.20 price no longer matches the catalog.

## 2. Changes (four edits, one retirement)

| # | model | action | price in/out (per 1M) | cached | tier | date |
|---|---|---|---|---|---|---|
| 1 | `minimax-m3` | **register** | $0.60 / $2.40 | — (omit) | 8 | 2026-10-22 |
| 2 | `minimax-m3.1-flash` | **register** | $0.15 / $0.60 | — (omit) | 4 | 2026-10-22 |
| 3 | `minimax-m2.7:cloud` | **retire** | — | — | — | — |
| 4 | `deepseek-v4-pro` | **reprice** | $1.32 / $3.96 (peak) | $0.033 | 8 | 2026-10-22 |

### Decision notes

- **deepseek-v4-pro peak pricing.** The registry carries one price; the ledger's
  governing principle is *never understate* (see `cost_of_call`). Register the peak
  rate ($1.32/$3.96) as the single figure and document the off-peak half-rate
  ($0.66/$1.98) in a comment + a follow-up note. Time-aware peak/off-peak costing is
  a **separate mechanism** (new schema column + schedule logic) and is explicitly
  **out of scope** here — noted as a follow-up, not built unrequested.
- **cached omitted for the new minimax models.** The vendor's cached-token rate for
  `minimax-m3` / `minimax-m3.1-flash` was not verifiable from the live pricing page, so
  no `cached` key is registered (same as the existing `minimax-m2.7:cloud`). Cached
  input tokens bill at the fresh `input` rate — the *never-understate* direction.
- **minimax-m3 tier 8.** Same capability band as deepseek-v4-pro (frontier
  coding/agentic, 1M ctx, thinking+multimodal). At $0.60 it is cheaper than
  deepseek-v4-pro at *either* price, so it becomes the cheapest tier-8 model and wins
  the tier-6/7/8 bands. This is the intended optimal-mix outcome. A shadow/canary
  validation (existing `monitor_split_canary.py` pattern) can demote it if quality
  underperforms — noted, not built here.
- **minimax-m3.1-flash tier 4.** Displaces `glm-5.3-flash` as the tier-4 pick. It is
  **$0.10/M more expensive on output** ($0.60 vs $0.50) but claims ~3× faster online
  inference + a generational leap in agentic/coding/cyber. A deliberate
  performance-vs-price trade, documented here rather than hidden.
- **kimi-k2.7-code, mistral-large-4, kimi-k3** are **not registered** in this change
  (noted, not prioritised; kimi-k2.7-code is cheaper than *peak* pro but pricier than
  off-peak pro, so it is not a clear win yet).

## 3. Routing consequence (pinned by RED→GREEN tests)

New `MODEL_COST_ORDER` (ascending input, then tier):

```
$0.00  llama3.1:8b(t1), dolphin3(t2)
$0.15  deepseek-v4.1-flash(t3), minimax-m3.1-flash(t4), glm-5.3-flash(t5.5)
$0.60  minimax-m3(t8)
$1.32  deepseek-v4-pro(t8)
$1.40  glm-5.3(t6), glm-5.2(t6.5)
$15+   gpt-5.5 .. gpt-6-astra(t10..14)
```

Band flips (cheapest model meeting min_tier):

| min_tier | before | after |
|---|---|---|
| 1 | llama3.1:8b | llama3.1:8b (same) |
| 2 | dolphin3 | dolphin3 (same) |
| 3 | deepseek-v4.1-flash | deepseek-v4.1-flash (same) |
| 4 | glm-5.3-flash | **minimax-m3.1-flash** (flip) |
| 5 | glm-5.3-flash | glm-5.3-flash (same) |
| 6 | deepseek-v4-pro | **minimax-m3** (flip) |
| 7 | deepseek-v4-pro | **minimax-m3** (flip) |
| 8 | deepseek-v4-pro | **minimax-m3** (flip) |
| 9/10 | gpt-5.5 | gpt-5.5 (same) |

Four bands flip (tier 4, 6, 7, 8). Tier 5 is unchanged: `minimax-m3.1-flash` sits at
tier 4, so `glm-5.3-flash` (tier 5.5) still wins the tier-5 floor. All four flips are
asserted in the test suite so a future price edit that silently moves them again fails
loudly.

## 4. Reference sweep — every file touching these models

Files to edit (registry/config/tests/docs):

1. `scripts/models.py` — `MODEL_REGISTRY` (add 2, retire 1, reprice 1), ratio comment block.
2. `~/.hermes/config.yaml` — `providers.biggie-llm.models` **and** `fallback_providers`:
   add `minimax-m3`, `minimax-m3.1-flash`; remove `minimax-m2.7:cloud`. (Two lists must
   stay in sync — enforced by `test_registry_config_alignment.py`.)
3. `tests/test_price_book_routing.py` — update band assertions (tier 4/5/7/8 flips;
   remove the `minimax-m2.7` cost-order assertion).
4. `tests/test_price_book_versioning.py` — update `minimax-m2.7` position assertion and
   the deepseek-v4-pro `$0.66` comments.
5. `tests/test_remove_dead_ollama_refs.py` — `test_live_catalog_names_all_registered...`
   asserts `minimax-m2.7:cloud` must *remain* registered; remove it from that list.
6. `tests/test_registry_config_alignment.py` — no code change expected (it is
   config-driven), but it must pass after the config edit.
7. `scripts/test_biggie_endpoint.py` — line 139 asserts mid-coding routes to
   `("minimax-m2.7:cloud", "glm-5")`; both names are retired/dead. Update to the
   post-change expectation (minimax-m3.1-flash / minimax-m3).
8. `scripts/check-routing-stats.py` — `canonical_model` alias map carries the dead
   `deepseek-v4-flash:cloud` alias; add `minimax-m3`/`minimax-m3.1-flash` aliases if
   state.db can carry the `:cloud` suffix.
9. `scripts/router.py` — `_session_compression_ladder` and any hardcoded model-name
   tuples referencing `minimax-m2.7` (see step 8).
10. `scripts/unit_economics.py` — no code change (prices flow from `MODEL_REGISTRY`
    via `seed_prices`); verify only.
11. `scripts/routing_table.yaml` — tier-reference comment block lists stale names
    (`deepseek-v4-flash`, `deepseek-v3.1:671b`, `glm-5`, `minimax-m2.7`); update the
    comment to reflect the post-change model set.
12. `SKILL.md` — model tables (lines ~195–210, ~340–350) listing `minimax-m2.7` and
    `deepseek-v4-pro` prices.
13. `ARCHITECTURE.md` — model/price tables.
14. `references/*.md` — historical plan docs may mention these names; **historical
    docs are frozen evidence and are NOT rewritten.** Only add the new plan and, where
    a doc is a *living* reference (e.g. `ollama-unit-economics.md`), update the live
    price table.

### Reference inventory (grep, 2026-10-22)

- `minimax-m2.7` — SKILL.md, scripts/test_biggie_endpoint.py, scripts/models.py,
  scripts/router.py, scripts/routing_table.yaml, ARCHITECTURE.md,
  tests/test_remove_dead_ollama_refs.py, tests/test_gpt56_gpt6_lanes.py,
  tests/test_registry_config_alignment.py (docstring), tests/test_price_book_versioning.py,
  tests/test_price_book_routing.py, + historical reference docs.
- `minimax-m3` — only `references/model-economics-2026-09-13.md` (historical).
- `minimax-m3.1-flash` — nowhere (new).
- `deepseek-v4-pro` — 35+ files (workhorse; only the price-relevant ones change).

### Additional sites found during implementation (not in the initial sweep)

- `tests/test_model_retirement.py` — `test_tier_7_band_resolves_to_deepseek_v4_pro`
  now resolves to `minimax-m3` (B46 flip; test renamed to `..._minimax_m3`).
- `tests/test_gpt_budget_routing.py` — `test_denied_routine_still_gets_a_capable_model`
  degradation target now `minimax-m3` (B46 flip).
- `scripts/router.py:525` — comment example `minimax-m2.7:cloud` → `minimax-m3:cloud`.
- `scripts/check-routing-stats.py` / `scripts/unit_economics.py` — verified: no
  `minimax-m2.7` reference (no edit needed).

## 5. Out of scope (explicitly deferred)

- Time-aware peak/off-peak costing (new schema column + schedule).
- Registering kimi-k2.7-code / mistral-large-4 / kimi-k3.
- Shadow canary to validate minimax-m3 quality before the tier-8 flip.

## 6. Verification

- `python3 -m pytest tests/` green (baseline 1019 passed / 13 pre-existing failures —
  the 13 must remain 13, no new failures).
- `test_registry_config_alignment.py` passes against the real config (two-list sync).
- Endpoint does **not** need a restart for config.yaml (hot-applies ~60s), but
  `MODEL_REGISTRY` lives in the endpoint process — **flag to the user** that a restart
  is required for the new price book to take effect, and it must be done externally.
