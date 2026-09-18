# Plan — Retire `deepseek-v4-flash` and `qwen3.5` (Ollama model retirement)

## Trigger

Ollama will retire two tags on **2026-09-25**:

- `deepseek-v4-flash:0731`
- `qwen3.5:397b`

A live probe of `ollama.com/v1/models` (2026-09-19) confirms both bare names in our
`MODEL_REGISTRY` are **aliases of those retiring tags** — there is no un-tagged
`deepseek-v4-flash` or `qwen3.5` "latest" to fall back to:

- `qwen3.5:397b` is the *only* qwen3.5 entry; no qwen replacement exists in the list.
- `deepseek-v4-flash:0731` is the *only* v4-flash entry.

Not retiring (verified present): `deepseek-v4.1-flash` (our session-compression
workhorse), `deepseek-v4-pro:0813`, `glm-5.3`, `glm-5.2`, `glm-5.3-flash`,
`minimax-m2.7`.

## Scope (larger than a 2-line delete)

Both models are load-bearing in more than the summariser ladder:

| Model | Registry | TOOL_CAPABLE | Ladder rung | Routing band |
|---|---|---|---|---|
| `deepseek-v4-flash` | line 158 | **line 123** | rung 2 | tier 3 (shadowed by v4.1-flash) |
| `qwen3.5` | line 166 | — | rung 5 | **tier 7** (normal-chat) |

`qwen3.5` is tier 7 and resolves the tier-6/7 normal-chat bands (pinned by
`tests/test_price_book_routing.py::test_tier_6_band_selects_the_cheaper_capable_model`
and the `(7, "qwen3.5")` band in `test_unaffected_bands_still_select_as_before`).

## Changes

### 1. `deepseek-v4-flash` → absorbed by `deepseek-v4.1-flash`

- Remove `"deepseek-v4-flash"` from `MODEL_REGISTRY` (`models.py:158`).
- Remove from `TOOL_CAPABLE_MODELS` (`router.py:123`).
- Remove from `_session_compression_ladder` (`router.py:1640`).

`deepseek-v4.1-flash` is already rung 1, cheaper ($0.15/$0.60 vs $0.22/$0.66), and
not retiring. No capability loss for tool-flagged requests: `deepseek-v4-pro` and
`glm-5.3` remain tool-capable ollama fallbacks.

### 2. `qwen3.5` → tier-7 band resolves to `deepseek-v4-pro`

- Remove `"qwen3.5"` from `MODEL_REGISTRY` (`models.py:166`).
- Remove from `_session_compression_ladder` (`router.py:1643`).

No qwen replacement exists. The next-cheapest tier ≥7 is `deepseek-v4-pro` (tier 8,
$0.66 — actually *cheaper* than the glm models at $1.40). The tier-7 band now
resolves to `deepseek-v4-pro`. This matches the existing
`model-economics-2026-09-13.md` note flagging qwen3.5 as a 7× latency outlier.

## Test changes

New test file `tests/test_model_retirement.py` asserts the invariants the fix depends on:

1. Neither `deepseek-v4-flash` nor `qwen3.5` appears in `MODEL_REGISTRY`.
2. Neither is routable: `_normalize_model_name`/`MODEL_COST_ORDER`/`MODEL_CAPABILITY_TIERS`
   have no trace of them.
3. Neither appears in `TOOL_CAPABLE_MODELS` (for v4-flash) nor the compression ladder.
4. The tier-7 band now selects `deepseek-v4-pro` (not qwen3.5).

Existing tests that must be updated (they pin the *old* behaviour, which retirement
inverts):

- `tests/test_price_book_routing.py` — tier-6/7 band assertions reference `qwen3.5`.
- `tests/test_router.py` — setup/expected-sets reference both names.
- `tests/test_b28_account_grouping.py` — `OLLAMA_MODELS` list includes `qwen3.5`.
- `tests/test_b29_ladder_invariants.py:82` — `qwen3.5` in a not-reserved assertion.
- `tests/test_report_success_rate.py` — `deepseek-v4-flash` used as fixture data.

## Procedure

TDD batch (per standing workflow):

1. RED — write `test_model_retirement.py` asserting post-retirement invariants;
   update the five existing test files to the post-retirement expectations.
2. Run suite → confirm the retirement tests + updated tests fail against current code.
3. GREEN — apply the registry/ladder/TOOL_CAPABLE edits.
4. Non-vacuity probe — re-introduce one name; confirm the new tests fail again.
5. Full suite green; commit.

## Rollback / safety

- Pure config edit (registry dict + three set/ladder lines + tests). No endpoint
  code-path change.
- Deploy requires an endpoint restart (sibling unit): `sudo systemctl restart
  biggie-llm-endpoint` then `deploy-check`.
- Deadline: must land before 2026-09-25 to avoid a hard routing break.
