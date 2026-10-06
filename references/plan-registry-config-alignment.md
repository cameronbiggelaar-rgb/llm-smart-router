# Plan: registry ↔ config.yaml alignment (glm-5.3-flash drift fix)

Defect class: **registry/config drift** — `MODEL_REGISTRY` (source of routing
truth) and `~/.hermes/config.yaml` (source of servable backends) are two
hand-maintained lists with no invariant test binding them. Three models drifted;
one routing decision leaked ~9× cost for two days.

## Evidence (2026-10-06)

1. `glm-5.3-flash` (tier 5.5, $0.15/$0.50) is registered and **selected by the
   router** (42 completed calls, 2026-10-05→06; 48 selection warnings/day in the
   endpoint journal) but absent from `config.yaml`. Every one silently fell back
   to `glm-5.3:cloud` ($1.40/$4.40): $0.337 charged over the window for tokens
   that cost $0.04 at flash prices (~8.5× multiplier).
2. Upstream probe (live, 2026-10-06): `glm-5.3-flash` → HTTP 200 (also
   `data/probe_glm53_flash_sanity.json`: tools, streaming, 45K ctx, all clean);
   `minimax-m2.7` → HTTP 200; `glm-5.1` → HTTP 410 "was retired at
   2026-09-25".
3. `minimax-m2.7:cloud` (tier 4, $0.30/$1.20) is registered, priced, servable,
   and never configured — the $0.30 rung between flash and glm-5.3 is missing
   from the live escalation ladder.
4. `glm-5.1` (tier 5, $1.00/$3.20) is registered but dead upstream (410) and
   never served once (`router_logs`: zero rows). A latent escalation bug of the
   same class the dead-refs retirement fixed for `glm-5`/`deepseek-v3.1:671b`.
5. `config.yaml` lines 34–35 hold a bare `- provider: ollama-cloud` stub —
   the B41 alias drop deleted the `model:` line but left the entry. Harmless
   today, but it masks adds and is drift residue.
6. No test loads the real config: `grep config.yaml tests/*.py` → 0 hits.
   `test_model_registry_consistency.py` only checks internal consistency.

## Fix (TDD: RED first, then GREEN)

1. **RED** — `tests/test_registry_config_alignment.py`, four invariants
   against the REAL config (recovered via repo path `parents[4]`, skipped when
   absent; conftest's HOME redirect is bypassed by patching
   `ep.HERMES_CONFIG_PATH` and clearing the backends cache):
   - every `MODEL_REGISTRY` key resolves to a discovered backend (exact or
     `:cloud`/`:local`/`:ollama` suffix bridge, mirroring the endpoint);
   - every non-local discovered backend (suffix-stripped) is in
     `MODEL_REGISTRY` (the B41 ghost direction);
   - no `fallback_providers` entry lacks a `model:` key (the stub class);
   - `providers.biggie-llm.models` == fallback_providers models + the two
     hardcoded locals (the two-list sync B41 maintained by hand).
   Expected RED: glm-5.3-flash, minimax-m2.7:cloud, glm-5.1 (registry→backend)
   and the stub (fallback_providers).
2. **GREEN-A (config)** — backup, then surgical text edit (never yaml.dump):
   replace the stub entry with `glm-5.3-flash:cloud` and `minimax-m2.7:cloud`
   fallback_providers entries; add both keys to `providers.biggie-llm.models`.
   Validate with `yaml.safe_load` before writing. Hot-applies via the
   endpoint's 60s discovery TTL — no restart.
3. **GREEN-B (registry retirement)** — remove `glm-5.1` from `MODEL_REGISTRY`
   (derived `MODEL_COST_ORDER`/`MODEL_CAPABILITY_TIERS` follow automatically).
   Collateral: drop glm-5.1 from `test_price_book_versioning.py`
   (`VENDOR_PUBLISHED`/`SUPERSEDED`), `test_remove_dead_ollama_refs.py`
   (sanity anchor), `test_price_book_routing.py` (order-pin tuple);
   update SKILL.md (2 rows), ARCHITECTURE.md (1 row), routing_table.yaml tier
   comment, feature_extractor.py comment. Leave `test_gpt56_gpt6_lanes.py`
   `ALL_MODELS` as-is (it already lists retired models as fixture-only names).
   Historical `references/*.md` and price-book rows stay untouched — history
   is never restated.
4. **Verify** — new tests green; full suite = 978+4 pass with only the 13
   pre-existing failures (monitor_split_canary 7, optimiser_evidence_honesty 4,
   shadow_execution 1, streaming_cost_gap 1); commit scoped files only
   (never `data/router_ops.log`, never the pre-existing dirty
   `candidate_routing.yaml` / plan-retire doc).
5. **Live** — after the 60s TTL: `/v1/models` lists both new backends; an
   explicit `{"model": "glm-5.3-flash"}` request serves 200; no new
   "not in discovered backends" warnings in the journal.

## Adjacent, out of scope

- `_pick_fallback_backend` call sites (~4406, ~4854) omit the `is_available`
  argument; production is safe only because the default injects
  `is_model_available`. Fragile, not faulty — separate fix.
- The nightly digest is blind to instant 503s (they never reach router_logs).
- Hermes gateway may cache its view of `providers.biggie-llm.models` until a
  restart; the endpoint (the serving path) hot-applies. Gateway restart not
  required for this fix.