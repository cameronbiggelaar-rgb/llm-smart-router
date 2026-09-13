---
name: llm-smart-router
description: "LLM Smart Router — data-driven model routing to minimise cost. Phase 1: data collection from Hermes sessions DB."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [llm, routing, cost-optimization, data-collection, ml]
    related_skills: [writing-plans, test-driven-development]
---

# LLM Smart Router

> **Goal:** A data-driven system that routes each task to the cheapest model that can successfully complete it, optimising for cost without sacrificing quality.

## Architecture

Two systems live in this skill:

1. **Active router (`biggie_llm_endpoint.py` + `router.py`)** — a live FastAPI endpoint that Hermes points to as a custom provider. It reads Hermes `config.yaml`, extracts task features, routes each request to the cheapest capable model, proxies to the backend, and handles rate limits / circuit breakers / limp-home mode.

2. **Data pipeline (Phase 1, `collector.py`)** — a read-only observer that reads Hermes session data and logs observations to the router SQLite DB for cost analysis.

```
Hermes → biggie_llm_endpoint.py (FastAPI, :8080)
             └─ router.py (route_task, escalate_on_failure)
             └─ proxy_to_backend → cheapest capable model
             └─ rate limits / circuit breakers / limp-home

state.db (Hermes) → RouterCollector → feature_extractor → router_logs.db (SQLite)
 ↓
 report.py (cost summary)
```

## Phase 1: Data Collection (current)

A cron job runs every 6 hours, reads the Hermes sessions DB, extracts features from each session, and writes observations to the router SQLite database.

### Files

| File | Purpose |
|---|---|
| `scripts/biggie_llm_endpoint.py` | **Live FastAPI router** — reads Hermes config, routes to cheapest capable model, proxies, SSE streaming, compression, circuit breakers, limp-home |
| `scripts/router.py` | `route_task`, `escalate_on_failure`, model health / circuit breakers, fallback chains |
| `scripts/models.py` | **`MODEL_REGISTRY`** (single source of truth) + derived cost tables, SQLite schema |
| `scripts/feature_extractor.py` | Task classification, feature extraction from session data |
| `scripts/collector.py` | RouterCollector — reads Hermes DB, writes to router DB |
| `scripts/report.py` | Cost savings report generator |
| `scripts/test_biggie_endpoint.py` | Endpoint-level tests (streaming, compression, tool finish_reason) |
| `tests/` | Unit + integration tests (streaming iterator reuse, structural compression, model-registry consistency) |
| `ARCHITECTURE.md` | Full architecture, module contracts, test strategy |

### Commands

```bash
# Run data collection manually
cd ~/.hermes/skills/llm-smart-router/scripts
python3 collector.py

# Generate cost report (last 7 days)
python3 report.py --days 7

# Generate cost report as JSON
python3 report.py --days 30 --json

# Generate report from a specific date
python3 report.py --since "2026-06-01"

# Run tests
cd ~/.hermes/skills/llm-smart-router
python3 -m pytest tests/test_router.py -v
```

### Cron Job

- **Name:** LLM Router Data Collection
- **Schedule:** Every 6 hours
- **Script:** `~/.hermes/scripts/run-llm-router-collector.sh`
- **Delivery:** Local (silent — only errors are reported)

## Phase 2: Classifier (planned)

After ~2 weeks of data collection, train a lightweight Random Forest classifier:

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=5)
clf.fit(X_train, y_train)
```

**Input features:** prompt_length, context_length, tool_call_count, task_type (one-hot), contains_code_blocks, has_keywords

**Target:** cheapest_successful_model

**Fallback:** If the cheap model fails, escalate to the next in the chain. Wrong predictions just cost a retry, never a bad result.

## Phase 3: Active Routing (planned)

Package as a Hermes skill that:
1. Loads on every session
2. Logs every interaction silently
3. Uses the classifier to route automatically
4. Reports savings periodically

## Model Pricing

> **Single source of truth:** `MODEL_REGISTRY` in `scripts/models.py`. Each model is one dict entry `{provider, ratio, input, output, tier, date}`. `DEFAULT_MODEL_COSTS`, `MODEL_COST_ORDER` and `MODEL_CAPABILITY_TIERS` are all **derived** from it so they can never drift. To add/change a model, edit `MODEL_REGISTRY` (and Hermes `config.yaml`) — the tables below are documentation only.

### Pricing model
All models are **flat-subscription or free** — there's no per-call cost:
- **gpt-5.5 (ChatGPT OAuth):** $20/mo flat, rate-limited
- **Ollama Cloud:** $100/mo flat
- **Local:** Free

The "cost" values are **relative compute units** — a dimensionless measure of how much subscription budget each model call consumes. This lets the router prefer cheaper models for simple tasks, preserving expensive model capacity for complex work.

### Relative compute ratios (deepseek-v4-flash = 1.0x baseline)
| Model | Ratio | Tier | Notes |
|---|---|---|---|
| llama3.1:8b | 0.0x | 1 | Local, free |
| dolphin3 | 0.0x | 2 | Local, free |
| deepseek-v4.1-flash | **0.68x** | 3 | Cheapest rung of the compression ladder (cheaper than v4-flash on input AND output) |
| deepseek-v4-flash | **1.0x** | 3 | Baseline — small, fast |
| minimax-m2.7:cloud | **2.0x** | 4 | Mid-size |
| glm-5 | **2.0x** | 4 | Mid-size |
| glm-5.1 | **2.5x** | 5 | Slightly larger |
| glm-5.2 | **3.0x** | 6 | Larger context |
| qwen3.5 | **3.5x** | 7 | Cloud |
| deepseek-v4-pro | **4.0x** | 8 | Premium tier |
| deepseek-v3.1:671b | **10.0x** | 9 | Massive 671B MoE |
| gpt-5.5 | **30.0x** | 10 | ChatGPT $20/mo, most capable, rate-limited |
| gpt-5.6-luna | **32.0x** | 11 | openai-codex, agentic-coding |
| gpt-5.6-terra | **34.0x** | 12 | openai-codex, agentic-coding |
| gpt-5.6-sol | **36.0x** | 13 | openai-codex, agentic-coding |
| gpt-6-astra | **40.0x** | 14 | openai-codex, **excluded from auto-routing** |

### Effective compute units per 1M tokens
| Model | Provider | Input units/1M | Output units/1M |
|---|---|---|---|
| llama3.1:8b | local | 0.00 | 0.00 |
| dolphin3 | local | 0.00 | 0.00 |
| deepseek-v4.1-flash | ollama-cloud | 0.34 | 1.35 |
| deepseek-v4-flash | ollama-cloud | 0.50 | 1.50 |
| minimax-m2.7:cloud | ollama-cloud | 1.00 | 3.00 |
| glm-5 | ollama-cloud | 1.00 | 3.00 |
| glm-5.1 | ollama-cloud | 1.25 | 3.75 |
| glm-5.2 | ollama-cloud | 1.50 | 4.50 |
| qwen3.5 | ollama-cloud | 1.75 | 5.25 |
| deepseek-v4-pro | ollama-cloud | 2.00 | 6.00 |
| deepseek-v3.1:671b | ollama-cloud | 5.00 | 15.00 |
| gpt-5.5 | openai-codex | 15.00 | 60.00 |
| gpt-5.6-luna | openai-codex | 16.00 | 64.00 |
| gpt-5.6-terra | openai-codex | 17.00 | 68.00 |
| gpt-5.6-sol | openai-codex | 18.00 | 72.00 |
| gpt-6-astra | openai-codex | 20.00 | 80.00 |

## Operating the live endpoint

`biggie-llm-endpoint.service` (systemd) runs `scripts/biggie_llm_endpoint.py` on `http://127.0.0.1:8080`. Hermes points its custom provider at this URL with model `biggie-router`.

```bash
# Restart after code changes
sudo systemctl restart biggie-llm-endpoint.service

# Health check
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/v1/models

# Live logs
journalctl -u biggie-llm-endpoint.service --since "5 min ago" --no-pager
```

**Key behaviours:**
- **Streaming:** preflight captures the httpx `aiter_lines()` iterator **once** and stores it on the preflight; resume continues the *same* iterator. Never call `aiter_lines()` twice (httpx responses are single-use — that caused "Let Let Let" dropped/repeated output).
- **Compression:** `compression_level_for_workload` gives a ≥50k-token `session_compression` **structural** compression even when the request carries tools. The `requires_tools → "off"` rule only applies to normal workloads.
- **Tool finish_reason:** the non-streaming → SSE wrapper (`_wrap_non_streaming`) emits `finish_reason="tool_calls"` when the message carries `tool_calls`, else `"stop"`. Emitting `"stop"` for a tool-calling turn caused repeated/leaked `tool_result`.
- **Rate limits / circuit breakers:** models are rate-limited and circuit-break after repeated failures; the router escalates through `fallback_chain` and enters **limp-home mode** (local only) when all cloud models are exhausted.
- **Malformed tool calls / empty content escalate WITHOUT tripping the breaker:** model-output quality failures (glm-5.2 leaking `skill_view(name='...')` inline syntax into the tool `name` field; empty preflight streams) are classified by the endpoint (`fail_type`) and passed to `escalate_on_failure` as `malformed_tool_call`/`empty_content`. These escalate to the next model but do NOT increment `consecutive_failures` or open the circuit — a reachable-but-sloppy model must not be taken out of rotation for 30 min. Only genuine `error` (provider/availability) or `rate_limit` failures count toward the breaker.
- **Tool schema forwarding (FIX 2026-08-12):** the OpenAI-compatible proxy paths (non-streaming `proxy_to_backend`, streaming `_preflight_openai_stream`, `_open_fresh_stream`) MUST forward `tools` and `tool_choice` from the incoming request to the backend. Without this, Ollama Cloud models never see tool schemas and emit tool-call syntax as prose instead of structured `tool_calls`. The Codex path already converted via `_responses_tools`; the Ollama Cloud path needs no conversion — standard OpenAI tool format is accepted natively.
- **Model registration:** edit `MODEL_REGISTRY` in `scripts/models.py` **and** Hermes `config.yaml` together — they must stay in sync.
- **The compression ladder's flash ceiling must match by FAMILY, not by literal name.** `_select_session_compression_model` gates flash out above `FLASH_MAX_CONTEXT_TOKENS` (100K) because flash degenerates on very large contexts. The original filter was `m != "deepseek-v4-flash"` — adding a second flash alias (`deepseek-v4.1-flash`) silently bypassed the ceiling, letting 125K-token jobs land on flash. Filter by family (`_normalize_model_name(m).startswith("deepseek-v4") and .endswith("-flash")`) so every flash alias inherits the ceiling.
- **That ceiling's stated rationale does NOT hold up (measured 2026-09-12) — see `references/ollama-unit-economics.md`.** The empty-stream bug is **output-budget**-induced, not context-size-induced: a 16K context with `max_tokens=256` fails the same way 173K does, and v4.1-flash served itself 3/3 at 141K `prompt_tokens` with a realistic budget. Production empty-stream rates for v4-flash are **flat** from <20K to 100K (0.17-0.27 pct), which falsifies "degenerates on very large contexts". Before trusting any behavioural gate, check the harness varies the variable it claims to test: the original `flash_ceiling_headtohead.py` logged `prompt_tokens: 14232` at *every* target size because the router compresses requests **before** routing — send `X-Compression-Level: off` to test raw context size.
- **Adding a ladder rung requires a live restart to take effect.** `models.py`/`router.py` edits do nothing to a running endpoint — the process holds the old registry, so a request for the new model falls through to local `llama3.1:8b`. `sudo systemctl restart biggie-llm-endpoint` and verify in `router_logs.db` that the new model is actually serving.
- **`discover_backends()` reads only `fallback_chain`/`default_model`, NOT `custom_providers`.** Adding a model to `custom_providers` alone leaves it undiscovered (`v4.1 discovered: False`); it must also appear in `fallback_providers` in `config.yaml`.
- **You cannot A/B models through the router.** For `task_type=session_compression` the router always overrides the client's requested model with the ladder rung, so every leg returns the same model. A/B must call the provider (`https://ollama.com/v1`) directly — `force_model` is not reachable from the HTTP API.

## Routing tiers, the clamp, and the exclusion design

- **The clamp is the real ceiling, not the routing table.** `_estimate_min_tier` returns `max(1, min(min_tier, 10))` (router.py line 1300) and complexity scoring caps at 10. So **no normal routing path can ever reach tier 11+** — the routing-table floors above 10 are unreachable by ordinary routing. Strong models (5.6, gpt-6) are reached ONLY via `escalate_on_failure` (uses `MODEL_CAPABILITY_TIERS` directly, no clamp) or `force_model`. To let an explicit high-tier trigger (e.g. rethink/rearchitect at tier 13) actually reach a strong model, raise the clamp ceiling (10→13) — gpt-6 stays at 14 so it's still never auto-selected.
- **A registered model is auto-reachable unless explicitly excluded.** `_build_fallback_chain` and `escalate_on_failure` walk ALL of `MODEL_COST_ORDER`, so any model in `MODEL_REGISTRY` appears in every fallback chain and is reachable via escalation — even one you intended as "explicit trigger only". To keep a model out of auto-routing, add it to `EXCLUDED_FROM_AUTO_ROUTING` (router.py) and filter it out of **all three** auto-routing paths: `_select_model` (main loop AND the `available[-1]` fallback), `_build_fallback_chain`, and `escalate_on_failure` (both the tool-escalation loop and the normal loop). `force_model` (line 855) bypasses these, so the model stays reachable explicitly.
- **`match_sub_type` returns the FIRST match.** When adding a more-specific/higher-stakes sub-type that overlaps a broader one (e.g. "rethink/rearchitect" vs "system design"), the specific sub-type must be ordered BEFORE the broader one in `routing_table.yaml`, or the broader match wins and the trigger never fires. Test with a prompt that contains the broader keyword (e.g. "rearchitect the system design") to catch the ordering bug.
- **`exact_keywords` = full-phrase substrings, not bare words.** A sub-type can carry an `exact_keywords` list (checked in `match_sub_type` alongside `keywords`) for triggers that must only fire on deliberate phrasing. "deep review" lives there (as "perform a deep review" / "do a deep review" / "deep review of the") so incidental mentions like "do a quick deep review of this diff" do NOT hit tier 13. Keep bare-word triggers in `keywords`, phrase-only triggers in `exact_keywords`.

## Firing and verifying the rethink/rearchitect lane

Use `scripts/probe_rethink_lane.py` (`--dry` to preview, default fires + verifies, `--deep` to exercise the "deep review" phrasing, `--count` to read the lane without sending): it POSTs a normal chat-completion whose prompt carries **both** a `planning` keyword and a rethink sub-type keyword (e.g. "rethink the architecture and plan the migration" or "perform a deep review of the system architecture") to `/v1/chat/completions` with model sentinel `biggie-router`. Because the keyword trigger routes through `match_sub_type`, this exercises the **normal-routing** tier-13 path to `gpt-5.6-sol` — distinct from `force_model` (which bypasses routing and would land in the force_model lane). Fire it to prove the lane records and to populate the rethink row for a user's major review.

**Lane-detection pitfall:** the recorded `routing_reason` string does **NOT** contain the literal word "rethink" — it is `"capability tier 13 needed, selected gpt-5.6-sol (tier 13)"`. So detect the rethink lane by **final model == the tier-13 target AND the tier number in the reason AND `escalated == 0` AND not a force_model string**, never by substring-matching "rethink". This matches how `check-routing-stats.py` buckets lanes (a rethink probe that fired correctly must show in the `rethink/rearchitect` count going 0 → 1 in `router_logs.db`, not in escalation or force_model).
- **Adding a model to `MODEL_REGISTRY` breaks tests that hardcode the most-expensive model** (`MODEL_COST_ORDER[-1] == "gpt-5.5"`). Update those assertions to the new top model — the invariant is "cheapest first", not a fixed name.
- **Adding a model to `TOOL_CAPABLE_MODELS` breaks tool-routing tests that assume the old most-capable tool model.** Tests that `mark_available()` a fixed set and assert the tool fallback path now pick the new higher-tier tool-capable model. Preserve their intent by `mark_rate_limited()`-ing the new tool-capable models in `setup_method`.

## Driving gpt-6-astra directly with reasoning effort (large reviews)

`force_model` reaches astra but the router's codex proxy hardcodes the request body and does NOT forward a `reasoning` param — so you get NO effort control through the router. To run a heavy/reasoning-guided review on astra, bypass the router and call the codex backend directly:

- **Endpoint:** `POST https://chatgpt.com/backend-api/codex/responses` (NOT `/v1/chat/completions` on the local endpoint, and NOT `/v1/responses`).
- **Auth:** `Authorization: Bearer <openai-codex access_token>` from `~/.hermes/auth.json` (credential_pool.openai-codex; skip entries with last_status == "exhausted" and last_error_code == 429).
- **Body:** `{model: ["gpt-6-astra"], input: [{role:"user",content:...}], store:false, stream:true, reasoning:{effort:"low"|"medium"|"high"}}`.
- **Stream is MANDATORY:** non-stream returns `400 {"detail":"Stream must be set to true"}`. Read SSE `data:` lines; capture `response.output_text.delta` events for text (NOT `response.content_part.done`/`.completed`'s output — deltas are where the words are), and `response.completed`'s `usage` field for token/reasoning stats.
- **`reasoning.effort` IS accepted:** verified `low` and `high` both return clean output (e.g. "ASTRA_DIRECT_OK").
- **Probe:** `python3 scripts/probe_astra_direct.py --effort high` (flags: `--stream`, `--effort <low|medium|high>`, `--no-reasoning`).

This is the path for the user's upcoming large review — hand over `probe_astra_direct.py` as the working template (it reads the token from auth.json, so no secrets in the script).

## Self-optimising layer (cost accounting, rollup, experiments)

Full design: `references/self-optimising-router-plan.md`. Modules:

- `scripts/unit_economics.py` — **real USD** per model/call-type/volume, with **versioned** prices (`model_pricing`, `price_for(model, at=...)`) and `Decimal` money. Unpriced ⇒ `None`, never a confident `$0`.
- `scripts/quality.py` — `fact_coverage_v1`, the *established* A/B scorer promoted to a module. One scorer, not a second opinion.
- `scripts/rollup.py` — schema migration, `rollup_day`/`rollup_range` → `daily_findings`, `purge_raw` retention, `vacuum_if_needed`.
- `scripts/traffic_split.py` + `scripts/experiments.yaml` — **directing production traffic to a candidate model**: `mode: shadow` (observe only, incumbent still serves) or `mode: split` with `percent`. Deterministic hash bucketing, sticky per session.
- `scripts/optimiser.py` — ranks models per workload by **cost per quality point**; emits `candidate_routing.yaml`. **Propose-only.**
- `scripts/router_ops.py` — CLI: `rollup | report | findings | propose | purge | audit | backfill | maintain`.

```bash
cd ~/.hermes/skills/llm-smart-router/scripts
python3 router_ops.py audit --days 30          # logging-health: unpriced rows, quality coverage
python3 router_ops.py report --days 14         # unit cost per (model, call type)
python3 router_ops.py backfill --yes           # reconstruct cost for pre-instrumentation rows
python3 router_ops.py propose --days 14        # write candidate_routing.yaml (never applies)
python3 router_ops.py purge --retention 30     # dry-run by default; --yes to delete
python3 router_ops.py maintain                 # what the daily timer runs
```

### To test a new model against production traffic

Edit `scripts/experiments.yaml` — no code change:

```yaml
experiments:
  - name: glm53flash-split
    enabled: true          # ships disabled
    model: glm-5.3-flash
    mode: split            # or: shadow
    percent: 10            # % of matching sessions
    match: { workload: [session_compression] }
```

The endpoint re-reads it within ~30s. The candidate must already be a discovered backend (present in `~/.hermes/config.yaml`), otherwise the request stays on the incumbent and the arm is logged `control` rather than silently reporting treatment outcomes it never served.

- **`shadow`** — incumbent answers the user; candidate is recorded (`is_shadow=1`) for offline scoring. Zero user-visible risk.
- **`split`** — the percentage of matching sessions genuinely get the candidate. This is the real-load A/B.

### Operating facts (measured on production data)

- Every call is priced inline and written to `router_logs.cost_usd` with `cost_unknown` and `pricing_version`. The logger computes the cost itself when the caller does not pass one, so a call site cannot go cost-blind by omission.
- The optimiser **refuses to propose without quality evidence** (`quality_n` from `daily_findings`). With no measured quality there is no evidence a cheaper model is safe, so nothing is proposed — that is the correct answer, not a bug. Populate quality via `mode: shadow` runs scored with `quality.py`.
- **Retention refuses to purge any day without a committed rollup** — a crash mid-rollup must never destroy unrolled raw rows. Purge is dry-run unless `--yes`.

### Before enabling ANYTHING against live traffic

```bash
cd ~/.hermes/skills/llm-smart-router
python3 scripts/preflight.py   # 16 end-to-end checks, throwaway DB, never touches prod
```

Preflight drives the real production code path (real init, real price seeding, real logger, real HTTP requests) against a throwaway DB. It exists because a single live enablement surfaced four defects that 266 unit tests did not catch. **Do not enable a shadow or split experiment if preflight fails.** CI runs it on every push (`.github/workflows/tests.yml`), alongside pytest.

Override the DB path with `BIGGIE_ROUTER_DB=/tmp/x.db` to point any of this at a throwaway file.

### Is self-optimisation live or manual?

Manual to *start*, and it cannot currently start at all. Precisely:

| layer | status |
|---|---|
| cost accounting / rollup | **automatic** — timer `biggie-router-ops.timer` rolls up daily and writes findings |
| logging health audit | **automatic** (`router_ops.py maintain`) |
| optimiser (`propose`) | **propose-only, never applies.** Writes `candidate_routing.yaml`; **zero production code reads that file** (verified). Promotion is a human edit. |
| traffic split / canary | **manual, config-driven** — `experiments.yaml` is the only interface, hot-reloaded within ~30s. Ships disabled. |
| shadow experiments | **automatic once enabled** — but only for a candidate that works |
| quality evidence | **not being generated.** All 70 scored rows are shadow rows; **zero production rows are scored.** |

The blocker is the last row. Ranking needs measured quality, quality is only scored inside the shadow path, and shadow needs a working candidate. So the loop is currently **open**: the router measures cost and reports honestly that it cannot rank anything.

### Pitfalls

- **Editing a price without moving its `"date"` is a silent no-op.** `seed_prices()` is idempotent on `(model, effective_from)` — it skips a model already recorded at that date. Change `MODEL_REGISTRY["x"]["input"]` and leave `"date"` alone and it writes **zero** rows: the stale price stays in `model_pricing`, live calls keep pricing at the old rate, and nothing raises. A correction is a new *versioned* row (new date), never an edit. Superseded rows stay on record, and `price_for(at=...)` resolves the rate in force, so history is not restated. Guarded by `tests/test_price_book_versioning.py`.
- **A price is a routing input, not bookkeeping.** `MODEL_COST_ORDER` and `get_available_models()` are derived by sorting `MODEL_REGISTRY` on input price, and the auto router picks the cheapest available model meeting `min_tier`. Correcting the book reorders the chain and can flip which model serves a tier band. Measure the flip and pin it (`tests/test_price_book_routing.py`) before shipping.
- **Importing `biggie_llm_endpoint` in a test reaches production.** It pulls in `unit_economics`, whose `_default_conn()` resolves `Path.home()/.hermes/skills/llm-smart-router/data/router_logs.db` and seeds the price book into it — the live database. `tests/conftest.py` now redirects `HOME` for the session to make that path unreachable; `tests/test_prod_db_isolation.py` guards it. Any new test that touches pricing without passing an explicit `conn` relies on that guard.
- **`cost_unknown` must default to `None` (compute it), not `1`.** Defaulting to 1 meant any call site that forgot the cost fields logged the request cost-blind — on production that silently hid 952 `session_compression` calls (68.4M tokens, the busiest workload) from every spend rollup. An omitted cost is now *computed*; "unknown" is reserved for a model that genuinely has no price. Guarded by `tests/test_cost_capture_parity.py` and `tests/test_streaming_cost_gap.py`.
- **A test that asserts the defective behaviour is a defect too.** `test_log_request_to_db_defaults_are_safe` asserted `cost_unknown == 1` as "safe", which is what locked the bug in. When fixing a contract, grep for tests that assert the old one.
- **`_ensure_log_columns` only ALTERs an existing table.** It cannot create `router_logs`. The endpoint's real init calls `rollup.migrate()` first; without that, a fresh DB has no `model_pricing` or `daily_findings` and cost capture degrades silently.
- **`estimated_cost_usd` in `router_logs` is dead.** It was never written by the endpoint (all 231k rows were 0.0). Cost now lives in `cost_usd` + `cost_unknown` + `pricing_version`. Do not report spend from `estimated_cost_usd`.
- **Backfilled costs are reconstructed, not measured.** They carry `pricing_version='backfill:<date>'`; filter `pricing_version LIKE 'backfill%'` to separate them.
- **Streaming completions record zero output tokens.** Both streaming completion loggers hardcode `output_tokens=0`, so 114,618 completed streaming rows carry $17,438.92 with no output cost counted (route-start rows legitimately have none — don't conflate them). Measured understatement is small (+1.6% compression, +13.5% chat) but it is *unlabelled*: the row reads as measured. Pinned by `tests/test_streaming_cost_gap.py`, whose docstring says what to do when it is fixed.
- **`saw_tool_calls` is not rolled up.** Written per request, absent from `daily_findings`, so the signal that produced the glm-5.3-flash verdict (tool_calls with no content when tools are offered) is invisible to every rollup report. Pinned in `tests/test_static_contracts.py`.
- **The optimiser's empty report has three meanings** (no traffic / nothing measured / incumbent genuinely wins). `Proposal.workloads_examined` + `workloads_unrankable` carry which one as data; do not parse the English.
- **The optimiser never edits `routing_table.yaml`.** It writes `candidate_routing.yaml` with `applied: false`; promotion is a human edit.
- **`aiter_lines()` is single-use:** capture the iterator once in preflight and reuse it. Calling it again on the same httpx response returns nothing/empty.
- **state.db is large (~2GB):** The collector queries with `mode=ro` (read-only) and uses indexed queries. It only reads sessions newer than the last collection timestamp.
- **Missing model prices:** Unknown models get `cost_unknown=1` and `cost_usd=0.0`. Add them to the price book via `unit_economics.seed_prices`.
- **Task classification is heuristic:** The keyword-based classifier is a starting point. Phase 2 will replace it with a trained model.
- **Cron runs silently:** The `no_agent` cron job only reports errors. Check `cronjob list` for status.
- **Runtime DB artifacts:** `biggie_router.db`, `data/*.db-shm`, `data/*.db-wal` are gitignored — never commit them.
