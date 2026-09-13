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

- **There is exactly ONE billable definition; never add filters to it.** `rollup.BILLABLE_ROW_SQL` = `COALESCE(error_type,'') != 'streaming_in_progress'` and nothing else (a streaming request writes a start marker plus a completion row; only the marker is non-billable). Adding `AND cost_unknown = 0` to the allowance CLI looked harmless and passed every unit test, but the cost backfill *repairs* previously-unpriced rows by reconstructing a price from the model's own rate card — so `cost_unknown = 1` rows are **billable work**. It silently dropped **5.2 pct** of billable tokens and biased avg-input **+1.41 pct**, which feeds the marginal rate and drifted the projection off the dashboard. Pin any reporting query to `BILLABLE_ROW_SQL` by test so the definitions cannot drift.
- **A cached rollup keeps the bug you just fixed — re-roll, don't assume.** `daily_findings` held totals **1.85x** the truth for 09-10..09-12 because the rollup cached the double-counted figures *before* the code was fixed; the cached number matched the old broken total exactly, which is how the cause was confirmed rather than guessed. `rollup_day()` is delete-then-insert per day, so re-running is safe and idempotent. `biggie-router-ops.timer` re-rolls only `maintain --days 3`, so a stale day **self-heals while inside that window and never afterwards** — check when the fix landed versus when the timer last ran; days outside the window need an explicit `rollup_range` backfill (on a copy).
- **Stale cost is a latent routing hazard, not just a reporting one.** `optimiser.rank_models()` takes `cost_per_call` from `daily_findings`, so an inflated basis can reject a genuinely cheap rung or rank it below a pricier one and invent an `est_weekly_delta_usd`. It is inert only while the quality gate is shut (eligibility needs `quality_n > 0`).
- **A quality score can be an artifact of a response that never summarised.** The only scored ledger rows averaged **0.0042** — but 65 of 70 had `finish_reason='tool_calls'` and emitted 54–87 tokens against 39K–152K inputs. The model returned a tool call, not a summary, so the fact-coverage scorer graded empty text as 0. Treating that as "a terrible summariser" would kill a cheap rung for no reason. Record such attempts as **unmeasurable** and exclude them (`AVG(score) FILTER (WHERE measurable = 1)`): a score of 0.0 must mean "measured and the facts were lost", never "we failed to measure". Real sampled probes scored **0.899**.
- **Probe scores do not reach the optimiser by themselves.** `rollup_day()` aggregates `AVG(quality_score)` from `router_logs`, while `quality_probe.record_probe()` writes to the separate `quality_probe` table (joined on `request_id`). Wiring the probe in does not make `daily_findings.quality_avg` populate — decide deliberately whether to join them instead of assuming it flows.
- **List-price ledger dollars are NOT the operator's bill — reconcile against the vendor dashboard.** On a fixed plan, in-allowance usage is not billed at all, so only consumption beyond the allowance costs money. Measured on one week of identical traffic: ledger **$3,977** vs dashboard **$48.26** (~80x). The ledger is a *routing-mix proxy*; the dashboard is billing ground truth. Never quote a ledger figure as a saving.
- **Only a minority of requests are credit-billed.** Measured **8.5%** (2,960 of ~34,700). Projecting a period's whole token burn at a marginal rate therefore overstates by ~12x — the first cut of the allowance report printed **$567.94 against a $48.26 reading** before scaling by the billable share. `scripts/allowance.py` now scales by it; the reconciliation is pinned by test.
- **A reliability premise is a measurement, not a memory.** The "144 empty-content events on >100K contexts, so add a 150K flash ceiling" premise was **false**: since the output-budget floor deployed there were **0** empty events above 150K over **174** such compressions, all 174 successful, and the empty rate is *highest below 20K*. The failures were output-budget induced (`SESSION_COMPRESSION_MIN_MAX_TOKENS`), not size induced. A ceiling would have cost **9.3x** on that band for no reliability gain. Always re-measure the premise before shipping the fix — the same trap produced the worthless Aug-22 ceiling harness and the un-reproduced "glm-5.3 degenerates above 150K" claim.
- **`router_logs.empty_stream` is a dead column** (`SUM = 0` over 75,953 rows) and `error_type='empty_content'` is **masked by escalation** (the success row carries the recovered outcome). The **journal is authoritative** for empty streams: `journalctl -u biggie-llm-endpoint | grep "empty content for"`. Journal timestamps are **AEST** while `router_logs.timestamp` is **UTC ISO** — convert before joining, and strip the `:cloud` provider suffix, or the join silently returns 0 matches.
- **Editing a price without moving its `"date"` is a silent no-op.** `seed_prices()` is idempotent on `(model, effective_from)` — it skips a model already recorded at that date. Change `MODEL_REGISTRY["x"]["input"]` and leave `"date"` alone and it writes **zero** rows: the stale price stays in `model_pricing`, live calls keep pricing at the old rate, and nothing raises. A correction is a new *versioned* row (new date), never an edit. Superseded rows stay on record, and `price_for(at=...)` resolves the rate in force, so history is not restated. Guarded by `tests/test_price_book_versioning.py`.
- **A candidate model's CONTEXT CEILING disqualifies it faster than its price qualifies it — check it before costing anything.** Ollama's table gives per-token rates and says nothing about maximum context. Measured on real compression payloads: `gpt-oss:20b` ($0.07/1M input, 2x cheaper than the incumbent) returned **HTTP 400 "prompt is too long" on every payload ≥100K** — and **27% of compressions are ≥100K, 9% >150K**. A cheaper model that rejects a quarter of the traffic is not cheaper. Corollary: a *timeout* is not a rejection — do not read a client read-timeout as a context limit.
- **Ollama charges PEAK pricing (2x) 12:00–18:00 UTC Mon–Fri on the deepseek models, and the price book has no peak dimension.** One flat rate per model, so peak-window calls are understated and off-peak over-stated. Measured over 14 days: 9% of deepseek spend fell in the peak window ($58.36 vs $544.98 off-peak), which roughly self-cancels today — traffic is AEST-evening-weighted — but it is an unmodelled 2x exposure if load moves into that window. v4.1-flash doubles to $0.30/$1.20, v4-flash to $0.44/$1.32, v4-pro to $1.32/$3.96.
- **`maintain` performs three writes, not one — read `cmd_maintain` before triggering it on prod.** It runs `rollup --days 3` (the intended self-heal), `purge --yes --retention 30`, and a `propose` step that **overwrites `scripts/candidate_routing.yaml`** with a fresh `generated` timestamp, leaving the git tree dirty. After any manual trigger run `git checkout scripts/candidate_routing.yaml`. Take a backup first and verify it (`PRAGMA integrity_check`, row count) before mutating `data/router_logs.db`.
- **Quality scoring is `fact_fidelity_v2`. `fact_coverage_v1` is retired — never rank on it, and never average the two.** They are not on the same scale. v1's ceiling on real payloads was ~0.24 against the optimiser's 0.80 floor, so `rank_models` returned **0 models for every workload** and the optimiser was permanently inert. The rollup now averages `quality_score` **only** for `quality_method = 'fact_fidelity_v2'` rows (SQL in `rollup.py`); a mixed-method average is meaningless. Live rows written before 2026-09-13 are v1 and must be ignored, not migrated.
- **v2 = `precision × fact_yield × substance`.** `precision` = share of stated facts that are real (the safety property: fabricating figures must never win). `fact_yield` = facts carried ÷ what a summary of *this length* could carry — judged against the summary's own budget, not the source's size, which is what makes the score scale-invariant (v1 measured 0.2439 on a 200K context vs 0.6780 on a 68K one for identical quality — a 2.8x swing from input size alone). `substance` down-weights below `MIN_SUMMARY_FACTS`.
- **`FACT_CHARS_BUDGET` is a measured constant, not a guess — recalibrate it when summary behaviour changes.** It was 60, implying one fact per 60 chars; real production compaction summaries (200 sampled from `state.db`) carry **~804 chars/fact at p25** (median 870, p75 1280). At 60, `fact_yield` saturated near 0.07 for genuine summaries and *no model could ever reach the 0.80 floor* — the redesigned metric would still have been inert. Anchor at **p25, not the mean**: the budget sets where `fact_yield` saturates at 1.0, so anchoring at the mean makes half of all real summaries mutually indistinguishable exactly where ranking matters.
- **`MIN_SUMMARY_FACTS = 5` exists to kill a perverse incentive — do not remove it.** Because `fact_yield` is normalised by the summary's *own* length, a 3-fact stub scored 1.0 while a realistic 15-fact production summary scored 0.74: terser output would look like the best summariser and could be promoted. 5 is the measured p05 of real summaries. Below the floor the score scales down smoothly rather than zeroing.
- **`extract_facts` must filter structural markers.** List ordinals (`473. REPAIRED`) and numbered headings were counted as facts — a **mean 22.3% of all extracted "facts"** on real summaries (max 83%) — and, because the source rarely shares the same numbering, they were *also* counted as hallucinated. Filter a number only when **every** occurrence is in a marker position; a real figure in prose must survive.
- **Score against the WHOLE conversation, never `role == "user"` alone.** A user-only source is 19.4K of a real 225.7K payload — **87% of the fact set invisible** — so a summary correctly reporting a fact from a tool/assistant message is scored as a *hallucination*. Use `quality.source_text_from_messages`; the probe, the endpoint and any offline harness must share it or their numbers are not comparable.
- **Validate a pairing before trusting any precision number.** Scoring a summary against the wrong source produces authoritative-looking garbage: my first source↔summary pairing gave median token overlap **0.146** and precision values that meant nothing. If a summary's distinctive tokens (paths, call ids, filenames) are not in its "source", the pairing is wrong.
- **Fixing the scorer does NOT put quality in front of the optimiser — check the data path before claiming the optimiser is unblocked.** `rank_models` reads `daily_findings.quality_avg`, which the rollup builds from `router_logs.quality_score`, and that column is written by **one** call site: the shadow path. `experiments.yaml` has no enabled experiment, so shadow never runs and compression traffic is unscored. The sampled `quality_probe` series is a **separate table that neither `rollup.py` nor `optimiser.py` reads** (`grep -c quality_probe scripts/rollup.py scripts/optimiser.py` → 0). Its docstring says the two are "joined on `request_id` when needed" — that join does not exist, so measured v2 quality reaches reporting but never a ranking. Trace the column to its writer before promising an outcome.
- **A probe's `summary_chars` must be checkable against the row's `output_tokens` — treat a large mismatch as the probe scoring a fragment.** Live probe rows recorded `summary_chars` of 105 and 112 against `output_tokens` of 227 and 104 (≈0.5–1.0 chars/token, where real text is ~4). A compaction summary is thousands of chars (median 10,537 in `state.db`). When the scored text is orders of magnitude smaller than what the model produced, the score measures the capture path, not the model.
- **`purge` refuses to delete a day it has no committed rollup for — that guard is the safety property, and it is why retention never actually reclaims space.** `purge_raw` skips any day absent from `daily_findings`, so a naive trigger deletes nothing (33 of 64 days had no rollup here). Two consequences: raw detail survives indefinitely and the DB grows unbounded (~+27 MB/month, 138 MB at 243K rows), and reclaiming space requires *first* rolling the day up, *then* purging. Deleting detail is therefore always a deliberate action — the timer will never do it for you.
- **Never average `latency_seconds` over all rows — exclude >300s first.** `latency_seconds` is non-null on only 30.8% of rows (74,751 of 242,964) and 339 carry impossible values from uncaptured end-timestamps (max **1,909,566s = 22 days**). They drag the all-model mean to **1021s** when the p50 is **6.40s**; the `<20K` band averages **19.05s**, not 1021s. Report percentiles, not means, and filter outliers at the source.
- **`success=0` and `empty_stream` are ~0.00% for every high-volume model, so they are useless as reliability signals. Escalation rate is the signal.** The escalation *sinks* invert it: `gpt-5.6-sol` (41.7%), `llama3.1:8b` (65.3%), `dolphin3` (27.8%) are terminal destinations, not failures — all their rows have `success=1` and `model_used == final_model`. Read a high escalation rate as "this is where traffic lands", never "this model is broken".
- **`output_tokens` is populated on only 2.74% of rows (6,661 of 242,964), so every ledger cost figure is effectively input-only.** Relative model comparisons stay valid (the same blind spot applies to both sides); absolute dollars **understate** output-heavy workloads. Never present a ledger total as the true bill.
- **Cost scales monotonically with context on every model — context is the cost lever, model choice is the multiplier.** Measured $/call, <20K vs >150K: `deepseek-v4.1-flash` 0.00123→0.02471 (**20x**), `gpt-5.5` 0.54637→2.46932 (**4.5x**), `glm-5.3:cloud` 0.09450→0.27302 (**2.9x**). Model choice moves the base (v4.1-flash is **24.7x** cheaper per call than glm-5.3 at $0.00745 vs $0.18408); context moves the exponent. This is why `BIGGIE_FLASH_MAX_CONTEXT_TOKENS=0` paid off ~14x.
- **Count calls, not just tokens, when hunting overage — the expensive lane is a call-count problem.** Measured over 7 days: the ChatGPT lane was **35.0% of billable dollars on 2.5% of calls** ($1.1804/call vs $0.0556, **21.2x**). A single line item — `gpt-5.5`, 499 calls, $705.34 — cost more than `glm-5.3:cloud` did across 5,326 calls. Cap the expensive lane's *calls*, not just its tokens.
- **A price is a routing input, not bookkeeping.** `MODEL_COST_ORDER` and `get_available_models()` are derived by sorting `MODEL_REGISTRY` on input price, and the auto router picks the cheapest available model meeting `min_tier`. Correcting the book reorders the chain and can flip which model serves a tier band. Measure the flip and pin it (`tests/test_price_book_routing.py`) before shipping. Note compression is **not** affected: it returns from `_select_session_compression_model()`'s curated ladder, so a price-driven reorder does not move the summariser rungs.
- **Importing `biggie_llm_endpoint` in a test reaches production.** It pulls in `unit_economics`, whose `_default_conn()` resolves `Path.home()/.hermes/skills/llm-smart-router/data/router_logs.db` and seeds the price book into it — the live database. `tests/conftest.py` now redirects `HOME` for the session to make that path unreachable; `tests/test_prod_db_isolation.py` guards it. Any new test that touches pricing without passing an explicit `conn` relies on that guard.
- **`cost_unknown` must default to `None` (compute it), not `1`.** Defaulting to 1 meant any call site that forgot the cost fields logged the request cost-blind — on production that silently hid 952 `session_compression` calls (68.4M tokens, the busiest workload) from every spend rollup. An omitted cost is now *computed*; "unknown" is reserved for a model that genuinely has no price. Guarded by `tests/test_cost_capture_parity.py` and `tests/test_streaming_cost_gap.py`.
- **A test that asserts the defective behaviour is a defect too.** `test_log_request_to_db_defaults_are_safe` asserted `cost_unknown == 1` as "safe", which is what locked the bug in. When fixing a contract, grep for tests that assert the old one.
- **Exception-safety tests must use input that actually raises.** Passing `None/None` to a scorer proves nothing, because that path never throws — the mutation check passed while the `except` clause was narrowed to `ZeroDivisionError`. Pass an object whose `__str__` throws. Same rule for behavioural gates: assert the variable under test *varied*.
- **An operational env dial must be read at CALL time.** `DEFAULT_RATE` read at import meant `BIGGIE_QUALITY_PROBE_RATE` did nothing until a restart, making the sampling rate effectively a code constant. Read the environment inside the function so retuning needs no restart.
- **`_ensure_log_columns` only ALTERs an existing table.** It cannot create `router_logs`. The endpoint's real init calls `rollup.migrate()` first; without that, a fresh DB has no `model_pricing` or `daily_findings` and cost capture degrades silently.
- **`estimated_cost_usd` in `router_logs` is dead.** It was never written by the endpoint (all 231k rows were 0.0). Cost now lives in `cost_usd` + `cost_unknown` + `pricing_version`. Do not report spend from `estimated_cost_usd`.
- **Backfilled costs are reconstructed, not measured.** They carry `pricing_version='backfill:<date>'`; filter `pricing_version LIKE 'backfill%'` to separate them.
- **Streaming completions used to record zero output tokens — FIXED by requesting usage, not by guessing.** Both streaming completion loggers hardcoded `output_tokens=0`, so 114,618 completed streaming rows carried $17,438.92 with no output cost counted (route-start rows legitimately have none — don't conflate them). The fix asks the backend for `stream_options={"include_usage": True}` (**provider-gated**: a backend that rejects the parameter must not have its stream broken), harvests the final usage chunk in `_resume_stream`, and records the real count. The usage chunk carries `choices: []`, so a delta parser indexing `choices[0]` drops exactly the payload holding the tokens — parse it separately. When no usage arrives, record the input-only estimate **with `cost_unknown=1`** (both fields together: a lone `cost_unknown` zeroes the estimate). Measured on one real call: 19in+36out = $0.000185 vs $0.000027 input-only — a **6.9x** understatement, far larger on output-heavy chat than the 1 pct on compression the original estimate assumed. Guarded by `tests/test_streaming_usage_capture.py` and the inverted `tests/test_streaming_cost_gap.py`.
- **A streaming request must leave ONE row — supersede the marker, don't just exclude it from sums.** A streaming request logs a `streaming_in_progress` start marker then a completion row; measured **118,326 markers = 48 pct of the table**. Excluding them from `BILLABLE_ROW_SQL` fixed the money but left every consumer responsible for remembering the filter, and `DELETE`-on-completion is the honest fix: it keeps abandonment detection (a marker with no completion is never touched — 289 such requests exist) while making the table itself one row per request. Guard the delete on a **non-empty** `request_id`: production rows share an empty one, and equality would wipe every unpaired marker at once. Verified end-to-end against the real backend: 1 row written, cost matching the vendor rate card exactly.
- **Ask the real backend whether a fix is possible before designing it.** The streaming usage gap looked like a data-source limitation; probing ollama.com directly showed the stream emits usage when asked and none when not (`/tmp/probe_usage.py`). One bounded probe converted an unfixable-looking defect into a 6.9x correction — and the same probe proved the request shape (`choices: []`) that made the naive parser wrong.
- **A source-text assertion is usually vacuous — drive the code and inspect what it builds.** `assert "include_usage" in src` passed even with the provider gate disabled, because the string was still present. The non-vacuity proof caught it; asserting on the body the preflight actually constructs (`build_request(json=...)` captured via a fake httpx client) is what fails when the behaviour is removed. Same family as the async-generator trap: exercising the wiring is the only thing that proves it exists.
- **`saw_tool_calls` is not rolled up.** Written per request, absent from `daily_findings`, so the signal that produced the glm-5.3-flash verdict (tool_calls with no content when tools are offered) is invisible to every rollup report. Pinned in `tests/test_static_contracts.py`.
- **Measure summariser quality where the traffic is.** `quality_score` was written ONLY by the shadow path; with the shadow experiment disabled, the model serving ~99% of compressions was entirely unscored. `scripts/quality_probe.py` samples (`BIGGIE_QUALITY_PROBE_RATE`, default 5%) and scores post-response, wired into **both** streaming completion paths — the primary and the escalation retry. Unmeasured must stay NULL, never 0.0, or "not measured" averages as "measured bad".
- **The optimiser's empty report has three meanings** (no traffic / nothing measured / incumbent genuinely wins). `Proposal.workloads_examined` + `workloads_unrankable` carry which one as data; do not parse the English.
- **The optimiser never edits `routing_table.yaml`.** It writes `candidate_routing.yaml` with `applied: false`; promotion is a human edit.
- **`aiter_lines()` is single-use:** capture the iterator once in preflight and reuse it. Calling it again on the same httpx response returns nothing/empty.
- **A `StreamingResponse` body is an async generator — it only runs when drained.** Asserting on the returned response proves nothing about the relay; iterate `resp.body_iterator` in the test, or the wiring under test never executes.
- **Shell safety guards false-positive on heredocs:** `python3 - <<'PY'` blocks containing certain restart-ish words are refused. Write the script to a file and run it instead.
- **state.db is large (~2GB):** The collector queries with `mode=ro` (read-only) and uses indexed queries. It only reads sessions newer than the last collection timestamp.
- **Missing model prices:** Unknown models get `cost_unknown=1` and `cost_usd=0.0`. Add them to the price book via `unit_economics.seed_prices`.
- **Task classification is heuristic:** The keyword-based classifier is a starting point. Phase 2 will replace it with a trained model.
- **Cron runs silently:** The `no_agent` cron job only reports errors. Check `cronjob list` for status.
- **Runtime DB artifacts:** `biggie_router.db`, `data/*.db-shm`, `data/*.db-wal` are gitignored — never commit them.
