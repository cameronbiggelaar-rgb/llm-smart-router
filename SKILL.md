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
| deepseek-v4-flash | **1.0x** | 3 | Baseline — small, fast |
| minimax-m2.7:cloud | **2.0x** | 4 | Mid-size |
| glm-5 | **2.0x** | 4 | Mid-size |
| glm-5.1 | **2.5x** | 5 | Slightly larger |
| glm-5.2 | **3.0x** | 6 | Larger context |
| qwen3.5 | **3.5x** | 7 | Cloud |
| deepseek-v4-pro | **4.0x** | 8 | Premium tier |
| deepseek-v3.1:671b | **10.0x** | 9 | Massive 671B MoE |
| gpt-5.5 | **30.0x** | 10 | ChatGPT $20/mo, most capable, rate-limited |

### Effective compute units per 1M tokens
| Model | Provider | Input units/1M | Output units/1M |
|---|---|---|---|
| llama3.1:8b | local | 0.00 | 0.00 |
| dolphin3 | local | 0.00 | 0.00 |
| deepseek-v4-flash | ollama-cloud | 0.50 | 1.50 |
| minimax-m2.7:cloud | ollama-cloud | 1.00 | 3.00 |
| glm-5 | ollama-cloud | 1.00 | 3.00 |
| glm-5.1 | ollama-cloud | 1.25 | 3.75 |
| glm-5.2 | ollama-cloud | 1.50 | 4.50 |
| qwen3.5 | ollama-cloud | 1.75 | 5.25 |
| deepseek-v4-pro | ollama-cloud | 2.00 | 6.00 |
| deepseek-v3.1:671b | ollama-cloud | 5.00 | 15.00 |
| gpt-5.5 | openai-codex | 15.00 | 60.00 |

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
- **Model registration:** edit `MODEL_REGISTRY` in `scripts/models.py` **and** Hermes `config.yaml` together — they must stay in sync.

## Pitfalls

- **`aiter_lines()` is single-use:** capture the iterator once in preflight and reuse it. Calling it again on the same httpx response returns nothing/empty.
- **state.db is large (~2GB):** The collector queries with `mode=ro` (read-only) and uses indexed queries. It only reads sessions newer than the last collection timestamp.
- **Missing model prices:** Unknown models get $0 cost. Add them to `MODEL_REGISTRY` in `models.py` (never edit the derived `DEFAULT_MODEL_COSTS` / `MODEL_COST_ORDER` directly).
- **Task classification is heuristic:** The keyword-based classifier is a starting point. Phase 2 will replace it with a trained model.
- **Cron runs silently:** The `no_agent` cron job only reports errors. Check `cronjob list` for status.
- **Runtime DB artifacts:** `biggie_router.db`, `data/*.db-shm`, `data/*.db-wal` are gitignored — never commit them.
