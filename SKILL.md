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

The router is a **read-only observer** — it reads existing Hermes session data (`state.db`) and builds a feature set from what's already tracked. No gateway patching required.

```
state.db (Hermes) → RouterCollector → feature_extractor → router_logs.db (SQLite)
                                                              ↓
                                                         report.py (cost summary)
```

## Phase 1: Data Collection (current)

A cron job runs every 6 hours, reads the Hermes sessions DB, extracts features from each session, and writes observations to the router SQLite database.

### Files

| File | Purpose |
|---|---|
| `scripts/models.py` | Dataclasses, SQLite schema, model pricing table |
| `scripts/feature_extractor.py` | Task classification, feature extraction from session data |
| `scripts/collector.py` | RouterCollector — reads Hermes DB, writes to router DB |
| `scripts/report.py` | Cost savings report generator |
| `tests/test_router.py` | 36 unit + integration tests |
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

### Pricing model
All models are **flat-subscription or free** — there's no per-call cost:
- **gpt-5.5 (ChatGPT OAuth):** $20/mo flat, rate-limited
- **Ollama Cloud:** $100/mo flat
- **Local:** Free

The "cost" values are **relative compute units** — a dimensionless measure of how much subscription budget each model call consumes. This lets the router prefer cheaper models for simple tasks, preserving expensive model capacity for complex work.

### Relative compute ratios (deepseek-v4-flash = 1.0x baseline)
| Model | Ratio | Notes |
|---|---|---|
| llama3.1:8b | 0.0x | Local, free |
| qwen3:14b | 0.0x | Local, free |
| deepseek-v4-flash | **1.0x** | Baseline — small, fast |
| minimax-m2.7:cloud | **2.0x** | Mid-size |
| glm-5 | **2.0x** | Mid-size |
| glm-5.1 | **2.5x** | Slightly larger |
| glm-5.2 | **3.0x** | Larger context |
| deepseek-v4-pro | **4.0x** | Premium tier |
| deepseek-v3.1:671b | **10.0x** | Massive 671B MoE |
| gpt-5.5 | **30.0x** | ChatGPT $20/mo, most capable, rate-limited |

### Effective compute units per 1M tokens
| Model | Provider | Input units/1M | Output units/1M |
|---|---|---|---|
| llama3.1:8b | local | 0.00 | 0.00 |
| qwen3:14b | local | 0.00 | 0.00 |
| deepseek-v4-flash | ollama-cloud | 0.50 | 1.50 |
| minimax-m2.7:cloud | ollama-cloud | 1.00 | 3.00 |
| glm-5 | ollama-cloud | 1.00 | 3.00 |
| glm-5.1 | ollama-cloud | 1.25 | 3.75 |
| glm-5.2 | ollama-cloud | 1.50 | 4.50 |
| deepseek-v4-pro | ollama-cloud | 2.00 | 6.00 |
| deepseek-v3.1:671b | ollama-cloud | 5.00 | 15.00 |
| gpt-5.5 | openai-codex | 15.00 | 60.00 |

## Pitfalls

- **state.db is large (~2GB):** The collector queries with `mode=ro` (read-only) and uses indexed queries. It only reads sessions newer than the last collection timestamp.
- **Missing model prices:** Unknown models get $0 cost. Add them to `models.py:DEFAULT_MODEL_COSTS` or insert directly into `model_costs` table.
- **Task classification is heuristic:** The keyword-based classifier is a starting point. Phase 2 will replace it with a trained model.
- **Cron runs silently:** The `no_agent` cron job only reports errors. Check `cronjob list` for status.
