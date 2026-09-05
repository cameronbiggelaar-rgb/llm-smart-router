# Plan — Register gpt-5.6 tiers + create a GPT-6 lane

**Status:** DRAFT — for review, not yet executed
**Date:** 2026-09-05
**Author:** Hermes Agent
**Scope:** `llm-smart-router` skill (live `biggie-llm-endpoint.service`)

---

## 1. Goal

Two changes to the Biggie LLM router:

1. **Lane 1 — update the tiers:** register the three gpt-5.6 agentic-coding models
   (`gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol`) as the new top routing tiers
   (11–13), above gpt-5.5 (tier 10). **Do NOT raise the existing routing-table
   floors** — routine work stays on ollama-cloud. The 5.6 models are reached two
   ways: (a) **escalation** when a lower model fails (rate limit, degeneration,
   malformed tool call), and (b) an **explicit rethink/rearchitect trigger** that
   routes deliberate high-stakes asks straight to the strongest model.
2. **Lane 2 — a GPT-6 lane:** make `gpt-6-astra` reachable as a **separate,
   explicit** path for whole-project autonomous work — **not** a per-prompt
   routing tier. This honours the GPT-6 evaluation verdict (do not auto-route to
   GPT-6; it is not cost-effective as a router workhorse) while giving the user a
   lane for architecture-review / find-failure-states / update-plans asks.

**Why gpt-6 is a lane, not a tier:** the router is prompt-in/prompt-out. GPT-6 is
problem-in/agent-out. Forcing it into the tier ladder would underuse it and burn
its metered capacity on routine prompts. A separate trigger path is the honest fit.

**Design principle (per user):** keep the $20/mo ChatGPT capacity for the
rethink/rearchitect points — the bigger problems the lower models can't solve.
Routine work must not drift onto the strong models just because they exist.

---

## 2. Current state (verified)

- **Single source of truth:** `MODEL_REGISTRY` in `scripts/models.py`. Adding a
  model = one dict entry `{provider, ratio, input, output, tier, date}`.
  `DEFAULT_MODEL_COSTS`, `MODEL_COST_ORDER`, `MODEL_CAPABILITY_TIERS` are all
  **derived** — they update automatically. Never edit them directly.
- **Backend discovery:** `discover_backends()` in `biggie_llm_endpoint.py` reads
  `fallback_providers` from `config.yaml` → `fallback_chain` → backends. A model
  must be in `fallback_providers` (provider `openai-codex`) to be reachable.
- **Routing:** `_select_model(min_tier)` returns the cheapest available model with
  `tier >= min_tier` (walks `MODEL_COST_ORDER`). `_build_fallback_chain` escalates
  to models with `tier > current_tier`. The routing table (`routing_table.yaml`)
  sets task-type/sub-type **floors**; complexity can escalate above.
- **⚠️ Critical clamp:** `_estimate_min_tier` returns `max(1, min(min_tier, 10))`
  (router.py line 1300), and complexity scoring also caps at 10. **No normal
  routing path can reach tier 11+.** The 5.6 models and gpt-6 are reachable ONLY
  via `escalate_on_failure` (uses `MODEL_CAPABILITY_TIERS` directly, no clamp) or
  `force_model`. This is exactly what we want: the strong models are reserved for
  escalation and explicit triggers, never auto-selected for routine work.
- **Tool gating:** `TOOL_CAPABLE_MODELS` in `router.py` gates tool-bearing
  requests. Consistency test requires `TOOL_CAPABLE_MODELS ⊆ MODEL_REGISTRY`.
- **Baseline:** `python3 -m pytest tests/` → **112 passed** (green).
- **Worktree:** clean.

---

## 3. Lane 1 — gpt-5.6 tiers (11–13)

### 3.1 `scripts/models.py` — MODEL_REGISTRY additions

Add three entries (provider `openai-codex`, flat $20/mo ChatGPT — same OAuth
credential as gpt-5.5). Input units set **above** gpt-5.5's 15.00 so
`MODEL_COST_ORDER` keeps gpt-5.5 cheapest for tier-10 work and only reaches the
5.6 models for tier 11+:

| Model | provider | ratio | input | output | tier | date |
|---|---|---|---|---|---|---|
| gpt-5.6-luna | openai-codex | 32.0 | 16.00 | 64.00 | 11 | 2026-09-05 |
| gpt-5.6-terra | openai-codex | 34.0 | 17.00 | 68.00 | 12 | 2026-09-05 |
| gpt-5.6-sol | openai-codex | 36.0 | 18.00 | 72.00 | 13 | 2026-09-05 |

*(Ratios/units are proposed defaults — adjust if you have a different capacity
model in mind. They only affect relative ordering, not money.)*

### 3.2 `config.yaml` — make them reachable

- Add all three to `fallback_providers` (provider `openai-codex`), **after**
  gpt-5.5 so the fallback chain stays ordered cheapest→most capable.
- Add all three to `custom_providers.biggie-llm.models` so Hermes' own model list
  knows about them.

### 3.3 `routing_table.yaml` — NO floor raises

**Do not change any routing-table floors.** Routine work (implement, debug,
plan) stays on ollama-cloud. The 5.6 models are reached only via escalation and
the explicit rethink/rearchitect trigger (3.4). This is the user's explicit
preference: keep the $20/mo capacity for the bigger problems, don't let routine
work drift onto the strong models.

Only change: update the tier-reference comment to add tiers 11–13.

### 3.4 Explicit rethink/rearchitect trigger (new)

Add a routing-table sub-type under `planning` that routes **deliberate
high-stakes asks** straight to the strongest 5.6 model — the "bigger problems the
lower models can't solve" lane. This is the user's stated purpose for keeping
$20/mo capacity.

| Task type / sub-type | Keywords | Tier | Routes to |
|---|---|---|---|
| planning → rethink/rearchitect | "rethink", "rearchitect", "redesign the architecture", "find failure states", "update the plans", "review the architecture" | 13 | gpt-5.6-sol |

**⚠️ But the clamp blocks this:** `_estimate_min_tier` caps at 10, so a tier-13
floor would be clamped to 10 and route to gpt-5.5, not gpt-5.6-sol. To make the
trigger actually reach the 5.6 models, we must **raise the clamp ceiling** from
10 to 13 (router.py line 1300) — OR route the trigger via `force_model` instead.

**Two sub-options (needs your call):**
- **3.4a — raise the clamp to 13:** `_estimate_min_tier` returns
  `max(1, min(min_tier, 13))`. This lets the rethink/rearchitect floor (13) reach
  gpt-5.6-sol. Risk: any future floor above 13 would also pass through — but
  gpt-6 stays at 14, so it's still never auto-selected. Low risk, keeps the
  trigger in the routing table where it's visible and testable.
- **3.4b — force_model trigger:** the rethink/rearchitect keywords route via
  `force_model="gpt-5.6-sol"` in the endpoint, bypassing the clamp entirely. No
  clamp change. Slightly more wiring in the endpoint, but zero risk to the
  routing ladder.

**Recommendation:** 3.4a (raise clamp to 13). It's one line, keeps the trigger
declarative in the routing table, and gpt-6 at tier 14 remains unreachable by
normal routing — preserving the "lane, not tier" boundary.

### 3.5 `router.py` — TOOL_CAPABLE_MODELS

Add the three 5.6 models **only after verifying** they emit clean structured
`tool_calls` (they are agentic-coding models, so expected — but must be proven,
mirroring the deepseek-v4-pro verification note). The consistency test
(`test_policy_sets_only_reference_registered_models`) will pass automatically
once they're in the registry.

### 3.6 Tests (TDD — RED first)

- **`tests/test_router.py`** — add routing assertions:
  - rethink/rearchitect planning ask → gpt-5.6-sol (via 3.4a clamp raise)
  - escalation from gpt-5.5 → gpt-5.6-luna → terra → sol
  - **regression guard:** routine coding "implement/build" still routes to
    ollama-cloud (glm-5.3), NOT gpt-5.6 — proves the floor was not raised
  - **regression guard:** routine debugging "crash/race/deadlock" still routes to
    ollama-cloud, NOT gpt-5.6
- **`tests/test_model_registry_consistency.py`** — should stay green (generic
  invariants); add explicit assertions that the three 5.6 models are registered,
  priced, and tiered.

---

## 4. Lane 2 — GPT-6 lane (separate, explicit)

### 4.1 `scripts/models.py` — register gpt-6-astra

| Model | provider | ratio | input | output | tier | date |
|---|---|---|---|---|---|---|
| gpt-6-astra | openai-codex | 40.0 | 20.00 | 80.00 | 14 | 2026-09-05 |

Registered at tier 14 but **NOT added to the routing table** → never auto-selected
by `_select_model` for normal tasks (floors cap at 13, complexity caps at 10).
Reachable only via an explicit trigger.

### 4.2 `config.yaml` — make it reachable

- Add to `fallback_providers` (provider `openai-codex`), after the 5.6 models.
- Add to `custom_providers.biggie-llm.models`.

### 4.3 Trigger mechanism (design decision — needs your call)

**Option A (recommended, minimal):** reachable via `force_model="gpt-6-astra"`
only. No routing-table entry, no new workload type. Safest — GPT-6 is used only
when explicitly requested. Zero risk of accidental auto-routing.

**Option B (fuller lane):** add a new workload type (e.g. `agentic_project`) or a
routing-table entry for whole-project asks ("review the architecture", "find
failure states", "update the plans") that routes to gpt-6-astra. More useful, but
adds a detection path and increases GPT-6 usage — needs a usage guard.

**Recommendation:** ship Option A first (reachable + explicit), measure real
demand, then build Option B only if whole-project asks actually occur.

### 4.4 Tests

- gpt-6-astra is **not** auto-selected for any normal task type (regression guard
  against accidental auto-routing).
- gpt-6-astra is reachable via `force_model="gpt-6-astra"`.

---

## 5. Deploy & verify

1. `python3 -m pytest tests/` — all green (new + existing).
2. Restart: `sudo systemctl restart biggie-llm-endpoint.service`.
3. Health: `curl -s http://127.0.0.1:8080/v1/models` shows the four new models.
4. Live smoke: force a coding task to `gpt-5.6-luna` and an architecture ask to
   `gpt-6-astra`; confirm routing logs show the right model/provider.
5. Confirm gpt-6-astra is **not** auto-selected for a routine qa task.
6. Confirm a routine "implement" task still routes to ollama-cloud (floor not
   raised).

---

## 6. Open decisions (blocking before execution)

1. **Tier/ratio values** for the 5.6 models (proposed above — confirm or adjust).
2. **Rethink/rearchitect trigger** — 3.4a (raise clamp to 13, declarative) vs
   3.4b (force_model, no clamp change). Recommend 3.4a.
3. **GPT-6 lane trigger** — Option A (force_model only) vs Option B (workload
   type). Recommend A first.
4. **Tool-capability verification** of the 5.6 models before adding to
   `TOOL_CAPABLE_MODELS`.

---

## 7. Execution order (TDD, one commit)

1. Write failing tests (3.6 + 4.4) → RED.
2. Add MODEL_REGISTRY entries (3.1 + 4.1) → tests GREEN.
3. Update config.yaml (3.2 + 4.2).
4. Update routing_table.yaml (3.3 comment + 3.4 trigger) and raise clamp (3.4a).
5. Verify tool-calling, then update TOOL_CAPABLE_MODELS (3.5).
6. Full suite green → commit as **one** commit.
7. Deploy + verify (5). **Stop for confirmation before deploy** (prod routing
   change).
