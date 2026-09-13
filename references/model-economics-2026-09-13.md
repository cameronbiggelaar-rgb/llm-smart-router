# Ledger insights — model economics, and what to test next

**Scope:** live `data/router_logs.db`, read-only, canonical `BILLABLE_ROW_SQL`
throughout. Companion to `references/ledger-insights.md` (B21), which covered the
64-day view; this one is the **model-economics and testing-agenda** cut, taken
2026-09-13 after the B23 writer fix landed (not yet live).

Latency is reported as **percentiles on rows with `0 < latency_seconds < 300`** —
see the `latency_seconds` pitfall in SKILL.md: 339 rows carry uncaptured-end
timestamps (max 22 days) that drag a naive mean to ~1021s when p50 is 6.4s.

## 1. Where we actually are now

The routing collapse onto `deepseek-v4.1-flash` (B21) is **complete and holding**:

| day | calls | spend | v4.1-flash share |
|---|---|---|---|
| 2026-09-11 | 6,139 | $426.92 (compression) | 35% |
| 2026-09-12 | 3,815 | $18.62 | 88% |
| 2026-09-13 | 1,856 | $22.71 | **99.2%** |

Per-call compression cost fell $0.0647 → $0.0045 over the same period (~14x),
which is the `BIGGIE_FLASH_MAX_CONTEXT_TOKENS=0` lever doing its job.

**Concentration risk is now the headline:** 1,841 of 1,856 calls today (99.2%)
land on one model. Escalation path is barely exercised (3 escalations today,
0.16%). The ladder has never been tested under a v4.1-flash outage.

## 2. Peak pricing — an unmodelled cost (NEW)

Ollama publishes **peak pricing at 2x** for the three deepseek models, applying
**12:00–18:00 UTC, Monday–Friday** (22:00–04:00 AEST). Measured exposure,
last 14 days:

| model | window | calls | input | cost at published rate |
|---|---|---|---|---|
| deepseek-v4-flash | PEAK | 1,879 | 119.2M | $52.43 |
| deepseek-v4.1-flash | PEAK | 335 | 17.5M | $5.31 |
| deepseek-v4-pro | PEAK | 8 | 0.3M | $0.62 |

**Peak = 9.7% of these models' spend.** Our price book models a **single flat
rate per model** and has no peak dimension, so:

- the ledger **understates** peak-window calls and **overstates** offpeak ones;
- net across the 14 days it is nearly self-cancelling ($610.73 recorded vs
  $603.33 published — 1.2% high), so it is **not** currently distorting totals;
- but it is a real modelling gap on a $100 plan with a $300 credit ceiling, and
  the hour profile explains why the exposure is small: traffic is
  AEST-evening-weighted (21:00 UTC is the peak hour at 5,072 calls) — the
  peak window catches only the 12:00–13:00 UTC tail.

**Actionable:** if peak load ever rises, shifting compression out of 12:00–18:00
UTC is a pure 2x saving with no quality trade. Worth a guard, not urgent.

## 3. New models on Ollama Cloud we do not route to

Published rates (USD/1M in/out), re-checked 2026-09-13:

| model | in | out | note |
|---|---|---|---|
| **nemotron-3-super** | **0.015** | 0.60 | cheapest input on the platform |
| gpt-oss:20b | 0.07 | 0.30 | |
| nemotron-3-nano | 0.06 | 0.24 | |
| gpt-oss:120b | 0.15 | 0.60 | same rate as v4.1-flash |
| gemma4 | 0.14 | 0.40 | |
| glm-5.3-flash | 0.15 | 0.50 | in shadow experiment |
| nemotron-3-ultra | 0.10 | 3.00 | cheap in, dear out |
| mistral-large-3 | 0.50 | 1.50 | |
| minimax-m3 | 0.60 | 2.40 | |
| kimi-k3 | 3.00 | 15.00 | 1M ctx, tool calling |
| kimi-k2.7-code | 0.95 | 4.00 | code-specialised |

**None of these are priced in our book or registered in the router.** We are
routing 4,573M input tokens of compression through a ladder whose cheapest rung
is $0.15/1M when a $0.015/1M model is offered.

**Counterfactual** (same input tokens, last 14d, ollama lane, $3,429.30 actual):

- all at `nemotron-3-super` ($0.015) → **$68.59** (saves $3,360.71)
- all at `gpt-oss:20b` ($0.07) → $320.09 (saves $3,109.21)
- all at `deepseek-v4.1-flash` ($0.15) → $685.90 (saves $2,743.40)

This is a **ceiling on the prize, not a forecast** — compression is
input-bound (4,573M in vs 0.4M out) so an input-cheap model is exactly the right
shape, but nothing here has been quality-tested for summarisation.

Note the ladder already **excludes** `minimax/glm-5/glm-5.1` as summarisers by
policy, and `glm-5.3`/`glm-5.2` are the two most expensive models in the
compression mix ($1,907.61 + $927.55 over 14d = **83% of ollama-lane spend** on
two models whose rate is 9.3x the cheapest rung).

## 4. Existing models: what the data says

**Latency** (bounded, 14d) — p50 / p95:

| model | n | p50 | p95 |
|---|---|---|---|
| deepseek-v4-flash | 40,761 | 5.20 | 32.94 |
| gpt-5.5 | 3,043 | 5.38 | 47.27 |
| deepseek-v4.1-flash | 7,349 | **6.40** | 41.81 |
| deepseek-v4-pro | 769 | 6.83 | 49.36 |
| glm-5.3 | 9,713 | 7.65 | 36.24 |
| glm-5.2 | 3,665 | 8.84 | 21.86 |
| gpt-5.6-sol | 544 | 8.52 | 48.35 |
| **glm-5.3-flash** | 190 | **10.12** | 46.88 |
| **qwen3.5** | 208 | **45.15** | 67.57 |
| llama3.1:8b | 86 | 63.33 | 246.99 |

`qwen3.5` is a **7x latency outlier** vs the field. It is tier 7, used only 208
times, and costs $0.60/1M — it occupies a rung almost never selected. Worth
questioning whether it earns its place.

**Reliability — `escalated`/`error_type` as a model signal, not success rate:**

| model | n (tools) | escalations |
|---|---|---|
| deepseek-v4-flash | 39,696 | 0.04% |
| deepseek-v4.1-flash | 7,086 | **0.06%** |
| glm-5.3 | 9,599 | 0.04% |
| gpt-5.5 | 2,891 | 2.0% |
| deepseek-v4-pro | 618 | 4.1% |
| **gpt-5.6-sol** | 542 | **41.0%** |

v4.1-flash — the model now carrying 99% of traffic — has the **lowest escalation
rate of any cloud model**. The collapse was not a reliability trade. `gpt-5.6-sol`
at 41% is a *sink*, not a failure (terminal fallback, all rows `success=1`), but
the 41% figure means it is where broken traffic lands.

## 5. The measurement gap is now the binding constraint

**`quality_probe` contains ZERO rows.** The B20 mechanism is deployed but has
never recorded a single probe.

- `BIGGIE_QUALITY_PROBE_RATE` is **not set** in the service environment, so the
  default `0.05` (5%) applies.
- The running service (PID 927, started 11:39) **predates the B19/B20 edits**
  (`quality_probe.py` mtime 16:29, `biggie_llm_endpoint.py` 17:39), so
  `should_measure`/`record_probe` are **not in the running process at all**.

So at 5% of ~1,856 daily calls we should expect **~90 probes/day** once live —
ample for the optimiser's `quality_n >= 30` gate within a day.

**The shadow experiment is also stranded.** `glm53flash-compression-shadow` ran
188 rows on 2026-09-12 only, at `avg_q = 0.0042` — that is the *tool_calls
artifact* (empty-text rows graded 0), not a real score. Measured cost/latency
from those rows:

- `glm-5.3-flash`: 188 calls, avg **99.7K** in, **516** out, $0.01522/call, 17.7s
- `deepseek-v4.1-flash`: 7,347 calls, avg **66.3K** in, **16.6** out, $0.00746/call

glm-5.3-flash is **3.5x more verbose** (516 vs 17 output tokens) on a summary
task and **2x the per-call cost** despite an identical input rate. That is a
quality question the artifact score cannot answer — which is exactly what a
working probe would resolve.

The experiment is `enabled: false` in `scripts/experiments.yaml`. Both entries
(shadow and the 10% split) are disabled.

## 6. Smaller findings

- **`:cloud` suffix pricing works.** `cost_of_call("deepseek-v4.1-flash:cloud")`
  returns $0.15 — the normaliser resolves it. The 2,119 rows flagged
  `cost_unknown=1` on 09-12/09-13 are a **backfill artifact**
  (`pricing_version='backfill:2026-09-12'` wrote them before the 09-12 price row
  existed); they are 28% of that model's rows and 0% of the 09-13 rows I sampled.
- **Only 6 of 20 offered models are configured backends.** Adding a model needs
  both a `MODEL_REGISTRY` entry and a configured backend — an experiment naming an
  unregistered model is skipped, never half-applied.
- **`gpt-5.5` served 667 compression calls** under
  `"native session_compression workload — selected cheapest available"` despite
  the code comment saying it is deliberately excluded. That reason string is the
  fail-open branch: it fires when the preferred ladder yields nothing. 667 calls
  ≈ $700-1,900. The exclusion comment and the fail-open path disagree.
- **Failover has effectively never been tested:** 50 `all cloud models exhausted`
  events in 14 days against ~60,000 calls (0.08%).

## 7. Recommended testing agenda (in order)

1. **Restart the service.** Until then: two rows per streaming request, zero
   output tokens on streams, no quality probing. Everything below is blocked on
   measurements that a restart starts producing.
2. **Confirm the probe fires** (~90/day expected) and that `quality_n` reaches 30
   for at least one model — that is the optimiser's only gate.
3. **Test candidate summarisers on real compression payloads**, in shadow, one at
   a time: `nemotron-3-super` ($0.015 in — the 10x input lever),
   `gpt-oss:20b` ($0.07), `glm-5.3-flash` ($0.15, already measured 3.5x verbose).
   Judge on **fidelity of the summary**, not cost — a cheaper summariser that
   loses context costs more in downstream re-reads.
4. **Question `glm-5.3`/`glm-5.2` as the compression fallback.** They are 83% of
   ollama-lane spend at 9.3x the cheapest rung, and exist because v4.1-flash was
   believed unsafe on large contexts. `MODEL_CONTEXT_CEILING` now enforces that
   separately, so the ladder may be able to start lower.
5. **Revisit `qwen3.5`** — 45s p50, 7x the field, 208 calls in 14 days.
6. **Add peak-pricing awareness** if compression load ever moves into
   12:00–18:00 UTC. Today it is a 9.7% self-cancelling effect, not a distortion.
7. **Decide the `gpt-5.5` fail-open.** Either the exclusion is real (and the
   fail-open should skip it too) or it is not (and the comment is wrong). 667
   calls of ambiguity is worth resolving.
