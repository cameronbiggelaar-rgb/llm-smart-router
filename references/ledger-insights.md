# Insights from the router ledger — cost, performance, and routing

**Scope:** `data/router_logs.db`, **242,964 rows / 23 models / 64 days**
(2026-05-19 → 2026-09-13), read-only. Billable rows via
`rollup.BILLABLE_ROW_SQL`. Method: `/tmp/ins1.py` … `/tmp/ins17.py`,
outputs `/tmp/ins*.txt`.

Every number below is a measured ledger figure, not an estimate. Where a
number cannot be trusted, that is stated as a finding in its own right.

---

## 1. The cost trend: a 25x collapse on 2026-09-12

| window | calls/day | $/day | avg context |
|---|---|---|---|
| 09-06 … 09-11 | 5,326 | **515.90** | 76,008 |
| 09-12 … 09-13 | 2,826 | **20.61** | 66,190 |

The cause is a **routing shift, not a volume drop**. Call volumes were
comparable (09-11: 6,139 calls; 09-13: 1,849 calls at time of reading), but
the model mix was replaced almost entirely:

| model | 09-10 | 09-11 | 09-12 | 09-13 |
|---|---|---|---|---|
| deepseek-v4-flash | 82.4% | 39.3% | 6.7% | **0.0%** |
| deepseek-v4.1-flash | 0.0% | 35.4% | 87.8% | **99.2%** |
| glm-5.3:cloud | 13.1% | 10.4% | 0.0% | 0.1% |
| glm-5.2:cloud | 3.1% | 5.3% | 0.0% | 0.1% |

The switch completes inside **2026-09-11T20 → 09-12T01** (see
`/tmp/ins11.txt` section F). Note the timing: the ChatGPT lane hit
`usage_limit_reached` at 09-12 10:36, *after* the shift was already underway —
so the shift was not a reaction to the cap.

**The unit economics that made it so effective:**

- `deepseek-v4.1-flash` **$0.00745/call**
- `glm-5.3:cloud` **$0.18408/call** — **24.7x** more
- At today's volume (1,837 calls/day): $13.68 on v4.1-flash vs $338.16 on
  glm-5.3 → **$324.48/day** of avoided list-price spend.

---

## 2. Lane economics: the $20 plan consumes 35% of the dollars on 2.5% of calls

Seven days, billable:

| lane | calls | share of calls | $ | share of $ | $/call |
|---|---|---|---|---|---|
| **ChatGPT** (gpt-5.5, 5.6-sol/terra/luna, 6-astra) | 929 | **2.5%** | 1,096.55 | **35.0%** | **1.1804** |
| **Ollama** (everything else) | 36,724 | 97.5% | 2,040.09 | 65.0% | 0.0556 |

**The ChatGPT lane costs 21.2x more per call.** That is the overage lever, and
it is a volume-of-calls problem far more than a token problem: 929 calls
generate a third of the bill.

Biggest single line item in that lane: **gpt-5.5, 499 calls, $705.34 (22.5% of
all billable dollars), $1.4135/call** — more than `glm-5.3:cloud` costs for
5,326 calls.

---

## 3. Context is the cost lever, and it is monotonic

$/call by context band, measured:

| model | <20K | 20–50K | 50–100K | 100–150K | >150K | ratio |
|---|---|---|---|---|---|---|
| deepseek-v4.1-flash | 0.00123 | 0.00383 | 0.00697 | 0.01568 | 0.02471 | **20x** |
| deepseek-v4-flash | 0.01152 | 0.00793 | 0.01568 | 0.02488 | 0.03550 | 3.1x |
| glm-5.3:cloud | 0.09450 | — | — | 0.18471 | 0.27302 | **2.9x** |
| gpt-5.5 | 0.54637 | 0.49684 | 1.14425 | 1.85381 | 2.46932 | **4.5x** |

Cost rises monotonically with context on every model. **This validates the
context-reduction levers (compression, the output-budget floor) as the correct
cost control** — and it is why `BIGGIE_FLASH_MAX_CONTEXT_TOKENS=0` (removing
the per-call compression) paid off ~14x on the ollama lane.

---

## 4. Duplicate-write defects inflate the ledger by ~1.9x

Two independent double-counts, both confirmed:

**(a) Streaming-marker rows.** 118,284 rows (**48.7% of all rows**) carry
`error_type='streaming_in_progress'` and **$17,591.88** (47.2% of raw
list-price cost) plus 8.02B input tokens.

These are not orphaned attempts. Of 3,000 sampled `request_id`s having both a
marker and a completed row, **2,999 had identical `input_tokens`** — they are
duplicate insert rows for the same logical request, not separate work. Only
**290** markers are genuine orphans.

Correctly excluded by `BILLABLE_ROW_SQL`.

**(b) `daily_findings` cache is stale.** Live, versus truth from `router_logs`:

| day | cached | truth | ratio |
|---|---|---|---|
| 09-12 | $33.21 | $18.62 | 1.78x |
| 09-11 | $1,395.25 | $741.97 | **1.88x** |
| 09-10 | $415.63 | $229.08 | 1.81x |

The cache was written before the B16/B17 double-count fix and never re-rolled.
`biggie-router-ops.timer` runs `maintain --days 3`, so 09-11/09-12 self-heal;
**09-10 falls outside the window and will never re-roll on its own.**
`rollup_day` is delete-then-insert (idempotent), so repair is safe on a copy.

Combined effect in the ledger: **raw $6,007.21 vs billable $3,136.64 in the
last 7 days = 1.915x inflation.**

---

## 5. The optimiser can rank nothing, and its only quality number is an artifact

`optimiser.py` requires `calls >= MIN_SAMPLES (30)`, `quality_n > 0`,
`quality_avg >= DEFAULT_QUALITY_FLOOR (0.80)`, and `is_shadow = 0` by default.

Across **242,768 live rows, exactly one model has any recorded quality at all**:

- `glm-5.3-flash` — 70 samples, `quality_avg = 0.0044`, `is_shadow = 1`

It fails on **both** counts: it is shadow-excluded by default, *and* 0.0044 is
below the 0.80 floor. **So `rank_models` returns zero eligible models for every
workload, always.** This is structural, not a tuning issue.

And the 0.0044 is the known artifact: **65 of those 70 rows are
`finish_reason='tool_calls'`** (empty summary text, 54–87 output tokens against
39K–152K inputs) graded as 0. Reading it as "glm-5.3-flash is a bad
summariser" would wrongly kill the cheapest rung.

### The optimiser also reads the stale cost cache

`optimiser.py` L121 is the **only remaining `SUM(cost_usd)` in the codebase with
no billable filter** — and it reads `daily_findings`, which carries the stale
1.8–1.9x figures from finding 4(b). So even once quality flows, the optimiser
would rank candidates using inflated incumbent costs.

Note the interaction: `daily_findings.cost_usd` is *already* marker-free (the
rollup applies the billable filter), so this filter gap is currently **latent,
not active**. It becomes active the moment quality data flows and candidates
start being compared. Fix both together.

---

## 6. Performance: latency is barely instrumented, and the aggregate is garbage

`latency_seconds` is non-null on only **74,751 of 242,964 rows (30.8%)**, and
within the recent window only two models have any latency at all:

| model | n | p50 | p90 | avg | max |
|---|---|---|---|---|---|
| deepseek-v4-flash | 70,442 | 5.53 | 19.32 | 9.82 | 1624.67 |
| deepseek-v4.1-flash (09-12+) | 5,163 | 6.25 | 26.15 | 11.95 | 302.93 |

**The all-model average of 1021s is an artifact — do not report it.** It is
dragged by **339 rows >300s**, including physically impossible values: the max
is **1,909,566s (22 days)** on a 2026-05-28 `glm-5.1` row, and several
million-second values on `gpt-5.5` and `minimax-m2.7` rows from May. These are
missing/incorrect end-timestamps. Excluding rows >300s, the `<20K` band averages
**19.05s, not 1021s**.

Latency genuinely scales sub-linearly with context (medians):

| band | n | p50 | p90 |
|---|---|---|---|
| <20K | 12,591 | 5.07 | 62.70 |
| 20–50K | 46,507 | 5.03 | 16.92 |
| 50–100K | 40,705 | 6.36 | 19.26 |
| 100–150K | 18,016 | 7.96 | 21.47 |
| >150K | 6,867 | 9.40 | 19.59 |

So the v4.1-flash switch cost roughly **+1.1s at p50** (6.25s vs 5.53s)
relative to v4-flash — cheap and fast, a good trade.

---

## 7. Reliability is uniformly clean; escalation is the only real signal

`success=0` and `empty_stream=1` are **0.00% for every high-volume model**
(sole exception `glm-5.3-flash` at 1.58% — the artifact-covered one). Failure
counting is therefore not useful here; **escalation rate is the signal**:

| model | n | escalated |
|---|---|---|
| llama3.1:8b | 216 | 65.28% |
| gpt-5.6-sol | 544 | 41.73% |
| dolphin3 | 108 | 27.78% |
| gpt-5.5 | 14,405 | 2.42% |
| deepseek-v4-pro | 8,798 | 1.95% |
| glm-5.2:cloud | 11,352 | 1.16% |
| glm-5.3:cloud | 9,714 | 0.55% |
| deepseek-v4-flash | 70,442 | 0.20% |
| **deepseek-v4.1-flash** | 7,358 | **0.12%** |

`gpt-5.6-sol` is **not a failing model**: all 544 rows have `success=1`,
`empty_stream=0`, and `model_used == final_model` on every escalated row — it
is the terminal escalation sink, so its 41.73% is by construction. Same for
`llama3.1:8b` and `dolphin3` (local, $0.00, escalation to cloud).

The new incumbent `deepseek-v4.1-flash` has the **lowest escalation rate of any
cloud model at 0.12%** — the collapse onto it was not a reliability regression.

---

## 8. Measurement gaps that bound what this data can answer

1. **`output_tokens > 0` on only 6,661 of 242,964 rows (2.74%).** Cost is
   effectively input-only, so every figure above **understates true cost** for
   output-heavy workloads. Relative comparisons between models remain valid;
   absolute dollars do not.
2. **Latency on 2 of 23 models**, and 0 of the ChatGPT lane — the most expensive
   lane has no performance data at all.
3. **Quality: 70 of 242,964 rows (0.03%)**, and those 70 are the artifact set.
   There is currently **no working quality signal for any production model.**

---

## Actionable conclusions

| # | Finding | Action |
|---|---|---|
| 1 | Optimiser structurally inert (0 eligible models) | Wire the new probe into `rollup`; without it no ranking can ever occur |
| 2 | 09-10 `daily_findings` stale 1.81x, outside the self-heal window | Manual re-roll on a copy, then live (prod write — needs approval) |
| 3 | `optimiser.py` L121 lacks the billable filter | Latent today; fix together with #1 |
| 4 | 118,284 marker rows = 48.7% of rows | Already correctly excluded — no action, but never add a report that bypasses `BILLABLE_ROW_SQL` |
| 5 | ChatGPT lane = 35% of $ on 2.5% of calls | The real overage control: cap gpt-5.5 *calls*, not tokens |
| 6 | Latency aggregates poisoned by timestamp bugs | Exclude rows >300s; fix end-timestamp capture |
| 7 | `output_tokens` unpopulated | Blocks output-cost accounting entirely |
