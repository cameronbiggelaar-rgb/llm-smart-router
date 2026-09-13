# Allowance-burn reporting + compression quality measurement

Scope agreed with the operator 2026-09-13 ("Yes do these", 3 items).
Repo convention: this plan is written to `references/` BEFORE execution, and
each batch is TDD (RED → GREEN → commit) with its own test file.

**Status: plan written. See "Item 2 — finding" below: the requested change is
NOT implemented because the evidence falsifies its premise.**

---

## Item 1 — allowance-burn report (the reconciliation method)

**Goal.** Early warning before overage, in the currency the operator pays
(the `$100` Ollama plan's included usage), not list-price ledger dollars.

**Why list prices are the wrong currency.** The ledger prices every token at
metered list rate. On a fixed plan with included allowance, in-allowance usage
is not billed at all. Measured gap: the old glm-heavy week logged **$3,977**
while the operator's dashboard showed **$48.26 week-to-date** for the same
traffic — ~80x. So the ledger is a *routing-mix proxy*; the dashboard is
billing ground truth.

**The method (operator-specified).**

    implied marginal $/token = dashboard_$ / (avg_input_tokens_per_request
                                              x dashboard_credit_billed_requests)

Units check: `$ / (tokens/request x requests) = $ / token`. This is a
*marginal* rate — only usage beyond the included allowance is billed, which is
why it lands far below list price.

**Calibration data (from the operator's supplied dashboard snapshots):**

| Snapshot (AEST) | Balance | Week-to-date $ | Credit-billed reqs |
|---|---|---|---|
| 2026-09-11 20:02 | 16.28 | 23.73 | 1,749 |
| 2026-09-11 23:14 | 21.88 | 38.11 | 2,578 |
| 2026-09-12 06:10 | 11.74 | 48.26 | 2,960 |

Per-model breakdown at the latest reading (2026-09-12 06:10):

| Model | Requests | $ | $/req |
|---|---|---|---|
| glm-5.3 | 431 | 24.91 | 0.05780 |
| glm-5.2 | 252 | 10.10 | 0.04008 |
| deepseek-v4-flash:0731 | 1,751 | 9.11 | 0.00520 |
| deepseek-v4-pro:0813 | 115 | 2.21 | 0.01922 |
| deepseek-v4.1-flash | 406 | 1.80 | 0.00443 |
| qwen3.5:397b | 5 | 0.13 | 0.02600 |
| TOTAL | 2,960 | 48.26 | 0.01630 |

**Deliverable.** `scripts/allowance.py`:

* `Snapshot` — a recorded dashboard reading (spend, credit-billed requests,
  optional balance + per-model split).
* A snapshot store (`data/allowance_snapshots.json`, gitignored) so readings
  accumulate; the report must never invent a dashboard number.
* `calibrate()` — the method above, per model and overall, joining each
  snapshot to the ledger's `avg_input_tokens_per_request` for the window.
* `burn_report()` — project the current period's spend from live token burn at
  the calibrated marginal rate, express it as % of the allowance, and warn
  BEFORE the allowance is consumed.
* `router_ops.py allowance` subcommand (text + `--json`).

**Honesty rules baked into the report.** Label the marginal rate as calibrated
(not list); state the snapshot date and that spend is projected from it; never
present a projected figure as an actual one. If no snapshot exists, say so and
refuse to project rather than default to list prices.

**Open question, must be flagged not guessed:** the vendor pricing page states
**$300 of usage credits/mo** on the `$100` tier, while this skill's reference
frames it as "the `$100` plan's included usage". The allowance is therefore a
**configurable** input (default from env) and the report prints which value it
used.

**Tests.** `tests/test_allowance_burn.py` — calibration arithmetic against the
real snapshot table above; refusal-to-project with no snapshot; marginal rate
strictly below list price; warning fires before the allowance is consumed.

---

## Item 2 — finding: the 150K flash ceiling is NOT applied (falsified premise)

The operator asked to "add the 150K ceiling to flash — 144 empty-content events
on >100K contexts is a reliability hole". That framing came from an earlier
message of mine and does not survive measurement.

**1. The empty-content rate does NOT rise with context size.** Journal events
(de-duplicated, matched to logged context sizes — see the join traps below):

| Band | Requests | Reqs % | Empties | Empties % | per-1k reqs |
|---|---|---|---|---|---|
| <20K | 177 | 3.6 | 7 | 9.9 | **39.5** |
| 20-50K | 1,850 | 37.4 | 35 | 49.3 | 18.9 |
| 50-100K | 2,024 | 40.9 | 22 | 31.0 | 10.9 |
| 100-150K | 604 | 12.2 | 5 | 7.0 | 8.3 |
| 150-200K | 260 | 5.3 | 1 | 1.4 | **3.8** |
| >200K | 36 | 0.7 | 1 | 1.4 | 27.8 |

The rate is **highest below 20K and falls monotonically to 150-200K**. Above
100K is 18% of requests but only 9% of empties. Context size is not the driver.

**2. The real cause is output budget, and it is already fixed.** Flash's
failure is `finish_reason=length` with reasoning consuming the whole budget —
it fires at a **16K** context with `max_tokens=256` and does not fire at 141K
with a realistic budget. `apply_session_compression_output_floor()`
(`BIGGIE_SESSION_COMPRESSION_MIN_MAX_TOKENS`, default 16384) addresses exactly
this and has been **live since 2026-09-12 07:43** (1,572 log lines).

**3. Zero failures above 150K since that deploy.** Post-floor empty events: 3
(all at 20-50K/50-100K). **Above 150K: zero**, while 425 of 5,995 requests
(7.1%) sit above 150K. There is no reliability hole to close.

**4. Applying it would be a cost regression.** The ceiling skips the whole
flash family above the threshold. The >100K band is **48.5% of all input
tokens**, and flash is **9.3x cheaper per token** than the glm rung it
escalates to. The measured 09-12 mix change (flash-first) took Ollama-lane
compression from **$426.92/day to $29.42/day**. The 150K ceiling would push
that traffic back up the ladder for no measured reliability gain.

**Therefore:** not implemented. Deliverable is instead a **regression pin** so
the floor cannot silently regress (it is the change that actually fixed the
failures), plus this recorded finding. The decision to add a size ceiling is
put back to the operator with the cost attached.

**Join traps that produced a wrong first answer** (cost hours if forgotten):
1. `router_logs.timestamp` is **UTC** ISO; the journal is **local (AEST)**. A
   naive string match yields 0/72 matches and a fake "no correlation" result.
2. `model_used` in the DB carries a provider suffix (`:cloud`); journal lines
   do not.
3. `router_logs.empty_stream` is a **dead column** (`SUM()=0` over 75,953
   rows) and `error_type='empty_content'` is **masked by escalation** (the
   recovery overwrites it). The journal is authoritative for empty streams.
4. A streaming compression writes **two rows** — always
   `COUNT(DISTINCT request_id)`.

**Tests.** `tests/test_compression_reliability_pin.py` — the floor applies to
session_compression only, applies through escalation, preserves larger caller
budgets, and the size-ceiling default stays disabled (0) with the reason
recorded.

---

## Item 3 — measure compression quality (live, zero incremental cost)

**What was asked.** "Start measuring compression quality — the shadow
experiment (glm53flash-compression-shadow) has been disabled since 09-12, and
without a quality signal the cheap rung is running unverified."

**Why NOT simply re-enable the shadow experiment.**

* Its question is **already answered**: on the assembled production prompt with
  tools present, `glm-5.3-flash` returned `finish=length` with **0 chars** while
  the incumbent returned 35,038 chars. Decisive for compression.
* Re-running it costs real money: **$2.86 per 188 calls ($0.0155/call)**, which
  at the observed compression rate mirrors **~$180/day**.
* It measures a *candidate*, whereas the stated concern is the **incumbent**
  cheap rung running unverified. Those are different questions.

**The insight that makes this cheap.** `quality.score_summary` is a local
`fact_coverage_v1` regex scorer — it needs no second generation. Both inputs
are already in memory on every compression call: the source messages and the
streamed summary text. So quality can be measured on **100% of compression
traffic at zero incremental token cost**, instead of on 1/26th of it at
$180/day.

**Implementation.**

* The streaming relay `_resume_stream` already accumulates the full streamed
  content for the degeneration guard — propagate it to the completion logger
  (`stats["content"]`); no new buffering.
* Score in the completion logger when `workload_type == session_compression`,
  persisting to `quality_score` / `quality_method` (both columns already
  exist and are already surfaced by `unit_cost`, `rollup` and `router_ops
  report`).
* Source = the messages the model actually received (`compressed_messages`),
  which isolates model fidelity from compression loss and matches the offline
  A/B harness semantics.
* Env gate `BIGGIE_COMPRESSION_QUALITY_SCORING` (default on); an unscored call
  stays `NULL`, never `0.0` — "not measured" must stay distinct from "measured
  bad", which is the existing `quality.py` contract.

**Tests.** `tests/test_live_quality_scoring.py` — scoring happens for
session_compression and NOT for other workloads; an unscored call stays NULL;
the source used is the post-compression messages; a degenerate empty summary
scores 0.0 while a faithful one scores high; scoring failure never fails the
request.

---

## Execution order

1. Batch 1 — `allowance.py` + `test_allowance_burn.py` + CLI.
2. Batch 2 — reliability/floor pins + `test_compression_reliability_pin.py`.
3. Batch 3 — live quality scoring + `test_live_quality_scoring.py`.
4. Full suite + preflight, docs (SKILL.md + reference), commit, push.
5. Report: what is live vs what needs a service restart, and the Item 2
   decision returned to the operator.

---

## Batch results (executed 2026-09-13)

### Batch 1 — allowance-burn report — DONE (`7df04bd`)

- `scripts/allowance.py` + `router_ops.py allowance` + `tests/test_allowance_burn.py` (13 tests).
- Records dashboard readings (`--record --spend --requests --balance`) into an append-only store.
- **Live result:** projected **$48.40** vs dashboard actual **$48.26** (0.3% off) — the method reconciles.
- **Bug caught by running it, not by a unit test:** the first cut projected the period's
  ENTIRE token burn at the marginal rate and printed **$567.94 against a $48.26 reading
  (~12x cry-wolf)**. Cause: only **8.5%** of requests are credit-billed (2,960 of ~34,700),
  because in-allowance usage is free. Fixed by projecting only the *billable share* of
  volume; `test_projection_reconciles_with_the_reading_it_came_from` pins it.
- Honesty rules under test: no snapshot => no projection (never falls back to list price);
  projection labelled `projected_from_snapshot` and carries its reading.

### Batch 2 — the 150K flash ceiling — NOT APPLIED (premise falsified)

The directive was based on "144 empty-content events on >100K contexts". Measured:

| evidence | result |
|---|---|
| empty events above **150K** since the output-budget floor deployed (09-12 07:43) | **0** |
| compressions above 150K in that window | **174** |
| success of those 174 >150K flash compressions | **174/174 (100%)** |
| empty-event rate by context band (matched 71/72 events) | **highest at <20K (39.5/1k), falling to 3.8/1k at 150-200K** |
| >100K share of requests vs share of empties | 18.3% vs 9.9% — size is not the driver |

The failures were **output-budget induced** and were fixed on 09-12 by
`SESSION_COMPRESSION_MIN_MAX_TOKENS=16384`; the last empty event was 09-12 13:11.
Adding a 150K ceiling now would cost **9.3x** on that band (flash $8.70 -> glm-5.2
$81.23 per day, ~$508/wk at list) and buy **zero** reliability. Not applied.

Also note `MODEL_CONTEXT_CEILING["glm-5.3"]=150000` already exists and was justified by a
"repeated-token garbage above 150K" claim that **did not reproduce** at 200,027 tokens.

### Batch 3 — compression quality measurement — DONE

- `scripts/quality_probe.py` + `router_ops.py quality` + 20 tests (2 files).
- Sampled (**default 5%**, `BIGGIE_QUALITY_PROBE_RATE`, read at call time so it retunes
  without a restart), scored **post-response** so it cannot touch user latency (~1 ms per
  37K-char source).
- Wired into the endpoint at **both** streaming completion paths (primary + escalation) via
  an `on_summary` callback threaded through `proxy_to_backend_streaming` -> `_resume_stream`.
- Verified on 50 real captured compression samples: 50/50 scored, avg 0.899.
- Unmeasured stays NULL (`measurable=0`), never 0.0.

### Bugs found while wiring batch 3 (all fixed, all now pinned)

1. `DEFAULT_RATE` was read at import, so the env dial did nothing on a running endpoint.
2. My first exception-safety test passed `None/None`, which never raises — vacuous. Replaced
   with an input whose `__str__` throws; the mutation check then failed as it should.
3. The escalation streaming path bypasses the primary callback, leaving the highest-value
   sample (the escalated retry) unmeasured.

### Not live until restart

Batches 1 and 3 add code to the running endpoint. The service (PID 927) predates them, so
**no probing happens until it restarts**. The `allowance` and `quality` subcommands work now
(they are CLI, not service code).
