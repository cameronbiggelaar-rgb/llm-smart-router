# Ollama Cloud unit economics — measured Sep 2026

## Model pricing (per 1M tokens, list) — re-verified against ollama.com/pricing 2026-09-12

| Model | Input/1M | Cached in | Output/1M | Context |
|---|---|---|---|---|
| deepseek-v4-flash | $0.22 | $0.007 | $0.66 | 1M |
| deepseek-v4-pro | $0.66 | $0.022 | $1.98 | 1M |
| **glm-5.3** | **$1.40** | $0.26 | **$4.40** | 1M |
| **glm-5.2** | **$1.40** | $0.26 | **$4.40** | 1M |
| **glm-5.3-flash** | **$0.15** | $0.03 | **$0.50** | 1M (vision/tools/thinking) |
| deepseek-v4.1-flash | not listed | — | — | 1M (library page exists) |

⚠️ **The router's `models.py` prices are stale and overstate cost by 1.5-2.5x.**
`models.py` carries v4-flash at $0.50 in (live $0.22) and glm-5.3 at $1.50 in
(live $1.40). Ratios derived from them (v4-flash 1.0, v4.1-flash 0.68, glm 3.0)
are therefore only usable for *ordering*, never for dollar estimates. v4.1-flash's
$0.34/$1.35 does not appear on the public pricing table at all — treat it as
vendor-quoted, unverified.

**glm-5.3 and glm-5.2 are priced IDENTICALLY** ($1.40/$4.40 input/output). Any
per-request cost difference between them is a *context-band* effect, not a price
effect — see the 5.3-vs-5.2 section.

**`glm-5.3-flash` ($0.15/$0.50) is not referenced anywhere in the router,**
config.yaml, or this skill. It is 9.3x cheaper on input than glm-5.3 and has a 1M
context — the obvious candidate rung for the >100K band that glm-5.3 currently
serves. Unvalidated for compression quality.

**Peak pricing** doubles only deepseek-v4-flash ($0.44 in) and v4-pro ($1.32 in),
12:00-18:00 UTC Mon-Fri. glm is unaffected, so daytime deepseek-heavy windows cost
~2x and glm-heavy ones do not.

v4.1-flash stays the compression workhorse on measured quality grounds (see the
compression A/B below), not on the stale ratio table.

Ground truth from the Ollama Cloud billing dashboard (`Usage credits`) combined with
`data/router_logs.db` request-level token data. **The router `estimated_cost_usd`
column is stale/placeholder — trust these figures, not that column.**

**How to reconcile dashboard $ with logged tokens:** the dashboard gives $ and request
count per model for the week; `router_logs.db` gives `input_tokens` per request. Divide
the two (`$ / (avg_input_tokens x dashboard_requests)`) to get an implied marginal
price per 1M input tokens. It lands well under list price because only usage beyond the
plan's included allowance is credit-billed. Use it to sanity-check any A/B cost claim —
A/B "credit units" come from the router's ratio table and can disagree with real
dollars by an order of magnitude, since units price tokens while dollars price *which
contexts* a model actually serves.

## What this week actually cost (Ollama, Sep 5-11 2026)

**Three dashboard snapshots of the same week** — the figure is week-to-date and grows as
the week runs, so the latest is the fullest picture:

| Snapshot (AEST) | Balance | Week-to-date $ | Credit-billed requests |
|---|---|---|---|
| 2026-09-11 20:02 | 16.28 | 23.73 | 1,749 |
| 2026-09-11 23:14 | 21.88 | 38.11 | 2,578 |
| 2026-09-12 06:10 | **11.74** | **48.26** | **2,960** |

Latest breakdown (2026-09-12 06:10):

| Model | Requests | $ | $/req | Share of spend |
|---|---|---|---|---|
| glm-5.3 | 431 | 24.91 | 0.05780 | 51.6 |
| glm-5.2 | 252 | 10.10 | 0.04008 | 20.9 |
| deepseek-v4-flash:0731 | 1,751 | 9.11 | 0.00520 | 18.9 |
| deepseek-v4-pro:0813 | 115 | 2.21 | 0.01922 | 4.6 |
| deepseek-v4.1-flash | 406 | 1.80 | 0.00443 | 3.7 |
| qwen3.5:397b | 5 | 0.13 | 0.02600 | 0.3 |
| **TOTAL** | **2,960** | **48.26** | 0.01630 | |

Credit-billed requests are the ones that exceeded the $100 plan's included usage, which
is why the dashboard's request counts run one to two orders of magnitude below
`router_logs.db` counts for the same models.

**Balance accounting across snapshots (the dashboard is internally consistent):**
20:02→23:14 the balance *rose* 16.28→21.88 while week-to-date usage rose 14.38 — implying
a **$20 top-up** in that window. 23:14→06:10 (6.93h) the balance fell 21.88→11.74
(−10.14) while week-to-date rose 38.11→48.26 (+10.15): the two agree to the cent, so no
top-up occurred and credits are consumed exactly as the usage counter advances.

**Burn rate in the last window: $1.46/h ≈ $35/day ≈ $246/wk** at the current mix — an
order of magnitude above the $6.65/day week-to-date average, so the mix got
glm-heavier, not the volume. $11.74 buys ~8h at that rate (zero ≈ 14:00 AEST
2026-09-12). Auto-reload was **Off** throughout.

glm-5.3 alone was **74%** of the last window's spend (+99 req, +$7.47) while
deepseek-v4.1-flash took +252 req for +$1.10 — a **13x** per-request gap.

- **The two glm models are the whole problem: 683 requests (23 percent of volume)
  burn $35.01 = 72.5 percent of all credit spend.** Every glm compression costs
  $0.040-0.058 against $0.0044 for v4.1-flash — **~12x more per request.** This is
  the single fact that justifies the trigger change: glm only earns its price above
  the 100K flash ceiling.
- **Weekly burn revised up twice: $23.73 → $38.11 → $48.26 at the latest reading** →
  monthly overage ≈ **$100+** on top of the $100 plan, and the trailing rate
  ($1.46/h ≈ $35/day) implies ~$246/wk if the current glm-heavy mix persists. Earlier
  figures were mid-week readings of the same growing counter.
- **v4.1-flash is live and cheap:** 154 requests for $0.70 — the cheapest per-request
  model in the fleet, cheaper than v4-flash ($0.00455 vs $0.00507).
- Note the per-request figures blend model price with *which contexts each model
  serves*, so they are not a pure model-efficiency measure — the A/B below is.
- **Why glm costs 11.6x more per request — it is TWO effects multiplied, not one.**
  glm-5.3's avg input is 127K tokens vs v4.1-flash's 52K (2.4x bigger contexts, because
  it serves everything above the 100K flash ceiling) **and** its price is ~4x higher.
  2.4 x 4 ≈ 11x. So moving a compression off glm onto flash saves on both axes at once;
  that is why the trigger change is worth more than a pure price swap would suggest.
- **Implied real price per 1M input tokens (credit-billed overage / logged tokens):**
  v4.1-flash $0.09, v4-flash $0.10, glm-5.2 $0.24, glm-5.3 $0.41, v4-pro $0.48. All sit
  well below list price because only the overage beyond the plan's included allowance is
  billed — treat these as *marginal* prices, not list.
- **Whole week (all models, all plans):** 76,003 logged rows = **38,089 distinct requests**
  (2x row inflation — see pitfalls), **2.83B input tokens**, avg 74,420 input tokens per
  request. Output is negligible (~0.8M/wk). Older notes quoting 72,005 requests /
  5.4B input tokens were the un-deduplicated counts.

## Ceiling head-to-head — RE-RUN 2026-09-12 (the Aug 22 artifact was worthless)

**The original `flash_ceiling_headtohead.py` proves nothing.** Two independent defects:

1. **The payload never reached the model.** The router applies request-level
   compression *before* routing, so a "105K-token" payload was collapsed to ~14K.
   Every Aug-22 row logged `prompt_tokens: 14232` at *every* target size — the
   give-away that context size was not being varied at all. A 420,953-char payload
   still logged `prompt_tokens: 14232`.
2. **No control leg.** It only tested 105-145K, so it could not distinguish "flash
   fails on large contexts" from "flash fails everywhere".

**Fix**: send `X-Compression-Level: off` (documented request header — no config or
service change) and measure real `usage.prompt_tokens`, calibrating chars/token
empirically (numeric filler is ~2.3-2.8 chars/token, not the 4.0 assumed).

### Result: the empty-stream bug is OUTPUT-BUDGET-induced, not context-size-induced

2x2 with forced v4.1-flash, `X-Compression-Level: off`:

| context | max_tokens | served itself | what happened |
|---|---|---|---|
| 16K | 256 | **no** | `finish=length`, reasoning only, 0 content |
| 16K | 8192 | **no** | `finish=length`, reasoning 16,385 chars, 0 content |
| 173K | 256 | **no** | `finish=length`, reasoning only, 0 content |
| 141K | 8192 | **yes 2/2** | `finish=stop`, 799-1113 content chars |

The failure follows the **budget**, not the context: it fires at a 16K context. At
141K prompt_tokens with a realistic budget, flash served itself 3/3 (plus 2/2 in the
band test at 95K/110K/130K). **A size-based ceiling therefore does not fix the
failure mode it was written for.**

### Result: glm-5.3's "repeated-token garbage above 150K" claim did NOT reproduce

At **200,027 prompt_tokens** glm-5.3 produced coherent, on-topic output
(871 chars, unique-word ratio 0.752, max word-run 1, no repeated n-grams).
Runs that did degrade were `finish=length` with 19-25K chars of reasoning consuming
the output budget — the same budget problem, not a context-size degeneration. The
`MODEL_CONTEXT_CEILING glm-5.3 = 150000` escalation to glm-5.2 (which costs the same
per token but serves bigger contexts) is therefore **not justified by the stated
reason**.

### Production evidence agrees

Empty-stream events parsed from the router journal (authoritative - see the DB traps
below) and matched to logged context sizes, Sep 5-12 production, de-duplicated to
distinct requests:

| model | band | events | requests | rate |
|---|---|---|---|---|
| deepseek-v4-flash | <20K | 12 | 1,099 | 1.09 pct |
| deepseek-v4-flash | 20-50K | 44 | 12,702 | 0.35 pct |
| deepseek-v4-flash | 50-100K | 59 | 11,986 | 0.49 pct |
| deepseek-v4-flash | >=100K | 0 | 3 | 0 pct |
| glm-5.3 | >=100K | 46 | 5,876 | 0.78 pct |

**v4-flash's empty-stream rate does not rise with context size.** It is *highest* below
20K (1.09 pct) and ~0.35-0.49 pct from 20K-100K: flat-to-declining. That falsifies the
ceiling's stated premise ("flash degenerates on very large contexts"). Only 3 production
v4-flash requests ever exceeded 100K - the ceiling diverts them - so there is no
production evidence at all for large-context flash failure.

Note the *absolute* failure counts are small (115 events over 25,790 requests, 0.45 pct)
and every one of them recovered by escalation, so this is a throughput/latency cost, not
a correctness cost.

**Two traps when counting empty streams from the DB:**

1. `router_logs.empty_stream` is a **dead column** - `SUM(empty_stream) = 0` over 75,953
   rows. Never query it.
2. `error_type='empty_content'` is **masked by escalation**. The DB writes the final
   outcome on the success row (`success=1`, `error_type=''`, `saw_content=1`) after the
   retry recovers, so a model that empty-streams constantly can look clean. In one 2h
   window the journal showed 59 empty streams for deepseek-v4.1-flash, 19 for glm-5.3
   and 11 for deepseek-v4-flash, while `error_type='empty_content'` recorded **zero**
   for all three. `error_type` only surfaces `empty_content` when the escalation ladder
   also failed (e.g. gpt-5.6-sol, 18 distinct).

**The journal is the authoritative source for empty-stream counts**:
`sudo journalctl -u biggie-llm-endpoint --since <t> | grep "empty content for"`.

**A compression request writes TWO rows** (a `streaming_in_progress` start row plus the
result row) - always `COUNT(DISTINCT request_id)`, never `COUNT(*)`.

### What the ceiling is actually worth (week, de-duplicated)

Above-100K compressions are **25.1 pct of requests but 48.5 pct of all input tokens**
(9,573 reqs / 1,365,616,270 tokens). At list prices that band costs $1,912/wk on
glm-5.3 vs $205/wk on v4.1-flash — a **9.3x** unit-cost ratio.

**Do not quote the dollar delta as a saving.** Dashboard spend is billing ground truth
and it cannot be predicted from logged tokens (see pitfalls: logged tokens predict
$2.38 for a window that actually cost $10.15). The defensible claim is the token share
and the list-price ratio, not a currency figure.

### Cost of running these tests

27 forced v4.1-flash requests (2,438,282 input tokens) + 1 forced glm-5.3 request at
200,027 tokens = **2,638,309 input tokens**, i.e. ~$0.40 at v4.1-flash list and ~$3.70 at
glm-5.3 list before plan credits. A ceiling re-run is cheap; do not skip it to save money.

### Recommendation (not yet applied - requires an explicit decision)

The evidence does **not** support keeping a size-based flash ceiling, and does not
support the glm-5.3 150K ceiling either. But both changes move production routing, so
they are the user's call:

1. **Raise/remove `BIGGIE_FLASH_MAX_CONTEXT_TOKENS`** - the failure mode it guards is
   budget-induced, not size-induced, and production rates are flat across bands. This
   sends the >100K compression band (48.5 pct of all input tokens) to a rung 9.3x
   cheaper per token.
2. **Re-target the guard at the real cause**: enforce/raise a minimum output budget for
   compression requests instead of gating by context size.
3. **Fix the harness's premise-check habit**: any behavioural gate test should assert
   that the variable under test actually varied (here: `prompt_tokens` scaling).

**Harness corrected** → `flash_ceiling_headtohead_v2.py` (compression off + measured
tokens + calibration). Supporting scripts: `ceiling_control_test.py`,
`ceiling_band_test.py`, `ceiling_crux_test.py`, `ceiling_final_test.py`. Results:
`ceiling_band_results.json`, `ceiling_crux_results.json`, `ceiling_final_results.json`.

## Deployment status — what is live (2026-09-12 06:10 AEST)

**Source tree is NOT committed**: `llm-smart-router` has 6 modified files (`SKILL.md`,
`feature_extractor.py`, `models.py`, `probe_rethink_lane.py`, `router.py`,
`routing_table.yaml`) plus untracked `data/` and the new `ab_*.py` scripts. HEAD is
`84e26d9` (2026-09-05). Nothing is pushed. This is a working tree, not a release.

**What IS running**: `biggie-llm-endpoint.service`, MainPID started **2026-09-11 21:57:10
AEST** (`/usr/bin/python3 .../scripts/biggie_llm_endpoint.py`, `Restart=on-failure`).
`router.py` mtime is 21:56:50 — **50 seconds before the restart**, so the running process
loaded the edited router. Confirmed live by behaviour, not by inspection:

| Change | Live? | Evidence |
|---|---|---|
| v4.1-flash added to compression ladder | **yes** | 592 v4.1-flash compression reqs post-restart vs 11 in the whole preceding week |
| Flash-family ceiling (`startswith("deepseek-v4") and endswith("-flash")`) | **yes** | 4 v4.1-flash reqs ran at 123-126K *before* 11:56; **zero** above 100K after |
| `exact_keywords` / `deep review` in routing_table | **untested** | no tier-13 route since restart. Fires only on "perform a deep review" / "do a deep review" / "deep review of the" — phrase-level, so incidental mentions don't trigger it |

The pre-restart 123-126K v4.1-flash requests are the smoking gun that the old code let
flash past the ceiling; they were replaced by glm-5.3 afterwards.

## Are we seeing the benefit of glm-5.3 over glm-5.2? (measured 2026-09-12)

**glm-5.3 and glm-5.2 are priced identically** ($1.40/$4.40 per 1M, live table). So the
question reduces to: does 5.3 do anything better per token?

**No measured quality difference exists.** Week to date, both models: 0 user corrections,
0 retries, 1 escalation each, 0 empty streams. Success 3617/3632 (99.6%) for 5.2 vs
5783/5837 (99.1%) for 5.3 — statistically indistinguishable, and 5.3 is nominally *worse*.
The compression A/B (below) puts glm-5.3 at **13 pct hallucinated numbers vs 3-4 pct**
for either flash model and 2.12x the credit units. No A/B artifact in `data/` compares
5.3 against 5.2 directly — the quality case for 5.3 is unmeasured.

**The cost gap is a routing artifact, not a model property:**
- glm-5.3 avg input **122,863** tokens; glm-5.2 avg input **169,170** tokens (1.38x bigger).
- At identical list prices, 5.2 should therefore cost *more* per request — and it does
  not in the dashboard ($0.0578 vs $0.0401), because 5.2 is served far less and sits
  deeper into the credit-billed tail.
- Band placement (week, distinct reqs): 5.3 owns **100-150K (5,809 reqs)**; 5.2 owns
  **150-200K (2,918)** and **>200K (486)**. **>150K is not a cheap lane** — glm-5.2
  escalated once with `empty_content` in the >150K band.

**Conclusion:** 5.3 is not buying measurable quality over 5.2, and both are the most
expensive way to summarise in the fleet. The tier-6 vs tier-6.5 ordering is a routing
preference, not evidence. The real lever is getting the 100-150K band off glm entirely
(see `glm-5.3-flash` below), not choosing between the two glms.

## The optimisation the data actually supports

**Band token concentration (week, distinct compression reqs):**

| Band | Reqs | Share of reqs | Tokens | Share of tokens |
|---|---|---|---|---|
| <40K | 10,171 | 27.7% | 305M | 11.0% |
| 40-60K | 7,781 | 21.2% | 379M | 13.7% |
| 60-80K | 5,252 | 14.3% | 365M | 13.2% |
| 80-100K | 3,985 | 10.9% | 358M | 12.9% |
| **100-150K** | **5,990** | **16.3%** | **740M** | **26.7%** |
| >150K | 3,529 | 9.6% | 620M | 22.4% |

**46 percent of all compression requests are under the 60K `threshold_tokens`** (median
61,444), and 74 percent are under 100K. The 100-150K band is 16 percent of requests but
**27 percent of tokens and the entire glm-5.3 bill.**

- **`glm-5.3-flash` ($0.15/$0.50) is the untried lever**: 9.3x cheaper input than
  glm-5.3, 1M context. Moving the 100-150K band alone would cut ~$1,036/wk of list-price
  input to ~$111. **Not referenced anywhere in the router or config.** Needs a quality
  A/B before it touches the ladder.
- **Ceiling changes are NOT validated yet.** `scripts/flash_ceiling_results.json` (Aug 22)
  is useless as evidence: every leg recorded `esc_away=False`, i.e. the harness's forced
  model was served anyway and nothing actually tested the ceiling. Re-run
  `flash_ceiling_headtohead.py` before moving the 100K line.
- **Raising `threshold_tokens` (60K) is the cheapest win** — 46 percent of compressions
  fire below it, each re-sending a full context for a summary. Compressing less often on
  larger contexts trades summary freshness for real token savings; needs a product call.

## The core economics fact

**Cost is driven by input-token re-sending (context), not output.** Avg input token
count per request by model (7-day): flash 52k, glm-5.3 123k, glm-5.2 169k, gpt-5.5
81k, gpt-5.6-sol 36k. Each router request re-injects a large context, so cheap tiers
burn enormous token counts for pennies — that's why Ollama's credit model survives.

## OpenAI subscription caps (official Codex pricing page, Sep 2026)

Caps are **messages per rolling 5-hour window**, not tokens. Plan multipliers
(Pro 5x = $100, Pro 20x = $200):

| Model | Plus | Pro 5x | Pro 20x |
|---|---|---|---|
| GPT-6 Astra | 5-45 | 25-225 | 100-900 |
| GPT-5.6 Sol | 10-100 | 50-500 | 200-2,000 |
| GPT-5.6 Terra | 25-200 | 125-1,000 | 500-4,000 |
| GPT-5.6 Luna | 250-2,000 | 1,250-10,000 | 5,000-40,000 |
| GPT-5.5 | 15-80 | 75-400 | 300-1,600 |
| GPT-5.4 | 20-100 | 100-500 | 400-2,000 |

Weekly limits may also apply. Local + cloud chats share the allowance. Large repos,
long prompts, and tool use drain the real per-5h count faster than the estimate.

## Does full-GPT tiering fit? (drive ALL load to GPT)

Tier mapping = same ladder shape: flash→Luna, glm-5.x→Terra, deepseek-pro→Sol,
current gpt-5.5→GPT-5.5. Measured weekly volumes (all→/7 = per-day):
- flash → Luna: need ~7,218/day. Pro 5x Luna cap 1,250-10,000/5h = 3,750-30k/day →
  **exceeds low estimate; blows through a quiet day.**
- glm-5.3+5.2 → Terra: need ~2,621/day. Pro 5x Terra 125-1,000/5h = 375-3,000/day →
  **over the low end by 7x.**
- deepseek-pro → Sol + gpt-5.5 → 5.5: ~86 + 188/day — **fits Pro 5x comfortably.**

**Conclusion: Pro 5x ($100) does NOT hold full-all-load-to-GPT under tiering** — the
flash/glm tiers are 2-7x beyond its Luna/Terra low estimates. Pro 20x ($200) fits
with headroom but is economically absurd: it costs $200/mo to carry context-heavy
(~50-170k token/req) sub-agent turns that Ollama does for ~$9/wk in credits.

**What DOES fit:** the true top-tier lane (gpt-5.5 + 5.6-sol = ~2,134 req/wk, peak
~288/hr). Plus already dies under that peak (Plus 5.5 = 15-80/5h). Pro 5x gives 5x
headroom on the tier-10+ lane. So: **upgrade to Pro 5x only for top-tier routing;
keep flash/glm/pro on Ollama credits.**

## Compression quality A/B — v4.1-flash vs v4-flash vs glm-5.3 (2026-09-11)

Measured with the **real production summariser prompt** (`agent.context_compressor
.ContextCompressor._build_summary_prompt`), identical input per session, at
production settings: turn block bounded to `_SUMMARY_INPUT_MAX_CHARS=160_000`,
**no `max_tokens` cap** (context_compressor.py:645 — "NEVER add a max_tokens wire
cap on the summary call"). Harness: `scripts/ab_direct.py`.

| Model | Mean fact-coverage | Mean hallucinated numbers | Mean units/session | vs v4.1 |
|---|---|---|---|---|
| **deepseek-v4.1-flash** | 26 | 4 | **30,103** | 1.00x |
| deepseek-v4-flash | 32 | 3 | 44,243 | 1.47x |
| glm-5.3 | 20 | 13 | 63,886 | 2.12x |

(Coverage and hallucination columns are percentages; units are Ollama credit units.)

Across 4 real sessions (133K-160K char inputs):

- **All three emit 9/10 template sections** (the 10th, `## Active Task`, is often
  folded into `## Historical Task Snapshot`). No prose dumps, no empty outputs.
- **v4.1-flash is the cheapest by 1.5-2.1x** and produces the *most* numbered
  completed-actions (16-61) of the three — denser, better-structured checkpoints.
- **glm-5.3 hallucinates 3-4x more numbers** than either flash model (13 vs 3-4
  percent) and is the only model that intermittently degenerates into reasoning
  prose instead of the template (observed at 310K-char input).
- v4-flash occasionally writes much longer summaries (37K output tokens / 121K
  chars on one session) — more exhaustive but 1.5x the cost for no coverage gain.

**Conclusion: v4.1-flash is a strict upgrade as the compression workhorse** —
cheaper, no quality regression, better structure, fewer hallucinations. It is the
correct first rung of the ladder. Keep the 100K `FLASH_MAX_CONTEXT_TOKENS` ceiling:
beyond it v4.1 degrades the same way (template abandoned, budget spent on reasoning).

## glm-5.3-flash vs deepseek flash for compression (2026-09-12)

Question asked: is `glm-5.3-flash` a better/cheaper alternative to the deepseek
flash models for session compression? Measured with the same harness and the same
real sessions as the A/B above (`scripts/ab_glm53_flash_compress.py`, direct
provider, production prompt, **no `max_tokens` cap**). Result:
`data/ab_compress_glm53flash.json`.

| Model | Mean $/call | Out tok | Reason tok | Latency | Coverage | Halluc. |
|---|---|---|---|---|---|---|
| deepseek-v4.1-flash | **$0.01217** | 9,893 | 3,236 | **29.2s** | 19 | 19 |
| deepseek-v4-flash | $0.01496 | 8,833 | 2,078 | 33.6s | 22 | 23 |
| glm-5.3-flash | $0.01795 | 23,746 | 16,198 | 197.6s | **23** | **6** |

(Coverage and hallucination are percentages; hallucination is *bad*, lower is better.)

**glm-5.3-flash is 1.48x MORE expensive per call than v4.1-flash, and 6.8x
slower.** Its headline output price is *lower* ($0.50 vs $0.60/1M) and its input
price is *identical* ($0.15/1M, ratio 1.0) — but it emits 2.4x the output tokens,
and **68 percent of those billed output tokens are reasoning** (16,198 thinking
tokens vs 7,548 answer tokens; v4.1-flash spends only 32 percent on reasoning).
A cheaper per-token price loses to a 2.4x-verbose, reasoning-heavy generation
profile. Never rank these models by the price sheet alone.

It *is* genuinely higher quality: +4.1 points fact-coverage and **12.9 points
fewer hallucinated numbers** (6 percent vs 19 percent) — the best of the three on
both axes. That is a real summariser-quality win, paid for with cost and latency.

- **Latency vs the timeout budget is the blocker.** Mean 197.6s against
  production's `compression.timeout: 120` (and
  `conversation_compression.DEFAULT_CONTEXT_TIMEOUT_SECONDS = 120.0`); the router's
  own upstream call allows 300s (`router.py:1146`). In the scale test
  glm-5.3-flash **hard-timed-out at 240s on a 28K-token prompt** where
  v4.1-flash took 24-29s, then succeeded at 168-207s on the retry — high variance.
  The idle deadline is progress-aware, so confirm against a *streaming*
  reproduction before ruling it in or out.
- **The saving is NOT on the flash rung.** Because input is priced identically,
  swapping glm-5.3-flash onto rung 1 is a pure loss (measured above). The money is
  on the **glm-5.3 + glm-5.2 rungs, which are $1.50/1M input and carry 87 percent
  of compression spend** ($1,458 + $930 = $2,388 of $2,732/wk). Repricing their
  real 7-day volume (1,591.6M input tokens) gives ~$4,094/wk -> ~$705/wk, i.e.
  **~$2,149/wk (~$9.3k/yr) — the entire cost case.** Their inputs average
  128K/170K tokens, well past flash's 52K/44K.
- **glm-5.3-flash does not degenerate in that large band, where glm-5.3 does.**
  At 155,702 prompt tokens glm-5.3-flash returned 36,285 chars, while **glm-5.3
  returned 0 chars** — consistent with the 150K `MODEL_CONTEXT_CEILING` that bars
  glm-5.3 from the band (`router.py:68`). Quality at that size is **NOT verified**:
  the coverage/hallucination scorer returned no valid numbers on the 600K-char
  payload, so do not claim quality holds above 150K without a re-score.
- **`FLASH_MAX_CONTEXT_TOKENS` is currently 0 (ceiling OFF)** in the live endpoint
  (`BIGGIE_FLASH_MAX_CONTEXT_TOKENS=0`), so flash is never skipped for size — yet
  glm-5.3/5.2 still absorb the large contexts. The trigger (availability vs
  escalation) was not pinned down; do not assume the ceiling is doing the routing.

**Verdict: not a deepseek-flash replacement — it is 1.48x more expensive and 6.8x
slower on that rung for a quality gain.** It is a candidate for displacing the
glm-5.3/glm-5.2 large-context rungs (~$9.3k/yr), gated on resolving latency
against the timeout budget and on scoring quality above 150K tokens.

### Measurement pitfalls (cost hours if forgotten)

- **`router_logs.db` double-logs every compression row.** Each request writes two rows
  (one `success=0, error_type=streaming_in_progress`, one `success=1`) under the same
  `request_id` — a measured **2.00x** inflation over the last 24h (17,059 rows /
  8,540 distinct `request_id`). Always `COUNT(DISTINCT request_id)` or divide by two
  before quoting request counts or token sums, or every volume figure is 2x high.
- **The dashboard's $/request cannot be back-derived from `input_tokens`.** In the
  6.93h window 23:14→06:10 AEST, de-duplicated tokens × the marginal prices above
  predict $2.38 while the dashboard moved $10.15 for the same window — a 4x gap. The
  implied-marginal-price method in the section above is therefore good only for
  ordering models by real cost, not for predicting a window's spend. Treat the
  dashboard $ as ground truth and `router_logs.db` as a routing-mix proxy.
- **Never cap `max_tokens` when reproducing production.** With a 10K cap, v4.1-flash
  spends ~31K chars of *reasoning* that count against the budget and returns only
  3/10 sections, looking broken. Uncapped (production), it completes 9/10 and wins.
  A capped harness will produce a false "v4.1 is worse" verdict.
- **Reasoning tokens count toward `max_tokens`** and land in the `reasoning` field,
  not `content` — always read both when scoring.
- **You cannot A/B models through the router.** For `task_type=session_compression`
  the router always overrides the client's model with the ladder rung, so every leg
  returns the same model. A/B must call the provider (`https://ollama.com/v1`)
  directly. `force_model` exists in router.py but is not reachable from the HTTP API.
- **Truncation to the flash ceiling is unfair to glm.** Comparing a 160K-char input
  against a 310K-char input compares payloads, not models — truncate every leg
  identically (`--truncate_all`).

## Operational guidance

- **Balance: $11.74 at 2026-09-12 06:10 AEST, auto-reload OFF.** At the trailing
  $1.46/h this implies ~8h of runway — dry around **14:10 AEST**. Nothing refills it.
  The balance is the purchased-credit balance; the dashboard's "usage credits used this
  week" ($48.26) moves it 1:1 (measured to the cent across 23:14→06:10).
- **Confirm the plan tier before sizing overage.** The live pricing page shows **Pro at
  $20/mo with $60 of included credits** and **Max at $100/mo with $300** (plus Max-only
  10 concurrent requests and early model access). Earlier notes calling this a "$100
  plan" may have conflated the Max price with the Pro tier. If the account were on Max,
  $48.26/wk ≈ $207/mo would fit inside $300 of included credits and the purchased
  balance should never drain — but it *is* draining, so included credits are exhausted
  (or the tier is Pro). Get this from the account page, not from inference.
- **Current weekly burn: $48.26/wk credits** on top of the flat plan fee. Earlier
  readings of $23.73 → $38.11 were mid-week values of the same growing counter — do not
  treat them as separate measurements. 72 percent of the overage is glm-5.3+glm-5.2.
- **The 100K `FLASH_MAX_CONTEXT_TOKENS` ceiling is what puts the bill on glm.** Every
  compression above 100K skips the flash rungs and lands on glm-5.3 ($1.40/1M, 9.3x the
  input price of glm-5.3-flash). The 100-150K band alone is 27 percent of compression
  tokens and 16 percent of requests.
- **Do not change the ceiling until the harness is re-run.** `flash_ceiling_results.json`
  is an Aug 22 artifact that recorded `esc_away=False` on every leg — the forced model
  was served anyway, so it never tested the ceiling at all. It cannot justify a change.
- Set up `POST /admin/subscription {"tier":"pro"}` (router) only as a runtime
  on/off toggle for peak days, not as a permanent full-GPT reroute.
### Live change applied after the ceiling re-run

After the corrected head-to-head showed that large-context failures tracked output budget rather than context size, production was changed to:

- disable the flash-family session-compression context ceiling with `BIGGIE_FLASH_MAX_CONTEXT_TOKENS=0`; and
- enforce `BIGGIE_SESSION_COMPRESSION_MIN_MAX_TOKENS=16384` for native `session_compression` requests.

Verification: `biggie-llm-endpoint.service` is active; `/health` is OK; systemd environment shows both values; an in-process 240K-token session-compression route selects `deepseek-v4.1-flash`; and `apply_session_compression_output_floor()` raises a 256-token compression budget to 16,384 while preserving larger budgets and non-compression requests.

`glm-5.3-flash` remains an opportunity rather than an active route: Ollama lists it at $0.15 input / $0.50 output per 1M tokens with a 1M context and claims it beats GLM-5.2 on coding/agentic benchmarks, but the local Hermes config currently exposes only `glm-5.3:cloud` and `glm-5.2:cloud`, not `glm-5.3-flash:cloud`.

## Shadow experiment result: glm-5.3-flash fails on real compression payloads

The offline A/B (above) measured glm-5.3-flash on the assembled production
prompt and found it 1.48x costlier and 6.8x slower. Running it as a **shadow
experiment against live traffic** produced a categorical result the offline
harness could not: on the shape the endpoint actually sends, the candidate
returns **no summary at all** when tools are offered.

Measured 2026-09-12 over 188 candidate calls / 602 experiment rows ($2.86):

- assembled production prompt (201,279 chars), tools offered:
  `deepseek-v4.1-flash` finish=stop, **35,038 chars**;
  `glm-5.3-flash` finish=length, **0 chars**
- assembled production prompt, no tools: incumbent 32,440 chars;
  candidate 25,127 chars
- live shadow calls: 174 of 181 `finish_reason=tool_calls`; only 67 of 181
  produced a scoreable quality figure

The first line is the decisive one: given production's real compression request
with the request's tools present, the candidate consumes its entire 16,384-token
budget and emits zero visible content, while the incumbent returns 35,038 chars
on the same payload. `finish=length` with 0 chars means the budget is spent
before any answer appears, so the call cannot serve compression at all.

Caveats that keep this honest:

- **Do not use the 253-message sample as a discriminator.** On that raw
  multi-message shape BOTH models returned `tool_calls` with empty content, so it
  separates nothing. Only the assembled single-user-message prompt — what
  `context_compressor.py:3186` actually sends — discriminates.
- 5 of the live shadow calls did finish with `stop`, so the candidate is not
  universally broken; it is specifically unable to serve large compression
  requests that carry tools.
- This is an observational shadow result on one payload class, not a full
  benchmark. It is decisive for compression, the only use case tested.

### Operational findings that only appear when it runs for real

1. **`model_pricing` was never seeded by the endpoint.** Every row logged
   `cost_unknown=1` forever, so a live experiment produced real spend with no
   cost evidence. `_get_db_connection()` now seeds prices at first use.
2. **A production row must never be stamped `is_shadow=1`.** The first live
   enablement put 34 genuine incumbent calls into the shadow bucket. Repaired by
   clearing the flag — they are real production calls, and deleting them would
   understate production permanently. 17 candidate rows with unrecoverable
   semantics were quarantined rather than fabricated.
3. **`tool_calls` is a response, not a failure.** Recording it as `success=0`
   misreported the candidate as broken. `finish_reason`/`saw_tool_calls` are now
   recorded so an unscoreable answer is explicable instead of silent.
4. **Shadow spend must be reported separately.** `unit_cost` and
   `optimiser.rank_models` exclude shadow rows by default, so production cost is
   not overstated and an un-promoted candidate is not ranked as if serving.

Cost of the experiment: $2.86 for 188 candidate calls ($0.0155/call). At the
observed compression rate a shadow run mirrors ~$180/day, so an experiment whose
question is answered should be switched off rather than left enabled.

## Reconciling the ledger against the vendor invoice (2026-09-12)

The Ollama dashboard and the router ledger disagreed by **51x** on the same
traffic. Working the difference down produced four measured defects, all of
which had been invisible because every test injected its own database and none
compared the total against an external source of truth.

### The reconciliation that found everything

    vendor (week to date)        $76.86
    our ledger (same window)   $3,525.50
    ratio                          51.5x

Two independent errors, both directionally the same (overstatement):

| cause | effect | fix |
|---|---|---|
| every streaming request logged twice, both rows carrying the full token count | **1.86x** on 97.99% of rows (all streaming traffic) | `rollup.BILLABLE_ROW_SQL`; markers explicitly unpriced |
| price book vs `ollama.com/pricing` | glm-5.3/5.2 7% high, deepseek-v4-pro 3.03x high, qwen3.5:397b absent | pinned by test; correction is the operator's call |
| no cached-input rate column | inputs billed at fresh rate; vendor's effective glm-5.3 rate was ~$0.41/1M vs $1.40 fresh — consistent with ~88% cache hits | structural; needs a third price column |
| `input_tokens` is a `chars//4` estimate of the **pre-compression** messages | bills the size before compression, not what was sent | not yet addressed |

### Why "streaming is logged twice" is worse than it looks

The start marker (`error_type='streaming_in_progress'`) is deliberately written
so an abandoned stream stays visible — `_find_abandoned_streams` depends on it.
The completion row repeats the same token counts. So `SUM()` over the table
counts every streaming request twice, and streaming is 97.99% of rows.

The dangerous part is the interaction with B12. B12 made the logger compute a
cost whenever a caller omits one. The start marker omits cost fields. So the
B12 fix, on restart, would have made the double-count **live** rather than a
historical artifact of a backfill. That is why the fix has two independent
guards — the marker is explicitly non-billable, *and* aggregation counts only
billable rows. Either alone is one edit away from regressing.

### Reconciliation is the test that was missing

None of these were logic errors in the sense unit tests catch. Every function
did exactly what it said. What was absent was a check that the *total* agrees
with the vendor's invoice. Any future change to pricing or logging should be
validated by re-reconciling against the dashboard, not only by the suite.

### Residual, deliberately unfixed

* **Cached-input pricing** — needs a `cached_input_usd_per_1m` column and a
  cache-hit signal from upstream. Largest remaining overstatement on
  cache-heavy workloads.
* **Pre-compression token estimates** — `context_tokens` is computed before
  compression, so the ledger overstates what was sent.
* **Streaming output tokens** — completions log `output_tokens=0`
  (see `test_streaming_cost_gap.py`); understates, opposite direction, small.
