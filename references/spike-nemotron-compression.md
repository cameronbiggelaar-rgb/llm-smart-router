# Spike — nemotron-3-super as a compression summariser

**Status: COMPLETE.** Scope: the operator asked for *"anything we should consider
changing or testing against our current models or new models on ollama cloud"*,
and accepted the recommendation to restart the endpoint and shadow-test
nemotron-3-super.

## Method — a spike, not an experiment

Run OUTSIDE the router: read captured compression payloads from
`data/compression_samples/`, call ollama.com directly, score with the repo's own
`quality.score_summary`. No routing change, no registry change, no experiment
config, no production risk — so the question "is this model usable at all?"
gets answered before any machinery is built for it.

This ordering matters. The alternative (register the model, enable a shadow
experiment, wait for production traffic) spends real money and real time to
answer a question a spike answers in minutes — and, as B25.1 found, registering
a model to run an experiment can itself make it routable.

## Rates — verified against ollama.com, not recalled

Prices per 1M tokens (retrieved 2026-09-13 from ollama.com/pricing):

| model | input | cached input | output |
|---|---|---|---|
| nemotron-3-super | **$0.015** | $0.015 | $0.600 |
| gpt-oss:20b | $0.070 | $0.035 | $0.300 |
| nemotron-3-nano | $0.060 | - | $0.240 |
| gemma4 | $0.140 | $0.050 | $0.400 |
| deepseek-v4.1-flash | $0.150 | $0.003 | $0.600 |
| gpt-oss:120b | $0.150 | $0.014 | $0.600 |
| glm-5.3-flash | $0.150 | $0.030 | $0.500 |

nemotron-3-super is **10x cheaper on input than the cheapest rung of the
curated ladder**. Since compression is overwhelmingly input-bound (4,573M in vs
0.4M out over 14 days), input rate dominates the comparison.

**Peak pricing** (12:00–18:00 UTC Mon–Fri, DeepSeek rows only): v4.1-flash
doubles to $0.30 input / $1.20 output, v4-flash to $0.44/$1.32, v4-pro to
$1.32/$3.96. Not modelled in our price book.

## Result

Raw run: `/tmp/b25_spike.txt`. 5 real payloads x 3 models.

| model | n | $total | $avg/call | in_tok_avg | out_avg | p50 lat | cov_avg | ratio |
|---|---|---|---|---|---|---|---|---|
| nemotron-3-super | 3 | $0.01083 | **$0.00361** | 139,326 | 2,533 | 32.5s | 0.054 | 0.206 |
| gpt-oss:20b | 2 | $0.01504 | $0.00752 | 103,256 | 979 | 15.4s | 0.051 | 0.159 |
| incumbent v4.1-flash | 5 | $0.13544 | $0.02709 | 152,745 | 6,962 | 40.8s | 0.238 | 0.367 |

Paired on identical payloads:

- nemotron-3-super: cost **0.14x** the incumbent, quality 0.017 vs 0.091
- gpt-oss:20b: cost **0.34x**, quality 0.022 vs 0.083

### THE DECISIVE NEGATIVE: gpt-oss:20b cannot serve this workload

It **rejected every payload >=100K outright**, on all three attempts:

    HTTP 400 The prompt is too long: 156992, model maximum context length ...
    HTTP 400 ... 163768 ...
    HTTP 400 ... 173835 ...

Measured context distribution of real compressions (last 7d, n=37,005):

| band | calls | share | input tokens |
|---|---|---|---|
| <20K | 1,227 | 3% | 15.7M |
| 20–60K | 16,069 | 43% | 650.7M |
| 60–100K | 9,867 | 26% | 773.0M |
| 100–150K | 6,493 | 17% | 799.6M |
| 150–200K | 2,944 | 8% | 496.3M |
| >200K | 405 | 1% | 86.7M |

So **27% of compressions are >=100K and 9% are >150K.** A hard 400 is
disqualifying: it is not a tuning issue, and a cheaper model that rejects a
quarter of the traffic is not cheaper. This is the single most important
qualification, and it is **invisible in any per-token price table**.

### nemotron-3-super: context is fine, LATENCY is the risk

**Its context ceiling is not the problem.** `/tmp/b25_ceiling.py` probed it
directly and it accepted every size, up to **231,028 prompt tokens**, with no
400 and in 6-26s on a simple payload:

| target | status | elapsed | actual prompt_tokens |
|---|---|---|---|
| 60K | 200 | 6.0s | 53,228 |
| 130K | 200 | 12.5s | 115,428 |
| 150K | 200 | 8.0s | 133,228 |
| 200K | 200 | 11.1s | 177,628 |
| 260K | 200 | 26.1s | 231,028 |

So the two timeouts in the spike were **not** a context limit — they were
generation latency on real summaries (both at ~150K real payloads, where the
incumbent finished in 41-74s). An earlier draft of this file misread them as a
ceiling; that is the trap this correction exists to record. **A timeout is not a
rejection.**

The consequence is still a real qualification: **nemotron exceeded a 180s budget
on 2 of 5 real payloads (40%).** It needs a per-model timeout rule — and because
it is *faster* on simple payloads but slower on real summaries, the guard has to
be time-based, not size-based. Compare `glm-5.3`, whose guard IS size-based
(`MODEL_CONTEXT_CEILING` = 150K).

### The quality numbers are mostly a METRIC artifact — do not rank on them

`fact_coverage_v1` scored **every** model low (0.054 / 0.051 / 0.238 coverage).
The metric is *share of source numbers reproduced*, so a 50K-char source has a
huge denominator and any summary under ~15K chars scores near zero. It also
**systematically penalises brevity**: nemotron (ratio 0.206) and gpt-oss (0.159)
are ~2x terser than the incumbent (0.367) and score proportionally lower.

The one comparison that survives this is the incumbent-vs-incumbent spread:
coverage 0.204-0.295 across similar payloads. **`fact_coverage_v1` cannot rank
compression summaries of large contexts.** That is a finding about the
optimiser's only quality signal, and it is the same class of problem as the
0.0042 shadow artifact — a number that looks like a measurement but is an
artifact of the metric's construction.

## Recommendation

1. **nemotron-3-super is the only candidate worth pursuing.** Context is
   settled (231K accepted — not a constraint). Its cost case is real: **0.14x**
   on the payloads it serves. The one open blocker is **latency**: it blew a
   180s budget on 40% of real payloads, so it needs a time-based guard before
   promotion, not a size-based ceiling.
2. **Rule gpt-oss:20b out for compression.** A hard 400 on 27% of traffic is
   disqualifying, not a tuning detail.
3. **Do not promote on cost alone.** The cheapest model measured also failed the
   most payloads. Any promotion needs the ceiling rule first.
4. **Fix the quality metric before trusting any comparison.** Ranking candidates
   on `fact_coverage_v1` for large-context compression would choose the wrong
   model, because terser summaries score lower by construction.
5. **Model peak pricing** if load moves into 12:00–18:00 UTC (DeepSeek rows
   double), and prefer scheduling batch compression outside that window.

## What was NOT done

- No model registered, no experiment enabled, no traffic moved. This answers the
  feasibility question only.
- nemotron's context ceiling is now established: **231,028 tokens accepted**,
  so context is not a constraint. Its latency under real summarisation load is
  the unresolved question.
- gpt-oss:120b was not tested (same price as the incumbent on input, so no cost
  case).
- No timeout/latency guard was implemented — that is the prerequisite for a
  shadow experiment on nemotron, and it is a code change, not config.

