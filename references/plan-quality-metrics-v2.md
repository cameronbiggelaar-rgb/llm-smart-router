# Plan — fix the compression quality metrics

Status: proposed → executing (B26)
Scope: `scripts/quality.py`, `scripts/quality_probe.py`, `scripts/biggie_llm_endpoint.py`
(one call site), `scripts/optimiser.py` (row filtering), tests. No schema change
to `router_logs`. No history rewrite.

## Why: the metric is not measuring what it claims

Triggered by the B25 spike, where every model scored 0.06–0.14 on real payloads.
That looked like "all models are bad". It is not. Three defects, all measured on
real captures (`/tmp/metric_diag.py`, `/tmp/metric_diag2.py`).

### Defect 1 — the source text excludes 87% of the source (root cause)

Both extraction sites join **only `role == "user"`** messages:

* `quality_probe._source_text` (L104–111)
* `biggie_llm_endpoint.py:581` (shadow path, inline duplicate)

On a real capture (`20260809T045522Z-7650c`, 370 messages, 12 user / 180
assistant / 177 tool):

| scope | chars | facts (>=100) |
|---|---|---|
| user-only (what the scorer sees) | 19,426 | 49 |
| all messages (the true input) | 225,862 | 372 |
| **invisible to the scorer** | **91%** | **323 (87%)** |

Consequence is not dilution, it is **inversion**: a summary that faithfully
reports a fact which originated in an assistant or tool message is counted as a
*hallucinated number* and heavily penalised.

```
summary: 'the measured value was 100'  -> hallucinated=1, score=0.0
                                          (100 IS present in the real input)
```

The scorer rewards summaries that ignore the conversation and punishes the ones
that report it. Every number produced by B25 — and the live 0.0042 shadow score —
is an artifact of this.

### Defect 2 — coverage cannot discriminate at this scale

`coverage = |src ∩ smy| / |src|` with `|src| = 372` facts. A summary of a
225K-char context contains tens of numbers, so coverage tops out near 0.1 for
*every* model. The metric has no usable dynamic range: it compresses
"excellent", "adequate" and "poor" into the same band. This is why the numbers
cluster at 0.06–0.14.

`MIN_FACT = 100` compounds it: on real captures it discards 75–89% of raw
numbers, leaving a sparse set dominated by version strings and counts.

### Defect 3 — a verbatim copy scores 1.0

```
score(source, source)            = 1.0   <- reproduces facts perfectly
score(all facts, no prose)       = 1.0
score(faithful concise summary)  = <=0.24
```

There is no compression-pressure term. The metric *rewards echoing the source
and penalises summarising it* — the exact inverse of the compression objective.

### Consequence for the optimiser

`optimiser.rank_models` requires `quality_avg >= 0.80`. On real payloads the
metric's ceiling is ~0.24, so **no model can ever be eligible**. The optimiser is
blocked by a metric defect, not by missing data.

### Already-contaminated rows

| source | rows | avg |
|---|---|---|
| `router_logs.quality_score` (method `fact_coverage_v1`) | 70 | 0.0042 |
| `quality_probe` | 2 | 0.0 |

These are not measurements. They must never drive a decision.

## Fix: `fact_coverage_v2`

Keep `fact_coverage_v1` and its tests intact — v1 stays exported and its
semantics stay pinned. Add v2 alongside, selectable by `quality_method`, so old
and new rows remain distinguishable.

1. **Full-context source.** One shared helper, all roles, `content` either a
   string or a list of parts (the endpoint already handles both shapes
   elsewhere). Both call sites use it. No second implementation.
2. **Reference-based recall.** The `reference` parameter already exists and is
   documented as "accepted for API symmetry ... currently ignored". Make it
   real: when a reference compression of the same source is available, score
   recall against the *reference's* fact set instead of the whole context. That
   is the decision actually being made — "does this model preserve the facts the
   incumbent preserves?" — and it has usable dynamic range (reference ~tens of
   facts, not hundreds).
   Absolute mode (no reference) keeps recall over the full source so the scorer
   still works standalone.
3. **Two axes, named honestly.** `coverage` (recall) and `faithfulness`
   (precision = share of the summary's facts that are really in the source).
   `score = F1` harmonic mean, so neither axis can be gamed alone and brevity
   carries no automatic penalty.
4. **Unmeasurable is not zero.** A summary with no extractable facts (prose) is
   **not** scored 0.0 — it returns `measurable=False`, and callers store NULL.
   Scoring it 0.0 is precisely the false-positive class the user called out
   ("not measured" must never read as "measured bad").
5. **Quarantine legacy rows.** Reporting and the optimiser filter to the current
   method, so contaminated v1 rows cannot enter an average or a promotion.

## Batches (TDD, RED → GREEN → commit, one test file each)

| # | test file | property |
|---|---|---|
| B26.1 | `tests/test_quality_source_completeness.py` | source includes ALL roles; a true fact from an assistant/tool message is never counted as hallucinated |
| B26.2 | `tests/test_quality_v2_discrimination.py` | good > adequate > poor with usable separation; verbatim copy does not win; scale-invariance across source size |
| B26.3 | `tests/test_quality_v2_unmeasurable.py` | factless summary → unmeasurable/NULL, never 0.0 |
| B26.4 | `tests/test_quality_method_quarantine.py` | aggregates + optimiser count only current-method rows |
| B26.5 | (harness, not unit) re-run the B25 spike on real payloads under v2 | produce the first trustworthy nemotron-vs-incumbent comparison |

## Non-goals

* No change to `router_logs` schema, no backfill, no rewrite of the 70 v1 rows.
  They stay as honest evidence of what v1 did.
* No model registration, no experiment enabled, no traffic moved.
* No new dependency.
