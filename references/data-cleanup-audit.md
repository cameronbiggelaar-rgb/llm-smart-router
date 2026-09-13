# Data-cleanup and false-positive audit (router ledger)

Audit performed 2026-09-13 against `data/router_logs.db` (read-only). Purpose:
find state that makes a report say something **wrong**, in either direction —
an alarm that is not real (false positive) or an all-clear that is not real
(false negative).

The governing rule: **never run a rollup, backfill, or purge against the live
`data/router_logs.db`.** Copy it (`shutil.copy2` to `/tmp/…`) and operate on the
copy. A read-only URI open is fine for inspection:
`sqlite3.connect("file:data/router_logs.db?mode=ro", uri=True)`.

## The canonical billable definition

There is exactly one, in `rollup.py`:

```python
BILLABLE_ROW_SQL = "COALESCE(error_type, '') != 'streaming_in_progress'"
```

A streaming request writes a **start marker** row plus a **completion** row.
The marker is not billable; the completion is. That exclusion — and nothing
else — is what makes a row billable.

**Do not add filters.** A `cost_unknown = 0` filter looks harmless and is not:
the cost backfill repairs previously-unpriced rows by reconstructing a price
from the model's own rate card, so `cost_unknown = 1` rows are billable work.
Adding it dropped 5.2% of billable tokens and biased avg-input by +1.41%,
which corrupted the marginal rate and drifted the projection off the dashboard.

Any query feeding a reported figure must use `BILLABLE_ROW_SQL` verbatim.
`tests/test_allowance_burn.py` pins the CLI query to it so the two cannot
drift apart again.

## Findings

### 1. Stale `daily_findings` — the main cleanup ✅

The rollup cached the **double-counted** totals (marker rows priced alongside
completion rows), then the double-count was fixed in code without re-rolling
the already-cached days.

| day | cached (stale) | recomputed | ratio |
|---|---|---|---|
| 2026-09-10 | $415.63 | $229.08 | 1.81× |
| 2026-09-11 | $1,395.25 | $741.97 | 1.88× |
| 2026-09-12 | $33.21 | $18.62 | 1.78× |

The cached value matched the old broken total **exactly**, confirming the cause
rather than guessing it.

**Fix:** re-run the rollup for the affected days. `rollup_day()` is
delete-then-insert per day (idempotent), so a re-run is safe and picks up
late-arriving rows. Verified on a copy: every day then reconciles with its raw
rows at ratio **1.0000**.

**Self-healing — partly.** `biggie-router-ops.timer` runs
`router_ops.py maintain --days 3` daily at 18:10, which re-rolls today and the
two preceding days. The double-count fix landed **2026-09-12 20:42**
(`84a0159`), and the timer last ran **2026-09-12 18:47** — *before* the fix. So:

- **2026-09-11 and 2026-09-12**: self-heal on the next timer run (they sit
  inside the 3-day window).
- **2026-09-10**: falls out of the window, so it will **never** be re-rolled
  automatically. A day cached under a since-fixed bug needs an **explicit
  backfill** if its history matters.

This is the general rule: a stale rollup does not expire on its own once it
leaves the maintain window. Re-roll the affected days deliberately:
`rollup.rollup_range(conn, "2026-09-10", "2026-09-12")` — on a **copy** first.

### 2. Latent cost error in routing proposals ⚠

`optimiser.rank_models()` derives `cost_per_call` from `daily_findings`, and
`propose()` projects savings from it. Stale (1.85×) cost therefore inflates a
model's apparent cost-per-quality — it can reject a genuinely cheap rung or
rank it below a pricier one, and the resulting `est_weekly_delta_usd` would be
a fabricated number.

**Currently inert** because the quality gate is shut: eligibility requires
`quality_n > 0`, and no non-shadow finding has any quality rows. It becomes
**active** the moment measured quality reaches `daily_findings`.

### 3. The quality signal is an artifact — do not act on it 🚩

The only scored rows in the ledger are 70 shadow rows, averaging **0.0042**.
65 of those 70 had `finish_reason='tool_calls'` and emitted 54–87 output tokens
against 39K–152K input tokens. The model returned a **tool call, not a
summarisation**, and the fact-coverage scorer graded the effectively-empty text
as 0.

So that 0.004 is a **measurement artifact, not a low-quality rung**. Reading it
as "glm-5.3-flash is a terrible summariser" would justify killing a cheap rung
for no reason.

The correct treatment, already implemented in `quality_probe.py`: attempts with
no scoreable summary are recorded with `measurable = 0` and **excluded from
every average** (`AVG(score) FILTER (WHERE measurable = 1)`). A probe score of
0.0 must mean "measured, and the facts were lost" — never "we failed to
measure".

For comparison, the probe scored 50 real captured samples at **0.899** with
that guard in place.

### 4. Probe scores do not reach the optimiser ⚠

`rollup_day()` aggregates `AVG(quality_score)` from **`router_logs`**.
`quality_probe.record_probe()` writes to the separate **`quality_probe`** table
(joined on `request_id` when needed).

So even with the probe running, `daily_findings.quality_avg` stays empty and
the optimiser stays blind — the very gap the probe was built to close. Decide
deliberately whether to join the probe into the rollup, rather than assuming
the wiring is complete.

### 5. Shadow rows are correctly quarantined ✅

`daily_findings.is_shadow` and the optimiser's `include_shadow=False` default
both work: `is_shadow=1` rows (188 rows, $2.86) are excluded from production
rankings. No cleanup needed.

### 6. Experiment re-enable risk — low ✅

`experiments.yaml` holds `glm53flash-compression-shadow` with `enabled: false`,
and the endpoint re-reads it (cached briefly) to drive traffic split. The
`experiments` DB table is **empty** and nothing inserts into it, so the YAML is
the single source of truth — there is no stale DB row that could silently
re-enable the shadow. Residue is 602 historical rows tagged
`experiment='glm53flash-compression-shadow'`, correctly separated by
`is_shadow`.

### 7. Backfill contamination — none in the recent window ✅

The cost backfill touched 234,211 rows all-time, but 114,279 of them (47.2%) are
non-billable markers — **correctly excluded** by `BILLABLE_ROW_SQL`, so no spend
report is inflated by them. All-time backfill dollar figures are not comparable
to recent windows (older rate cards differ); do not put them in the same table.

### 8. Minor: empty `request_id` understates the request count (0.45%)

170 billable rows shared an empty `request_id`, so `COUNT(DISTINCT request_id)`
collapsed them into one. Effect on the projection: $48.26 → $48.48 (+0.45%).
Negligible; note it, don't fix it.

## Checklist for any reported figure

1. Does the query use `BILLABLE_ROW_SQL` verbatim (no extra filters)?
2. Is `daily_findings` fresh for the window, or does it predate a fixed bug?
3. Is the quality number measured, or an artifact of a non-summarising
   response? Check `finish_reason` and `measurable`.
4. Are shadow rows excluded (`is_shadow = 0`) for production claims?
5. Does a projection carry the reading it was calibrated from?
