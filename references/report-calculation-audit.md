# Report Calculation Audit

`scripts/check-routing-stats.py` emits the nightly digest. Every number it
prints is a candidate for a silent bug — the two defects found to date both
read as healthy while being wrong. Audit any change to this script (or its
fixtures) against this checklist.

## Bug classes that recur

### Numerator/denominator population mismatch
A rate is `counter_A / total_B` where `counter_A` is counted over one row set
and `total_B` is summed over a *different* population (often a different DB).

- "Equivalent gpt-5.5 calls saved" was `gpt55_units / 30 - total_calls` while
  `gpt55_units = total_calls * 30`, so the expression was `total_calls −
  total_calls` — structurally 0 on every run.
- "Correction rate" divided `corrections` (counted only over endpoint
  observations in `router_logs.db`) by `total_calls` (all API calls summed
  from `state.db`), diluting the rate ~25% whenever non-endpoint traffic exists.

Rule: for every derived rate/percentage, name the population the numerator is
counted over and make the denominator that same population. `router_completion_rows`
is the endpoint population; `total_calls` is the all-API population. They are
NOT interchangeable.

### Unguarded `sqlite3.Row` column access in a per-row loop
`r["col"]` raises IndexError when `col` is absent from the schema. Inside a loop
wrapped in a broad `except Exception: pass`, that aborts the loop and silently
zeroes every stat the loop was accumulating (completion success, failures,
lanes, selected models). Guard reads the same way sibling columns already are:
`r["col"] if "col" in r.keys() else 0`.

## Test harness (`tests/test_report_calculations.py`)

- Build `state.db` (sessions + session_model_usage) and `router_logs.db`
  (full schema incl. `user_corrected`, `cheaper_model_would_work`,
  `budget_class`, `governor_fail_open`) as isolated fixtures.
- Set `crs.SNAPSHOT_FILE` to a per-test `tmp_path` path on every run, or the
  day-over-day baseline leaks across tests (a second test's "first run" then
  sees a stale baseline).
- A `fit_cols=False` fixture omits `user_corrected` / `cheaper_model_would_work`
  to exercise the missing-column guard path.
- Pin *specific* fixture values (e.g. assert `"Correction rate:       10.00%"`,
  not just `"> 0"`) so a subtly wrong divisor still fails the test.

## Smoke-testing against production
Do not `cp` the production `state.db` into `/tmp` — it is multi-GB and the
`/tmp` tmpfs is ~6G; the copy fills it. Verify the production schema instead
with `sqlite3 <db> "PRAGMA table_info(router_logs)"` and let the unit fixtures
carry the logic checks.
