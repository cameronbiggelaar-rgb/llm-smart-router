# Testing strategy: stop discovering defects in production

Plan written before execution, per the established refactor workflow
(scoped plan in `references/` → TDD batch-by-batch, RED → GREEN → commit, each
batch gets its own test file).

## The problem, stated precisely

On 2026-09-12 a single live enablement of a shadow experiment surfaced four
defects plus one reporting defect. None were subtle; all reached production.

Why the existing 266 tests did not catch them:

1. **Tests inject fakes through the seam they are meant to verify.** Every
   endpoint test replaces `_get_db_connection` with a fixture connection and
   monkeypatches `proxy_to_backend`. That is legitimate for unit tests, but it
   means the *wiring* — path resolution, schema creation, price seeding, the
   real INSERT, the real call site — is never executed by any test.
2. **The production DB path is hardcoded**, so a test *cannot* point the real
   path at a throwaway DB. Two consequences: no test ever runs the real
   initialisation, and an exploratory probe silently wrote stub rows into the
   production database because overriding the path was ignored.
3. **Logging failures are swallowed.** `_log_request_to_db` wraps everything in
   `except Exception: logger.warning(...)`. A call site passing a keyword the
   function does not accept (exactly what `finish_reason` did) raises
   `TypeError`, logs a warning, and returns 200 to the user with the row lost.
   The request looks healthy; the evidence never lands.
4. **No CI.** There is no `.github/workflows`, so the suite runs only when
   someone remembers to run it.

The common thread: **nothing tested the seams between modules, and the seams
failed silently.** Defects 1–4 from the live run all lived there.

## What each live defect would have needed

| defect | class of test that catches it |
|---|---|
| `asyncio` never imported at module level | import the module, drive a real request |
| `rollup.migrate()` schema missing 10 logger columns | schema ⇄ writable-column parity |
| `finish_reason` not a parameter of the logger | static call-site ⇄ signature check |
| `model_pricing` never seeded by the endpoint | real init path, then assert priced |
| `_ensure_log_columns` masked the schema gap | run the real init path, not a hand-rolled CREATE |
| probe wrote to the hardcoded production DB | injectable path + a test that asserts non-leakage |
| optimiser says "incumbent is optimal" when nothing is measured | message must distinguish *no evidence* from *no change needed* |

## Batches

### B11 — Make the production path testable (no behaviour change)
Add an environment override for the router DB path, defaulting to exactly the
current production path. This is the enabling change: with it, tests can drive
the *real* initialisation and write path against a throwaway file instead of
replacing the function under test.
Tests: default is unchanged; override is honoured; the real logger writes to the
override and never to production.

### B12 — Preflight harness: "does it actually work end to end?"
`scripts/preflight.py` — boots the real production code path (real
`_ensure_log_columns`, real `seed_prices`, real `_log_request_to_db`) against a
throwaway DB, drives real HTTP requests through the ASGI app with only the
network stubbed, then asserts:
* rows written == requests made (the **no-silent-drop invariant**)
* every column the logger can write exists in the table it writes to
* prices are seeded, so a logged call carries a real cost rather than
  `cost_unknown=1`
* an experiment whose candidate cannot be resolved is *reported*, not silent
This is the gate to run before enabling anything against live traffic.

### B13 — Static contracts
* AST-walk every `_log_request_to_db(...)` call site in the endpoint and assert
  each keyword is a real parameter of the function. Catches the
  `finish_reason` class of defect without executing anything.
* Pin the test fixture's schema to the production init path, so a fixture can
  never silently drift from what production builds.
* Assert the endpoint's writable columns and `rollup.NEW_LOG_COLUMNS` agree
  (they currently differ by 3 known compression columns — make that an explicit,
  documented allowlist rather than an unnoticed drift).

### B14 — Truthful reporting when evidence is absent
`optimiser.format_proposal` currently prints "No cheaper qualified routing
identified — incumbent is optimal" when in truth **zero** production rows have
measured quality (0 of 205,587), so *nothing was eligible*. That sentence
asserts a conclusion the data does not support.
Fix: distinguish "no model measured → cannot rank" from "measured, and the
incumbent already wins". Test both branches.

### B15 — CI
`.github/workflows/tests.yml` running the suite on push, so the suite is a gate
rather than a habit.

## Non-goals

* Not replacing the existing unit tests — they are fine at what they do.
* Not adding a CI service that needs secrets; the suite is offline by design.
* Not making the endpoint auto-apply anything. The optimiser stays propose-only.

## Discipline

Every batch: failing test first, then the fix, then commit. Run the full suite
before each commit. If a batch reveals that an earlier "fix" was wrong, correct
it in a test rather than in prose.
