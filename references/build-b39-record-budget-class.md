# Build B39 — record the governor's class on every routed request

## What shipped

The plan governor now records `budget_class` on **every** routed request while
armed, not only when a plan model is selected. Previously the class was written
only inside the `if _is_plan_model(selected)` gate, so while GPT was dark (all
five plan models `available=0`) nothing ever reached the plan, no class was
written, and `budget_class` stayed `''` on every row. The nightly digest's
`COALESCE(NULLIF(budget_class,''), 'ungoverned')` then folded the whole table into
`ungoverned`, which read as "the governor looks inert".

That was a reporting artifact — the gate was never bypassed. This build makes the
report tell the truth: it can now distinguish **armed-but-idle** from **off**.

## Changes

1. `scripts/router.py` — `_apply_plan_governor` reordered:
   - load config + `enabled` check first (unchanged rollback lever),
   - **classify before** the plan-model test,
   - return `(selected, klass, False, False)` when `klass` is empty or the
     selected model is not a plan model (nothing to gate, class still recorded),
   - only then probe the account (`fetch_reading` + `evaluate_policy`) — the
     expensive probe is now skipped for the ~99% of traffic that never touches a
     plan model.

2. `scripts/router.py` — Hook 2 (ordinary capability path) dropped the
   `if _is_plan_model(selected)` guard around `_apply_plan_governor`, so the
   non-tool path also records the class. The `denied` branch is a no-op for
   non-plan picks (the governor returns `denied=False` for them).

3. `scripts/router.py` — `_build_tool_fallback_chain` hoisted the per-model
   `_plan_denied_for_class(klass)` call out of the candidate loop. It is a
   property of the request class, not of each model; recomputing it per candidate
   scaled cost with the model count (~14 × 87 µs ≈ 1.2 ms on a 667 µs route).
   Now computed once, and guarded by `bool(klass)` so an empty class (governor
   off / unclassified) does not arm containment — the §5.5 failover-leak fix.

4. `scripts/check-routing-stats.py` — `_plan_budget_summary` now emits `armed`
   (from `load_governor_config`) and `governed` (rows with a recorded class), and
   the render block prints `Governor: ARMED`, a `Governed (class recorded)`
   count, and an explicit `(armed, no governed traffic in window — idle, not
   inert)` line when armed with zero governed rows. The ambiguous "looks inert"
   state now has a named, distinct rendering.

## The coupled §5.5 defect (handled deliberately)

Recording the class universally arms the fallback-chain filter, because the chain
builder checks `klass is not None`. With `klass` now non-empty on every armed
request, a routine request whose chain walks up in tier would — without the
`bool(klass)` guard — have its plan-account models excluded only when the class
is non-empty *and* the plan is actually denied. The empty-string class previously
short-circuited this, leaking all four plan models into the chain. The hoist +
`bool(klass)` guard keeps containment exactly where it was for governed denials,
and leaves the chain unfiltered only when there is genuinely no class (governor
off / classifier failed), matching the old `klass is None` behaviour.

## Verification

- `tests/test_budget_class_recording.py` — 7 tests, RED-first then GREEN:
  routine/architectural/non-tool classes recorded while the plan is dark;
  recorded class keeps plan models out of a routine chain; chain verdict
  computed once not per-model; governor-off still returns empty class (rollback
  contract); a non-plan pick is never denied.
- Full governor suite (173 tests) green, including the B38 tool-tier-floor tests.
- Full suite: 967 passed / 10 failed — the 10 are pre-existing data drift
  (canary-monitor + optimiser-evidence + streaming-cost-gap), confirmed by
  stashing B39 and re-running on HEAD (same 10).
- Live smoke: governor-off → empty class; armed + plan dark → routine coding
  request selects `deepseek-v4.1-flash` with `budget_class='routine'`.

## Not done / follow-ups

- The report still backdates its period to 2026-07-31 (cumulative). Day-over-day
  delta needs a period anchor fix — separate from this build.
- `ungoverned` will now shrink to rows written before the B39 deploy (or while
  the governor is off). It is no longer a proxy for "everything".
