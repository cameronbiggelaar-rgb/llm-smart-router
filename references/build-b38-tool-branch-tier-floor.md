# Build notes — B38: honour `min_tier` in the tool-capable pick (§9.4)

**Date:** 2026-09-25
**Commit:** `2ad68dd` (implementation) · `22872d3` (scoped plan)
**Plan:** `references/plan-b38-tool-branch-tier-floor.md`
**Scope:** `scripts/router.py` + tests. No config change, no governor policy change.
**Status:** COMPLETE. Tests green, impact measured, no restart required yet.

## What was wrong

`_select_tool_capable_model()` took no arguments and returned the **highest-tier
available** tool-capable model — the docstring said so explicitly ("prefer the MOST
CAPABLE available … so tool work keeps the best reasoning quality"). Correct for a
world where the top model is cheap; wrong when the top model is a metered
ChatGPT-plan model the operator has reserved for the architectural lane. Every
tool-bearing request, however ordinary, took `gpt-5.6-sol` whenever it was up.

### Measured, scoped to rows the tool branch actually decided

| route to `gpt-5.6-sol` | calls |
|---|---|
| tool branch (`routing_reason LIKE 'tool-bearing request%'`) | **1,267** |
| escalation | 489 |
| genuine tier-13 architectural lane | **3** |

A tier-13 model was reached by *ordinary tool work* **422×** more often than by the
lane it exists for. 4 of 1,760 total `sol` calls are unaccounted for in this split.

### Corrected attribution (an earlier claim of mine was wrong)

Previously reported as "the 4 `tier 13 needed → tier 12` rows are the tool branch."
**False.** All 15 `tier 13 needed` rows in 30d have `requires_tools=0` — those 4
terra rows are the **non-tool** tier matcher degrading when `gpt-5.6-sol` was
unavailable. Unrelated to §9.4, which is the tool branch stealing `sol`.

Also note: an initial impact query (`requires_tools=1`) showed **98,566**
`deepseek-v4-flash:cloud` calls apparently moving to `pro`, implying a massive
demotion. That was an artifact — `requires_tools=1` is far broader than the tool
branch, and the flash rows are **canary treatment** rows whose model is overridden
by `apply_experiment()` *after* `route_task` returns. Scoping to the branch's own
reason string and excluding `experiment_arm='treatment'` gives the true denominator
of **19,686** and the true churn of **10.0%**.

## What changed

1. **`_select_tool_capable_model(min_tier: int = 0)`** — returns the **cheapest**
   tool-capable model meeting the floor, walking `get_available_models()` (already
   in `MODEL_COST_ORDER`) exactly like `_select_model(min_tier)`. On an unreachable
   floor it **degrades to the most capable** available tool model rather than failing
   closed (design line 131 — `min_tier` is a quality floor, not a capability wall).
   Returns `""` only when nothing tool-capable is available.

   **"Cheapest" is price-based, not tier-based** — verified against the price book,
   and this matters: `deepseek-v4-pro` (tier 8, $0.66/$1.98) is *cheaper* than
   `glm-5.3` (tier 6, $1.40/$4.40). An implementation that sorted by tier would
   return the pricier model. Pinned by `test_cost_order_is_price_based_and_pinned`,
   which reads the live price book so a reorder fails loudly.

2. **`_architectural_floor(prompt)`** — a `max()` composition term, and the
   subtle part of this batch. `_estimate_min_tier` → `match_sub_type(task_type,
   prompt)` scans **only the request's own task_type**; the tier-13 sub-type lives
   **solely under `planning`** in `routing_table.yaml`. So the canonical
   architectural ask tagged `coding`/`debugging`/`research`/`testing` scores **6**
   (measured), not 13 — and a naive floor would have demoted genuine architectural
   work off `sol`, i.e. broken the one lane the governor exists to protect.

   `_architectural_floor` uses `gpt_budget.classify_request`, which scans **all**
   task types (rev 4 fixed exactly this), returning `_ARCHITECTURAL_MIN_TIER` (13)
   or 0. This keeps routing and budget classification in agreement — the B37h
   principle, one shared notion of "architectural", so the two cannot disagree.
   Deliberately **independent of governor state** (a capability floor is not an
   entitlement) and **fails safe to 0** if the classifier raises.

3. **`min_tier` hoisted above the tool-branch early return.** It was computed
   *after* the branch, so the branch could never see it. Now computed once, before
   the tool gate, and the duplicate computation further down was removed.

4. **Tool branch** now uses `max(min_tier, _architectural_floor(prompt))`.

5. **`_reselect_without_plan(min_tier)`** — was "highest-tier non-plan", now honours
   the floor. Its docstring asserted *"``min_tier`` is accepted for signature
   symmetry but the tool branch has no tier floor of its own"* — true before this
   change, false after; corrected rather than left as a stale claim.

6. **Reason strings record the floor** (`… capability floor N`) so the leak is
   measurable from the DB rather than inferred from model names.

## Impact (30d replay, tool-branch-decided rows)

```
picks that change: 1,969 / 19,686  (10.0%)
  gpt-5.6-sol    -> deepseek-v4-pro   1,267
  glm-5.3:cloud  -> deepseek-v4-pro     476
  gpt-5.6-terra  -> deepseek-v4-pro     224
  gpt-5.6-luna   -> deepseek-v4-pro       2

cost  $2,131.83 -> $1,144.11   delta -$987.71
```

**This is an upper bound.** Prompts are not stored (B37k limitation), so only the
estimator term could be replayed; the architectural term can only *raise* floors,
so some of these 1,969 demotions would not occur on a real run. Direction is safe
(over-states the change).

Note the `glm-5.3:cloud` → `pro` moves: `glm-5.3:cloud` has **no price-book entry**,
so its cost delta is excluded from the dollar figure — the real saving is slightly
larger. Floor for ordinary coding is 4; pro (8) satisfies it and is cheaper.

## Tests

- `tests/test_tool_tier_floor.py` — **NEW, 26 tests.** Floor honouring (parametrized),
  degradation, fail-closed on a genuine capability gap, non-tool-capable never
  selected, astra exclusion, price-order coherence, `_architectural_floor`
  behaviour (fires / silent / all-task-type / governor-independent / armed /
  fails-safe), and two end-to-end `route_task` checks.
- **Two pre-existing tests encoded the defect and were updated**, not deleted:
  - `test_gpt_budget_tool_branch.py::test_governor_disabled_matches_pre_b37_behaviour`
    asserted `selected_model == "gpt-5.6-sol"` with the comment *"Pre-B37:
    highest-tier tool-capable model"*. Now asserts governor **neutrality** (selected
    model == what the ungoverned selector returns) instead of pinning a model name.
  - `test_router.py::TestOllamaToolCapableModels::test_tool_routing_prefers_most_capable_tool_model`
    asserted `"gpt-5.5"` unconditionally. Rewritten to assert a sufficient
    tool-capable model, plus a new companion asserting ordinary tool work no longer
    reaches the top model.
- **Governor stub convention:** the new module uses an autouse `ungovened_plan`
  fixture (governor OFF), matching `_ungoverned_plan` in `test_gpt56_gpt6_lanes.py`
  established at `11d4de8`. Without it, the armed governor correctly *denies* an
  architectural request and reselects it off the plan — a policy outcome, not a
  routing-floor one, and it made the end-to-end test fail only in the full suite
  (order-dependent, because an earlier governor test primes the live reading cache).
  Policy belongs to `test_gpt_budget_*`.

**Full suite: 964 passed / 6 failed / 4 deselected** — up from 936 passed, with
exactly the same 6 pre-existing failures (5 canary-monitor data-dependent, 1
production-DB streaming-cost rollup).

## Pitfalls encountered

- **`Reading` shape.** Wrote a `_reading()` helper with invented fields
  (`used_percent`, `resets_at`, `window`, `plan`…). The real shape is `Reading`
  wrapping two `WindowReading`s (primary 5h / secondary 7d). Always copy the helper
  from an existing governor test rather than reconstructing the dataclass.
- **Invented a number in a docstring.** Claimed the estimator yields 11 for a
  `coding`-tagged architectural prompt; measured it and it is **6**. Docstring and
  test docstrings corrected. Measure before writing a figure into prose.
- **`router_logs` column names:** `model_used`, not `model`. No `tier` column.
- **The `success` column remains unusable** (83,956/83,961 non-success rows are the
  streaming artifact) — use token sums + `routing_reason`.
- Order-dependent test failures from the armed governor are expected; check with a
  targeted run *and* the full suite, since they can pass alone and fail in-suite.

## Follow-ups (open)

- **Restart required** to serve this change: `biggie-llm-endpoint.service` was last
  restarted serving `723a406`, so B38 is **not live** until it is restarted.
- **Interaction with the armed governor.** Both mechanisms are now in place and
  target the same leak from different directions: the floor stops ordinary tool
  work *reaching* the plan, the allowlist stops whatever still arrives. B38 shrinks
  the governor's routine class substantially. Worth re-running the calibration
  (`calibrate_plan_budget.py --days 30`) after restart to see the combined denial
  volume.
- **`match_sub_type` keyword gap (unchanged):** the canonical ask *"design the
  architecture for it and implementation and test plans"* still scores tier 8, not
  13. `force_model` remains the escape hatch. B38 does not address this, and
  `_architectural_floor` inherits the same detector — so prompts the classifier
  misses get no architectural floor either.
- **`glm-5.3:cloud` has no price-book row** — cost reporting for that model is
  incomplete.
