# Plan — B38: honour `min_tier` in the tool-capable pick (§9.4)

**Date:** 2026-09-25
**Scope:** `scripts/router.py` + tests. No config, no governor policy change.
**Status:** scoped; awaiting operator review before batch 1 (design §9.4: "deserves
its own batch + review rather than riding in on the governor").

## 0. Correction to the record first

Last session I attributed the 4 "tier 13 needed, served tier 12 (`gpt-5.6-terra`)"
rows in the B37k calibration replay to `_select_tool_capable_model`. **That
attribution was wrong, and the query below disproves it.**

```
rows whose logged reason says 'tier 13 needed': 15
  2026-09-05 planning  tools=0 cplx=0.05 -> gpt-5.6-sol    (tier 13)  ✓ met
  2026-09-14 planning  tools=0 cplx=0.45 -> gpt-5.6-terra  (tier 12)  ✗ 4 rows
  2026-09-16 coding    tools=0 cplx=0.90 -> deepseek-v3.1:671b (9)    ✗ 3 rows
  2026-09-19 planning  tools=0 cplx=0.30 -> glm-5.2        (6.5)      ✗ 7 rows
```

Every one of the 15 rows has **`requires_tools=0`**, and the string
`capability tier {N} needed` is emitted at exactly one place —
`router.py:1251`, the **non-tool** capability path. The tool branch writes a
different string (`tool-bearing request — selected proven tool-capable model …`)
with no `tier N needed` clause. So those rows are the non-tool path **degrading
predictably** when `gpt-5.6-sol` was unavailable — which is design line 131
working as intended, not the §9.4 defect.

## 1. The actual §9.4 defect

`_select_tool_capable_model()` takes **no arguments** and picks the
**highest-tier available** tool-capable model, ignoring `min_tier` entirely. Its
docstring is explicit that this is deliberate quality policy:

> "Among those we prefer the MOST CAPABLE available (highest tier) so tool work
> keeps the best reasoning quality (gpt-5.5 first)"

Consequences, measured on 30d of `router_logs`:

- Every tool-bearing request takes the top plan model whenever it is available,
  **however ordinary the request**. `tool-bearing request — selected proven
  tool-capable model gpt-5.6-sol (tier 13)` = **1,267 calls / 49.4 Mtok**.
- It **steals architectural traffic**: of 1,760 `sol` calls, 1,267 arrived via
  the tool branch, 489 via escalation, and only **3** via the genuine
  architectural tier-13 lane. A tier-13 model was reached by ordinary tool work
  422× more often than by the architectural lane it exists for.
- Tool-bearing requests on plan models total **13,376 calls / 506.2 Mtok** in
  30d (3.49% of all routed tokens).

The branch returns at `router.py:974`, **before** `_estimate_min_tier()` is
called at `router.py:1085` — so the floor does not exist at that point in the
function. Hoisting it is part of this batch.

## 2. The safety constraint that shapes the design

**Do not simply feed `_estimate_min_tier()` into the tool branch.** That would
under-serve exactly the lane the governor exists to protect, because the two
mechanisms scope their table scan differently:

| mechanism | scan scope | `"rearchitect the storage layer"` tagged `coding` |
|---|---|---|
| `gpt_budget.classify_request` (governor) | **every** task type's sub-types | `architectural` |
| `_estimate_min_tier` → `match_sub_type(task_type, …)` | **only** the request's own `task_type` | tier **11** (not 13) |

The tier-13 sub-type lives **only** under `planning` in `routing_table.yaml`
(line 93). A genuine architectural ask that the classifier tags `coding` or
`debugging` — the ~30%-of-traffic case rev 4 fixed for classification — yields a
floor of 11 from `_estimate_min_tier`:

```
prompt                                       task_type   class          min_tier(own tt)
rearchitect the storage layer end to end     coding      architectural                11
rearchitect the storage layer end to end     debugging   architectural                11
redesign the architecture of the router      coding      architectural                11
find failure states in the payment flow      debugging   architectural                11
review the architecture of the ingest path   research    architectural                11
```

So a naive hoist gives floor 11 → picks `gpt-5.6-luna` (11) and **silently
demotes a real architectural request off `sol` (13)**. The tool floor must
therefore be `max(min_tier, architectural_floor)`, reusing the **same
all-task-type scan** as the classifier — consistent with B37h's "one shared lane
classifier so routing and reporting can't disagree".

## 3. Design

1. `_select_tool_capable_model(min_tier: int = 0) -> str` — mirror
   `_select_model()`'s semantics exactly, which is the point (the tool branch is
   currently the *only* selector without a floor):
   - iterate `get_available_models()` (already in `MODEL_COST_ORDER`) → return the
     **cheapest tool-capable model meeting the floor**;
   - nothing meets it → **degrade predictably to the most capable available**
     (design line 131), never fail closed on a tier miss;
   - floor 0 → cheapest tool-capable (the un-floored callers keep working).
2. Hoist the `min_tier` computation **above** the tool-branch early return so the
   tool path can use it. `_estimate_min_tier` and its callees are pure
   (table reads only, no state writes) — verified — so the hoist is safe.
3. `_architectural_floor(prompt) -> int` — returns 13 when the prompt matches a
   tier ≥13 sub-type under **any** task type, else 0. Reuses
   `gpt_budget.classify_request` / `_ARCHITECTURAL_MIN_TIER`; **independent of
   governor `enabled`** (this is a capability floor, not an entitlement).
4. Tool floor = `max(min_tier, _architectural_floor(prompt))`.
5. `_reselect_without_plan(min_tier)` — honour the same floor, so the
   governor's re-selection doesn't drop a high-tier routine request to pro when a
   tier-appropriate non-plan tool model exists. (It currently documents "the tool
   branch has no tier floor of its own" — that premise is what this batch
   removes.)
6. Reason strings record the floor for observability, e.g.
   `… selected deepseek-v4-pro (tier 8; floor 4)`.

## 4. What this changes on live traffic (honest sizing)

The flash canary (`v41flash-tool-canary`, percent 100) is applied **after**
`route_task` in the endpoint, so it overrides the tool branch's pick for all
eligible tool requests. B38's live effect is therefore confined to tool requests
**not** served by the canary — `exclude_task_type: [debugging, testing]` — plus
any non-eligible traffic. That is the "debugging/testing held on pro" line item,
not the 98k canary-served requests.

With the governor **armed**, the visible effect is smaller still: routine tool
requests are already denied the plan, so B38 mostly moves *where* the decision is
made (capability floor instead of entitlement denial) rather than *which model
serves the request*. Its real value is:

- **defence in depth** — the leak is closed at source, so flipping the governor
  off (the rollback lever) no longer re-opens the tool-branch burn;
- it makes the governor's routine class much smaller, which is §9.4's stated
  benefit;
- it stops a tier-13 model being reached by ordinary tool work at all.

**Observability consequence:** routine tool requests will no longer be *denied* —
they will simply never be routed to the plan. So `plan_budget_denied` should drop
toward 0 for tool traffic and the denial counter becomes a non-tool signal. This
is expected and must be stated in the digest, not read as the governor breaking.

## 5. The open decision (operator's call)

The highest-tier preference is **deliberate quality policy**, and reversing it is
a real quality trade-off, not just a bug fix. The evidence to price that trade-off
**does not exist**:

- `success` is unusable as a quality proxy — `success=0` is 83,956 of 83,961 rows
  and is **entirely** the `streaming_in_progress` artifact (5 rows are other
  errors). Confirmed: `success=0 AND error_type='streaming_in_progress'` = 83,956;
  `success=1 AND streaming_in_progress` = 0.
- `quality_score` is **NULL for every tool-bearing row** (0 of 20,180 pro, 0 of
  1,756 sol).
- So there is **no measurement** that says gpt-5.5/sol is better *at tool work*
  than pro — and none that says it isn't.

Options:
- **(a)** Implement B38 with the floor (recommended — it is the root-cause fix and
  is a strict improvement in coherence: the tool branch stops being the one
  selector without a tier floor).
- **(b)** Implement only the *architectural-floor* part (never demote a genuine
  architectural tool request) and leave the routine preference alone.
- **(c)** Defer until the tool-agreement instrument produces quality data.

## 6. Batches (TDD: RED → GREEN → commit each)

| batch | change | test file |
|---|---|---|
| B38a | `_select_tool_capable_model(min_tier)` honours the floor; cheapest-meeting; degrade predictably | `tests/test_tool_tier_floor.py` |
| B38b | hoist `min_tier` above the tool-branch early return | same |
| B38c | `_architectural_floor()` + `max()` composition; architectural prompt tagged `coding` still reaches `sol` | `tests/test_tool_architectural_floor.py` |
| B38d | `_reselect_without_plan(min_tier)` honours the floor | same |
| B38e | reason strings record the floor | same |
| B38f | full suite green (same 6 pre-existing failures) + design doc §9.4 marked resolved | — |

## 7. Rollback

`git revert` of the B38 commits. No config, no data migration, no governor
change — so the governor's own lever (`enabled: false`) stays orthogonal.
