# Plan — the governor's silent fail-open (why routine GPT traffic kept being served)

## What I told you first, and what was wrong

Earlier I said the tool-bearing branch "returns before the governor is consulted".
**That is wrong.** `router.py:1057` calls `_apply_plan_governor` on the tool branch,
and `router.py:1224` calls it on the ordinary capability path. B37e (`cfc09e5`)
and B37f (`7e47a99`) added both hooks on 2026-09-25. The gate is wired in.

The correct statement is narrower and worse: **the gate is wired in, and it still
never refused anything.** `plan_budget_denied = 1` appears on **132 rows, all of
them on a single day (2026-09-27, 04:06:34 → 12:00:21 UTC)** — during a hard-cap
window, not an allowlist denial. Since then: **5,211 GPT rows served with
`plan_budget_denied = 0`**, of which 4,392 carry `budget_class = 'routine'`.

## The defect

`_apply_plan_governor` has four exits. Three of them are indistinguishable in
`router_logs` from a deliberate allow:

| exit | klass | denied | logged? |
|---|---|---|---|
| `enabled=false` (rollback) | `""` | False | n/a — not governed |
| classifier failed | `""` | False | warning |
| exception | `""` | False | warning |
| **reading unreadable → fail open** | **`'routine'`** | **False** | **NOTHING** |

That last row is the one to explain. `_read_usage_payload()` (gpt_budget.py:207)
swallows **every** exception and returns `None`; `fetch_reading()` then returns
`Reading(source='unavailable')` (gpt_budget.py:267); `evaluate_policy()` returns
`VERDICT_OPEN` at gpt_budget.py:572 — before the allowlist is ever consulted — and
`_apply_plan_governor` returns `(selected, klass, False, False)` at router.py:1577.

Result: **the plan model is served, the class is stamped, `denied=0`, and not one
line is logged.** That is byte-for-byte the Oct-4/Oct-5 row signature.

### Why this is the only candidate left

Every other exit is excluded by the data:

* `enabled=false` → would leave `budget_class=''`; 37,598 rows carry a class, so
  the governor is armed and classifying. Excluded.
* classifier failure / exception → both log a warning; journal count for
  `plan governor unavailable` and `classifier unavailable` is **0**. Excluded.
* `_is_plan_model(selected)` false → the DB holds **bare** names
  (`gpt-5.5`, `gpt-5.6-terra`), and `_is_plan_model` returns **True** for those.
  The `:codex` suffix theory is dead: there is no `:codex` in the data, and
  `get_available_models()` returns bare gpt names. Excluded.
* `binding_window(reading) is None` → only reachable with no windows, which
  `parse_reading` already labels `source='unavailable'` (gpt_budget.py:163).
  Subsumed by the fail-open. Excluded.
* `evaluate_policy` reaching the allowlist → returns **DENIED** for routine
  (verified live: routine → `'denied'`, architectural → `'open'`). Excluded.

Direct probes confirm the healthy path works: `_apply_plan_governor('gpt-5.6-terra', routine_prompt)`
→ `('', 'routine', True, False)` **denied**, today, right now.

So the plan was served on Oct 4/5 because **the account probe was unreadable and
the governor failed open**, silently.

### Honest limit on this claim

I have **not** directly observed the probe failing inside the endpoint process —
it succeeds from my shell now, and the persisted reading is `source='live'`. The
mechanism is established by elimination, not by capture. That limitation is itself
the defect: a silent fail-open cannot be diagnosed after the fact. **The first
thing the fix must do is make it observable**, so the next occurrence is evidence
rather than inference.

## Why failing open is the wrong direction *for this policy*

The fail-open invariant (`gpt_budget.py:17`, `:561`, `:1450`) reads: *"an unreadable
probe must never starve the plan."* That is a sound rule for a **deny-list** — when
you cannot verify a cap, let the traffic through and let the account's own 429
catch it.

But `budget_policy.yaml` is an **allowlist**: the plan is reserved for the
architectural lane. Under an allowlist, an unverifiable entitlement is not consent.
Failing open re-creates precisely the leak the allowlist exists to close, and it
does so at the moment the governor is least able to notice — 519 GPT rows in one
evening.

The invariant predates the allowlist semantics (B37c, 2026-09-25) and was never
revisited when the policy became an allowlist.

## Change

### 1. `evaluate_policy` — split "unreadable" from "open" (gpt_budget.py)

Add `VERDICT_UNVERIFIED = "unverified"` next to `VERDICT_OPEN` / `VERDICT_CLOSED` /
`VERDICT_DENIED`. Replace the bare fail-open at line 572:

```python
if reading.source == "unavailable":
    # An unverifiable entitlement is not consent. Under an allowlist the plan is
    # reserved for the named classes; anything else must not ride a blind probe
    # onto the plan. Classes the allowlist DOES name keep the old fail-open, but
    # are reported as unverified rather than allowed.
    allowed = _known_classes(cfg.allowlist)
    return VERDICT_UNVERIFIED if klass in allowed else VERDICT_DENIED
```

`enabled=false` still returns `VERDICT_OPEN` first, so the rollback contract
(governor off ⇒ nothing is governed) is untouched.

### 2. `_apply_plan_governor` — a fourth return value, and no false claim (router.py)

Return `(model, klass, denied, closed, fail_open)`. `VERDICT_UNVERIFIED` →
`(selected, klass, False, False, True)`. `VERDICT_DENIED`/`VERDICT_CLOSED` →
`("", klass, True, closed, False)`.

### 3. Stop the reason string from asserting a verdict that never happened

Two append sites currently write `"; plan governor allowed class '{budget_class}'"`
whenever a class is truthy — `router.py:1109` (tool branch) and `router.py:1279`
(ordinary path). Both fire on the short-circuit and on the fail-open, so the row
claims the governor *allowed* the class when it never evaluated entitlement.

* verdict reached, class entitled → `; plan governor allowed class 'X'`
* fail-open on an entitled class → `; plan governor could not read the account — plan entitlement UNVERIFIED (fail-open)`
* no plan model in play → `; plan governor armed — class 'X' recorded (no plan model in play)`

A routine GPT row in `router_logs` must never again read "allowed".

### 4. Plumb `governor_fail_open` to the row

`RoutingDecision` gains `governor_fail_open: bool = False`. The endpoint writes it
from `decision.governor_fail_open` at all **five** call sites that already write
`plan_budget_denied` (biggie_llm_endpoint.py:4627, 4675, 4773, 4932, 5011), plus the
`_log_request_to_db` signature (:2824), the schema map (:307), and a
`ALTER TABLE` migration for the existing `router_logs.db`.

Without this the fix is invisible in exactly the way B39 was: the digest would
still report `routine_denied = 0` and read as "the governor looks inert", when the
truth is "the governor was blind for N requests".

### 5. `_plan_denied_for_class` must be explicit, not `!= OPEN`

`router.py:1495` is `_governor_verdict(klass) != gpt_budget.VERDICT_OPEN`. With a
new verdict value that silently flips to "denied" and starts containing fallback
chains for classes we are in fact serving. Make it a membership test over
`{VERDICT_DENIED, VERDICT_CLOSED}` so chains agree with the routing decision.

### 6. Reporting

`_plan_budget_summary` gains `fail_open` (rows with `governor_fail_open=1`), and
the render distinguishes *denied* / *served* / *blind*. A window with
`fail_open > 0` must say so and must not report the leak as closed.

## Batches

* **B44a** — `evaluate_policy` verdict + `_apply_plan_governor` return shape +
  `_plan_denied_for_class` membership. `tests/test_governor_fail_open_visibility.py`
  (RED first, this batch).
* **B44b** — reason-string honesty at both append sites.
* **B44c** — `RoutingDecision.governor_fail_open` + endpoint plumbing + schema
  migration (5 log sites, `_log_request_to_db`, `ALTER TABLE`).
* **B44d** — reporting: blind vs denied vs served; design/build notes.

Each batch commits on its own, RED before GREEN.

## Acceptance

* An unreadable reading denies the plan to `routine` (no GPT served).
* An unreadable reading still serves `architectural`, flagged UNVERIFIED —
  the plan is not starved.
* No `router_logs` row ever says `plan governor allowed class` unless the
  allowlist was actually consulted.
* A blind window is countable from the row data alone, without journal access.

## Decision needed from the operator

Change 1 reverses a documented safety invariant. It is the right direction under
an allowlist, and it is the only change that closes the leak in the blind case —
but it is your call, and it is reversible: `budget_policy.yaml enabled: false`
still restores exact pre-B37 behaviour.

**Recommendation: accept.** The failure mode it protects against ("plan starved
while the probe is down") costs quality on architectural work and is visible in
the report; the failure mode it currently allows costs money and is invisible.
