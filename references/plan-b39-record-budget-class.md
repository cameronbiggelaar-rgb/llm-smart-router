# Plan B39 — record the governor's class on every routed request

## The finding this fixes

`router.py:1487` makes the governor a **no-op for reporting** whenever the
selected model is not on the ChatGPT plan:

```python
if not _is_plan_model(selected):
    return selected, "", False, False   # klass = "" always
```

`RouterDecision.budget_class` therefore stays `""` unless a plan model was
*actually selected*. While GPT is dark (all five plan models rate-limited until
the Sunday reset) nothing selects one, so **every row in `router_logs` carries
`budget_class = ''`** — 195,139 of 195,139 rows.

The reporting layer renders `''` as `ungoverned`, so the nightly digest prints:

```
Routine denied (leak closed):      0
Architectural calls served:        0
Ungoverned (pre-governor):   195,178
```

and the operator reads "the governor looks inert". It is not: the gate works
(direct probe denies both classes correctly). The metric simply cannot express
the state the system is actually in.

That is a **false-zero** in the exact shape `database-attribution-evidence`
rule 5 describes: an unwritten column reported as a count. It also violates
rule 10 — a flag is only a denominator once you know which paths write it.

## Live evidence (not a simulation)

Post-restart rows select a *non-plan* model at a floor it cannot meet:

```
1443 | deepseek-v4-pro | tool-bearing ... (tier 8, capability floor 13)
```

Floor 13 with only tier-8 available is the **degrade path** in
`_select_tool_capable_model` — proof that the plan models were unavailable at
request time, so the governor was never handed a plan model and never classified
anything. This is why an earlier synthetic probe (which forced availability)
disagreed with the database: the probe was wrong, the database was right.

## Three states the report must be able to tell apart

| state | meaning | today |
|---|---|---|
| governor off / pre-arm | no policy ran | `''` → `ungoverned` ✅ |
| armed, request classified, plan not reached | idle — nothing to gate | `''` → `ungoverned` ❌ |
| armed, plan reached | denied or served | class recorded ✅ |

Only the middle state is broken, and it is the one the system has been in for
days.

## Change

### 1. Classify whenever the governor is armed (`_apply_plan_governor`)

Move the classification above the plan-model test. A non-plan pick still returns
`denied=False`, but it now carries its class.

* `enabled=false` → still `""`. **The rollback contract is unchanged**: with the
  governor off, no row is reported as governed.
* A classifier failure must not invent a class — the existing
  `_plan_classify` fail-safe returns `architectural` on exception, which would
  over-report the protected lane. Fail to `""` here instead.

### 2. Call it unconditionally on the ordinary capability path

`router.py:1210` guards the call with `if _is_plan_model(selected):`. Remove the
guard so the class is recorded there too; denial handling is already keyed off
the returned `denied` flag, which stays False for non-plan picks.

### 3. Hoist the verdict out of the per-model loop (required, not optional)

`_build_tool_fallback_chain` currently calls `_plan_denied_for_class(klass)`
**inside** `for model in get_available_models()`. Measured: **87 µs per call**
(dominated by `fetch_reading`, which re-reads persisted state every time).

Today that loop is only entered with a class for plan-selected requests — rare.
Recording the class universally puts it on the hot path for *every* tool request:
~14 models × 87 µs ≈ **1.2 ms**, against a measured `route_task` of **667 µs** —
a ~3x routing regression. Compute the verdict once per chain build.

This is a direct consequence of change 1, so it ships in the same batch.

### 4. Reporting: name the idle state

`_plan_budget_summary` gains `classified` (rows with a non-empty class) and the
render distinguishes:

* `classified == 0` → **ungoverned** (pre-arm; genuine no-policy rows)
* `classified > 0` and no plan traffic → **"armed — N classified, plan not
  reached"**, i.e. explicitly idle, not inert
* denials/served as today

`architectural_served` must count only rows the **plan actually served**
(`provider = 'openai-codex'`), otherwise an architectural request served on
`pro` because the plan is dark would inflate the protected lane — the same
class of error, pointing the other way.

### 5. §5.5 containment now applies to non-plan picks (behaviour change)

Call sites pass `klass=budget_class or None`. Recording the class arms
`_contain_plan_in_chain` / `_build_tool_fallback_chain` for ordinary requests
too, so a routine `pro` request no longer carries `gpt-5.5/5.6-*` in its
escalation chain. That is the §5.5 intent — the leak re-entering through
failover — and is pinned by a non-vacuity test. `force_model` (the §3.3 escape
hatch) is untouched.

## Batches

* **B39a** — record the class + hoist the verdict.
  `tests/test_budget_class_recording.py` (RED first).
* **B39b** — reporting: idle vs ungoverned, plan-served architectural.
  extends `tests/test_routing_stats_plan_budget.py`.
* **B39c** — design doc note + build notes.

## Acceptance

Sunday's reset is the real test: once the plan returns, routine denials must
climb while architectural stays served, and the digest must stop saying
"ungoverned" for traffic that was classified all along.
