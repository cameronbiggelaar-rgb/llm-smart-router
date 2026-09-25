# Plan-budget governor — implementation notes

Companion to `references/design-gpt-budget-governor.md` (the design). This records
what the *build* taught, so the next batch doesn't rediscover it.

## Where the hooks live

`route_task()` in `scripts/router.py` has two enforcement points, and hooking only
one is the classic mistake (the design's rev 1 did exactly that):

1. **The tool-bearing early return** (`requires_tools` branch) — the real leak.
   `_select_tool_capable_model()` returns the *highest-tier* available tool-capable
   model with no complexity gate. 97.6% of the plan's burn arrived here.
2. **The ordinary capability path** (`_select_model(min_tier)`) — ~1% of burn,
   but still a leak: routine tier-10 work lands on `gpt-5.5`.

`force_model` is honoured **before** the governor in both paths: it is the
operator's escape hatch, so it must bypass the allowlist.

## Pitfalls that cost real time

**`available = False` alone does not take a model out of rotation.** `route_task()`
calls `probe_all_recovering()` at the top, which restores cloud models whose
cooldown expired. Tests must set a far-future cooldown:

```python
st = router._MODEL_STATUSES[m]
st.available = False
st.rate_limited_until = time.time() + 3600   # must also set rate_limited_until
```

**`_reselect_without_plan()` is tool-gated; the non-tool path needs its own
helper.** One reselect function shared by both hooks will silently restrict the
ordinary capability path to `TOOL_CAPABLE_MODELS`.

**Denial must *not* fail closed on the capability path.** The design (§131) says
degrade predictably to `deepseek-v4-pro` (tier 8). Fail closed only in the tool
branch, where tool capability is an actual wall and papering over it would be
worse than an error.

**A routine tool request that is denied has an EMPTY escalation chain.** All
higher-tier tool-capable models are plan models, so containment empties the chain.
This is intended (fail visibly rather than escalate onto the reserved plan) — pin
it in a test so it reads as a decision rather than a bug.

**Escalation must inherit the class, not re-derive it.** `escalate_on_failure()`
takes `budget_class=` from the originating decision. Without it, a failed routine
request escalates straight into the plan via the tool ladder.

**Guard the `prefer_model` path too.** The canary's declared fallback is
ollama-cloud today, but a future experiment configured with a plan fallback would
smuggle routine work onto the plan by preference.

**`classify_lane()` must be shared, and denials are never architectural.** A
denial reason quotes the allowlist ("architectural allowlist"), so a naive
substring test buckets every routine refusal as architectural work. Check the
`plan account denied` marker *before* the tier test.

**Read the tier the request NEEDED, not the tier it got.** Reasons record
`capability tier N needed` and `selected X (tier N)`. The tool branch can select a
tier-13 model for a tier-8 request; parsing the selected tier makes replay claim
the governor spares traffic it denies.

**Prompts are not stored.** `router_logs` keeps `prompt_length` only, so
calibration replay classifies from `routing_reason`. That can only *under*-detect
architectural work, making the denial count an upper bound — say so in output
rather than implying precision.

**A broad `except Exception: pass` hides new instrumentation.** The
`router_logs` block in `check-routing-stats.py` swallows everything, so the
digest read must open its own connection *outside* that block. Otherwise the one
line reporting whether the governor works silently never appears.

**`main()` in these scripts returns `None`, not 0.** Tests asserting an exit code
must assert "did not raise" plus a content check instead.

**Test fixtures: count placeholders from the column list.**
`) VALUES (` + `"?" * (12 + n)` hardcoded counts drift the moment a column is
added; build the list, then `len(cols)`.

**Migrate a fresh path, not a hand-rolled partial table.** `rollup.migrate()`
creates the full baseline only when `router_logs` is absent; a stub table missing
`workload_type` makes `NEW_TABLES_SQL` fail. To model a real upgrade, migrate a
scratch DB then `ALTER TABLE ... DROP COLUMN` the new columns.

## Rollout state

`scripts/budget_policy.yaml` ships **`enabled: false`** (design §6): full coverage,
no behavioural change. Arming it is a separate, deliberate operator step —
`enabled: false` is the rollback lever and is hot-reloaded (~30s TTL).

A missing policy file defaults to **enabled** (architectural-only allowlist).

## Verification that actually discriminates

Prefer assertions that fail when the guard is removed:

* For containment, assert the *unfiltered* chain DOES contain a plan model first —
  otherwise the test passes trivially whenever the chain is empty.
* For the classifier, use a tier-13 denial: that is the only configuration where
  the naive and correct implementations disagree.
* Exercise the real `route_task` and feed its actual `reason` through
  `classify_lane()`, rather than asserting on hand-written literals that can't
  catch a reason-string change.
