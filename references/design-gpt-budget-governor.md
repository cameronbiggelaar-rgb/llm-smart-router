# Design — ChatGPT plan budget governor (B37)

Status: **design rev 4 / not implemented**. Supersedes rev 3's routine-ceiling +
architectural-reserve split; §10 records what changed and why. Awaiting the
decisions in §9.

**Policy under design (operator's decision, rev 4):** gpt is reserved for the
**architectural lane only** — "just keep it for the architectural lane." There is
no routine share: routine traffic, including all ordinary tool-bearing requests,
never reaches the plan.

## 1. Problem

The ChatGPT Plus plan is the scarcest resource in the estate: a fixed $20/mo
subscription with a hard, account-wide cap. Today the router spends it
reactively and then loses it entirely.

### 1.1 What actually drained the window (measured — corrects rev 1)

Replaying the capping window (2026-09-19 20:45 → 2026-09-21 04:52 UTC), by token
share:

| lane | model | calls | Mtok | share |
|---|---|---|---|---|
| **tool-bearing** | **gpt-5.6-sol** | **630** | **33.0** | **98.5%** |
| non-tool | gpt-5.5 | 116 | 0.4 | 1.1% |
| non-tool | gpt-5.6-luna | 12 | 0.2 | 0.4% |
| | **total** | **758** | **33.5** | |

**The leak is the tool-bearing branch, not complexity escalation.** This
overturns rev 1's core assumption (`complexity >= 0.85 → gpt-5.5` was believed to
be the main path).

Mechanism, verified in `router.py`:

- A tool-bearing request **returns early** (~line 994) via
  `_select_tool_capable_model()` — *before* `_estimate_min_tier()` is called
  (line 1005). The log reason proves it: tool requests record
  `tool-bearing request — selected proven tool-capable model …` with **no
  `tier N needed` clause**, while non-tool requests carry one.
- `_select_tool_capable_model()` deliberately picks the **highest-tier available**
  tool-capable model and **ignores `min_tier` entirely** (docstring: prefer "the
  MOST CAPABLE available … so tool work keeps the best reasoning quality
  (gpt-5.5 first)").
- So **every** tool-bearing request takes the top plan model whenever available,
  however ordinary the request. Tool-bearing volume is **3,374/day** (47,244
  over 14d). There is no complexity gate on this path at all.

**And it does worse than leak — it steals architectural traffic.** `gpt-5.6-sol`
is tier 13, so it always wins the highest-tier tool pick. Of **1,760** `sol` calls
in 30d:

| how `sol` was reached | calls |
|---|---|
| **tool branch** (ignores `min_tier`) | **1,267** |
| escalation | 489 |
| **architectural tier-13 lane** | **3** |
| `force_model` | 1 |

`sol` is *the* architectural model, and the architectural lane accounts for
**0.2%** of its calls. **83%** of `planning` traffic is tool-bearing (4,377 of
5,296). So when you ask for a rethink/rearchitect, the request is normally
claimed by the tool branch *before* the tier-13 architecture rule is ever
consulted — it lands on `sol` by accident, at 47.7K tokens/call, for reasons that
have nothing to do with the request being architectural.

That is the boom-bust engine: one ordinary tool-bearing workload can drain a
week of plan capacity in a day, and the fix cannot be tier-based alone because
the tier is computed *after* the branch that matters.

### 1.2 Why no internal capacity model is possible (validates §3.1)

Two cap events, same account, same plan:

| cap event | trailing-7d consumption |
|---|---|
| 2026-09-21 04:52 UTC | **46.7 Mtok** (1,406 calls) |
| 2026-09-12 ~23:59 UTC | **163.4 Mtok** (2,674 calls) |

A **3.5× spread** in what "100%" means. The budget is not a fixed token quota —
it is weighted (cached vs uncached input, model scope, 5h/7d interaction).
**Any internal counter would be wrong by up to 3.5×**, so the only trustworthy
signal is the endpoint's own `used_percent`. This is the single most important
design constraint, and it is why the governor is built on a probe, not a counter.

Supporting evidence:

- **Live `/usage` read** (`chatgpt.com/backend-api/wham/usage`, authoritative):
  `plan_type=plus`, `limit_reached=true`, `primary_window` 18000s (5h) at
  **0% used**, `secondary_window` 604800s (7d) at **100% used**, reset in ~46h.
- **Boom-bust in served traffic** (`gpt*` calls/day): `Aug 30: 1731 ·
  Aug 31: 1115 · Sep 5: 970 · Sep 6: 748 · Sep 11: 801 · Sep 12: 11 ·
  Sep 13: 424 · Sep 14: 613 · Sep 15: 35 · Sep 19: 151 · Sep 20: 494 ·
  Sep 21: 113 · (then dark)`.
- **Expensive but not fatal.** During the 4-day dark period 7,773/7,775 calls
  still succeeded (99.97%) by falling to `deepseek-v4-pro`. The current cost is
  *lost capability*, not lost service.
- **Architectural work is genuinely token-heavy:** `gpt-5.6-sol` averages
  **52.2K tokens/call**; the tier-13 and `force_model` lanes average ~50–62K,
  versus 3.1K for gpt-5.5 on non-tool work. The window must therefore be measured
  in **budget share**, never call count.

Existing partial mechanisms (do not duplicate):

- `CHATGPT_PLAN_RESERVED_FROM_COMPRESSION` / `COMPRESSION_RESERVED_PROVIDERS` —
  bars the plan from summarisation. Orthogonal: a *workload-class* ban, not a
  volume governor.
- `EXCLUDED_FROM_AUTO_ROUTING = {gpt-6-astra}` — explicit-lane only.
- `mark_rate_limited` / `mark_account_rate_limited` — *reactive*; fires only
  after a 429. The observed cooldown was **489,205s (~5.66 days)**, so the
  reactive path costs ~6 days of capability per event.
- `allowance.py` — reporting only, priced in Ollama credit dollars. Not a router
  gate; different account, different currency.

## 2. Goals / non-goals

**Goals**

1. Enforce the operator's decision: **gpt is reserved for the architectural lane
   only.** Routine work — including all ordinary tool-bearing traffic — never
   reaches the plan.
2. Make the boundary **config-driven** (the architectural tier / keyword set lives
   in `routing_table.yaml`), no code change or restart to adjust.
3. Close the actual leak (§1.1): govern the tool-bearing branch, which is what
   drains the window.
4. Protect the architectural lanes: tier-13 rethink/rearchitect and `force_model`
   always reach the plan. Escalation **inherits the originating request's class** —
   a failed architectural request may escalate into the plan; a failed routine
   request escalates only to non-plan models (otherwise the leak re-enters through
   the failover path).
5. Degrade predictably to `deepseek-v4-pro` (tier 8) — the downgrade already
   proven at 99.97% success.
6. Be observable and **fail open**.

**Non-goals**

- Not a cost governor. The plan is fixed-price; the lever is *what is entitled to
  it*, not dollars.
- Not a replacement for the 429 path. The governor gates by class; `mark_rate_limited`
  remains the reactive backstop.
- Not applied to `force_model` (operator intent wins) or compression (barred).
- Not a fix for the tool branch's highest-tier preference itself — see §9.4. That
  follow-up is now **done** as B38 (commit `2ad68dd`); the governor and the floor
  fix are complementary: the floor stops *ordinary* tool work reaching the plan,
  the allowlist stops whatever still arrives.

## 3. Architecture

### 3.1 Signal: authoritative window state, not an estimate

`fetch_reading()` reads the Codex `/usage` endpoint for per-window detail.
Justified by §1.2: capacity is unmodellable to ±3.5×.

**Integration detail (verified):** `_probe_codex_quota_restored` returns only a
`bool` — it collapses both windows to `worst_used < 100.0` and discards
per-window detail, so it cannot be used as-is. `fetch_reading()` must issue the
same request (`_codex_usage_probe_url` + `codex_account_headers(token)`) and parse
the `rate_limit.primary_window` / `secondary_window` and
`additional_rate_limits[]` blocks itself. Reuse the **URL/header helpers**, not
the bool wrapper. The wrapper also treats any model-scoped
`additional_rate_limits` at 100% as "not restored"; the reader must preserve that
per-model distinction rather than merging it away.

New module `scripts/gpt_budget.py`:

```python
@dataclass(frozen=True)
class WindowReading:
    used_percent: float          # 0..100
    limit_window_seconds: int
    reset_after_seconds: int
    reset_at: int                # absolute epoch

@dataclass(frozen=True)
class PlanReading:
    allowed: bool                    # endpoint's own verdict
    primary: WindowReading | None    # 5h
    secondary: WindowReading | None  # 7d
    scoped: dict[str, WindowReading] # additional_rate_limits, by model
    has_credits: bool
    fetched_at: float
    source: str                      # 'live' | 'cache' | 'stale' | 'unavailable'
```

- **Cache:** one reading per `probe.min_interval_seconds` (default 300). The
  governor must add **no per-request network call** — asserted by test (§5.2).
- **Staleness:** older than `probe.stale_after_seconds` (1800) → `unavailable`
  → fail open (§3.5).
- Persisted to `router_state.db` so a restart does not force a cold probe.

### 3.2 Policy: allowlist — the plan is for the architectural lane, full stop

**Operator decision (rev 4, supersedes the 50/50 split entirely): "Just keep it
for the architectural lane."** There is no routine ceiling to tune and no routine
share to pace, because **routine is not entitled to the plan account at all**:

```
plan_account_allowlist:
  - architectural            # the ONLY class permitted on the plan
routine_ceiling_percent: 0   # explicit: routine gets nothing
```

The whole of §3.2's earlier pacing apparatus (linear ramp, daily checkpoints,
`slack`, schedule overrides) existed to answer *"how much routine is allowed?"*.
The answer is now **none**, so that machinery is **deleted, not tuned**. A
`final_stretch` tail release is likewise removed: releasing the tail to *routine*
was the point of it, and routine is no longer eligible.

**Verdict, in this order — one rule now:**

```
architectural (allowlisted class):
    used >= hard_cap_percent (95)  -> closed     # genuine hard stop, all classes
    otherwise                       -> open      # never paced; always available
else (routine):
    used >= hard_cap_percent (95)  -> closed
    otherwise                       -> denied     # select outside the plan account
```

`hard_cap_percent` stays because an allowlist is not a licence to run the plan to
literal 100%: at 95% the last 5% absorbs in-flight requests that cannot be
redirected mid-call, and a hard 429 on a live architectural session is the exact
failure this design exists to prevent. It is the *only* number left to calibrate.

**What survives from earlier revisions:** the signal (§3.1, authoritative
`/usage`), account-keyed enforcement (§3.3), fail-open (§3.5), and the hard cap.
**What is deleted:** routine pacing, the daily ramp table, `slack`, and the
`schedule:` variability knob — all of which only made sense while routine had a
share to divide. `routine_ceiling_percent` is retained as an explicit `0` so the
intent is legible in config rather than implied by absence.

**Reserve is not a budget any more, it is a whole.** With no routine draw the plan
window serves architectural demand only, so "will the reserve last?" becomes "will
architectural demand alone exhaust the window?" — measured at **~0 Mtok/wk today**
(§3.4: the lane is dormant), against 46.7–163.4 Mtok/wk of capacity. That is the
*safest* configuration available at this capacity, and it is why removing routine
is better than rationing it: there is no rationing arithmetic left to get wrong.

**The architectural class must be detected from the prompt, not from `min_tier`.**
The canonical architectural signal is a *sub-type keyword match* in
`routing_table.yaml`:

```yaml
planning:
  default_tier: 7
  sub_types:
    - keywords: ["rethink", "rearchitect", "redesign the architecture",
                 "find failure states", "update the plans", "review the architecture"]
      tier: 13
    - keywords: ["architecture", "system design", "production", "scalable", "migration"]
      tier: 8
```

**But the detector as written is keyed on `task_type`, and that is a defect this
design must fix — measured:**

| prompt (identical) | classified `task_type` | `match_sub_type` |
|---|---|---|
| `"rearchitect the budget governor"` | `planning` | **13** |
| `"rearchitect the budget governor"` | `debugging` | **None** |
| `"rearchitect the budget governor"` | `coding` | **None** |
| `"design the architecture for it and implementation and test plans"` | `planning` | **8** (not 13) |

Two failures, both fatal to an allowlist:

1. **`match_sub_type` is scoped to one `task_type`.** It reads
   `table.get(task_type)` and returns `None` for everything else. Real traffic is
   **~30% `debugging`** and only **~11% `planning`** (last 13d; compression is the
   largest share at ~42%). An architectural
   request classified as `debugging` — the common case, since architectural work
   usually starts as "this is wrong, fix it" — is **invisible** to the detector.
   Production evidence: the tier-13 capability clause has fired **once in 208,530
   logged calls** (the single `"capability tier 13 needed"` row) — it is effectively
   dormant. Every other `sol` selection (1,760 total) is either the tool-bearing
   branch (1,267) or a tool-failover escalation (~490). An allowlist gated on a
   dormant detector would deny the plan to almost all genuine architectural work
   and silently starve the exact lane the operator asked to protect.
2. **The operator's own canonical sentence scores tier 8, not 13.**
   `"design the architecture for it and implementation and test plans"` matches the
   tier-8 `architecture` keyword, not the tier-13 list. It is a near-miss on
   wording, and wording is exactly what an allowlist must not depend on.

**Fix — the governor scans the sub-type table globally, not per-`task_type`:**

```
architectural(prompt) := any sub-type with tier >= 13 matches, across ALL
                         task types in routing_table.yaml
                      OR force_model is set
```

The tier-13 keyword list itself is unchanged (`rethink`, `rearchitect`, `redesign
the architecture`, `find failure states`, `update the plans`, `review the
architecture`, `perform a deep review`), but it is **hoisted out of the `planning`
namespace** so a `debugging`-classified request can still match it. This is a
small, testable change with a large blast radius if omitted: without it the
allowlist protects a class that almost never fires.

**The residual risk is real and must be stated:** a genuine architectural request
phrased as `"fix the tool failures"` still matches nothing and is treated as
routine. Keywords cannot be complete. Two mitigations, both cheap:
- **`force_model` is an unconditional escape hatch** — `force_model: gpt-5.6-sol`
  always reaches the plan, so the operator is never *blocked* from gpt on demand.
- **Calibration (§5.7) reports how often routine requests were denied a plan model
  they would previously have taken**, so over-denial is visible rather than silent,
  and the keyword list can be extended from evidence.

This is why the earlier `schedule:` ceiling knob is **deleted rather than
tuned** — with a dormant detector, a tunable ceiling would have looked
healthy while denying almost everything. Fixing the detector is the actual work.

### 3.3 Enforcement points — two hooks, because one branch returns early

rev 1 hooked only `_select_model(min_tier)`. **That misses the leak** (§1.1). Two
hooks are needed in `route_task`:

```
route_task()
 ├─ if requires_tools:                       # <-- THE ACTUAL LEAK (98.5% of the burn)
 │     tool_model = _select_tool_capable_model()   # highest tier; ignores min_tier
 │     [NEW] _apply_plan_governor(tool_model)  # classify from prompt → deny if routine
 │     return RoutingDecision(...)                 # early return, ~line 994
 │
 └─ else:
       min_tier = _estimate_min_tier(...)
       selected = _select_model(min_tier)
       [NEW] _apply_plan_governor(selected)   # classify from prompt → deny if routine
       return RoutingDecision(...)
```

**Hook 1 is the one that matters.**

Plus the escalation paths, so a denied plan model cannot be reached by
fallback either:

- `escalate_on_failure(...)` — respect the governor when `prefer_model` targets a
  plan model.
- `_build_tool_fallback_chain(...)` — must not *contain* a denied plan model.
  Care needed: the chain walks *up* in tier (`if tier > current_tier`), so for a
  routine request served by `deepseek-v4-pro` it would otherwise escalate
  straight into gpt — **the leak re-entering through the back door**.

**Account-keyed, not name-keyed.** `get_model_account(model)` → `"openai-codex"`;
`models_in_account("openai-codex")` → the whole family. A new plan model is then
governed automatically with no governor edit, mirroring the provider-level
`COMPRESSION_RESERVED_PROVIDERS` idiom.

**Class determination** — one class is permitted, so the classifier's only job is
to decide *architectural or not*, and it must be right for the operator's sake:

1. **Architectural** — a tier ≥13 sub-type matches **scanning every `task_type` in
   `routing_table.yaml`** (not one namespace — see the defect in §3.2), **or**
   `force_model` is set. Computed from `prompt`, so it is available on the tool
   branch too.
2. **Routine** — everything else, including every ordinary tool-bearing request.
3. **Escalation inherits** the originating request's class: a failed architectural
   request may escalate into the plan; a failed routine request may not (the
   escalation chain must not contain a denied plan model — §3.3, §5.5).

The lane names used for reporting (`force_model`, `escalation`,
`rethink_rearchitect`) come from the classifier already in
`check-routing-stats.py` (reason-string based). Extract that to a shared helper so
routing and reporting cannot disagree — but note the **routing** classifier must
additionally use the prompt match, because the reason string is not written yet at
enforcement time, and because the tool branch never records a tier clause.

When denied, re-select with the plan account excluded. Never return an empty
decision — fall through to the existing `all_exhausted` path.

Reason string must name cause and class, e.g.:

```
tool-bearing request — plan account denied to class 'routine' by plan governor
  (architectural allowlist; gpt is reserved for architectural loads)
  — selected deepseek-v4-pro (tier 8)
architectural request — plan account allowed (hard cap 95%, at 41.2%)
  — selected gpt-5.6-sol (tier 13)
```

### 3.4 The honest consequence (measured, current era)

With the allowlist, **routine never reaches the plan**, so the relevant question is
no longer "how much routine can we afford?" but "what does the plan serve today,
and what will it serve after?" Measured on the **clean era** — Sep 15–21, the 6
days *after* compression closed and *before* gpt went dark — because any earlier
window is polluted by compression, which was then the plan's single largest draw
(39.1 Mtok in Sep 12–15 alone, 56% of that era) and is now gone:

| lane served on the plan (clean era, 6d) | Mtok | share | after the allowlist |
|---|---|---|---|
| **tool-branch (theft)** | 28.6 | **97.6%** | **denied** → `deepseek-v4-pro` |
| escalation (tool-failover) | 0.1 | 0.5% | denied (it is the tool branch failing over, not architecture) |
| other tier-gated (planning/debugging via `min_tier`) | 0.6 | 1.9% | denied |
| **genuine architectural demand** | **0.0** | **0.0%** | **this is what the plan is for — and it is currently dormant** |

Three facts the operator should see plainly, because they are the whole argument:

1. **The plan serves essentially one thing today: the tool branch, 97.6%.**
   Denying routine removes virtually all of the plan's current draw at a stroke —
   which is exactly the operator's instruction, and the single biggest change this
   design makes.
2. **Compression needs no governor action.** It *was* the largest draw (39.1 Mtok
   in Sep 12–15), but the reserved-provider rule closed it: **1,341 gpt compression
   calls / 103.5 Mtok in Sep 10–15, and 0 since Sep 15**. It must not be
   double-counted as demand the governor must manage.
3. **The architectural lane is currently dormant — 1 call in 30 days.** The
   tier-13 `rethink`/`rearchitect` sub-type clause has fired **once across
   208,530 logged calls** (the single `"capability tier 13 needed"` row);
   escalation, once its compression component is stripped, is ~0.1 Mtok/6d and is
   itself tool-failover, not architecture. This is the finding the allowlist
   depends on: **"keep gpt for the architectural lane" is not a capacity problem
   at all** — the lane has no draw to ration. The plan will sit nearly idle
   (correctly — that is the point of reserving it) and be available in full the
   moment a genuine architectural session begins.

**Therefore the sizing conclusion inverts:** there is no headroom arithmetic to get
wrong, because genuine architectural demand is ~0 today, against 46.7–163.4 Mtok/wk
of capacity. The **residual risk is not capacity but detection** — that a genuine
architectural request is phrased `"fix the tool failures"`, matches no keyword, and
is denied while the plan sits idle. That is why §5.7 calibration reports denial
counts by class, and why `force_model` is the unconditional escape hatch: the
operator must never be blocked from gpt on demand, because the whole point of the
allowlist is to have gpt ready *for* those sessions, not to ration them.

### 3.5 Failure modes

**Revised by B44a (see §3.5.1).** The rule below held while the policy was a
*deny-list*: with no allowlist to consult, refusing on an unreadable probe would
have starved the plan for no gain.

| Failure | Behaviour |
|---|---|
| Probe network error / timeout | Last good cache; if none or stale → **`unverified`** (B44a) |
| Token expired (401) | Refresh once (reuse `_refresh_expired_codex_probe_token`); on failure → `unverified` |
| Unparseable body | Log once → `unverified` |
| `rate_limit.allowed == false` | **closed** — authoritative, overrides the allowlist |
| State DB unwritable | In-memory for the process; never block routing |
| Governor *bug* (exception in the hook) | **open** — a code fault must never wedge live traffic; logged |

Stance (original, deny-list era): **an unreadable probe must never starve the
plan.** Silent under-use is worse than boom-bust because it is invisible.

#### 3.5.1 B44a revision — under an allowlist, an unreadable probe is not consent

The stance above is right about the failure it names and wrong about the remedy
once an allowlist exists. "Never starve the plan" was written to protect the
*architectural* lane — the one class the policy explicitly names. As implemented
it protected **every** class: `evaluate_policy` returned `VERDICT_OPEN` on an
unreadable reading *before consulting the allowlist*, so routine traffic rode a
blind probe onto the plan. That is the leak (5,211 GPT rows served,
`plan_budget_denied = 0`, 4,392 of them routine) and it was invisible, because a
blind allow and a deliberate allow are the same row.

Revised rule: an unreadable reading yields a **third verdict**,
`VERDICT_UNVERIFIED`, and the allowlist still decides.

| reading | class | verdict | served? |
|---|---|---|---|
| unreadable | named by the allowlist (architectural) | `unverified` | **yes** — the plan is not starved |
| unreadable | not named (routine) | `denied` | no |
| readable, under cap | named | `open` | yes |
| readable, over cap | any | `closed` | no |
| any | any, `enabled: false` | `open` | yes — rollback unchanged |

An unverifiable entitlement is not consent; the lane the policy names is still
served, so the original intent is preserved where it was meant to apply. The
blind case is now **countable** (`router_logs.governor_fail_open`, B44c) rather
than invisible, and the report refuses to call the leak closed for a window it
could not check.

Rollback from the revision is the same single lever as from the governor itself:
`enabled: false` restores exact pre-B37 behaviour byte-for-byte.

### 3.6 Configuration surface

New `scripts/budget_policy.yaml`, hot-reloaded like `experiments.yaml`:

```yaml
governor:
  enabled: true                     # master switch; false == today's behaviour
  accounts:
    openai-codex:
      allowlist: [architectural]    # the ONLY class permitted on the plan (§3.2)
      routine_ceiling_percent: 0    # explicit: routine gets none
      hard_cap_percent: 95          # absolute ceiling for all classes
  probe:
    min_interval_seconds: 300
    stale_after_seconds: 1800
```

Rollback = `enabled: false` (hot-reloaded ~30s). Same lever shape as the canary —
one operational idiom in this repo, not two.

### 3.7 Observability

- Routing reason carries class + decision (§3.3). **B44b:** the reason string
  only claims the governor *allowed* a class when a verdict was actually
  reached; a blind probe says UNVERIFIED, and an ungoverned request says the
  class was recorded without a verdict.
- New columns `router_logs.plan_budget_denied` (INTEGER DEFAULT 0) and
  `router_logs.budget_class` (TEXT), so both are queryable. **B44c:** plus
  `router_logs.governor_fail_open` (INTEGER DEFAULT 0) — 1 marks a request
  served while the reading was unreadable, which is what makes a blind window
  countable instead of invisible.
- `recovery_log` events `plan_denied_routine` / `plan_closed` / `plan_probe_open`.
- CLI `router_ops.py plan-budget`: current reading, hard cap, class allowlist,
  window used %, denial count by class today. **B44c:** plus a `blind` column, a
  per-window blind count, and a report that refuses to say "leak closed" for a
  window containing blind rows (`--json` carries `blind_calls`).
- Nightly digest line: window used %, denial count by class, architectural call
  count, **and the blind count** — a digest that reports `routine denied: 0`
  without the blind count cannot distinguish a closed leak from an unchecked one.

## 4. Implementation plan

TDD throughout — RED first per batch, one test file per batch, commit per batch.

| Batch | Deliverable | Test file |
|---|---|---|
| **B37a** | `gpt_budget.py`: readings, `fetch_reading()`, cache/staleness, fail-open. No routing wiring | `test_gpt_budget_reading.py` |
| **B37b** | **`classify_request()`** — architectural vs routine, scanning **all** task types for a tier ≥13 sub-type (fixing the `planning`-only scoping defect, §3.2), plus `force_model` and escalation-inheritance | `test_gpt_budget_class.py` |
| **B37c** | `evaluate_policy(reading, cfg, *, klass) -> Verdict` — pure function, §3.2 truth table | `test_gpt_budget_policy.py` |
| **B37d** | `budget_policy.yaml` loader, hot reload | `test_gpt_budget_config.py` |
| **B37e** | **Hook 1** — `_apply_plan_governor` on the tool-bearing early-return path (the leak) | `test_gpt_budget_tool_branch.py` |
| **B37f** | **Hook 2** — `_select_model(min_tier)` path | `test_gpt_budget_routing.py` |
| **B37g** | Escalation + `_build_tool_fallback_chain` containment | `test_gpt_budget_escalation.py` |
| **B37h** | Shared lane classifier extracted from `check-routing-stats.py` (reporting side) | `test_budget_classifier.py` |
| **B37i** | Observability: columns, events, `router_ops.py plan-budget` | `test_gpt_budget_observability.py` |
| **B37j** | Nightly digest line | `test_routing_stats_plan_budget.py` |
| **B37k** | Calibration replay; confirm the allowlist denies routine without starving architecture | `test_gpt_budget_calibration.py` |

Ordering: B37a–d are inert (no routing risk). **B37b lands before the hooks because
class detection decides whether "keep gpt for architectural loads" actually
works** — if an architectural keyword is misclassified as routine it is denied the
plan, and the operator's requirement fails silently. **B37e lands before B37f
deliberately** — the tool branch is the actual leak and the highest-value hook.
Each batch: new tests green **plus** full suite green before commit (currently
782 passed / 1 pre-existing prod-DB failure, deselected as known-bad — not
"fixed").

**Not in B37, deliberately:** the §9.4 fix to `_select_tool_capable_model` (honour
`min_tier`). It is a routing-behaviour change that fixes the "steals architectural
traffic" defect at source, and it deserves its own batch + review rather than
riding in on the governor.

**RESOLVED — implemented as B38 (commit `2ad68dd`).** `_select_tool_capable_model`
now takes `min_tier` and returns the cheapest tool-capable model meeting it (cost
order is `MODEL_COST_ORDER`, which is price-based: `deepseek-v4-pro` tier 8 is
cheaper than `glm-5.3` tier 6). `min_tier` is now computed *above* the tool-branch
early return, and the branch composes it with a new `_architectural_floor(prompt)`
term — necessary because `_estimate_min_tier` scans only the request's own
`task_type` while the tier-13 sub-type lives solely under `planning`, so a genuine
architectural ask tagged `coding` scored 6 and would have been demoted off
`gpt-5.6-sol`. See `references/build-b38-tool-branch-tier-floor.md`.

## 5. Test plan

### 5.1 Unit — policy truth table (B37c, highest value)

- **Allowlist, architectural:** `used=0..94` → open; `used=96` → **closed**.
- **Allowlist, routine:** `used=0` → **denied** (the decisive change — routine is
  refused even in a pristine window, because it is not entitled to the plan at all).
- **Routine is never `open`:** for `used` in 0..94 and every `reset_after`, routine
  → denied. *Regression test against re-introducing a routine share.*
- `used=96` → closed, **all classes** including architectural.
- `used=100, elapsed=19.11, reset_after=165300` — the real 2026-09-21 state →
  **closed**. *Regression test for the actual incident.*
- `limit_window_seconds=0` → open for architectural (division guard; never
  throttle on bad input).
- `enabled=false` → open for **all** classes (rollback == today's behaviour).
- **Deleted-concept guards (assert the machinery is gone, not merely unused):**
  no `slack`/`final_stretch`/`schedule` key is read; a config carrying them still
  loads and they have **no effect** on the verdict.
- **Architectural detection (the operator's requirement, tested directly):**
  - `"rearchitect the budget governor"` → tier 13 → **architectural**.
  - `"let's rethink the budget approach"` → 13 → architectural.
  - `"review the architecture of the endpoint"` → 13 → architectural.
  - **The cross-`task_type` fix (§3.2 defect) — the highest-value tests here:**
    the *same* `"rearchitect the budget governor"` prompt must classify
    architectural when the request is classified `debugging`, `coding`,
    `session_compression`, or `research` — not only `planning`. A
    `planning`-only regression fails these.
  - `"write a quick plan for the canary"` → 5 → **routine** (must not be
    over-classified; a list that only greps "plan" would wrongly protect it).
  - `"refactor the monitor script"`/`"fix the failing test"` → routine.
  - Plain tool-bearing request with no architectural keyword → routine.
  - **Tool-bearing + architectural keyword → architectural** (the case the branch
    was stealing — §1.1). Must reach the plan, not be denied.
  - `force_model` → architectural regardless of prompt (the escape hatch).
  - Classifier reads `routing_table.yaml`, so a sub-type promoted to tier ≥13 by
    config becomes architectural **without a code change**; a test asserts a
    config-driven promotion is honoured.
  - **Known limitation asserted as a limitation:** `"fix the tool failures"` →
    routine. Pinned so the gap is documented, not discovered in production.

### 5.2 Unit — probe and cache (B37a)

- Valid body → parsed reading with both windows.
- `additional_rate_limits` → preserved per-model, not merged away.
- `allowed=false` → closed (authoritative override).
- 401 → refresh once, then fail open.
- Timeout → fresh-enough cache, else open.
- Missing `secondary_window` → primary governs; both missing → open.
- `source` reflects `live`/`cache`/`stale`/`unavailable` honestly (assert we never
  label a cached value `live`).
- **Hot-path guard:** second call within `min_interval_seconds` makes **zero** HTTP
  calls (counting fake client).

### 5.3 Config surface (B37d)

- Allowlist missing → defaults to `[architectural]`, never "all classes".
- Unknown class name in `allowlist` → architectural-only retained, warning logged
  (never fail open into permitting routine).
- `allowlist: []` → **no class permitted**; assert this is rejected as a
  misconfiguration (an empty allowlist would silently black-hole the plan).
- Tightening/loosening takes effect on reload without restart.
- Malformed YAML → ignore the file, keep last good config, warning (never fail
  closed on bad config, never fail *open into routine* either).

### 5.4 Integration — the leak (B37e, the critical batch)

- **Tool-bearing + routine → `deepseek-v4-pro`, not gpt.** *The* test for the
  reported problem: a plain tool-bearing request must not take `gpt-5.6-sol`.
- Tool-bearing + architectural (rearchitect prompt) → **still gpt-5.6-sol**.
- Tool-bearing + architectural + `used=96` → closed → non-plan model.
- Tool-bearing + `force_model=gpt-5.5` → honoured (bypasses the allowlist).
- `plan_budget_denied=1`, `budget_class='routine'` recorded.
- Tool branch with governor disabled → identical to pre-B37 behaviour (captured
  baseline decision set).
- Non-tool path (B37f) — same matrix.
- **No empty decision:** only a plan model available and it denied → verify the
  `all_exhausted` fallback, not a crash or empty model.

### 5.5 Regression guards

- **`_build_tool_fallback_chain` containment:** for a routine request served by
  `deepseek-v4-pro`, the chain must **not** contain a denied plan model.
  (Back-door leak — the chain escalates upward in tier.)
- `CHATGPT_PLAN_RESERVED_FROM_COMPRESSION` still holds, governor on or off.
- `EXCLUDED_FROM_AUTO_ROUTING` (`gpt-6-astra`) never auto-selected.
- Tool-capability gate still fails closed when no tool-capable model exists — the
  governor must not paper over it with a non-tool model.
- Flash canary (`v41flash-tool-canary`, percent 100) unaffected — governor touches
  plan-account models only.
- Future plan models governed **by account**, asserted with a synthetic registry
  entry.

### 5.6 Live verification (read-only)

1. `gpt_budget.fetch_reading()` vs a direct `/usage` curl must agree.
2. `router_ops.py plan-budget` shows the same used % as the endpoint.
3. While gpt is still dark: zero behaviour change for architectural class; all
   routine traffic continues to pro.
4. **Post-reset acceptance test:** the first gpt call lands, and the window is
   **not** at 100% within 24h — routine did not re-drain it. *Acceptance test for
   the whole change.*
5. Measure real capacity against §3.4's table and correct it from observation.

### 5.7 Calibration (B37k)

Replay recorded history through the policy; report what it *would* have served per
day by class. Success = a smoothed curve (no 700-then-6) **with the architectural
lane never denied, and routine denial counts tracked per day** so the operator can
see both the leakage removed and any over-denial of genuine architectural work. If
the architectural lane is ever denied during replay, the detector's keyword
coverage is too narrow and must be widened before enabling (§3.2).

## 6. Rollout

1. Land B37a–d with `enabled: false`. Inert.
2. Land B37e–k with `enabled: false`. Full coverage, still inert.
3. **Observe-only** (`enabled: false` + shadow log of decisions): real-traffic
   counterfactual before it can block anything. Recommended, cheap.
4. Enable at the **post-reset window start** (Sunday), so the first governed week
   begins from a clean window.
5. Watch `plan_budget_denied` by class + digest for one full window; confirm
   routine denial is high and architectural denial is zero.
6. Rollback: `enabled: false`.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Routine tool work loses gpt entirely (§3.4) | This is the operator's decision, the whole point of the change; observe-only quantifies it first |
| **Over-denial: a genuine architectural request missed by the keyword detector** | Calibration §5.7 reports denial-by-class; the detector gap is the residual risk (§3.4); `force_model` is the escape hatch |
| Back-door leak via the fallback chain | §5.5 containment test |
| Probe treats token expiry as "capped" (as it does today) | Fail-open on 401 + refresh; `source` visible |
| Governor masks a capability regression by silently downgrading | Every denial logged with class + decision, counted, surfaced nightly |
| Second mechanism diverging from `mark_rate_limited` | Composed by design: governor = class gate, 429 = reactive backstop |
| Detector scoped to one `task_type` and misses others (§3.2) | Cross-`task_type` scan; test matrix in §5.1 |
| Allowlist edit misfires (typo, empty list) | §5.3 config-surface tests; malformed/empty rejected, never fail-open into routine |

## 8. Open questions

1. **5h window semantics.** Design gates the 7d window. Should the 5h window get
   its own treatment (it moves ~4.8× faster per unit time), or is the 429 on it
   sufficient? Current answer: rely on the reactive path for the 5h, gate only the
   7d.
2. **Reset-credit interaction.** 3 spare "reset usage" redemptions exist. When
   `closed` with >24h remaining, should the governor *recommend* redeeming one?
   Out of scope for B37; candidate follow-up.
3. **Denial-onset alert.** One-shot notification when the first routine request is
   denied in a window (i.e. the allowlist is actually blocking), rather than only
   in the nightly digest.
4. **Should the tool branch keep its highest-tier preference when *not* denied?**
   See §9.4 — **RESOLVED as B38 (`2ad68dd`)**: it should not. The branch now honours
   `min_tier`, composed with an architectural floor. Consequence: 10.0% of
   tool-branch picks change (sol 1,267 / glm-5.3 476 / terra 224 / luna 2 → all
   `deepseek-v4-pro`), worth −$987.71 over the 30d window measured.

## 9. Decisions

1. **Policy model — RESOLVED by operator: "Just keep it for the architectural
   lane."** Supersedes the earlier "burn up to X% and reserve the rest" split.
   The design is now an **allowlist**: `[architectural]` only, routine gets zero.
   The earlier (a)/(b)/(c) ceiling-semantics question is moot — there is no routine
   share to pace.
2. **RESOLVED — architectural detection.** Class is architectural iff a tier ≥13
   sub-type matches scanning **all** `task_type`s (§3.2), or `force_model` is set.
   Escalation inherits the originating request's class (a failed architectural
   request may escalate into the plan; a failed routine one may not). The
   `planning`-only scoping defect is fixed as part of this batch (§3.2, §5.1).
3. **Observe-only phase** before enabling, or enable directly post-reset?
   *Recommendation: observe-only — it quantifies §3.4 on real traffic at zero
   risk, and GPT is dark until Sunday anyway.*
4. **Separate fix for the tool branch?** `_select_tool_capable_model` ignores
   `min_tier`, so *every* tool request prefers the top plan model — including
   1,267 of the 1,760 `sol` calls that should have been tier-gated. This is
   arguably the cleaner root-cause fix (honour `min_tier` as a floor, keeping the
   architectural tier-13 lane reachable) and would make the governor's routine
   class far smaller. Independent of the governor; scoped as a follow-up if
   wanted. **It is also the change that fixes the "steals architectural traffic"
   defect in §1.1**, so it is worth doing regardless of the governor.

## 10. Revision history

**rev 5 — B44a/b/c: the governor's silent fail-open, closed and made countable.**

- **The defect, stated correctly.** The governor was armed and wired in on both
  hooks (B37e/B37f), and it recorded `plan_budget_denied = 0` on every one of the
  5,211 GPT rows ever logged — 4,392 of them `budget_class='routine'`. It was not
  missing; it was **blind**. `_read_usage_payload()` swallows every exception →
  `Reading(source='unavailable')` → `evaluate_policy()` returned `VERDICT_OPEN`
  at the `unavailable` branch **before the allowlist was consulted**. The plan was
  served, the class stamped, nothing logged. `plan_budget_denied = 1` exists on
  132 rows, all on 2026-09-27, all from a hard-cap window, never from the
  allowlist.
- **Every other exit was excluded by the data:** `enabled=false` would leave the
  class empty (37,598 rows carry one); both exception exits log a warning and the
  journal has none; `_is_plan_model` is **True** for the bare names the DB holds,
  which kills the `:codex`-suffix theory (there is no `:codex` in the data).
  Established by elimination over the exit set — **not** by observing the probe
  fail inside the endpoint process, which is precisely why making it observable
  (below) is the load-bearing part.
- **§3.5 revised** (new §3.5.1): under an allowlist an unreadable probe is not
  consent. `VERDICT_UNVERIFIED` is the third verdict; the allowlist still decides.
  Routine is denied, the named architectural lane is served but flagged, and
  `enabled: false` still opens everything byte-for-byte.
- **Reason strings (§3.7) no longer assert a verdict that never happened.** The
  old suffix appended `; plan governor allowed class 'X'` whenever a class was
  truthy — on the short-circuit and the fail-open both. That sentence is how
  4,392 served routine GPT rows read as deliberate allows. Three facts now get
  three sentences: UNVERIFIED / allowed / recorded-without-a-verdict.
- **Observability (§3.7) gained the third state:**
  `router_logs.governor_fail_open` (both migration paths, additive, default 0 so
  historical rows never read as blind), threaded from
  `RoutingDecision.governor_fail_open` at all five log sites, plus a `blind`
  column and count in `router_ops.py plan-budget` and `--json`. The report now
  refuses to print "leak closed" for a window containing blind rows. Verified
  against a copy of the production DB: 61 → 62 columns, 145,427 rows preserved,
  idempotent.
- **`_plan_denied_for_class`** became a membership test over
  `{denied, closed}` instead of `!= open` — the negated form silently grew a
  third answer with the new verdict and would have stripped plan models from
  fallback chains we were still using.
- **Honest limit:** the blind probe was never captured failing inside the
  endpoint process. The fix's value is that it can now be seen when it happens.

**rev 4 — policy becomes an allowlist: architectural lane only.**

- **Policy changed** from rev 3's routine-ceiling + architectural-reserve split to
  an **allowlist**: `[architectural]` only, `routine_ceiling_percent: 0`. Routine
  traffic — including ordinary tool-bearing requests — is denied at any window
  level, not merely paced (§3.2). This is the operator's decision, taken verbatim:
  *"Just keep it for the architectural lane."*
- **Deleted the pacing apparatus:** the linear ramp, daily checkpoints, `slack`,
  `final_stretch` tail release, and the `schedule:` variability knob. None of them
  answer a question that still exists — with routine fully excluded there is no
  routine share to pace. `hard_cap_percent` remains as the only arithmetic (§3.2,
  §3.6).
- **Fixed a classification defect the allowlist would have made fatal.** The
  detector keyed architectural detection on `task_type='planning'`, so a
  `"rearchitect …"` prompt classified `debugging`/`coding`/`session_compression`
  (~30% `debugging` alone) was invisible — and an allowlist built on it would *deny*
  the requests it exists to protect. The classifier now scans **all** `task_type`s in
  `routing_table.yaml` for a tier ≥13 sub-type (§3.2, §3.3, §5.1).
- **Consequence re-measured on the clean era** (Sep 15–21, after compression closed
  and before gpt went dark): the plan serves the tool branch (97.6%), then
  escalation (~0.5%, itself tool-failover), then tier-gated planning/debugging
  (1.9%). **Genuine architectural demand is ~0 — the tier-13 clause has fired once
  across 208,530 calls.** Compression (once the largest draw) is already
  closed by the reserved-provider rule (0 calls since Sep 15). The residual risk is
  detection, not capacity: over-denial of a mis-worded architectural request (§3.4).
- **Escalation reclassified.** Escalation is tool-failover, not architecture, so it
  no longer counts as architectural; it *inherits* the originating request's class
  (a failed routine request must not fail over into the plan — the leak re-entering
  through the back door).
- **Known limitation documented rather than hidden:** `"fix the tool failures"`
  matches no keyword and is denied (§5.1); `force_model` remains the escape hatch.

**rev 3 — architectural class from prompt, reserve resize, batch reorder.** (Changes
were folded into the text at commit `295aed6`: architectural class detected from
`prompt` via `match_sub_type` ≥13, reserve re-sized to 12.6 Mtok/wk, B37b reordered
to the classifier.)

**rev 2 — policy, enforcement point, and reserve sizing all
corrected after measurement.**

- **Policy changed** from even-burn (symmetric, `slack=15pp`) to the operator's
  **routine ceiling + architectural reserve** (50/50 default, variable by
  schedule), because the operator specified a reserve model, not symmetry. `slack`
  dropped 15pp → 3pp accordingly (§3.2).
- **Enforcement point corrected.** rev 1 hooked only `_select_model(min_tier)` and
  assumed `complexity>=0.85 → gpt-5.5` was the main path. Measurement showed
  **98.5% of the capping window was the tool-bearing branch** (`gpt-5.6-sol`),
  which **returns early before `min_tier` is computed** and ignores `min_tier`
  entirely. rev 2 adds Hook 1 on that path (§3.3) — the actual fix.
- **Found that the tool branch also *steals* architectural traffic.** `sol` is
  tier 13 so it always wins the highest-tier pick; of 1,760 `sol` calls, **1,267
  came from the tool branch and only 3 from the architectural tier-13 lane**
  (§1.1). 83% of `planning` is tool-bearing. This is why the architectural class
  must be detected from `prompt`, not `min_tier` (§3.2.1).
- **Class detection specified concretely:** `match_sub_type(task_type, prompt)`
  returning a tier ≥13 sub-type — the `planning` rethink/rearchitect keywords —
  rather than a new keyword list (§3.2.1, §3.3).
- **Reserve sizing corrected (was wrong).** rev 2's earlier 14.4 Mtok/wk figure
  was measured from *served* traffic and therefore mostly counted the tool-branch
  leak. Re-measured on the genuine lanes: **12.6 Mtok/wk** (escalation 10.5 +
  force_model 2.0 + architecture sub-type 0.1) → **1.9× headroom** at the low
  capacity estimate. §3.4 also notes escalation is the reserve lane to watch and
  why §5.5's fallback-chain containment test protects it.
- **Capacity modelling abandoned as unsound.** Two cap events imply 46.7 vs 163.4
  Mtok for "100%" — a **3.5× spread** — so rev 1's token arithmetic is not usable
  for control; only the endpoint's `used_percent` is. (This *confirms* rev 1's
  signal choice while invalidating its estimates.)
- **Added §3.4** quantifying the honest consequence for routine tool traffic;
  operator accepted it (§9.2).
- **Kept from rev 1:** endpoint-probe signal, account-keyed enforcement, fail-open
  semantics, reserve concept, and the §3.2.1 constant-conflict finding.
