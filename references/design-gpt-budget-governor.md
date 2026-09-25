# Design — ChatGPT plan budget governor (B37)

Status: **design rev 2 / not implemented**. Supersedes rev 1's even-burn policy;
§10 records what changed and why. Awaiting the decisions in §9.

**Policy under design (operator's ask):** burn up to **50%** of the plan window on
routine work, **reserve the other 50%** for architectural work, and make that
percentage **variable** when the shape of upcoming work is known.

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

That is the boom-bust engine: one ordinary tool-bearing workload can drain a
week of plan capacity in a day.

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
  versus 3.1K for gpt-5.5 on non-tool work. The reserve must be measured in
  **budget share**, never call count.

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

1. Enforce the operator's split: routine work may consume at most **50%** of the
   plan window; **50% is reserved** for architectural work.
2. Make that percentage **variable by known work shape** (config + dated
   overrides), no code change or restart.
3. Close the actual leak (§1.1): govern the tool-bearing branch, which is what
   drains the window.
4. Protect the architectural lanes: tier-13 rethink/rearchitect and `force_model`
   are never paced out — they draw on the reserve.
5. Degrade predictably to `deepseek-v4-pro` (tier 8) — the downgrade already
   proven at 99.97% success.
6. Be observable and **fail open**.

**Non-goals**

- Not a cost governor. The plan is fixed-price; the lever is *when* capacity is
  spent, not dollars.
- Not a replacement for the 429 path. The governor paces; `mark_rate_limited`
  remains the reactive backstop.
- Not applied to `force_model` (operator intent wins) or compression (barred).
- Not a fix for the tool branch's highest-tier preference itself — see §9.4, a
  separate decision worth making anyway.

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

### 3.2 Policy: routine ceiling, architectural reserve, variable by shape

Expressed against the window (100% = full 7-day plan window):

```
routine_ceiling_percent   (default 50)   routine may consume up to this
architectural_reserve     (default 50)   = 100 - routine_ceiling_percent
```

**Routine pacing.** Routine consumption is paced linearly toward its ceiling, so
the ceiling is a *rate* limit rather than a cliff:

```
routine_allowance_now = routine_ceiling_percent
                        * (1 - reset_after_seconds / limit_window_seconds)

routine verdict (in this order):
  used >= hard_cap_percent (95)                      -> closed     (all classes)
  used > routine_allowance_now + slack (3)           -> paced out  (routine only)
  reset_after_seconds <= final_stretch_seconds (10h) -> open       (spend the tail)
  otherwise                                          -> open
```

**Daily form** — the operator's "by end of day" checkpoint. The window is 7 days,
so the daily pacing share is `routine_ceiling_percent / 7`:

| checkpoint | routine window budget allowed |
|---|---|
| end of day 1 | 7.1% of window |
| end of day 2 | 14.3% |
| end of day 4 | 28.6% |
| end of day 7 (window end) | **50.0%** ← the ceiling |

So "burn up to 50% and reserve the rest" is enforced continuously; by window end
routine has had its 50% and architectural retained 50% throughout. The reserve is
**never paced** — available from hour one, which is the point of reserving it.

**Architectural reserve** — available immediately, always, up to `hard_cap`:

| class | members | governed by |
|---|---|---|
| **architectural (reserve)** | tier-13+ rethink/rearchitect lane; `force_model` | hard cap only |
| **routine** | every other auto-routed plan-account request, **incl. the tool-bearing branch** | routine ceiling, paced |

**Variability by work shape** (the operator's explicit ask). Dated overrides
raise or lower the ceiling when the shape of work is known:

```yaml
routine_ceiling_percent: 50
schedule:
  - { from: 2026-09-28, ceiling_percent: 35, reason: "design sprint — hold reserve for architecture" }
  - { from: 2026-10-05, ceiling_percent: 70, reason: "bulk routine tool work expected" }
```

Resolved as: the **last** schedule entry whose `from` date has passed wins;
absent any, the top-level value applies. **Hot-reloaded** (same ~30s mechanism as
`experiments.yaml`), so the ceiling moves mid-window without a restart. A
tightening override takes effect immediately — the useful direction when work
shape changes.

**Why `slack` is now 3pp, not 15pp.** In rev 1 slack absorbed "don't throttle a
legitimate burst". With an explicit *ceiling* the operator has already chosen the
burst allowance, so slack only absorbs pacing noise; a large slack would silently
spend the reserve. The reserve boundary is now the control and must be tight.

#### 3.2.1 A constant conflict this design must avoid

`final_stretch_seconds` and `hard_cap_percent` are compatible only for certain
values. On an even burn the used-percentage at the start of the final stretch is
`100 * (1 - FINAL_STRETCH/window)`. If that exceeds the hard cap, the cap closes
first and **the final-stretch rule can never fire** — the tail is silently wasted
every window. Measured for the 7-day window at a 95% cap:

| `final_stretch` | even-burn used at stretch start | outcome |
|---|---|---|
| 2h | 98.8% | dead |
| 6h | 96.4% | dead |
| 8h | 95.2% | dead |
| **10h** | **94.0%** | **live** |
| 12h | 92.9% | live |

**Constraint: `final_stretch > (1 - hard_cap/100) * window_seconds`** — for 7d at
95% that is **> 8.4h**. Default is **10h**. This is an **asserted test invariant**
(§5.1), not a comment: the constants live in different config blocks and would
otherwise drift apart, and the failure mode is invisible.

### 3.3 Enforcement points — two hooks, because one branch returns early

rev 1 hooked only `_select_model(min_tier)`. **That misses the leak** (§1.1). Two
hooks are needed in `route_task`:

```
route_task()
 ├─ if requires_tools:                       # <-- THE ACTUAL LEAK (98.5% of the burn)
 │     tool_model = _select_tool_capable_model()   # highest tier; ignores min_tier
 │     [NEW] _apply_plan_governor(tool_model, klass=routine)
 │     return RoutingDecision(...)                 # early return, ~line 994
 │
 └─ else:
       min_tier = _estimate_min_tier(...)
       selected = _select_model(min_tier)
       [NEW] _apply_plan_governor(selected, klass=routine)
       return RoutingDecision(...)
```

**Hook 1 is the one that matters.**

Plus the escalation paths, so a paced-out plan model cannot be reached by
fallback either:

- `escalate_on_failure(...)` — respect the governor when `prefer_model` targets a
  plan model.
- `_build_tool_fallback_chain(...)` — must not *contain* a paced-out plan model.
  Care needed: the chain walks *up* in tier (`if tier > current_tier`), so for a
  routine request served by `deepseek-v4-pro` it would otherwise escalate
  straight into gpt — **the leak re-entering through the back door**.

**Account-keyed, not name-keyed.** `get_model_account(model)` → `"openai-codex"`;
`models_in_account("openai-codex")` → the whole family. A new plan model is then
governed automatically with no governor edit, mirroring the provider-level
`COMPRESSION_RESERVED_PROVIDERS` idiom.

**Class determination** reuses the classifier already in
`check-routing-stats.py` (distinguishes `force_model`, `escalation`, and tier-13+
`rethink_rearchitect`), extracted to a shared helper so routing and reporting
cannot disagree.

When paced out, re-select with the plan account excluded. Never return an empty
decision — fall through to the existing `all_exhausted` path.

Reason string must name cause and class, e.g.:

```
tool-bearing request — gpt-5.6-sol paced out by plan governor
  (routine used 71.2% vs ceiling 50% paced to 38.4%) — selected deepseek-v4-pro (tier 8)
```

### 3.4 The honest consequence (understand before enabling)

Routine tool-bearing demand is **3,374/day** (47,244 over 14d). A 50% routine
ceiling does not fund that. Sizing from the measured cap at ~52K tokens per plan
tool call:

| capacity estimate (§1.2) | 50% routine allowance | ≈ plan calls/wk | ≈ plan calls/day | share of routine tool demand |
|---|---|---|---|---|
| 46.7 Mtok/wk (Sep-21 event) | 23.4 Mtok/wk | ~447 | ~64 | **1.9%** |
| 163.4 Mtok/wk (Sep-12 event) | 81.7 Mtok/wk | ~1,564 | ~223 | **6.6%** |

**So a 50% ceiling effectively makes gpt an architectural-only resource.** Routine
tool work moves to `deepseek-v4-pro` almost entirely. That is the *intended*
tradeoff of "reserve the rest for bigger architectural things" — but it is a
large behavioural change and the operator should confirm it is what they mean
(§9.2). The 3.5× capacity uncertainty is exactly why the governor reads
`used_percent` instead of predicting from this table; these figures set
expectations only.

**Reserve adequacy:** architectural demand measures **14.4 Mtok/week** (tier-13+
lane 11.6 + force_model 2.8, from 30d of traffic) — **1.6× headroom** at the low
capacity estimate, **5.7×** at the high one. *At the low estimate the reserve is
adequate but not generous*, which is why `routine_ceiling_percent` must be a
tuning knob and calibration (§5.7) must check it.

### 3.5 Failure modes — fail open

| Failure | Behaviour |
|---|---|
| Probe network error / timeout | Last good cache; if none or stale → **treat as open** |
| Token expired (401) | Refresh once (reuse `_refresh_expired_codex_probe_token`); on failure → open |
| Unparseable body | Log once → open |
| `rate_limit.allowed == false` | **closed** — authoritative, overrides pacing |
| State DB unwritable | In-memory for the process; never block routing |

Stance: **an unreadable probe must never starve the plan.** Silent under-use is
worse than boom-bust because it is invisible. Every fail-open emits a warning log.

### 3.6 Configuration surface

New `scripts/budget_policy.yaml`, hot-reloaded like `experiments.yaml`:

```yaml
governor:
  enabled: true                     # master switch; false == today's behaviour
  accounts:
    openai-codex:
      routine_ceiling_percent: 50   # operator's lever; reserve = 100 - this
      hard_cap_percent: 95          # absolute ceiling for all classes
      slack_percent: 3              # pacing noise only (see §3.2)
      final_stretch_seconds: 36000  # 10h; must exceed (1 - hard_cap/100) * window
      reserve_lanes: [force_model, rethink_rearchitect]
  probe:
    min_interval_seconds: 300
    stale_after_seconds: 1800
  schedule: []                      # dated ceiling overrides (see §3.2)
```

Rollback = `enabled: false` (hot-reloaded ~30s). Same lever shape as the canary —
one operational idiom in this repo, not two.

### 3.7 Observability

- Routing reason carries class + pacing arithmetic (§3.3).
- New columns `router_logs.paced_out` (INTEGER DEFAULT 0) and
  `router_logs.budget_class` (TEXT), so both are queryable.
- `recovery_log` events `plan_paced_out` / `plan_closed` / `plan_probe_open`.
- CLI `router_ops.py plan-budget`: current reading, ceiling, paced allowance,
  reserve consumed, today's throttle count by class.
- Nightly digest line: window used %, routine ceiling, reserve headroom,
  throttled count.

## 4. Implementation plan

TDD throughout — RED first per batch, one test file per batch, commit per batch.

| Batch | Deliverable | Test file |
|---|---|---|
| **B37a** | `gpt_budget.py`: readings, `fetch_reading()`, cache/staleness, fail-open. No routing wiring | `test_gpt_budget_reading.py` |
| **B37b** | `evaluate_policy(reading, cfg, *, klass) -> Verdict` — pure function, §3.2/§3.3 tables | `test_gpt_budget_policy.py` |
| **B37c** | `budget_policy.yaml` loader, hot reload, schedule resolution | `test_gpt_budget_config.py` |
| **B37d** | **Hook 1** — `_apply_plan_governor` on the tool-bearing early-return path | `test_gpt_budget_tool_branch.py` |
| **B37e** | **Hook 2** — `_select_model(min_tier)` path | `test_gpt_budget_routing.py` |
| **B37f** | Escalation + `_build_tool_fallback_chain` containment | `test_gpt_budget_escalation.py` |
| **B37g** | Shared lane classifier extracted from `check-routing-stats.py` | `test_budget_classifier.py` |
| **B37h** | Observability: columns, events, `router_ops.py plan-budget` | `test_gpt_budget_observability.py` |
| **B37i** | Nightly digest line | `test_routing_stats_plan_budget.py` |
| **B37j** | Calibration replay; set `slack`; confirm reserve adequacy | `test_gpt_budget_calibration.py` |

Ordering: B37a–c are inert (no routing risk). **B37d lands before B37e
deliberately** — the tool branch is the actual leak and the highest-value hook.
Each batch: new tests green **plus** full suite green before commit (currently
782 passed / 1 pre-existing prod-DB failure, deselected as known-bad — not
"fixed").

## 5. Test plan

### 5.1 Unit — policy truth table (B37b, highest value)

- `used=0, elapsed=0` → open.
- Exactly at allowance (`used == ceiling*elapsed`) → open (boundary inclusive).
- `used > allowance + slack` → paced out (routine only).
- **Ceiling arithmetic:** `ceiling=50`, `elapsed=50` → allowance 25%, so
  `used=30` → **paced out** (a naive even-burn test would wrongly pass this).
- `used=96` → closed, **all classes** including architectural.
- `used=100, elapsed=19.11, reset_after=165300` — the real 2026-09-25 state →
  **closed**. *Regression test for the actual incident.*
- `used=94, reset_after=36000` → open (final stretch).
- **Invariant (§3.2.1):** assert
  `final_stretch_seconds > (1 - hard_cap/100) * secondary_window_seconds` for
  every account in the shipped config.
- `limit_window_seconds=0` → open (no division-guard throttle).
- **Architectural class while routine is paced out → open** (reserve wins).
- **Architectural class at `used=96` → closed** (hard cap beats reserve).
- `enabled=false` → open, always.

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

### 5.3 Schedule / variability (B37c)

- No entries → top-level ceiling.
- One passed entry → its ceiling.
- Multiple passed → **last** wins (ordering test).
- Future-dated entry → ignored (must not take effect early).
- Tightening override mid-window takes effect on reload without restart.
- Malformed entry → ignored with a warning, top-level ceiling retained (never fail
  closed on bad config).

### 5.4 Integration — the leak (B37d, the critical batch)

- **Tool-bearing + routine + paced out → `deepseek-v4-pro`, not gpt.** *The* test
  for the reported problem: a plain tool-bearing request must not take
  `gpt-5.6-sol` once routine is paced out.
- Tool-bearing + architectural (rethink/rearchitect prompt) + paced out →
  **still gpt-5.6-sol**.
- Tool-bearing + `force_model=gpt-5.5` + closed → honoured.
- `paced_out=1`, `budget_class='routine'` recorded.
- Tool branch with governor disabled → identical to pre-B37 behaviour (captured
  baseline decision set).
- Non-tool path (B37e) — same matrix.
- **No empty decision:** only a plan model available and it paced out → verify the
  `all_exhausted` fallback, not a crash or empty model.

### 5.5 Regression guards

- **`_build_tool_fallback_chain` containment:** for a routine request served by
  `deepseek-v4-pro`, the chain must **not** contain a paced-out plan model.
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

### 5.7 Calibration (B37j)

Replay recorded history through the policy; report what it *would* have served per
day by class. Success = a smoothed curve (no 700-then-6) **with the architectural
lanes never throttled**, and reserve consumption ≤ 50% so headroom is retained. If
the architectural lane is ever paced out during replay, the ceiling is too high and
must come down before enabling. Set `slack` from observed pacing noise.

## 6. Rollout

1. Land B37a–c with `enabled: false`. Inert.
2. Land B37d–j with `enabled: false`. Full coverage, still inert.
3. **Observe-only** (`enabled: false` + shadow log of decisions): real-traffic
   counterfactual before it can block anything. Recommended, cheap.
4. Enable at the **post-reset window start** (Sunday), so the first governed week
   begins from a clean window.
5. Watch `paced_out` by class + digest for one full window; confirm the reserve is
   not exhausted by architectural work.
6. Rollback: `enabled: false`.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Routine tool work loses gpt almost entirely (§3.4) | Surfaced for decision (§9.2); `routine_ceiling_percent` is the dial; observe-only quantifies it first |
| Under-use: throttling too hard wastes paid capacity | Calibration §5.7; `final_stretch` releases the tail; nightly reserve-headroom line |
| Back-door leak via the fallback chain | §5.5 containment test |
| Probe treats token expiry as "capped" (as it does today) | Fail-open on 401 + refresh; `source` visible |
| Governor masks a capability regression by silently downgrading | Every throttle logged with class + arithmetic, counted, surfaced nightly |
| Second mechanism diverging from `mark_rate_limited` | Composed by design: governor = predictive pacing, 429 = reactive backstop |
| Constant drift (§3.2.1) | Asserted invariant test |
| Ceiling override misfires (wrong date, bad edit) | Schedule resolution tests; malformed entry ignored, never fail-closed |

## 8. Open questions

1. **5h window semantics.** Design paces the 7d window. Should the 5h window get
   its own routine ceiling (it moves ~4.8× faster per unit time), or is a 429 on
   it sufficient? Current answer: rely on the reactive path for the 5h, pace only
   the 7d.
2. **Reset-credit interaction.** 3 spare "reset usage" redemptions exist. When
   `closed` with >24h remaining, should the governor *recommend* redeeming one?
   Out of scope for B37; candidate follow-up.
3. **Throttle-onset alert.** One-shot notification when routine first paces out in
   a window, rather than only in the nightly digest.
4. **Should the tool branch keep its highest-tier preference when *not* paced
   out?** See §9.4 — independent of the governor.

## 9. Decisions needed

1. **Ceiling semantics — confirm the reading.** "Burn up to 50% by end of day" was
   read as **(a): the routine ceiling is 50% of the 7-day window, paced linearly
   (~7.1%/day), leaving 50% reserved.** Alternatives are ~3.5× different in
   reserve size:
   - **(a) as implemented** — ceiling 50% of the *window*, paced. Reserve 50%.
   - **(b) per-day even share** — routine capped at ~7.1% of window *per day*,
     i.e. ~93% reserve.
   - **(c) front-loaded** — 50% of *remaining* budget each day (drains the week in
     ~3 days).
   One config value differs between (a) and (b); (c) is a different policy
   function. **Please confirm (a), or name the intended one** — the clarify prompt
   timed out, so (a) is what is specified above.
2. **Confirm §3.4 is acceptable:** a 50% ceiling makes gpt effectively
   architectural-only (~2–7% of routine tool calls keep it). Intended?
3. **Observe-only phase** before enabling, or enable directly post-reset?
4. **Separate fix for the tool branch?** `_select_tool_capable_model` ignores
   `min_tier`, so *every* tool request prefers the top plan model. Complexity-gating
   it (e.g. only ≥0.85 tool work may take a plan model) would reduce the leak at
   source and is arguably the cleaner fix — but it is a routing-behaviour change
   independent of the governor. Want it scoped as a follow-up?

## 10. Revision history

**rev 2 (this document) — policy and enforcement point both corrected after
measurement.**

- **Policy changed** from even-burn (symmetric, `slack=15pp`) to the operator's
  **routine ceiling + architectural reserve** (50/50 default, variable by
  schedule), because the operator specified a reserve model, not symmetry. `slack`
  dropped 15pp → 3pp accordingly (§3.2).
- **Enforcement point corrected.** rev 1 hooked only `_select_model(min_tier)` and
  assumed `complexity>=0.85 → gpt-5.5` was the main path. Measurement showed
  **98.5% of the capping window was the tool-bearing branch** (`gpt-5.6-sol`),
  which **returns early before `min_tier` is computed** and ignores `min_tier`
  entirely. rev 2 adds Hook 1 on that path (§3.3) — the actual fix.
- **Capacity modelling abandoned as unsound.** Two cap events imply 46.7 vs 163.4
  Mtok for "100%" — a **3.5× spread** — so rev 1's token arithmetic is not usable
  for control; only the endpoint's `used_percent` is. (This *confirms* rev 1's
  signal choice while invalidating its estimates.)
- **Reserve sized from measured demand:** architectural need is **14.4 Mtok/week**
  (tier-13+ 11.6 + force_model 2.8) → **1.6× headroom** at the low capacity
  estimate, tight enough to flag for calibration.
- **Added §3.4** quantifying the honest consequence for routine tool traffic.
- **Kept from rev 1:** endpoint-probe signal, account-keyed enforcement, fail-open
  semantics, reserve concept, and the §3.2.1 constant-conflict finding.
