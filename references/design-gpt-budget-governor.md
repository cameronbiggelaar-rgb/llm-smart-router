# Design — ChatGPT plan budget governor (B37)

Status: **design / not implemented**. Awaiting approval on the decisions in §9.

## 1. Problem

The ChatGPT Plus plan is the scarcest resource in the estate: a fixed $20/mo
subscription with a hard, account-wide usage cap. Today the router spends it
reactively and then loses it entirely.

Evidence (measured, 2026-09-25):

- **Live `/usage` read** (`chatgpt.com/backend-api/wham/usage`, authoritative):
  `plan_type=plus`, `limit_reached=true`,
  `primary_window` 18000s (5h) at **0% used**,
  `secondary_window` 604800s (7d) at **100% used**, reset in ~46h.
- **Boom-bust pattern** in served traffic (`final_model`):
  `Sep 1: 339 · Sep 2: 680 · Sep 3: 115 · Sep 4: 387 · Sep 5: 490 ·
   Sep 6: 288 · Sep 7: 16 · Sep 8: 3 · Sep 9: 6 · Sep 10: 10 ·
   Sep 11: 716 · Sep 12: 6 ...`
  A heavy day is followed by near-zero for days.
- **No governor exists.** `_select_model` routes to `gpt-5.5` whenever
  `complexity_score >= 0.85` and the model is available. Nothing throttles it
  before the cap. Verified: no daily/period budget check in `router.py`.
- **The cap is expensive but not fatal.** During the 4-day dark period,
  7,773/7,775 calls still succeeded (99.97%) by falling to `deepseek-v4-pro`.
  So the cost of the current design is *lost capability*, not lost service.

Existing partial mechanisms (do not duplicate):

- `CHATGPT_PLAN_RESERVED_FROM_COMPRESSION` / `COMPRESSION_RESERVED_PROVIDERS` —
  bars the plan from summarisation. Orthogonal: that is a *workload class* ban,
  not a volume governor.
- `EXCLUDED_FROM_AUTO_ROUTING = {gpt-6-astra}` — explicit-lane only.
- `mark_account_rate_limited` — *reactive*, fires only after a 429.
- `allowance.py` — reporting only, and priced in Ollama credit dollars. Not a
  router gate; different account and different currency.

## 2. Goals / non-goals

**Goals**

1. Stop the boom-bust: spread plan capacity across the whole window instead of
   burning it in a day and losing it for the rest of the week.
2. Protect the top-tier lanes: the rethink/rearchitect trigger (tier 13,
   `gpt-5.6-sol`) and explicit `force_model` keep access regardless of pacing.
3. Degrade gracefully and predictably to `deepseek-v4-pro` (tier 8) when paced
   out — the downgrade the estate already proved at 99.97% success.
4. Be observable: every throttle decision must be visible in the routing reason
   and in a queryable counter.
5. Fail **open** — an unreadable probe must never starve gpt; it must revert to
   today's behaviour.

**Non-goals**

- Not a cost governor. The plan is fixed-price; the lever is *when* capacity is
  spent, not how many dollars it costs.
- Not a replacement for the 429 path. The governor paces; `mark_rate_limited`
  still handles an unexpected upstream cap.
- Not applied to `force_model` (operator intent always wins) or compression
  (already barred).
- Not applied to Ollama accounts in this batch (the shape is generic; the first
  consumer is the ChatGPT plan).

## 3. Architecture

### 3.1 Signal: authoritative window state, not an estimate

The decisive design choice: **do not estimate capacity.** The Codex `/usage`
endpoint already reports both windows' `used_percent`, `reset_after_seconds`
and `limit_window_seconds`. Reuse the proven probe path
(`_probe_codex_quota_restored` / `_codex_usage_probe_url` in
`hermes_cli/auth_codex.py`).

**Integration detail (verified):** `_probe_codex_quota_restored` returns only a
`bool` — it collapses both windows to `worst_used < 100.0` and discards the
per-window detail. It cannot be used as-is; `fetch_reading()` must issue the same
request (`_codex_usage_probe_url` + `codex_account_headers(token)`) and parse the
`rate_limit.primary_window` / `secondary_window` and
`additional_rate_limits[]` blocks itself. It should *reuse the URL/header
helpers*, not the bool-returning wrapper. Note also that the wrapper treats any
model-scoped `additional_rate_limits` at 100% as "not restored" — the reader must
preserve that per-model distinction rather than merging it away.

Why this matters: any internal request counter would need a hand-calibrated
"calls per window" constant, which drifts as prompt sizes change. The endpoint
is ground truth and needs no calibration.

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
    allowed: bool                # endpoint's own verdict
    primary: WindowReading | None    # 5h
    secondary: WindowReading | None  # 7d
    has_credits: bool
    fetched_at: float            # local monotonic-ish epoch
    source: str                  # 'live' | 'cache' | 'stale' | 'unavailable'
```

- **Cache:** one reading per `PROBE_MIN_INTERVAL` (default 300s). The
  credential pool already throttles its own probe; the governor must not add
  hot-path network calls per request.
- **Staleness:** a reading older than `STALE_AFTER` (default 1800s) is treated
  as `unavailable`, which fails open (§3.5).
- **Single-writer:** the probe result is cached in-process and persisted to
  `router_state.db` so a router restart does not force a cold probe.

### 3.2 Policy: even-burn pacing with a protected reserve

For each *binding* window (the one with the higher `used_percent`, normally the
7-day secondary):

```
elapsed_percent = 100 * (1 - reset_after_seconds / limit_window_seconds)
ahead_by        = used_percent - elapsed_percent      # +ve = burning too fast
```

Decision table (evaluated per request, before selecting a plan-account model):

| Condition | Verdict | Rationale |
|---|---|---|
| `used_percent >= HARD_CAP` (default 95) | **closed** | Leave headroom for P0 |
| `ahead_by > SLACK` (default 15pp) | **paced out** | Burning faster than the window |
| `reset_after_seconds <= FINAL_STRETCH` (default 10h) | **open** | Window about to expire; unused capacity is lost, so spend freely |
| otherwise | **open** | On or behind the even-burn line |

Rules are evaluated **in the order shown**; the hard cap is checked first so a
capped window can never be reopened by the final-stretch rule (see §3.2.1).

The `FINAL_STRETCH` rule is deliberate: allowance does not roll over, so
withholding capacity in the last hours of a window is pure waste. It also gives
a natural recovery point each week.

#### 3.2.1 A conflict this design must avoid (found by validating against the real incident)

`FINAL_STRETCH` and `HARD_CAP` are only compatible for a certain window length.
On an *even* burn, the used-percentage at the start of the final stretch is
`100 * (1 - FINAL_STRETCH/window)`. If that exceeds `HARD_CAP`, the hard cap
closes the window first and **the final-stretch rule can never fire** — the
allowance tail is silently wasted on every window.

Measured against the 7-day window with `HARD_CAP=95`:

| `final_stretch` | even-burn used at stretch start | outcome |
|---|---|---|
| 2h | 98.8% | stretch dead (cap closed) |
| 4h | 97.6% | stretch dead |
| 6h | 96.4% | stretch dead |
| 8h | 95.2% | stretch dead |
| **10h** | **94.0%** | stretch live |
| 12h | 92.9% | stretch live |

**Constraint: `final_stretch > (1 - HARD_CAP/100) * window_seconds`**, which for a
7-day window at 95% is **> 8.4h**. Default is therefore **10h**, not 6h. This
must be an asserted invariant in tests (§5.1), not a comment — the two constants
live in different config blocks and will otherwise drift apart.

This policy *would have prevented the current outage*. Replayed against the real
burn (window start 2026-09-19 20:45 UTC, cap hit 2026-09-21 04:52 UTC = 32.11h
in, i.e. 100% used at 19.11% elapsed, `ahead_by` = **80.89pp**), a 15pp slack
throttles at **t+5.97h** (used 18.6%, elapsed 3.6%) — preserving **26.1h** of
window that was actually lost to the cap.

**The core tradeoff, stated plainly:** pacing means deliberately sending fewer
requests to gpt early in the window than capacity would allow, in exchange for
having it available all week. That is the entire point of the change — it trades
peak capability for continuous capability. The slack sweep is the dial:

| slack | throttles at | preserves |
|---|---|---|
| 5pp | t+2.0h (used 6.2%) | 30.1h |
| 10pp | t+4.0h (used 12.4%) | 28.1h |
| **15pp** | **t+6.0h (used 18.6%)** | **26.1h** |
| 25pp | t+9.9h (used 30.9%) | 22.2h |
| 30pp | t+11.9h (used 37.1%) | 20.2h |

Note the shape: the slack dial trades *how much of the window stays usable*
against *how much burst capacity is allowed before throttling*. Calibration
(§5.6) should set it from real data rather than this single replay.

### 3.3 Priority classes

| Class | Members | Governor applies? |
|---|---|---|
| **P0 — reserved** | `force_model`; rethink/rearchitect lane (tier 13); `EXCLUDED_FROM_AUTO_ROUTING` explicit lane | **No** — always allowed |
| **P1 — paced** | auto-selected plan-account models, i.e. `complexity >= 0.85` → gpt-5.5, and any future auto-reachable plan model | **Yes** |
| **P2 — barred** | compression workloads | N/A (already barred upstream) |

P0 bypass keeps the operator's directive intact: the $20/mo stays available for
"the bigger problems" even when the governor is throttling routine escalation.

### 3.4 Enforcement point

**One choke point, keyed by account — not by model name.**

```
route_task()
  └─ _select_model(min_tier)            # unchanged: cheapest model >= min_tier
       └─ [NEW] _apply_plan_governor()  # post-filter on the candidate
```

Insert the governor *after* selection and *before* returning, plus inside the
two escalation ladders (`escalate_on_failure`, `_build_tool_fallback_chain`
construction) so a paced-out plan model cannot be reached by escalation either.

Keying by account reuses machinery that already exists:

- `get_model_account(model)` → `"openai-codex"` for every plan model.
- `models_in_account("openai-codex")` → the whole family.

Consequence: **a new plan model added to `MODEL_REGISTRY` is governed
automatically**, with no governor edit. This mirrors the deliberate
provider-level design of `COMPRESSION_RESERVED_PROVIDERS`.

When a candidate is paced out, the governor re-selects with the plan account
excluded from the candidate set. It must **not** return an empty decision — if
nothing remains, fall through to the existing `all_exhausted` path.

Reason string must name the cause, e.g.:

```
capability tier 10 needed, gpt-5.5 paced out by plan governor
  (used 71% vs 38% elapsed, slack 15pp) — selected deepseek-v4-pro (tier 8)
```

### 3.5 Failure modes — fail open

| Failure | Behaviour |
|---|---|
| Probe network error / timeout | Use last good cache; if none or stale → **treat as open** (today's behaviour) |
| Token expired (401) | Attempt refresh once (reuse `_refresh_expired_codex_probe_token`); on failure → open |
| Endpoint returns unparseable body | Log once, → open |
| `rate_limit.allowed == false` | **closed** — this is authoritative and overrides pacing |
| State DB unwritable | Operate in-memory for the process; never block routing |

Explicit stance: **an unreadable probe must never starve the plan.** Silent
under-use is a worse failure than the boom-bust it replaces, because it is
invisible. Every fail-open must emit a warning log.

### 3.6 Configuration surface

New `scripts/budget_policy.yaml`, hot-reloaded like `experiments.yaml` (the
pattern the operator already uses and can roll back without a restart):

```yaml
governor:
  enabled: true                 # master switch; false == today's behaviour
  accounts:
    openai-codex:
      hard_cap_percent: 95      # never cross; leaves P0 headroom
      slack_percent: 15         # allowed lead over the even-burn line
      final_stretch_seconds: 36000   # last 10h: spend freely (must exceed
                                     # (1 - hard_cap/100) * window; asserted in tests)
      reserve_priority: true    # P0 (force_model, tier 13) always allowed
  probe:
    min_interval_seconds: 300
    stale_after_seconds: 1800
```

Rollback = `enabled: false` (hot-reloaded ~30s). Same lever shape as the canary,
so there is one operational idiom in this repo, not two.

### 3.7 Observability

- **Routing reason** (§3.4) on every throttle — visible in
  `router_logs.reason` with no schema change.
- **New column** `router_logs.paced_out` (INTEGER DEFAULT 0) so volume is
  queryable: `SELECT count(*) ... WHERE paced_out=1`.
- **`recovery_log`** events `plan_paced_out` / `plan_closed` / `plan_probe_open`,
  reusing the existing table.
- **CLI:** `router_ops.py plan-budget` printing the current reading, the
  even-burn line, the verdict, and today's throttle count.
- The existing `router-lanes-nightly` digest gains a line: plan window used %,
  even-burn line, throttled count.

## 4. Implementation plan

TDD throughout — RED test first per batch, one test file per batch, commit per
batch (the operator's stated convention).

| Batch | Deliverable | Test file (new unless noted) |
|---|---|---|
| **B37a** | `gpt_budget.py`: `PlanReading`/`WindowReading`, `fetch_reading()`, cache + staleness, fail-open paths; **no routing wiring** | `test_gpt_budget_reading.py` |
| **B37b** | `evaluate_policy(reading, cfg, *, priority) -> Verdict` — pure function implementing §3.2/§3.3 decision table | `test_gpt_budget_policy.py` |
| **B37c** | `budget_policy.yaml` loader with hot reload, mirroring `load_experiments()` | `test_gpt_budget_config.py` |
| **B37d** | Wire `_apply_plan_governor()` into `route_task` post-selection; account-keyed exclusion; reason string | `test_gpt_budget_routing.py` |
| **B37e** | Wire into `escalate_on_failure` + fallback-chain construction (plan model unreachable while paced) | `test_gpt_budget_escalation.py` |
| **B37f** | Observability: `paced_out` column, `recovery_log` events, `router_ops.py plan-budget` | `test_gpt_budget_observability.py` |
| **B37g** | Nightly digest line | `test_routing_stats_plan_budget.py` |
| **B37h** | Calibration pass: confirm the policy reproduces a *smoothed* week against recorded history; tune slack | `test_gpt_budget_calibration.py` |

Ordering rationale: B37a–c are pure/inert (no live-routing risk), so the risky
wiring (B37d–e) lands only after the policy is pinned by tests.

Each batch: `python3 -m pytest tests/<new file> -q` green, **plus** the full
suite green before commit (currently 782 passed / 1 pre-existing prod-DB
failure, which must be deselected as a known-bad assertion, not "fixed").

## 5. Test plan

### 5.1 Unit — policy truth table (B37b, highest value)

Table-driven over the §3.2 grid. Non-negotiable cases:

- `used=0, elapsed=0` → open (start of window, not "ahead").
- `used=50, elapsed=50` → open (exactly on the line; boundary is inclusive-open).
- `used=66, elapsed=50` → paced out (16pp > 15pp slack).
- `used=65, elapsed=50` → open (exactly at slack; must not throttle).
- `used=96` → closed regardless of elapsed.
- `used=100, elapsed=19.11, reset_after=165300` (the real 2026-09-25 state,
  replayed from the actual window start) → **closed**. *Regression test for the
  actual incident.*
- `used=94, reset_after=36000` → open (final stretch beats an otherwise-closing
  window).
- **Invariant test (§3.2.1):** assert
  `final_stretch_seconds > (1 - hard_cap_percent/100) * secondary_window_seconds`
  for every account in the shipped config. This must fail loudly if the two
  constants are ever edited apart, because the failure mode (a permanently dead
  final-stretch rule) is otherwise invisible.
- `limit_window_seconds=0` → open (cannot compute elapsed; never throttle on a
  division guard).
- P0 request while closed → open (reserve wins).
- `enabled=false` → open, always.

### 5.2 Unit — probe and cache (B37a)

- Valid body → parsed `PlanReading` with both windows.
- `allowed=false` → `verdict closed` (authoritative override).
- HTTP 401 → refresh attempted once, then fail open.
- Timeout / connection error → stale cache if fresh enough, else open.
- Body missing `secondary_window` → primary governs; both missing → open.
- `source` field reflects `live` / `cache` / `stale` / `unavailable` honestly —
  a test asserting we never label a cached value `live`.
- Cache: second call within `min_interval_seconds` makes **no** HTTP call
  (assert via a counting fake client — this is the hot-path guard).

### 5.3 Integration — routing (B37d/e)

- `complexity=0.9` + closed → selects `deepseek-v4-pro`, reason contains
  `paced out`, `paced_out` column set to 1.
- `complexity=0.9` + open → selects `gpt-5.5` (no regression when open).
- rethink/rearchitect prompt + closed → still `gpt-5.6-sol` (P0 bypass).
- `force_model=gpt-5.5` + closed → honoured.
- Escalation from `flash` with `prefer_model=gpt-5.5` while closed → lands on
  `deepseek-v4-pro`, **not** gpt, and not an empty decision.
- Fallback chain for a non-plan model must not *contain* a paced-out plan model.
- `enabled=false` → byte-identical decisions to pre-B37 behaviour (guard test:
  compare against a captured baseline decision set).

### 5.4 Regression guards (must not break)

- `CHATGPT_PLAN_RESERVED_FROM_COMPRESSION` still holds — compression never
  selects plan models, with the governor on or off.
- `EXCLUDED_FROM_AUTO_ROUTING` (`gpt-6-astra`) still never auto-selected.
- Tool-capability gate still fails closed when no tool-capable model is
  available (the governor must not paper over that with a non-tool model).
- The flash canary (`v41flash-tool-canary`, percent 100) is unaffected — the
  governor touches plan-account models only.

### 5.5 Live verification (read-only, post-merge)

1. `python3 -c "from gpt_budget import fetch_reading; print(fetch_reading())"`
   — compare against a direct `/usage` curl. They must agree.
2. `router_ops.py plan-budget` shows the same used % as the endpoint.
3. With the governor on and gpt **still capped** (until Sunday), confirm zero
   behaviour change for P0 and that P1 traffic continues to pro.
4. After Sunday's reset: confirm the first gpt call lands, and that the
   even-burn line is honoured — i.e. we are not at 100% again within 24h.
   **This is the acceptance test for the whole change.**

### 5.6 Calibration (B37h)

Replay recorded history through the policy and report what it *would* have
spent per day. Success = a smoothed curve (no 700-then-6), with the weekly total
of served gpt calls **not materially lower** than the historical peak day —
because the goal is redistribution, not reduction. If smoothing costs total
capacity, `slack_percent` is too tight and must be loosened before enabling.

## 6. Rollout

1. Land B37a–c with `enabled: false`. Inert; no behaviour change.
2. Land B37d–h with `enabled: false`. Full test coverage, still inert.
3. **Observe-only mode** (optional intermediate): `enabled: false` plus a shadow
   log of what the governor *would* have decided. Gives a real-traffic
   counterfactual before it can block anything. Recommended, cheap.
4. Enable during the *post-reset* window (Sunday), so the first governed week
   starts from a clean window.
5. Watch `paced_out` counts + the nightly digest for one full 7-day window.
6. Rollback: `enabled: false`.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Under-use: throttling too hard, wasting paid capacity | §5.6 calibration; `final_stretch` releases the tail; `slack_percent` is the single tuning knob |
| Probe adds latency to the hot path | Cached, ≥300s, persisted; never per-request |
| Probe treats a token expiry as "capped" (it already does today) | Fail-open on 401 + refresh once; `source` field makes it visible |
| Governor masks a real capability regression by silently downgrading | Every throttle is logged with reason + counted; nightly digest surfaces it |
| Second governor mechanism diverging from `mark_rate_limited` | They compose by design: governor = predictive pacing; 429 path = reactive backstop. Documented in §2. |
| `gpt-6-astra` or a future model bypasses pacing | Account-keyed, not name-keyed — new plan models are covered automatically |

## 8. Open questions

1. **Slack default.** 15pp is a starting value, now backed by the replay in
   §3.2.1 rather than a guess. Calibration (§5.6) should still set it from a full
   week of real data. Too tight wastes capacity; too loose re-creates the bust.
2. **Which window binds?** Design uses max(`used_percent`) across windows. The
   5h primary at 0% while the 7d is at 100% is the current state — max() picks
   the 7d correctly. Confirm this holds on a bursty day where the 5h is the
   binding constraint, and whether the 5h needs its own separate slack (a 5h
   window's even-burn line moves ~4.8× faster than the 7d's per unit time).
3. **Reset-credit interaction.** The account has 3 spare "reset usage"
   redemptions. Should the governor *recommend* redeeming one when closed with
   >24h remaining? Out of scope for B37, but worth a follow-up flag.
4. **Notify on throttle-onset?** A one-shot alert when the governor first paces
   out in a window (rather than only in the nightly digest) may be worth it.
5. **`additional_rate_limits` (per-model scopes).** The endpoint exposes
   model-scoped windows. Design currently evaluates only account-wide windows
   for the pacing decision. Confirm per-model scopes should gate only that model
   (recommended) rather than counting as an account-wide close.

## 9. Decisions needed from the operator

1. **Approve the policy shape** (§3.2 even-burn + hard cap + final stretch)?
2. **Approve account-keyed enforcement** (§3.4) rather than per-model names?
3. **Run the observe-only phase** (§6.3) before enabling for real?
4. **Slack default:** accept 15pp as the starting value, tuned by calibration?
