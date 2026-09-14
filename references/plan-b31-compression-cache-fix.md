# B31 — Compression / Cache Destruction Fix

Status: PLANNED (write-first per repo convention)
Author: assistant, 2026-09-14
Scope: config-only (no core Hermes patch) + 2 router-side code changes (changes 4, 5)

## 1. Problem statement

Session compression is running away and destroying the prompt cache. Measured,
24h window ending 2026-09-14T12:40Z:

| metric | value |
|---|---|
| `session_compression` calls | 6,059 (5,991 in a single 24h query) |
| input tokens from compression | 798,171,007 (~798M) |
| input tokens from real chat | 3,360,284 (~3.4M) |
| compression share of ALL input | **99.6%** |
| recorded cost (compression) | $534.65 / 24h |
| recorded cost (normal_chat) | $47.25 / 24h |
| recomputed at CORRECT prices | $519.41 / 24h  -> **~$15.6k/30d** |
| distinct request_id | 6,068 of 6,068 -> amplification 1.00x (NOT retries) |
| avg input per compression call | ~132,000 tokens |
| avg output per compression call | ~1,371 tokens |
| avg self-reported savings | 3.1% |
| `compression_level` | structural 3,130 / **off 2,281** / lite 648 |

Amplification is 1.00x: these are 6,000+ DISTINCT compaction requests, not
retries of one request. Each one re-sends a ~132K-token prompt to a summariser.

## 2. Root cause (three coupled defects)

### 2.1 Threshold is set far below the model window
`config.yaml`:
```
compression:
  threshold: 0.5          # -> hygiene path: int(256000 * 0.5) = 128,000
  threshold_tokens: 60000 # -> preflight path: absolute cap
```
Hermes' own default for `threshold` is 0.50 but with a **raise-only floor of
0.75 for windows < 512K** (`context_compressor.py:988-989`,
`_SMALL_CTX_THRESHOLD_PERCENT = 0.75`). `biggie-router` presents a 256K window,
so the floor applies and the preflight threshold becomes
`int((256000 - max_tokens) * 0.75)` — BUT `threshold_tokens: 60000` is an
absolute cap that then **clamps it back down**:
```
agent_init.py:1437  # threshold_tokens: absolute cap (lower of ratio threshold and this)
context_compressor.py:2245 _apply_threshold_tokens_cap():
    _effective_cap = min(self.threshold_tokens_cap, self.context_length)
    if _effective_cap < self.threshold_tokens: self.threshold_tokens = _effective_cap
```
So the effective cap is **60,000 against a 256,000 window** (23%), and the
hygiene path is `0.5 * 256000 = 128,000`. Two different, both-too-low triggers.

### 2.2 `proactive_prune_tokens: 48000` rewrites the prefix
Default is `0` (off). Hermes' own config comment (config_defaults.py:554-558):
> "Each committed prune rewrites sent history and **breaks the prompt-cache
> prefix** — the min_reclaim gate below keeps those breaks episodic."

At 48,000 this fires routinely on a session that lives above 60K, so the prefix
is rewritten continuously — exactly what defeats server-side caching.

### 2.3 The compaction loop is a no-op that still rewrites
Live log evidence (agent.log*):
- `Session hygiene: compressed 650 -> 650 msgs, ~219,730 -> ~282,783 tokens` (tokens UP)
- **237 of 697** hygiene compactions removed ZERO messages
- **265 of 697** did not reduce tokens at all
- `1,290` warnings: `Context is over the compression threshold (~82,455 of
  60,000 tokens) but reclamation did not run: prune:nothing_eligible`
- `60` `compression skipped: another path is compressing session=...`

Because the threshold (60K) sits below the **incompressible floor** (system
prompt + 50+ tool schemas + `protect_first_n`/`protect_last_n`), each pass
"shrinks messages by a healthy margin yet leaves the prompt over the line, so
the next turn compacts again, forever." (quoted from `context_compressor.py`
comment at ~2390.) Hermes has an `_ineffective_compression_count` guard with a
threshold of `>= 2` (`context_compressor.py:2513`), but `_HygieneSettings`
**only honours 5 knobs** (`run_turn.py:541-547`: hard_msg_limit, timeout,
total_ceiling, max_turn_hold, failure_cooldown) — it never reads
`threshold_tokens` or `threshold`. The hygiene path's `threshold_pct` is
hardcoded `0.85` and not configurable.

### 2.4 Cache is measurable and is being thrown away
Live probe against the router (`glm-5.3`):
```
cold            prompt=2,019  cached=0     (0.82s)
warm identical  prompt=2,019  cached=1,984 (1.67s)
rewritten prefix prompt=225   cached=0
warm again      prompt=2,019  cached=1,984
```
Upstream returns `prompt_tokens_details.cached_tokens`. Caching is automatic
and requires no opt-in; **it is defeated by rewriting the prefix**, and at
`cached` rates the discount is large (deepseek-v4.1-flash: $0.15 in vs
$0.003 cached — 50x).

## 3. Change set

### Change 1 — raise the thresholds (config only)
- `threshold_tokens: 60000` -> `180000`
  (70% of the 256K window; keeps ~76K headroom for output. Must stay BELOW the
  window and above the incompressible floor.)
- `threshold: 0.5` -> `0.75`
  (match Hermes' own raise-only floor for windows < 512K, so the hygiene path
  also stops firing early: 0.75*256000 = 192,000)
- `hygiene_hard_message_limit: 400` -> `1200`
  (default is 5,000; 400 forces compaction on message COUNT regardless of
  tokens, which is what drives the no-op churn on a legitimately long session)

### Change 2 — stop rewriting the prefix (config only)
- `proactive_prune_tokens: 48000` -> `0` (restore the default: OFF)
  Rationale: the default is off precisely because each committed prune breaks
  the prompt-cache prefix. Keep `proactive_prune_min_reclaim_tokens` untouched.
- Leave `protect_last_n: 20` / `protect_first_n: 3` at their **defaults** (they
  already are). No change; documented so it is clear they were reviewed.

### Change 3 — per-session compression cap (router-side, code)
`_HygieneSettings` cannot express a cap and the hygiene `threshold_pct` is
hardcoded. A core patch to root-owned-ish Hermes core is out of scope for this
change (and riskier). Instead enforce the cap where the calls are actually
counted: the router.

New config block in `config.yaml`:
```
compression_guard:
  enabled: true
  per_session_max_calls: 40      # per rolling 60 min, per session
  window_minutes: 60
```
Implementation: in `scripts/biggie_llm_endpoint.py`, before the classifier
returns `session_compression`, consult a module-level rolling counter keyed by
session identity. If the cap is exceeded, the request is **still served** (we
never fail a request) but:
- it is exempted from the summariser lane selection (routed to the cheapest
  available model rather than the compaction lane), and
- it is logged with `error_type='compression_budget_exceeded'` and
  `workload_type='session_compression_overflow'` so the churn is visible.

Because `session_id` is blank at the endpoint for every compaction row (verified:
`session_compression||4929` — empty session), the key is derived from a stable
hash of the first N characters of the prompt (the session prefix), which is
stable across turns for the same session and cheap to compute.

### Change 4 — record `cached_tokens` (router-side, code)
`router_logs` has **no cache column** (verified against `PRAGMA table_info`).
Add:
```
ALTER TABLE router_logs ADD COLUMN cached_input_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE router_logs ADD COLUMN cache_hit INTEGER NOT NULL DEFAULT 0;
```
Migration must be additive + idempotent (check `PRAGMA table_info` first).
Populate from the upstream usage block (`prompt_tokens_details.cached_tokens`)
where present. Then `compute_cost_fields` (`biggie_llm_endpoint.py:518`) passes
`cached_input_tokens` into `cost_of_call(...)`, which already accepts it
(`unit_economics.py:181`) and bills `input_tokens - cached_input_tokens` at full
rate and the cached portion at the cached rate.

### Change 5 — fix the stale prices (router-side, code)
`model_pricing` holds **duplicate rows** for the same model at different
`effective_from`, and `price_for()` takes the newest, so some records are
3x overstated:
- `deepseek-v4-pro`: 2026-07-01 $2.00/$6.00 (stale) vs 2026-09-13 $0.66/$1.98 (correct)
- `deepseek-v4-flash`: `model_costs` says $0.50/$1.50 vs correct $0.22/$0.66
- `glm-5.3`: 2026-08-26 $1.50/$4.50 vs 2026-09-13 $1.40/$4.40
- `qwen3.5`: 2026-08-08 $1.75/$5.25 vs 2026-09-13 $0.60/$3.60
`MODEL_REGISTRY` (scripts/models.py) is the single source of truth; make
`seed_prices` write a **correction row** when the registry price differs from
the newest recorded price, instead of only inserting when
`(model, effective_from)` is unseen (current behaviour at `unit_economics.py:108-113`
silently keeps stale rows because `effective_from` differs).

## 4. Risk assessment

| # | risk | severity | mitigation |
|---|---|---|---|
| R1 | Raising threshold lets the prompt grow until the provider 400s | HIGH | 180K is 70% of the 256K window; `max_tokens=16384` reserved, leaving ~76K headroom. Preflight still fires before the window is full. Verify with a live long-context request. |
| R2 | A legitimate long session now grows much larger, raising per-call cost | MED | That is the *point*: those input tokens are now ~99% cacheable at a ~50x discount, vs paying full price for 6,000 useless compactions. Net must be measured after the change. |
| R3 | Disabling proactive_prune lets old tool output be re-sent every turn | MED | That is the default. Re-sent tokens are cached; the prune path was actively destroying the cache. Measure cache-hit rate as the acceptance test. |
| R4 | Schema migration on the live DB could lock/break it | HIGH | Additive `ALTER TABLE ... DEFAULT 0` only; `PRAGMA table_info` guard for idempotency; back up `router_logs.db` first; run migration out-of-band, not on the request path. |
| R5 | Price correction changes historical cost attribution | MED | `price_for(at=...)` is time-aware. A correction row is written with a NEW `effective_from`, so historical rows keep the price that was in force then. Do NOT UPDATE old rows. |
| R6 | Per-session cap could starve a session that genuinely needs compaction | MED | Cap = 40/hour and the request is still served (routed cheap, never rejected). Set generously; it exists to break a runaway, not to ration normal use. |
| R7 | Config is read at different times by different paths | MED | `threshold*`/`proactive_prune_tokens` are read at **agent construction (cold)**; hygiene knobs are read **per turn (hot)**. A gateway restart is required for the cold keys. Verify by reading the effective values back at runtime. |
| R8 | Lowering prices could mask a real cost problem | LOW | Costs are diagnostic; the fix makes them *correct*, and cache tokens are recorded so the discount is now visible instead of assumed. |

## 5. Test plan (must pass before deploying)

Config/logic assertions (router suite, TDD, new file per batch):
1. `test_b31_compression_thresholds.py` — effective threshold resolution:
   180,000 wins over the ratio when lower; asserts the value is > the
   incompressible floor and < the window.
2. `test_b31_compression_budget_cap.py` — cap counter: 40 calls allowed, 41st
   is flagged + routed cheap + logged, and the request is NEVER rejected;
   window expiry resets.
3. `test_b31_cache_columns.py` — migration is idempotent (run twice = 1 column
   added), `cached_input_tokens` is populated from the usage block, and
   `cost_of_call` bills `in - cached` at full rate + cached at cached rate.
4. `test_b31_price_correction.py` — a registry price change writes a NEW
   effective row; `price_for(at=<old date>)` still returns the OLD price;
   `price_for()` with no `at` returns the corrected price.
5. Regression: full router suite must stay green (currently 551 passed).

Live verification (production, after deploy):
- cache-hit rate on normal chat before/after (expect a large rise)
- `session_compression` calls/hour before/after (expect a large fall)
- zero new `prune:nothing_eligible` / `still ~N after compression` warnings
- a long-context request still returns 200 (no provider overflow)
- `cached_input_tokens > 0` on warm rows

## 6. Rollback
- config: restore `config.yaml.bak-compression-fix-<ts>` (taken before change)
- code: `git revert` the B31 commits (router suite is the gate)
- schema: additive columns are inert if unused; no destructive down-migration
  needed. DB backup taken before migration (`router_logs.db.bak-b31-<ts>`).

## 7. Explicitly out of scope
- Patching Hermes core (`agent/context_compressor.py`, `gateway/run_turn.py`)
  to make `threshold_pct` configurable or to honour `threshold_tokens` on the
  hygiene path. That is the correct long-term fix but is a core change with a
  much larger blast radius; the router-side cap (change 3) contains the
  runaway now. Recommend raising it upstream separately.
- `/compact` semantics, `target_ratio`, `tail_mode`, `max_attempts`.
