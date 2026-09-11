# Self-Optimising Router — Architecture & Build Plan

**Goal:** Turn the smart router from a *static* cost-preferring dispatcher into a
*closed-loop* one: measure real cost and real quality per call, run controlled
experiments on live traffic, roll findings up, and propose cheaper call trees —
without regressing quality.

**Status:** design fixed; built in batches (see §9). Read this before touching
`router.py` or `biggie_llm_endpoint.py`.

---

## 1. What the audit found (facts, not assumptions)

Measured against the live `data/router_logs.db` (231,362 rows, 2026-05-14 → 09-11):

| Claim | Reality | Evidence |
|---|---|---|
| "We account for unit cost" | **FALSE** — `estimated_cost_usd = 0.0` on 231,348 / 231,362 rows | only 431 rows (the offline collector) ever wrote a cost |
| "We know output volume" | **PARTIAL** — streaming writes `output_tokens=0`; total 13.9M output vs 15.9B input | `_log_stream_complete()` passes `output_tokens=0` |
| "Pricing is real USD" | **FALSE** — `model_costs` holds Phase-1 *relative compute units* (deepseek-v4-flash `0.5/1.5`), not USD, and omits glm-5.3 / v4.1-flash / gpt-5.6 / gpt-6-astra | 10 rows, all stale |
| "We measure quality" | **FALSE** — no quality column, no scorer on any live path | 46 cols, none quality |
| "We optimise the call tree" | **FALSE** — `_select_session_compression_model()` returns a hardcoded list; `train.py` never written | no trainer, no optimiser |
| "Logs don't grow forever" | **FALSE** — no retention, no rollup, no VACUUM; 101 MB and ~11k rows/day ≈ 1.7 GB/yr | no prune code, no findings table |

So five of the six asks are genuinely absent, not partial. Reuse targets exist
though: the 46-column event log, `MODEL_REGISTRY`, `compression_sampler.py`,
the offline `ab_*` harnesses + their fact-coverage scorer, and a 138-test suite.

**The one partial thing worth keeping:** `compression_sampler.py` already
captures real production payloads to disk for offline A/B. That is the seed for
live experiments (§5), not a separate system.

---

## 2. Architecture

```
                    ┌──────────────────────────────────────────────┐
   Hermes client ──▶│  biggie_llm_endpoint.py  /v1/chat/completions │
                    │                                              │
                    │  1. classify workload  ──▶ call_type          │
                    │  2. route_task()       ──▶ chosen model       │
                    │  3. experiment hook    ──▶ [traffic_split]    │  ◀── NEW
                    │  4. serve (+ shadow leg)                     │
                    │  5. capture usage      ──▶ in/out tokens      │  ◀── FIX
                    │  6. price the call     ──▶ [unit_economics]   │  ◀── NEW
                    └───────────────┬──────────────────────────────┘
                                    ▼
                        ┌───────────────────────┐
                        │  router_logs.db       │
                        │   router_logs (raw)   │  46 cols + 7 new
                        │   model_pricing       │  ◀── NEW (real USD, versioned)
                        │   experiments         │  ◀── NEW (canary defs)
                        │   daily_findings      │  ◀── NEW (rolled up, kept forever)
                        │   rollup_state        │  ◀── NEW (watermarks)
                        └───────────┬───────────┘
                                    │  nightly
                    ┌───────────────▼──────────────┐
                    │  rollup.py   (aggregate+purge) │  ◀── NEW
                    └───────────────┬──────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  optimiser.py  → candidate     │  ◀── NEW
                    │  routing table + experiments   │
                    └───────────────┬───────────────┘
                                    │ proposes only
                                    ▼
                    experiments.yaml  ──▶ next canary  (human promotes)
```

Dependency rule: `models` ← `unit_economics` ← `rollup` ← `optimiser`.
`traffic_split` depends only on `models`. Nothing imports the endpoint.

---

## 3. Module contracts

### 3.1 `unit_economics.py` — real cost per call

```python
@dataclass(frozen=True)
class Price:
    model: str; provider: str
    input_usd_per_1m: Decimal; output_usd_per_1m: Decimal
    effective_from: str; source: str

def price_for(model: str, at: str | None = None) -> Price | None
def cost_of_call(model: str, input_tokens: int, output_tokens: int,
                 cached_input_tokens: int = 0, at: str | None = None) -> Decimal
def unit_cost(conn, since: str, until: str | None = None) -> list[UnitCostRow]
    # UnitCostRow: model, call_type, provider, calls, input_tokens,
    #              output_tokens, cost_usd, cost_per_call, cost_per_1k_in,
    #              quality_avg, quality_n, cost_per_quality_point
def record_price(conn, price: Price) -> None
def seed_prices(conn) -> int          # idempotent, from MODEL_REGISTRY
```

Rules:
- **Cost is computed at ingress and stored immutably** on the row (`cost_usd`).
  Reporting never re-derives it from today's price list.
- `price_for` resolves the price *in force at* `at`, so historical rows re-cost
  correctly when a provider changes rates.
- Unknown model ⇒ `None` price ⇒ `cost_usd = 0` **and** the row is flagged
  `cost_unknown=1` so unaccounted spend is visible rather than silently zero.
- `Decimal`, not float — money.

### 3.2 `quality.py` — response quality

```python
@dataclass(frozen=True)
class QualityScore:
    score: float          # 0..1
    coverage: float       # key-fact recall
    hallucinated_numbers: int
    method: str           # 'fact_coverage_v1'

def score_summary(prompt: str, response: str, reference: str | None = None) -> QualityScore
def quality_column(conn, since, until, model=None, call_type=None) -> list[QualityRow]
```

- Reuses the **established** fact-coverage + hallucinated-number scorer from
  `scripts/ab_direct.py` (`fact_coverage_v1`) — one scorer, not a second opinion.
- **Never on the hot path.** Scoring costs an LLM call. It runs:
  (a) offline over captured samples, and (b) online only for *sampled* production
  calls, gated by the same sampler flag as §5. Inline scoring would pay a second
  generation to measure the first.
- Rows carry `quality_score`, `quality_method`; NULL means "not measured", which
  is distinct from 0.0 ("measured, bad").

### 3.3 `traffic_split.py` — canary & shadow (the "point traffic at a test model")

```python
@dataclass(frozen=True)
class Experiment:
    name: str; enabled: bool
    model: str                       # the model under test
    mode: str                        # 'shadow' | 'split'
    percent: float                   # 0..100, split only
    match_workload: tuple[str, ...]  # e.g. ('session_compression',)
    match_min_tier: int
    started: str; notes: str

def load_experiments(path: str | None = None) -> list[Experiment]
def select_arm(exp: Experiment, request_id: str, session_id: str) -> str
    # 'control' | 'treatment' — deterministic
def pick_experiment(workload: str, tier: int, exps) -> Experiment | None
```

- **`shadow` mode is the safe default**: production serves the control answer,
  the endpoint *additionally* calls the test model on the real payload, logs its
  cost/latency/quality, and discards the output. Real production load, zero risk.
- **`split` mode** serves `percent` of matching traffic from the test model.
  Requires `percent` and is the deliberate promotion step.
- Bucketing is a **deterministic hash of `session_id`** (falling back to
  `request_id`), so a session stays in one arm — no flapping mid-conversation,
  and the split is reproducible.
- Config lives in `scripts/experiments.yaml` — exactly the "model name + test
  feature + flag/percentage" shape requested. Hot-reloaded on mtime change.
- Fail-closed: an experiment naming an unregistered/unreachable model is
  skipped, never half-applied.

### 3.4 `rollup.py` — findings + retention

```python
def rollup_day(conn, day: str) -> int              # idempotent, upsert into daily_findings
def purge_raw(conn, keep_days: int, dry_run=True) -> PurgePlan
def vacuum_if_needed(conn, min_free_pages: int = 2000) -> bool
def findings(conn, days: int = 7) -> list[Finding]  # human-readable
```

- **Order is inviolable: roll up → commit → then purge.** Never purge a day
  whose rollup is not committed, or the data is gone for good.
- `rollup_state` holds the last-rolled-up day and last-purged day so re-runs are
  idempotent (`INSERT ... ON CONFLICT DO UPDATE`, not blind append).
- `daily_findings` is ~1 row per (day × workload × model) — trivially small, kept
  forever. Raw rows default to 30 days.
- Purge is `dry_run=True` by default; deletion is `DELETE`+`VACUUM` in bounded
  batches so a 100 MB DB doesn't lock for minutes.

### 3.5 `optimiser.py` — the self-optimising call tree

```python
def propose(conn, days: int = 14, quality_floor: float = 0.8) -> Proposal
    # Proposal: per workload, current chain, candidate chain, est weekly delta,
    #           quality risk, and the experiment that would validate it
def write_candidate(proposal, path) -> None   # candidate_routing.yaml, never live
```

- Reads `daily_findings`, and for each workload class ranks models by
  `cost_per_quality_point` subject to `quality_avg >= quality_floor` and a
  minimum sample size `n >= 30` (no promoting a model on 3 calls).
- **Emits a candidate, never applies it.** Promotion is: candidate → experiment
  (shadow) → evidence → human edits `routing_table.yaml`. This keeps a
  quality-regressing auto-change from ever reaching production silently.
- Names the *specific* experiment to run to close each gap.

---

## 4. Data model

```sql
-- NEW: real USD pricing, versioned
CREATE TABLE model_pricing (
    model TEXT NOT NULL, provider TEXT NOT NULL,
    input_usd_per_1m REAL NOT NULL, output_usd_per_1m REAL NOT NULL,
    effective_from TEXT NOT NULL, source TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (model, effective_from)
);

-- NEW: canary definitions mirrored to DB for auditability
CREATE TABLE experiments (
    name TEXT PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT 0,
    model TEXT NOT NULL, mode TEXT NOT NULL DEFAULT 'shadow',
    percent REAL NOT NULL DEFAULT 0, match_json TEXT NOT NULL DEFAULT '{}',
    started TEXT NOT NULL DEFAULT '', notes TEXT NOT NULL DEFAULT ''
);

-- NEW: rolled-up findings (kept forever)
CREATE TABLE daily_findings (
    day TEXT NOT NULL, workload_type TEXT NOT NULL, model TEXT NOT NULL,
    call_type TEXT NOT NULL DEFAULT '', calls INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0, cost_unknown_calls INTEGER NOT NULL DEFAULT 0,
    success_calls INTEGER NOT NULL DEFAULT 0, escalated_calls INTEGER NOT NULL DEFAULT 0,
    latency_p50 REAL NOT NULL DEFAULT 0, latency_p95 REAL NOT NULL DEFAULT 0,
    quality_avg REAL, quality_n INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, workload_type, model, call_type)
);
CREATE INDEX idx_daily_findings_day ON daily_findings(day);

-- NEW: watermarks for idempotent rollup/purge
CREATE TABLE rollup_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
```

Alterations to `router_logs` (idempotent `_ensure_log_columns`-style migration):

```sql
ALTER TABLE router_logs ADD COLUMN cost_usd REAL NOT NULL DEFAULT 0;
ALTER TABLE router_logs ADD COLUMN cost_unknown INTEGER NOT NULL DEFAULT 0;
ALTER TABLE router_logs ADD COLUMN pricing_version TEXT NOT NULL DEFAULT '';
ALTER TABLE router_logs ADD COLUMN experiment TEXT NOT NULL DEFAULT '';
ALTER TABLE router_logs ADD COLUMN experiment_arm TEXT NOT NULL DEFAULT '';
ALTER TABLE router_logs ADD COLUMN is_shadow INTEGER NOT NULL DEFAULT 0;
ALTER TABLE router_logs ADD COLUMN quality_score REAL;          -- NULL = unmeasured
ALTER TABLE router_logs ADD COLUMN quality_method TEXT NOT NULL DEFAULT '';
CREATE INDEX idx_router_logs_workload_ts ON router_logs(workload_type, timestamp);
```

Retention: raw 30 d (config `ROLLUP_KEEP_DAYS`), `daily_findings` forever.
At ~11k rows/day this caps the raw table near 330k rows instead of growing to
4M/yr.

---

## 5. Traffic-splitting config (`scripts/experiments.yaml`)

```yaml
experiments:
  - name: glm53flash-compression-shadow
    enabled: true
    model: glm-5.3-flash
    mode: shadow            # observe only — production answer unchanged
    match:
      workload: [session_compression]
    notes: "measure cost/latency vs deepseek-v4.1-flash on real compressions"

  - name: v41flash-to-glm53-compression-split
    enabled: false
    model: glm-5.3-flash
    mode: split
    percent: 10             # 10% of matching traffic actually served by it
    match:
      workload: [session_compression]
```

Adding a model under test is a config edit — no code change.

---

## 6. Error handling

| Failure | Behaviour |
|---|---|
| Unknown model price | `cost_usd=0`, `cost_unknown=1`; rollup reports unaccounted spend |
| Experiments file missing/malformed | log warning, return `[]`, route normally |
| Experiment names unreachable model | skip that experiment, keep serving control |
| Shadow leg fails/times out | log the failure; **never** affect the client response |
| Rollup fails mid-day | watermark not advanced; day re-rolled next run (idempotent) |
| Purge attempted with no committed rollup | refuse, log error |
| Quality scorer unavailable | `quality_score` stays NULL; never fabricate a score |

---

## 7. Test strategy (layers)

| Layer | Covers | Fixture |
|---|---|---|
| **Schema** | migrations idempotent, WAL, all new cols/tables exist | temp DB, run migrate ×2 |
| **Unit** | `cost_of_call` math; `price_for` version resolution; hash bucketing determinism; `select_arm` distribution ±2%; scorer on known pairs; retention day selection | pure functions |
| **Contract** | every module's public API against a fixed fixture DB | `tests/fixtures/router_fixture.db` |
| **Integration** | ingress → row with non-zero `cost_usd`; streaming writes real `output_tokens`; shadow leg logs without touching the response; rollup → purge ordering | temp DB + fake backend |
| **E2E/regression** | full day rollup → findings → candidate proposal; **existing 138 tests still pass** | seeded DB |

Non-negotiable: a test that asserts a cost must assert a *specific* number
computed by hand, not `> 0`.

---

## 8. ADRs

- **ADR-1 Cost at ingress, stored immutably.** Token counts for streaming are
  complete only at stream end; report-time derivation would need the price list
  frozen forever. Store the cost, keep the pricing version on the row.
- **ADR-2 Real USD replaces relative compute units.** Ratios can order models
  but cannot answer "what did this week cost" — the actual question. Ratios stay
  as a fallback ordering signal only.
- **ADR-3 Quality measured offline/sampled, never inline.** Inline scoring needs
  a second LLM generation per call: it would roughly double cost and blow the
  120 s compression deadline to measure the thing being optimised.
- **ADR-4 Roll up before purge.** Purge without a committed rollup is
  irreversible data loss.
- **ADR-5 Optimiser proposes, human promotes.** An auto-applied routing change
  that regresses quality is worse than a stale-but-good routing table.
- **ADR-6 Shadow before split.** Observe the real model on real load with zero
  user-visible risk, then promote by percentage.
- **ADR-7 Deterministic hash bucketing.** Sticky per session; random sampling
  would flap sessions between arms and corrupt quality comparison.

---

## 9. Build batches

Each batch: RED test → GREEN → commit. One test file per batch.

| # | Batch | Files | Tests |
|---|---|---|---|
| B1 | Schema + migration | `scripts/rollup.py` (schema part) | `tests/test_rollup_schema.py` |
| B2 | Unit economics | `scripts/unit_economics.py` | `tests/test_unit_economics.py` |
| B3 | Quality scorer | `scripts/quality.py` | `tests/test_quality.py` |
| B4 | Traffic split | `scripts/traffic_split.py`, `scripts/experiments.yaml` | `tests/test_traffic_split.py` |
| B5 | Rollup + retention | `scripts/rollup.py` | `tests/test_rollup.py` |
| B6 | Optimiser | `scripts/optimiser.py` | `tests/test_optimiser.py` |
| B7 | Ingress wiring | `scripts/biggie_llm_endpoint.py` (cost, streaming tokens, shadow) | `tests/test_ingress_accounting.py` |
| B8 | Reporting | `scripts/report.py` / `check-routing-stats.py` extension | `tests/test_cost_report.py` |

Batches B2–B6 are independent once B1 lands, and touch disjoint files.
B7 is the only batch that modifies the hot path, and is last for that reason.
