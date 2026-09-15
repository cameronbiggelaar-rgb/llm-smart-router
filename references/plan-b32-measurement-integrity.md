# B32 — Measurement integrity: session attribution + timestamp handling

**Status:** plan (pre-implementation)
**Author:** assistant (this thread, 56467)
**Scope:** `llm-smart-router` skill only. No hermes-agent core changes.
**Prereq for:** B33 (phantom-floor / stale-thinking design)

---

## 1. Why this exists

Three separate cost/behaviour analyses in this thread produced **wrong numbers**, each
time from the same underlying flaw: comparing values whose format or window did not
align. Two were mine (assistant analysis), one was in the data itself.

| # | Claim | Reality | Root cause |
|---|-------|---------|-----------|
| 1 | "$562/day from compression" | billing artifact | `cached_tokens` hardcoded 0 → every cache hit invoiced at full rate |
| 2 | "428/hr → 2.4/hr, 26× cost drop" | unchanged | compared an **idle 5h window** to a busy one |
| 3 | "compaction escalates to gpt-5.6-sol, $840/day" | $0 post-fix; $17/day | SQL filter used `'2026-09-14 18:20'` (**space**) against stored `'2026-09-14T18:20'` (**T**). `'T' > ' '`, so **every row compared greater** — a "6h window" matched 7,205 rows ≈ 24h |

Plus one timezone error: logs are AEST (+10), DB is UTC, compared directly.

And one data gap: **`session_id` is blank on every row**, so "940 compactions from one
session" was inferred from the *blank key*, not proven. Churn cannot be attributed.

**Goal:** make these classes of error impossible-by-construction where cheap, and
detectable where not.

---

## 2. Findings (evidence)

### 2.1 session_id is hardcoded blank
`biggie_llm_endpoint.py:2176`
```python
"",  # session_id — not available at endpoint level
```
That comment is **wrong**. The value is available:
- `chat_completions()` at ~3495 already extracts `body.get("session_id") or body.get("user")`
- `_compression_session_key()` (1552) already derives a stable `pfx:<sha256[:32]>`
  prefix key for the budget guard, documented as "stable across turns for a given
  session and distinct between sessions"

So the plumbing exists; it just isn't wired into the log row.

### 2.2 Hermes does not send session identity on chat-completions
Verified from `request_dump_*.json`: chat-completions bodies carry only
`model, messages, tools, max_tokens, reasoning_effort`; inbound headers at the router
are `Authorization` + `Content-Type` only.

**The Codex path already sends it**:
```json
"extra_headers": {"session_id": "20260911_070402_732083ca",
                  "x-client-request-id": "pck_07cf2880b3229c04b19b8147"}
```
and `prompt_cache_key` (`pck_...`) exists there too. The chat-completions path has no
equivalent. That is a **core-side data capture gap**, not a router bug.

### 2.3 Timestamps
- DB rows: `datetime.now(timezone.utc).isoformat()` → `YYYY-MM-DDTHH:MM:SS+00:00` (correct, tz-aware)
- Logs: local AEST, **no offset marker** → indistinguishable from UTC by inspection
- The `T`-vs-space comparison trap is silent: no error, just a wrong window

---

## 3. Design — Task A

### A1. Session attribution (router-side)

**Source priority** (first non-empty wins), recorded with provenance:

| Priority | Source | `session_id_source` |
|----------|--------|---------------------|
| 1 | header `x-hermes-session-id` | `header` |
| 2 | header `x-session-id` | `header` |
| 3 | body `session_id` | `body` |
| 4 | body `metadata.session_id` | `body` |
| 5 | body `user` | `body` |
| 6 | `_compression_session_key(messages)` prefix hash | `derived` |
| 7 | (none possible) | `none` |

**Why a source column, not just the value.** A real session id and a derived prefix
hash are **not equivalent**:
- real id: stable per *session*
- derived: stable per *context prefix* — two sessions sharing an opening prompt collide

Storing the value alone would silently conflate them and I would repeat the attribution
error in a new form. `session_id_source` makes the distinction queryable, so
"940 compactions from one session" becomes a claim with a confidence level.

**Never raises.** Guard/logging bookkeeping must not break a live request (matches the
existing `_compression_session_key` contract).

**Wire all six** `_log_request_to_db` call sites (3791, 3834, 3930, 4080, 4158, and the
shadow path at 712).

### A2. Timestamp safety (analysis-side)

The `T`-vs-space bug lives in *analysis queries*, so the fix belongs where analysis is
written — a shared helper, not a comment.

Ship `scripts/timerange.py`:
```python
def db_window(hours: int, *, now: datetime | None = None) -> tuple[str, str]:
    """ISO-T UTF-8-sortable UTC bounds matching router_logs.timestamp exactly."""
```
- returns `(lo, hi)` in `%Y-%m-%dT%H:%M:%S` form, never space-separated
- `assert_same_clock()` helper for mixed log/DB comparisons (AEST vs UTC)
- refuses to build a window from a space-separated string

### A3. Regression tests (non-vacuity required)

`tests/test_b32_measurement_integrity.py`
1. `session_id` sourced from header → stored, `source=header`
2. from body → stored, `source=body`
3. absent → derived prefix hash, `source=derived`
4. **two distinct sessions with identical opening prompts** → same derived key,
   provenance still `derived` (documents the collision honestly)
5. `db_window()` output sorts correctly: `lo <= '2026-09-14T18:20:00' <= hi` semantics
6. **the bug itself pinned**: a space-separated bound must NOT be accepted as a valid
   window (fails loudly)
7. every `_log_request_to_db` call site passes a session_id (drift detector)
8. non-vacuity: inject the space-form bug → test 6 fails

---

## 4. Design — Task B (phantom floor) — OUTLINE ONLY

Full detail deferred until A produces data. Current evidence:

- Hermes charges `reasoning_content` against the context budget:
  `_NEWEST_TURN_ONLY_BUDGET_KEYS = "reasoning", "reasoning_content"`
- Measured on a real request: **808,941 chars (55%)** of the payload is
  `reasoning_content` across 418 assistant turns
- **The provider never receives it.** Probe: 300,000 chars of `reasoning_content`
  billed **48 input tokens**; the same volume as `content` billed 4,540
- So Hermes believes a session is ~366K tokens while the provider bills a median of 2,693
- Compaction fires against a **phantom**, cannot win, and re-fires: the loop

**Suspect:** `turn_context.py:64`
```python
charge_stale_thinking=_agent_stale_thinking_on_wire(agent),
```
→ `stale_thinking_reaches_wire(...)`, documented "True (conservative full charge) when
route facts are unavailable" — `except: return True`. Our route is a **custom**
provider (`api_mode: custom`, `127.0.0.1:8080/v1`), which it cannot classify.

**Open questions A should answer:**
- Is it ONE session looping or many? (needs A1)
- What is the true floor split: phantom vs genuinely-prunable?
- Does the billed input track the local estimate at all, per session?

---

## 5. Additional data we should capture (future state)

Called out per the design brief. These are **not** implementable router-side today —
they are core-side asks or new columns:

| # | Data | Why | Owner |
|---|------|-----|-------|
| 1 | `x-hermes-session-id` header on chat-completions | real attribution; Codex path already does it | **core** |
| 2 | `reasoning_token_estimate` vs `billed_input_tokens` per call | quantify the phantom directly | core + router |
| 3 | `stale_thinking_on_wire_resolved: bool` | today it silently defaults True | **core** |
| 4 | session message count + turn count on each row | attribute churn to session growth vs loop | router (have it) |
| 5 | per-session compaction count in a rolling window | already computed for the guard — persist it | router |
| 6 | `timezone` offset marker on log lines | kill the AEST/UTC class of error | **core** |
| 7 | `session_id_source` | distinguish real vs derived | router (this task) |

Items 1, 3, 6 need core changes (root-owned, external restart). Item 3 is the B33 fix
itself.

---

## 6. Risks

| Risk | Mitigation |
|------|-----------|
| Derived key collision conflates sessions | `session_id_source` makes it explicit; never claim real attribution from `derived` |
| Adding a column breaks fixtures | `rollup.py` precedent: schema-tolerant read path |
| Logging change breaks a live request | all new code wrapped, must never raise |
| Shared repo, concurrent thread | check `git status` before each batch; commits stay unpushed (repo convention) |
| Adding `session_id` changes group-by results | re-baseline before/after; no historical backfill claims |

## 7. Out of scope
- Core changes (1, 3, 6 above) — designed, not implemented here
- Historical backfill of `session_id`
- Any change to routing/escalation behaviour (A is measurement-only)
