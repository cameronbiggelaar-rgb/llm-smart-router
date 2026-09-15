# BUILD PLAN — compaction loop, attribution integrity, and the guard (post-B32)

**Status:** proposed (awaiting review/replay)
**Author:** assistant, thread 56467
**Supersedes/extends:** `plan-b33-route-blind-preflight-charge.md` (B33),
`plan-b34-derived-key-is-system-prompt.md` (B34)
**Evidence base:** B32 deployed 2026-09-15 02:13Z; monitored 60 min (30/30 samples,
0 failures); all numbers below measured, not inferred.

---

## 0. Where we are

| item | state | evidence |
|---|---|---|
| B32 measurement integrity | **SHIPPED + LIVE** | PID 2527599; 338/338 rows attributed; 0 blank; preflight 16/16; 595 tests |
| B33 route-blind charge | **design only** | 364,368 vs 158,603 tokens on the real dump |
| B34 derived key = system prompt | **CONFIRMED, unfixed** | `sha256(system[:512])`; 347/416 rows (83%) on ONE key; 3 of 4 parallel sessions collide (`state.db.system_prompt_hash`) |
| no-op compaction loop | **REAL, 100% of sample** | 17/17 `compressed N → N` in agent.log |

**The single most important correction in this revision:** B32's own `derived`
key is the *system prompt*, so "one session" was never true and the B31 guard is
**global**, not per-session. This changes the priority order below.

---

## 1. The two defects are independent — do not conflate them

**B33 (trigger):** `_preflight_request_tokens()` returns the anchored estimate
before the route-aware path, and `anchored_context_tokens()` defaults
`charge_stale_thinking=True` while its callers omit it. Result: the gating number
charges reasoning on routes where Hermes strips it — **364,368 vs 158,603** against
a 180,000 threshold. Compaction fires and cannot win.

**B34 (accounting):** the per-session cap keys on a constant (the system prompt),
so it is a single global 40/hr budget shared by all concurrent conversations.
Measured post-deploy: **289 overflow rows on ONE key** vs 95 genuine compression
rows on 2 keys — so most "runaway" labels are shared-budget relabels.

Either alone produces churn; B33 causes the churn, B34 mislabels and mis-budgets it.
Fixing B33 without B34 leaves the guard penalising innocent sessions; fixing B34
without B33 leaves the loop running.

---

## 2. Phased plan

### Phase 1 — B34 fix (router-side, this repo) ← **promote to FIRST**
*Rationale: cheapest, reversible, fully testable here, and it currently corrupts
the metric we use to judge everything else. Do not measure churn through a broken
lens.*

1. **Stop hashing the system prompt.** Exclude `role == "system"` from the joined
   prefix text. Proven discriminating by measurement (§2 of B34 plan: with the
   system message removed, two conversations produce *different* keys).
2. **Never enforce a per-session cap on a non-discriminating key.** If no real
   identity is available *and* the derived key cannot discriminate, log and serve —
   do not cap. A cap that cannot attribute silently penalises innocent sessions.
3. **Preserve the real-identity path.** `header`/`body` resolution already works
   (`b32smoke001` → `source=header`); it must win whenever present.
4. **Tests (TDD, RED first):**
   - two conversations sharing a system prompt → **different** keys;
   - same conversation across turns (growing tail) → **stable** key;
   - `derived` + non-discriminating → cap **not** enforced;
   - real identity present → used verbatim.
5. **Non-vacuity:** restore the system-prompt hash; confirm the collision test goes RED.

### Phase 2 — B33 fix (core, root-owned, external restart)
Unchanged from the existing design: make `charge_stale_thinking` explicit at the
four gating call sites (or fail-safe-per-route), and guard the invariant that the
trigger estimator and the budget walk share one wire-truth predicate. Core change
⇒ separate review gate, external restart, and a before/after measurement on the
sessions currently pinned at 180,236–184,427.

### Phase 3 — close the measurement gap that made this hard

**Status: 3.2 DONE (mechanism proven, plugin installed, verified in-process).
Activation pending a gateway restart.**

1. **Compaction outcome must be queryable.** `compressed N → M msgs` + token
   before/after exists ONLY in `agent.log` (AEST), so proving a no-op requires
   grepping logs on a different clock from the DB (UTC). Add a table/columns so
   "is this loop a no-op?" is a SQL query, not a log correlation. **Still open.**

2. **`session_id` on the wire — DONE (B35).** The premise changed during recon:
   core ALREADY sends `prompt_cache_key` on chat-completions, gated on the
   provider profile's `supports_prompt_cache_key`. The generic `custom` profile
   leaves it `False`, so the route had no identity. A **user plugin** at
   `$HERMES_HOME/plugins/model-providers/biggie-llm/` overrides the route and
   sets the flag — **no core patch needed** (which is why this moved ahead of
   B33). Proven read-only: 2/2 session-scoped keys emitted, distinct per session,
   turn-stable, 0/2 without the flag. `cache_key` is in
   `_SESSION_ID_ENFORCEABLE_SOURCES`, so this also **restores guard
   enforceability** on the main path. Artefact: `deploy/model-providers/biggie-llm/`.

3. **A collision-safety flag — largely delivered (B32/B34).**
   `session_id_source` plus `_SESSION_ID_ENFORCEABLE_SOURCES` already encode
   "does this key discriminate". A separate flag would be redundant.

**Phase 3.2's consequence:** because B34 made the guard refuse a
non-discriminating key, `session_compression_overflow` stopped firing for
chat-completions traffic — correct labelling, but **containment on that path is
now zero until B33 lands**. B33 is therefore the only remaining containment.

**Order revised:** B35 (done) → **B33** → verify the guard actually fires on
real runaways.

---

## 3. Data we need to capture (explicit list, per the design brief)

**Router-side (this repo):**
- `session_id_source` — **DONE (B32)**: `header` / `body` / `derived` / none.
- key-discriminates flag (or distinct-prefix count) — **NEW, Phase 3**.
- compaction msg-count before/after — **NEW, Phase 3** (currently log-only).
- tokens before/after per compaction — **NEW, Phase 3**.

**Core-side (`hermes-agent`):**
- which estimator produced the gating number (anchored vs native vs generic) and
  whether stale thinking was charged — the telemetry logs the *symptom*
  (`current_estimated_tokens`) and never the cause. **NEW, Phase 3.**
- `session_id` transmitted on the chat-completions path. **NEW, Phase 3.**

**Cross-DB hygiene:**
- `state.db.last_activity_at` is **mixed-type** (`null` 704, `text` 575, `real` 36).
  My first concurrency query returned 0 sessions because it compared an ISO string
  to floats. Any join must normalise types first, and there is no `updated_at`
  column despite the schema suggesting one.

---

## 4. Risks

| risk | mitigation |
|---|---|
| Excluding the system message narrows the key → false "distinct" for same session | test growing-tail stability explicitly (Phase 1.4) |
| Dropping the cap on derived keys removes a safety net | cap was already global/mislabelled; log loudly instead of silently penalising |
| B33 core change alters compaction for ALL routes | before/after on the pinned sessions; separate review gate |
| Router restart mid-flight | preflight gate + verified DB backup (established practice) |

---

## 5. Verification plan (replay before starting)

Each phase must show, before merge:
1. RED test first, then GREEN (TDD batch-by-batch, one test file per batch).
2. **Non-vacuity**: reintroduce the defect, confirm RED, restore, confirm GREEN.
3. Full suite + preflight 16/16.
4. Deploy ⇒ **post-restart round trip** (send request, read the row back) — a
   `/v1/models` 200 also answers from the OLD process.
5. Attribution coverage on new rows = 100%, and **distinct keys ≥ concurrent
   sessions** (the check that would have caught B34 on day one).

**The regression check that matters:** after Phase 1, the number of distinct
derived keys must be ≥ the number of concurrent Hermes sessions in the same
window (4 at the time of writing). That single assertion would have caught B34.

---

## 6. Open questions for review

1. Phase order — I have promoted B34 above B33. Agree? (Argument: B34 corrupts the
   instrument; B33 corrupts the system. Measure first.)
2. For B34, is excluding `role == "system"` acceptable, or should the router
   *refuse* to run any per-session cap until a true id is on the wire?
3. B33 touches core and needs an external restart — batch it with any other
   pending core work?
4. Should the compaction-outcome capture (Phase 3.1) live in `router_logs`, or stay
   in Hermes core where the msg counts actually exist?
5. Replay HAS been run and is recorded in `plan-b34-derived-key-is-system-prompt.md`
   §7 — C1/C3 RED today, C4 shows the proposed fix discriminates. Confirm these
   Phase-1 checks are the ones you want gated on before I start.
6. Cross-DB note for Phase 1 verification: `state.db.system_prompt_hash` lets the
   collision check be written in SQL against live sessions, so the regression test
   can assert "distinct derived keys >= distinct system_prompt_hash" on real data.
