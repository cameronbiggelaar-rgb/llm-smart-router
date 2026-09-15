# B34b — implementation plan for the derived-key / guard fix

**Status:** approved by user (2026-09-15): (1) fix B34 first, (2) exclude the
system message AND never enforce a per-session cap on a key that cannot
discriminate.
**Scope:** router-side only (`scripts/biggie_llm_endpoint.py`). No core change.
**Parent:** `plan-b34-derived-key-is-system-prompt.md`.

## Measured facts this design rests on

From 92 production request dumps (34 chat-completions payloads):

1. **The system prompt is always > 512 chars** (min 9,088, median 27,847,
   max 39,375). So the current key is *always* a pure system-prompt hash —
   never a partial prefix of the conversation. Not an edge case; the only case.
2. **Dropping `role == "system"` takes distinct keys from 3 → 21** on those 34
   payloads. Real improvement, but not sufficient — see 3.
3. **Compaction traffic still collides after dropping the system message.** Four
   *different* sessions (`20260731_064006_8d5cf822`, `20260906_073233_f6bb3957`,
   `20260911_114940_51fff757`, `20260913_074633_bb2d423b`) share one key because
   every compaction payload opens with the same constant
   `[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted…`
   wrapper. **Any prefix hash on compaction traffic is a wrapper hash, not a
   session identity.** No prefix strategy can fix this.
4. **A real identity DOES exist on the wire — for the Codex path.**
   `body.prompt_cache_key` is present on 58 of 92 dumps and **42 match a
   `state.db` session id exactly**. It is a genuine session id, not an opaque
   marker: 48 of 58 are `YYYYMMDD_HHMMSS_hex` shaped, 6 are `pck_…`, 4 are
   `bg_…`.
5. **The chat-completions path carries NO id.** Across all 33 chat-completions
   dumps (ollama.com, local 11434, the router's own 8080) `prompt_cache_key` is
   **absent** (pck=0), and headers carry only `Authorization` +
   `Content-Type`. The dump *envelope* has `session_id`; the body does not.
6. **The guard does not currently see header identity at all.** The call site
   builds its key with `_compression_session_key(messages, session_id=body.get("session_id") or body.get("user") or "")`
   — it never consults headers, and never consults `prompt_cache_key`, even
   though B32 added `resolve_session_identity()` for exactly that.

## Design

### 1. Identity resolution gains `prompt_cache_key`

Add to `resolve_session_identity`, after body `session_id`/`userId` fields and
before the derived fallback:

```
body.prompt_cache_key  ->  source "cache_key"
```

Priority becomes: `header` → `body` → `cache_key` → `derived` → `none`.
`SESSION_ID_SOURCES` grows by `"cache_key"`.

### 2. Derived prefix excludes the system message

`_compression_session_key` skips `role == "system"` when building the prefix.
Keeps the 512-char budget over the remaining messages. Rationale and measured
effect in fact 2. This is a strict improvement and cannot make discrimination
worse (the system message was provably the *only* thing being hashed).

### 3. The guard enforces ONLY on a real identity

New property: a per-session cap is meaningless without a per-session key.
`header`, `body` and `cache_key` are real identities → enforce.
`derived` → **do not enforce; log and serve normally**, flagged explicitly so the
non-enforcement is visible rather than silent.

This is the safety property the user approved. Justification, stated plainly:
the cap cannot attribute, so enforcing it does not contain a runaway session —
it penalises whichever innocent sessions share the key (measured: 289 overflow
rows on one key while 95 genuine compression rows sat on 2 keys).

**Trade-off, flagged not hidden:** containment for derived traffic is lost, so a
runaway on a derived session will buy the summariser rung until B33 fixes the
trigger. Accepted because (a) the summariser rung is the cheap
`deepseek-v4.1-flash` (total post-deploy compression spend ≈ $0.63), (b) the cap
demonstrably did not stop the loop (241 compactions/hr continued with it
enabled), and (c) B33 removes the trigger. If containment must be retained, the
correct mechanism is a **global** cap with a *measured* threshold — not a
per-session cap wearing a global key. That needs its own measurement; not in
this batch.

### 4. No new workload label

The overflow relabel keeps the single `session_compression_overflow` label, so
there is no second branch to update (the B31 pitfall about labels falling out of
matching branches). It simply fires far less often — only for real identities,
where it is accurate.

## Batches (TDD, RED first, one test file per batch)

**Batch 1 — `tests/test_b34_session_key.py`**
- `prompt_cache_key` resolves as `cache_key`, and outranks a derived key.
- A real header still outranks `prompt_cache_key`.
- Two conversations sharing a system prompt get **different** derived keys.
- Same conversation across turns stays **stable** (the property the fix must not
  break).
- A payload longer than 512 chars of non-system text never falls back to a
  system-prompt hash.
- `SESSION_ID_SOURCES` contains `cache_key`.

**Batch 2 — guard enforcement (`tests/test_b34_guard_scope.py`)**
- Real identity → cap enforced; exceeding it relabels overflow (existing
  behaviour preserved).
- Derived identity → cap **not** enforced, however many calls arrive; no
  overflow relabel; a warning is emitted.
- Guard key at the call site uses `resolve_session_identity` including headers,
  so a header-identified session is capped even though the body has no id.
- Non-vacuity: restore the old behaviour (enforce on derived) → the "derived not
  enforced" test goes RED.

## Verification before deploy

1. RED → GREEN per batch; commit per batch.
2. Non-vacuity injection per the skill's rule (restore each defect, confirm red).
3. Full suite + `scripts/preflight.py` 16/16.
4. **The regression assertion that would have caught B34 on day one:** distinct
   derived key count on live rows must be ≥ the number of concurrent Hermes
   sessions in the same window. Checkable in SQL via
   `state.db.system_prompt_hash` / session count.
5. Deploy = restart + **post-restart round trip** (send a request, read the row
   back), not `/v1/models` 200.
