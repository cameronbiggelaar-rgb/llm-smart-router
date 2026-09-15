# B34 — The "per-session" key is the SYSTEM PROMPT: the compression guard is global

**Status:** defect CONFIRMED and measured. Fix not yet written.
**Discovered:** 2026-09-15, during B32 post-deploy review — the user challenged a
"one session" claim that came from B32's own `derived` key.
**Severity:** the B31 guard does not do what its name says, and its accounting is
wrong in both directions.

## 1. What the user's challenge exposed

B32 added `session_id_source`. Post-deploy, the derived keys showed a dominant key
carrying **347 of 416 rows (83%)**, and I reported that as "one looping session".

The user pushed back: *two agent conversations were running in parallel* — why one
session? They were right. Hermes `state.db` shows **four** concurrent sessions in
the same window:

| Hermes session | thread | msgs | system_prompt_hash | router key |
|---|---|---|---|---|
| `20260913_212533_7f161b21` | 56495 | 1,093 | `bae8626232269081` | **collides** |
| `20260913_214955_da11827f` | 56467 | 125 | `bae8626232269081` | **collides** |
| `20260913_224247_cc840c4d` | 56463 | 307 | `bae8626232269081` | **collides** |
| `20260915_132557_12573f95` | 167978 | 7 | `f63e190d34f7b1b1` | unique |

**3 of 4 parallel sessions share ONE router key.** Decisive: the router key is
`sha256(system[:512])`, and state.db records that these three sessions have the
*identical* `system_prompt_hash`. This is the collision the user predicted.

## 2. Root cause — proven, not inferred

`_compression_session_key()` derives the key as
`sha256(first 512 chars of the joined message text)`. Message order puts the
**system message first**, and the system prompt is **9,589 chars** — so the hash
covers *only the system prompt*, which is identical for every session on the agent.

Verified three ways against the real dump:

```
manual sha256(system[:512])          = pfx:c1f9dd4aae8d0260c098f1a7d333f1e6
_compression_session_key(real msgs)  = pfx:c1f9dd4aae8d0260c098f1a7d333f1e6
dominant DB key                      = pfx:c1f9dd4aae8d0260c098f1a7d333f1e6  (n=331)
MATCH = True
```

And behaviourally — two *different* conversations:

```
with a shared system prompt : A -> pfx:41fe10d9…  B -> pfx:41fe10d9…  COLLIDE = True
without the system message  : A -> pfx:15d6268d…  B -> pfx:c9a17cd2…  COLLIDE = False
```

**The key is the system prompt, not a session.** Note the irony: the
`_compression_session_key` docstring claims *"the prefix is stable across turns
for a given session … and distinct between sessions"*. The first half is true and
that is exactly why the second half fails — it is stable because it is *constant*.

## 3. Consequences

1. **The guard is GLOBAL, not per-session.** B31 shipped
   `COMPRESSION_GUARD_PER_SESSION_MAX = 40` per rolling hour. With one shared key,
   **all concurrent conversations draw on one 40/hr budget.** One chatty session
   starves every other session's compaction treatment. The cap fires on sessions
   that were never looping.
2. **`session_compression_overflow` is inflated — MEASURED.** A row is relabelled
   overflow when the *shared* budget is exhausted, so the label says "this session
   is runaway" when it may mean "a different session used the budget". Post-deploy
   with the guard enabled (`COMPRESSION_GUARD_ENABLED=True`, cap 40/3600s):
   `session_compression_overflow` = **289 rows on ONE key**, versus
   `session_compression` = 95 rows on 2 keys. So the overwhelming majority of
   post-deploy compaction traffic is *relabelled* traffic on a single shared key,
   not genuinely-runaway sessions. Any analysis built on that label (including my
   own earlier churn numbers) over-attributes, and a healthy thread loses
   compaction treatment because a noisy sibling exhausted the shared cap.
3. **`derived` rows can never prove session isolation.** B32's skill note already
   warned that `derived` is not identity; this shows the failure is not a rare
   collision but the *default* state.

## 4. Ground truth: the loop IS real, but under-counted

`agent.log` is authoritative for compaction outcome (`router_logs` has no
msg-count column). Sampled outcomes:

```
total 'compressed N → M msgs' lines: 17
exact no-ops (N → N msgs):          17   <-- 100%
```

The no-op loop is real and total in this sample (`592→592`, `971→971`, `203→203`,
…). So B33 (route-blind charge) remains the cause of the churn; B34 is *separate* —
it corrupts the guard's accounting, not the trigger.

**Two independent defects, one visible symptom.**

## 5. The fix (proposed, needs review)

**Do not hash the system prompt.** Options, in preference order:

1. **Use the real identity when present, else refuse to cap.** The router already
   resolves `header`/`body` identities. For `derived` rows, the honest move is to
   *not* enforce a per-session cap at all (log it, serve normally) rather than
   apply a global cap mislabelled as per-session. A cap that cannot attribute is
   worse than no cap: it silently penalises innocent sessions.
2. **Salt the key with session-discriminating content** — e.g. hash the *tail*
   (newest turn) rather than the head, or exclude `role == "system"` from the
   joined text. Excluding the system message is the minimal change and is testable
   (the "without a system message" case above already shows it discriminates).
3. If Hermes can pass a real session id on the chat-completions path (it does on
   the Codex path via `prompt_cache_key`; the dump carries `session_id` at top
   level), option 1 becomes exact and the derived path is only a fallback.

Recommended: **(3) preferred, (2) as the fallback, (1) as the safety property** —
i.e. never enforce a per-session cap on a key known to be non-discriminating.

## 6. Additional data we need to capture (feeds the build plan)

- **Session identity on the wire.** Measured: chat-completions requests carry only
  `Authorization` + `Content-Type`. The `session_id` exists in the dump envelope but
  is not sent as a header. Until it is, per-session anything is approximate.
- **A collision-safety flag on the key.** `session_id_source='derived'` is not
  enough — it does not say *whether the key discriminates*. Record the number of
  distinct prefixes observed, or a boolean "key known non-discriminating".
- **Compaction outcome (N → M msgs, tokens before/after) must reach a queryable
  table.** It currently exists only in `agent.log`, so proving a no-op loop requires
  grepping logs with a different clock (AEST) from the DB (UTC). This is the single
  biggest gap in "is the data good?".
- **`state.db.last_activity_at` is MIXED-TYPE** (`null` 704, `text` 575, `real` 36)
  — my first concurrency query returned 0 sessions because I compared an ISO string
  to floats. Any cross-DB join must normalise types first.


## 7. Replay evidence (run against extracted production source, not a reimplementation)

`/tmp/replay_56467.py` pulls the *actual* `_compression_session_key` +
`_message_text` functions out of `scripts/biggie_llm_endpoint.py` and execs them,
so the replay tests production code rather than a paraphrase of it.

```
constants: COMPRESSION_GUARD_KEY_PREFIX_CHARS=512
           COMPRESSION_GUARD_PER_SESSION_MAX=40
           COMPRESSION_GUARD_WINDOW_SECONDS=3600

C1  two different conversations -> SAME key      RED   (defect confirmed)
C2  same conversation, 2 vs 3 turns -> same key  GREEN (stability preserved)
C3  key == manual sha256(system[:512])           CONFIRMED
C4  dropping the system message discriminates    YES   (proposed fix works)
C5  explicit session_id -> "sid:b32smoke001"     YES   (real id wins)
```

C1 and C3 are the defect; C2 is the property the fix must NOT break; C4 validates
the proposed fix; C5 confirms the preferred (real-identity) path already works.

**Real truncation is 32 hex chars** (`hexdigest()[:32]`), not 24.

## 8. Measured collision, final numbers

- post-deploy derived rows: **416**, across **5 distinct keys**
- dominant key: **347 rows (83%)**
- parallel Hermes sessions active in the same window: **4**
- of those, **3 share one key** (confirmed via `state.db.system_prompt_hash`)

So the guard's "per-session 40/hr" budget is drawn from **one pot by three
concurrent conversations** — and the fourth session, with a different system
prompt, is the only one correctly isolated.
