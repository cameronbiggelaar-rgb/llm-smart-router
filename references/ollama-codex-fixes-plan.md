# Ollama/Codex fixes — plan (approved for post-restart, 2026-09-16)

Scope: `scripts/router.py`, `scripts/biggie_llm_endpoint.py` in this skill.
Convention: this doc first, then TDD batch-by-batch (RED → GREEN → commit per batch).
Production edits need explicit user sign-off + an OUT-OF-PROCESS restart
(assistant runs inside cgroup `/system.slice/hermes-gateway.service`).

---

## Recon findings (2026-09-16, all measured)

### Deployment reality (corrects two earlier beliefs)
- `biggie-llm-endpoint.service` runs the REPO file:
  `ExecStart=/usr/bin/python3 /home/hermes/.hermes/skills/llm-smart-router/scripts/biggie_llm_endpoint.py`
  (WorkingDirectory = `.../scripts`, User=hermes, PID 2686923, up since 2026-09-15 16:50 AEST).
- Earlier claim "running endpoint is a DIFFERENT build than the repo file" was WRONG.
  The failure-log template is `"Backend %s request failed: %s"` (provider substituted),
  so grepping the literal string `"Backend local request failed"` in the repo naturally found nothing.
- `_MODEL_STATUSES` is populated at import: `router.py:293  _MODEL_STATUSES = _load_model_states()`.
  State DB (live) = `data/router_state.db`. There is a 0-byte decoy at
  `llm-smart-router/router_state.db` (no `model_state` table) — ignored by code.

### FIX 1 ALREADY EXISTS. The premise is wrong.
Account-wide short-circuit is implemented (B28) and DEPLOYED and FIRING:
- `mark_account_rate_limited(account, ...)` (router.py:506) marks every model on the account
  with `count_failure=False`, then logs `"Account openai-codex rate-limited by upstream cap — 5 models held: ..."`.
- Fired **2,912× in 24h**; each event holds the full family.
- Measured state right now (all 5 held, correct absolute reset):
  `gpt-5.5/5.6-luna/5.6-terra/5.6-sol/6-astra` → `available=0`, `held=+83.4h`,
  `rate_limited_until` set from upstream `resets_at` / `resets_in_seconds` (300,647 → 299,897 as time passes).
  The 1-hour backoff cap is correctly NOT applied (router.py:319-335 documents this).
- Live `/status` agrees: the 4 siblings show `🟡 RATE LIMITED 299897s`.

**In-process reproduction against real state** (`/tmp/fix_repro_56467.py`, no mocks):
```
gpt-5.5  available=False held=+83.4h is_available()=False
...(all five held)
get_available_models() -> 13 models, codex entries: []
route_task(normal_chat, cpx=0.45) -> glm-5.3-flash   (correctly avoids codex)
route_task(session_compression)   -> deepseek-v4.1-flash
```
So the family IS excluded from normal selection. "Walk each one" is NOT happening.

### The actual defect: `force_model` bypasses availability
`route_task()` has an early-return `if force_model:` branch that builds a
`RoutingDecision` for the forced model with **no `is_model_available()` check**.
Proven:
```
route_task(force_model="gpt-5.5") -> selected_model='gpt-5.5', reason='forced model: gpt-5.5'
```
even though gpt-5.5 is held. All-time `routing_reason` shows forced codex:
`forced model: gpt-5.5` ×8, `forced model: gpt-6-astra` ×2.
This is the surviving hole that lets a capped account keep being called.

### Unresolved (stated honestly)
Logs show routing to `openai-codex/gpt-5.5` at 12:06:13 while the same process reports
gpt-5.5 `CIRCUIT OPEN` — i.e. codex is dispatched despite the hold, ~1–3×/min,
while `router_logs` records **zero codex rows in 24h** (failures 502 out before the row is written).
Log lines interleave across concurrent requests, so a single pairing is not proof.
There is exactly ONE codex call site (`BUILTIN_PROVIDER_URLS["openai-codex"]`, endpoint.py:1026),
reached only via `proxy_to_backend`. Candidate dispatchers: (a) `force_model`,
(b) the non-streaming SSE fallback candidate list (`_non_streaming_fallback_to_sse`,
`alternative_backends`, endpoint.py ~3349), (c) an unlogged/error path.
**Fix: make the dispatch site LOG the request_id + dispatcher + target** (batch 2) so this is
attributable instead of inferred — this is the same telemetry gap as FIX 4.

### FIX 2 root cause
Local's declared context is **not** in this repo. Ollama's own metadata says:
```
curl /api/show {"model":"llama3.1:8b"} -> model_info["llama.context_length"] = 131072
```
while the running server is started `-c 4096 -np 1 ... --context-shift --keep 4`
(CPU-only, `size_vram: 0`). Hermes' context resolver trusts the declared value
(`agent/model_metadata.py::_probe_local_context_length` / `_reconcile_local_cached_context_length`,
with `MINIMUM_CONTEXT_LENGTH` invalidation and a documented vLLM/`--max-model-len` under-report case at :2017).
`--context-shift` then silently drops the OLDEST tokens, so big compactions produce
truncated/empty summaries → empty-content death spiral → retries.
The router endpoint does NOT declare 131072 anywhere (grep clean), and neither the endpoint unit
nor ollama.service sets `OLLAMA_CONTEXT_LENGTH`.

### FIX 3 site
`_get_httpx_client()` (endpoint.py:934) is ONE shared client at `timeout=300.0` for all backends —
so a saturated 1-slot local server holds connections ~300s and `api_max_retries: 3` multiplies it.
Local needs its own short-timeout client.

### FIX 4 site
Six identical swallowing sites, e.g. endpoint.py:2555/2669/2707/3075:
```
logger.error("Backend %s request failed: %s", provider, e)
```
`str(e)` is empty for many httpx errors → log reads `"Backend local request failed:"` with no type.

---

## Batch plan (TDD: RED → GREEN → commit per batch)

### Batch 1 — `force_model` must respect the account hold  (fixes 1)
- RED: `tests/test_force_model_availability.py` — held model ⇒ forced selection must not
  return the held model; expect fail-closed (empty selected + `all_exhausted=True`) or
  demotion to the best available, per user's choice.
- GREEN: in `route_task`'s `if force_model:` branch, consult `is_model_available(force_model)`.
  Same guard in `escalate_on_failure` (router.py:1143) so escalation cannot re-enter a held family.

### Batch 2 — attributable dispatch telemetry  (fixes 4 + closes the unresolved)
- RED: tests asserting (a) the log line carries the exception TYPE (`type(e).__name__`) and
  `repr(e)`, and (b) every dispatch logs `request_id` + which dispatcher + resolved target.
- GREEN: at all six `Backend %s request failed` sites log `type(e).__name__` + `repr(e)`;
  add the dispatcher/target log at `proxy_to_backend`.
- Deliverable: the 1–3/min codex calls become attributable → then fix the real caller.

### Batch 3 — local declared context (fixes 2) — NEEDS USER DECISION
Option A (declare 4K, fail fast): treat local as a 4,096-token model; refuse/skip local for
oversized prompts instead of silently truncating. No RAM/perf change; local stops being handed
work it cannot do.
Option B (raise server): give Ollama a real window (`OLLAMA_CONTEXT_LENGTH`, e.g. 16,384–32,768).
WARNING: CPU-only, `-np 1`, ~5.66/6 cores busy, load ~8 — a large `num_ctx` will slow and
memory-inflate the box. 131,072 is NOT advisable here.
Option A is recommended; B is only worth it if local must genuinely serve large compactions.

### Batch 4 — fast-fail local (fixes 3)
- RED: tests that a local request uses a short timeout (~60s), not the shared 300s, and that
  local is circuit-broken when queue depth exceeds a threshold.
- GREEN: dedicated `httpx.AsyncClient(timeout=60.0)` for `provider == "local"`; optional
  bounded circuit-break keyed on in-flight/queue depth.

## Ordering / constraints
- Batches 1–2 are safe, additive, and testable now.
- Batches 3–4 touch live latency/timeout behaviour → land with the restart window.
- Do NOT use bare pytest: CORE repo rule is `scripts/run_tests.sh`.
- The restart to activate `model.base_url` (already applied, `ctx=128000 source=config`) is
  STILL PENDING: gateway `hermes-gateway.service` MainPID=3106321, started 2026-09-15 20:58 AEST.


---

# COMPLETION RECORD (2026-09-16)

All four fixes implemented and committed. The plan above was written from recon;
the notes below record where reality differed.

| Batch | Fix | Commit | Scope |
|-------|-----|--------|-------|
| 1 | #1 `force_model` bypass | `003df92` | router.py |
| 2 | #1 true dispatcher (reachable-backend tier-walk) | `6493da4` | biggie_llm_endpoint.py |
| 3 | #2 local declares 4,096 (Option A) | `67bd8e5` | router.py |
| 4 | #3 fast-fail local + #4 log exception type | `970da2a` | biggie_llm_endpoint.py |

Full suite: 653 passed. Each batch: RED → GREEN → non-vacuity probe → restore
byte-identical → commit.

## Where the plan was wrong

1. **Fix #1 had TWO causes, not one.** The plan named the `force_model`
   early-return (endpoint.py:3608). That hole is real but was NOT the live
   dispatcher — zero codex rows in `router_logs` for 24h, only 16 forced
   requests all-time. The actual ~75 cap events/hour came from the
   "not in discovered backends → fell back to best reachable" tier-walk, which
   selected HELD gpt-5.5 by capability tier with no availability check, and was
   **duplicated** on the escalation path. Both are now fixed; batch 2 extracted
   the duplication into one availability-aware helper, because the copy is why
   the bug existed twice.

2. **Fix #4 was "three log sites", actually eleven.** Only three used the
   `"Backend %s request failed: %s"` wording; related sites used different
   phrasing (stream preflight, 502 details). All eleven now name the exception
   class. The root problem was subtler than "missing type": httpx timeout
   errors stringify to the EMPTY string, so the log read
   `Backend local request failed:` — nothing after the colon.

3. **Fix #3 did not need a second client.** The plan proposed a dedicated
   `httpx.AsyncClient(timeout=60.0)`. There is exactly ONE local dispatch site
   (the streaming path delegates to it), so a per-provider `_timeout_for()` is
   enough — no second client, no extra connection pool.
   `_timeout_for()` always returns a float, never None-for-default: httpx reads
   `timeout=None` as "never time out", so None would silently disable timeouts.
   The optional **queue-depth circuit-break was NOT implemented** — the 60s
   fast-fail makes the queue self-limiting (callers escalate instead of
   parking), and a depth threshold would need tuning evidence we do not have
   yet. Left as a follow-up.

4. **Option A needed a third gate the plan did not anticipate: limp-home.**
   Beyond the fallback chain and escalation, local is also reached in limp-home
   mode (cloud fully exhausted). Oversized requests there now fail closed
   rather than returning a summary silently missing its oldest tokens
   (`--context-shift` drops from the front). This matches how the same function
   already fails closed for tool-required work in limp-home. **This is a
   behaviour change worth watching: a very large compaction during total cloud
   exhaustion now errors instead of degrading.** That trade was chosen
   deliberately — silent truncation is undetectable downstream, a loud failure
   is not.

## Still pending

- **Restart to activate everything**: the four batches (endpoint changes are
  not live — running PID still serves pre-batch-2 code) plus the already-applied
  `model.base_url` fix (`ctx=128000 source=config`). Issue from OUTSIDE the
  gateway cgroup.
- Follow-up: local queue-depth circuit-break, if evidence shows 60s is not
  sufficient.
