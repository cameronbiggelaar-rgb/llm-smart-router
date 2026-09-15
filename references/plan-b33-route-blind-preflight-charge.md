# B33 — Compaction loop: route-blind charge in the anchored preflight estimator

**Status:** design (NOT implemented — core change, root-owned, needs external restart)
**Author:** assistant (this thread, 56467)
**Depends on:** B32 (measurement integrity) — `aab2c04`
**Scope:** `hermes-agent` core. NOT router-side.

---

## 1. Root cause (confirmed by measurement, not inference)

Hermes' compaction trigger charges `reasoning_content` against the context budget
**unconditionally on the anchored path**, even when the active route strips that
reasoning before it reaches the wire.

`stale_thinking_reaches_wire()`'s own docstring predicts this exact failure:

> "The single wire-truth predicate the compaction TRIGGER estimator and the
> tail-budget walks must share: if they disagree, a reasoning-heavy session can
> look over-threshold to preflight yet fully tail-protected to the walk — **an
> infinite compaction loop**."

That disagreement is present. `agent/turn_context.py`:

```python
def _preflight_request_tokens(agent, messages, system_prompt) -> int:
    anchored = anchored_context_tokens(messages, getattr(agent, "_usage_anchor", None))
    if anchored is not None:
        return anchored                      # <-- WINS, route-blind
    ...
    return estimate_request_tokens_rough(
        messages, system_prompt=..., tools=tools,
        charge_stale_thinking=_agent_stale_thinking_on_wire(agent),   # <-- route-aware
    )                                        # <-- NEVER REACHED
```

`anchored_context_tokens()` is declared:

```python
def anchored_context_tokens(messages, anchor, *, charge_stale_thinking: bool = True)
```

**`True` is the default, and the preflight callers omit the argument.** So the
gating estimate charges reasoning regardless of route — while the generic
estimator below it, which *does* ask the route, is unreachable whenever a usage
anchor exists (the normal case on a long session).

### Measured on the real session (dump `20260913_074633_bb2d423b`)

| Estimator | Tokens | Over 180,000? |
|---|---|---|
| route-**blind** (what preflight uses) | **364,368** | **YES** |
| route-**correct** (what it should use) | **158,603** | no |
| inflation | **205,765** | 100%+ of the threshold |

Route: `provider=biggie-llm`, `model=biggie-router`, host `127.0.0.1`.
`stale_thinking_reaches_wire(...)` = **False** (no `deepseek` substring, provider not
in the echo family set, host not `api.deepseek.com`) — so Hermes *itself* strips
`reasoning_content` on the way out. The estimator charges it anyway.

### Independent corroboration from production telemetry

`agent.log` `compression_attempt` events show `current_estimated_tokens` pinned
just *above* the threshold, repeatedly, on the same sessions:

```
current_estimated_tokens=181490  effective_threshold=180000  aborted
current_estimated_tokens=180608  effective_threshold=180000  aborted
current_estimated_tokens=180236  effective_threshold=180000  aborted
current_estimated_tokens=305982  effective_threshold=180000  aborted
```

That is loop-bounding behaviour, not a coincidence: the charge is ~205K larger
than reality, so no amount of pruning gets the estimate under a line it never
genuinely crosses. Consistent with the earlier probe showing the provider billed
**48 tokens for 300,000 characters** of `reasoning_content`.

### Why the earlier hypothesis was wrong

I previously claimed the phantom was `_NEWEST_TURN_ONLY_BUDGET_KEYS` /
`_STALE_REPLAY_PRUNE_KEYS` not stripping reasoning, and that the fix was to stop
charging it globally. **That was wrong.** `charge_stale_thinking` already exists
and is already threaded correctly in the *generic* path. The defect is narrower:
one default argument plus callers that omit it, on the path that actually gates.

---

## 2. Fix (single mechanism, four call sites)

**2.1 Change the default to fail-safe-per-route.** Rather than `True`, make the
anchored helper ask the route when not told:

```python
def anchored_context_tokens(messages, anchor, *, charge_stale_thinking: bool | None = None):
    if charge_stale_thinking is None:
        charge_stale_thinking = True   # preserve current behaviour unless route known
```

The safer shape is to **leave the default alone** but make each gating caller pass
the route truth explicitly — a default change silently alters four unrelated
callers (`image_token_cost`, `context_breakdown`, `turn_preflight`,
`turn_request_assembly`), each of which has its own reason to pick a value.

**Recommended: explicit at the gating sites.**

| File | Line | Current | Change |
|---|---|---|---|
| `agent/turn_context.py` | 43 | omits flag | `charge_stale_thinking=_agent_stale_thinking_on_wire(agent)` |
| `agent/turn_preflight.py` | 273 | omits flag | same |
| `agent/turn_request_assembly.py` | 247 | omits flag | same |
| `agent/context_breakdown.py` | 156 | omits flag | keep `True`? — **display only, decide separately** |

`context_breakdown` feeds the *status display*, not the trigger. If it charges
differently from the trigger the UI will disagree with the router; if it matches,
the displayed number drops ~205K. **Recommendation:** make it match the trigger,
so the operator sees the number the decision was made on. Call this out for
review rather than assuming.

`image_token_cost.py:127` is measuring per-image cost from real vs text-only
usage — leave it; it is calibration, not gating.

**2.2 Guard the invariant.** The docstring says the two paths *must* share the
predicate. Add a test that asserts they agree, so a future caller cannot
reintroduce the split silently.

---

## 3. Risks

| Risk | Assessment |
|---|---|
| Under-charging a route that DOES echo reasoning | The predicate is the existing, shipped wire-truth function. If it says False, `apply_reasoning_content_policy` already strips the field, so nothing is sent. Charging it would be charging for bytes that never leave. |
| A *fallback* route that does echo | The predicate is evaluated per active route, and fallback activation re-reads it (`_reasoning_echo_flag` is re-read on `switch_model`). Fallback to `deepseek-*` correctly returns True. |
| Sessions that genuinely exceed the threshold | Unaffected — their `content` alone is over the line. |
| Triggering *less* compaction hides a real problem | Acceptable and correct: the compaction was not fixing anything. 237 of 697 logged compactions removed ZERO messages. |
| Core is root-owned, needs external restart | Same as B30/B31 precedent. Requires user sign-off + out-of-process restart. |

**Expected effect:** the sessions currently pinned at 180,236–184,427 will drop to
~155–160K and stop compacting. That is the loop, not a behaviour change.

---

## 4. Additional data we should capture (per the design brief)

Some of these are the *reason* the defect survived this long — the telemetry
records the symptom and never the cause.

### 4.1 Core-side (new)

| # | Data | Why | Where |
|---|---|---|---|
| 1 | `estimate_source` on `compression_attempt`: `anchored` \| `native` \| `generic` | The telemetry logs `current_estimated_tokens` but **not which estimator produced it**. The route-blind path and the route-aware path are indistinguishable in the log — which is exactly why a 205K disagreement read as a plausible number for months. | `agent/turn_context.py` → existing telemetry dict |
| 2 | `charge_stale_thinking` (bool) on `compression_attempt` | Names the defect directly. With #1, one query proves the disagreement. | same |
| 3 | `estimate_route_blind` vs `estimate_route_correct` (both ints) | Quantifies the phantom per attempt in production, not just in a captured dump. | same |
| 4 | `x-hermes-session-id` header on chat-completions | Real attribution. The Codex path **already sends it** (`extra_headers.session_id`); chat-completions sends nothing, forcing the derived key. | `agent/` request assembly |
| 5 | `stale_thinking_on_wire_resolved: bool` | Distinguishes "route says strip" from "route unknown, failed conservative". Today a failure to classify is invisible and indistinguishable from a genuine echo route. | `agent/turn_context.py:68` `except: return True` |
| 6 | Timezone offset marker on `agent.log` lines | Logs are local AEST with no marker; the DB is UTC. Direct comparison is off by ten hours and looks plausible. (This session's own error.) | `hermes_cli/logs.py:51` |

### 4.2 Router-side (this repo)

| # | Data | Status |
|---|---|---|
| 7 | `session_id` populated + `session_id_source` | **DONE — B32** (`aab2c04`) |
| 8 | Per-session compaction count in a rolling window | computed for the guard — **persist it** so churn is auditable without log scraping |
| 9 | Session message count + turn count per row | derivable from the payload; makes churn attributable to *growth* vs *loop* |
| 10 | Billed input vs locally-estimated input, per call | closes the loop: proves the phantom end-to-end in production data rather than a one-off probe |

### 4.3 What B33 needs from B32 (the ordering matters)

Items **#1–#3** are unanswerable without **#7**: you cannot attribute an estimate
disagreement to a session when `session_id` is blank. That is why B32 shipped
first.

---

## 5. Open questions for review

1. Should `context_breakdown` (status display) match the trigger, or keep
   charging conservatively? Matching makes the UI consistent with the decision.
2. Should the anchored default become `None` (ask the route) rather than staying
   `True`? Safer against future callers, riskier to four existing sites.
3. Is there a legitimate reason a preflight estimate should *over*-charge
   reasoning? Over-charging only causes spurious compaction, which is what we are
   trying to stop — but confirm no safety margin depends on it.
