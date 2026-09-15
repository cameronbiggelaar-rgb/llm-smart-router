# B33 — UPDATE: route-blind charge in the anchored preflight estimator

**Status:** design UPDATED (2026-09-15). Not implemented — core change, own gate.
**Supersedes:** `plan-b33-route-blind-preflight-charge.md` (first draft)
**Author:** assistant, thread 56467
**Depends on:** B32 (`session_id_source`), B34 (guard contract), B35 (wire identity)
**Scope:** `hermes-agent` core — 3 gating call sites.

## 0. What changed since the first draft

Three things, from reading the actual code rather than the earlier reconstruction:

1. **The charge is in the DELTA, not the base.** `anchored_context_tokens`
   computes `total = anchor["prompt_tokens"] + anchor["completion_tokens"]`
   (provider-EXACT) and then adds
   `estimate_messages_tokens_rough(delta, charge_stale_thinking=charge_stale_thinking)`
   for messages appended since the anchor. The provider-exact base is never the
   problem; the route-blind inflation is entirely in the delta estimate.
   Consequence for the fix: passing the flag changes only the delta, so the
   anchored figure stays provider-anchored. This makes the change smaller and
   safer than "replace the estimator".

2. **There are exactly 3 GATING sites, not 4**, plus 3 non-gating consumers that
   want different values. A blanket default flip would break the others (§3).

3. **The predicate already exists and is already used** — on the *fallback*
   path only. `turn_context.py:62-64` passes
   `charge_stale_thinking=_agent_stale_thinking_on_wire(agent)`. So the codebase
   already decided the route-aware answer is correct; the anchored path just
   doesn't use it. This is an omission, not a missing mechanism.

## 1. Root cause (unchanged, restated precisely)

`_preflight_request_tokens()` prefers the anchored estimate when the anchor is
fresh:

```python
anchored = anchored_context_tokens(messages, getattr(agent, "_usage_anchor", None))
agent._request_pressure_anchored = anchored is not None
if anchored is not None:
    return anchored
```

`anchored_context_tokens` declares `charge_stale_thinking: bool = True`, and
this caller **omits the argument**, so the route-BLIND value wins over the
route-aware fallback three lines below it.

Measured on the real dump (`request_dump_20260913_074633_bb2d423b`):
**route-blind 364,368 vs route-correct 158,603 → 205,765-token inflation.**
`stale_thinking_reaches_wire()` returns **False** for the primary route, so the
inflation is charged for content the provider never sees. Production telemetry
pinned `current_estimated_tokens` at 180,236 / 180,608 / 181,490 / 184,427 —
just above the 180,000 line, repeatedly, on 3 sessions. That is the loop.

## 2. The exact call sites

### Gating (must pass the flag) — the fix

| file:line | context |
|---|---|
| `agent/turn_context.py:43` | `_preflight_request_tokens()` — returns the figure that gates preflight compression |
| `agent/turn_preflight.py:273` | `_real_tokens = _anchored` when non-None; feeds `_compressor.should_compress(_real_tokens)` |
| `agent/turn_request_assembly.py:247` | `request_pressure_tokens = _anchored_pressure`; also sets `agent._request_pressure_anchored` |

These three are the compaction TRIGGER. They must agree, or the trigger and the
tail-budget walk disagree — which the `stale_thinking_reaches_wire` docstring
names as the cause of an infinite loop.

### Non-gating (do NOT flip blindly)

| file:line | why it differs |
|---|---|
| `agent/context_breakdown.py:153` | already passes `charge_stale_thinking=False` for one branch; this is a **display** figure and deliberately shows the uncharged number |
| `hermes_cli/cli_status_bar_mixin.py:318` | status-bar display; must match what the user is told, not what gates |
| `agent/image_token_cost.py:127` | accounting of image token cost, not pressure |

`cli_status_bar_mixin` and `context_breakdown` were already reviewed as display
consumers — the first draft flagged `context_breakdown` as "for review"; it is
now settled: leave both display paths alone, fix only the gating sites.

## 3. Fix

**Recommended: explicit at the 3 gating sites.** Do NOT change the default —
three callers legitimately want `True`, and a default flip is a silent
behavioural change for every future caller.

```python
# agent/turn_context.py:43
anchored = anchored_context_tokens(
    messages,
    getattr(agent, "_usage_anchor", None),
    charge_stale_thinking=_agent_stale_thinking_on_wire(agent),
)
```

Same shape at `turn_preflight.py:273` and `turn_request_assembly.py:247`.
`turn_preflight` and `turn_request_assembly` need the helper imported; it lives
in `agent/turn_context.py` — either import it or promote it to a shared module
(prefer promoting: it is route truth, not turn-context internals).

**Alternative (weaker):** change the default to route-aware. Rejected: it breaks
the display consumers silently.

**Guard the invariant:** add a test asserting the trigger estimator and the
tail-budget walk pass the SAME `charge_stale_thinking` value for a given route.
That is the property the docstring demands and nothing currently enforces.

## 4. Expected effect

The sessions pinned at 180,236–184,427 should drop to roughly the
route-correct figure (~158,603 on the measured dump) — i.e. **below** the
180,000 line, so preflight stops firing on every turn. Predicted: the
compaction rate collapses from ~240/hr toward real traffic.

**Measure it, do not assume it:** capture before/after
`current_estimated_tokens` and compactions/hr on the same sessions. If the
figure does NOT move, the fix is in the wrong place and must be reverted before
anything else is concluded.

## 5. Risks

| risk | mitigation |
|---|---|
| Predicate wrong for a route → under-charge → sessions exceed the real window | The predicate is the SAME one the fallback path already trusts; it is not new. Verify `stale_thinking_reaches_wire` on the live route before/after. |
| Trigger and tail-budget walk disagree → the exact infinite loop being fixed | The invariant test in §3. This is the highest-risk failure and must be tested, not reasoned about. |
| Hidden 4th gating consumer | Enumerated by grep (§2). Re-grep at implementation time; the list is 6 total, 3 gating. |
| Compaction becomes too permissive → context overflows | `should_compress` still compares against the real threshold; only the INPUT to it changes. Watch for provider-side context errors post-change. |

## 6. Verification plan

1. **Before:** record `current_estimated_tokens` and compactions/hr for a window.
2. Unit-test the invariant: trigger and tail-budget walk agree per route.
3. Non-vacuity: revert the flag at one site → the invariant test must go RED.
4. **After restart:** same measurement. Expect a step change, not a drift.
5. Confirm no new provider context-length errors in the journal.
6. **Revert criterion:** if `current_estimated_tokens` is unchanged, revert and
   re-diagnose rather than layering another fix.

## 7. Relationship to B34 / B35

- **B34** removed the guard's ability to cap a non-discriminating key. So B33 is
  now the ONLY containment for the loop — it fixes the cause rather than
  relabelling the symptom. This raises B33's priority.
- **B35** puts a real session identity on the wire, so post-B33 the loop is
  attributable per compression lineage and the guard is enforceable again.
- Order: B35 (done) → B33 → re-verify the guard actually fires on real runaways.

## 8. Open questions

1. Promote `_agent_stale_thinking_on_wire` to a shared module, or import it from
   `turn_context`? (Prefer promoting.)
2. Is `charge_stale_thinking` the ONLY route-dependent input to the delta
   estimate, or is there a second one that also needs threading?
