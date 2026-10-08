# Plan B45 — GPT by request only

**Status:** implemented (B45, commit `02c284f`); source-only, inert until an
external endpoint restart.
**Predecessor:** B44a–d (the silent fail-open, closed and made countable).
**Operator directive (2026-10-08):** *"No that's ok for now I think we can make it
ask for the gpt model."* — decline the classifier tightening, and instead make
explicit request the entitlement.

## 1. Why this supersedes the classifier tightening

B44 closed the governor's silent fail-open, but left the *entitlement rule* as
"architectural class ⇒ entitled". I measured that rule's weak point: the
classifier is a bare substring match, so ordinary phrasing classifies
architectural —

```
rethink the wording of this email   -> architectural
rearchitect my resume               -> architectural
let's rethink dinner plans          -> architectural
update the plans for the weekend    -> architectural
```

Tightening the keywords would fix the examples, not the class of defect: any
substring heuristic can be tripped by ordinary language. Making the **explicit
ask** the entitlement removes the classifier from the trust path entirely — a
false positive can no longer admit a request to the plan, because classification
alone stops granting anything.

## 2. The mechanism, verified

Two probe results pin this (scratch: `probe_explicit_bypass.py` →
`explicit_bypass.txt`).

**Fact 1 — the explicit path already bypasses the governor.** Naming a plan
model returns before the governor hook in both branches:

| Request | Selected | Class | Governor |
|---|---|---|---|
| `force_model="gpt-5.5"` (ordinary) | `gpt-5.5` | architectural | **BYPASSED** |
| `force_model="gpt-5.5"` (tool) | `gpt-5.5` | *(empty)* | **BYPASSED** |
| auto `summarise this email` | `llama3.1:8b` | routine | consulted |
| auto `rethink the routing design` | `glm-5.3-flash` | architectural | consulted |
| auto tool `add a column to the schema` | `deepseek-v4.1-flash` | routine | consulted |

So "explicit only" is not a new mechanism — it is *narrowing the automatic
entitlement* so the escape hatch is the **only** way through. The operator's
existing guarantee ("never blocked from gpt on demand", §3.3) is preserved by
construction.

**Fact 2 — a second live leak: escalation with an empty class.** The escalation
and chain guards are written `if budget_class and _is_plan_model(model) and
_plan_denied_for_class(budget_class)`. The `budget_class and` conjunct means an
**empty** class disables the guard. Probed directly:

```
class=''              -> gpt-5.6-sol     (guard skipped)
class='routine'       -> deepseek-v4-pro (guard fires)
class='architectural' -> gpt-5.6-sol     (entitled today)
```

Production already shows this firing — 431 rows escalated into `gpt-5.6-sol`
with **no class**, plus 14 `gpt-5.6-terra`, 13 `gpt-5.5`, 6 `gpt-5.6-luna`:

```
class=(none)  gpt-5.6-sol    431
class=(none)  gpt-5.6-terra   14
class=(none)  gpt-5.5         13
class=(none)  gpt-5.6-luna     6
```

Under B45 an unclassified automatic request must be treated as **not entitled**,
which fixes this path as a side effect of the same rule.

## 3. Design

### 3.1 Policy (`scripts/budget_policy.yaml`)

New account key, default `false` so the change is opt-in and reversible:

```yaml
accounts:
  openai-codex:
    explicit_only: true      # entitlement is the explicit ask
    allowlist: [architectural]   # governs NON-explicit paths when explicit_only is false
    hard_cap_percent: 95
```

Entitlement, stated once:

- `explicit_only: true` → **entitled iff the request explicitly named a plan
  model.** The allowlist is moot for automatic paths.
- `explicit_only: false` → today's behaviour (`klass in allowlist`). The
  rollback lever `enabled: false` still opens every class.

### 3.2 `gpt_budget.py`

- `GovernorConfig` gains `explicit_only: bool = False`; `_parse_policy` reads it
  (`bool(account.get("explicit_only", False))`), `_default_config` leaves it off.
- `evaluate_policy(reading, cfg, *, klass, explicit=False)` — one new kwarg.
  Precedence:

  1. `enabled=false` → `OPEN` (rollback, unchanged).
  2. unreadable probe → `UNVERIFIED` if entitled else `DENIED`
     (under `explicit_only`, a blind probe does **not** admit a non-explicit
     request — closing the leak's own escape route).
  3. `allowed is False` → `CLOSED`.
  4. `used >= hard_cap` → `CLOSED`.
  5. entitled → `OPEN`, else `DENIED`.

  Entitlement helper, single definition:

  ```python
  def _entitled(klass, cfg, explicit):
      if cfg.explicit_only:
          return bool(explicit)
      return klass in _known_classes(cfg.allowlist)
  ```

### 3.3 `router.py`

- `_plan_explicit(force_model)` → `_is_plan_model(force_model)`: the request
  asked for a plan model by name. This is the *only* source of entitlement.
- `_apply_plan_governor(selected, *, prompt, force_model)` → gains `explicit`,
  passes it to `evaluate_policy`.
- `_plan_denied_for_class(klass, explicit=False)` and `_governor_verdict(...)`
  gain the kwarg so chains, escalation and the governor all agree.
- `RoutingDecision` gains `plan_explicit: bool` so escalation and chains inherit
  it exactly as they inherit `budget_class`.
- **Remove the `budget_class and` conjunct** from the three guards
  (`escalate_on_failure` prefer-model ~1930, tool ladder ~1961, capability
  ladder ~1991, degrade branch ~2013). Replacement:
  `if _is_plan_model(model) and _plan_denied_for_class(budget_class, plan_explicit)`.
  An empty class must no longer mean "unrestricted".
- `_contain_plan_in_chain(chain, klass, explicit)` and both chain builders take
  `explicit`; non-explicit chains are stripped of plan models.
- The forced tool-branch early return (~1037) records `budget_class` and
  `plan_explicit=True` so an explicit request is visible in reporting (today it
  logs an empty class).

### 3.4 Observability

No new column. `classify_lane` already buckets an explicit request as
`LANE_FORCE_MODEL`, which is exactly the countable signal for "reached gpt by
request"; the reason string gains the explicit/non-explicit wording.

## 4. Test plan (`tests/test_gpt_by_request_only.py`, RED first)

Policy truth table:
1. `explicit_only` + auto architectural → `DENIED` (the point of B45).
2. `explicit_only` + explicit (any class) → `OPEN`.
3. `explicit_only` + blind probe + non-explicit → `DENIED`, not `UNVERIFIED`.
4. `explicit_only` + blind probe + explicit → `UNVERIFIED`.
5. `explicit_only: false` + architectural → `OPEN` (regression: B44 behaviour kept).
6. `enabled: false` → `OPEN` for every class and every explicit value.

Routing:
7. auto architectural prompt cannot reach a plan model.
8. explicit `force_model="gpt-5.5"` reaches it, class recorded, `plan_explicit`.
9. escalation with class `''` and non-explicit cannot reach the plan (the
   431-row hole).
10. escalation with class `''` and explicit may reach it.
11. fallback chain built for a non-explicit request contains no plan models.

## 5. Rollback

Two independent levers, no code change, hot-reloaded in ~30s:

- `explicit_only: false` → restores the architectural-lane allowlist (B44 state).
- `enabled: false` → restores pre-B37 behaviour entirely.

## 6. Out of scope

- Tightening the classifier keywords (declined by the operator; moot under B45).
- A `plan_explicit` DB column — `LANE_FORCE_MODEL` already carries the signal.
- Deploy/restart. Source-only until an external endpoint restart is authorised.
