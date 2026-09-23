# Plan — split-canary monitor in the post-flip (no-control) era

## Context

B36d flipped `v41flash-tool-canary` to **percent: 100** (commit `e0a5500`). The
eligible cohort — tool-bearing `normal_chat`, excluding `debugging`/`testing` —
is now served **entirely** by `deepseek-v4.1-flash:cloud`. Pro is retained only
as the escalation fallback (`fallback_model: deepseek-v4-pro`).

`scripts/monitor_split_canary.py` was written for the **split** era: it compares
flash (treatment) against pro (control) in the same window. With the flip there
is no control traffic, so every comparison line degrades to `n/a` and the
escalation/failure guardrails — which are currently only evaluated for the
treatment arm against fixed thresholds — lose their reference point. The hourly
watchdog therefore stops being able to answer "is flash behaving like the pro
baseline it replaced?".

## Goal

Keep the hourly `split-canary-monitor` watchdog meaningful after the flip:

1. **Flash-only health** — always report flash volume, escalation rate,
   hard-failure rate, cost/completed for the window.
2. **Baseline comparison** — compare flash against the **pre-flip pro
   baseline** (the cohort's historical pro behaviour), not against a live
   control arm, when no control rows exist in the window.
3. **Delta-based guardrails** — flag when flash's escalation or hard-failure
   rate exceeds the pro baseline by more than a configured delta, in addition
   to the existing absolute thresholds.
4. **No control rows = expected, not an error.** Exit 0 with a report.
5. **Preserve split-era behaviour** — if control rows are present in the
   window (e.g. a future ramp-down), report the two-arm comparison as today.

## Design

- A **baseline** is computed from `router_logs` over a wide lookback window
  (default 7d) restricted to the *same eligible cohort predicates* the canary
  matches on (`requires_tools=1`, `workload_type='normal_chat'`,
  `task_type NOT IN ('debugging','testing')`), taking rows that were served by
  `deepseek-v4-pro` **and are not canary treatment rows**. This is the
  "what the cohort cost/failed like on pro" reference.
- Baseline is reported with its own sample size so a thin baseline is visibly
  thin rather than silently authoritative.
- Guardrails:
  - **absolute** (existing): escalation > 10%, hard-failure > 5%.
  - **delta** (new): escalation rate more than `+5pp` over baseline, or
    hard-failure more than `+2pp` over baseline — only evaluated when both the
    window and the baseline have enough volume (>= 30 requests) to be meaningful.
- Output gains a `baseline` block and an `arm_mode` field
  (`"split"` when control rows exist, `"flash_only"` otherwise) so downstream
  consumers (nightly digest, tests) can branch without guessing.

## Files

- `scripts/monitor_split_canary.py` — add baseline collection, `arm_mode`,
  delta guardrails, flash-only reporting.
- `tests/test_monitor_split_canary.py` — NEW. RED first:
  - flash-only window reports flash health and does not error
  - `arm_mode == "flash_only"` with no control rows, `"split"` with control rows
  - baseline is scoped to the eligible cohort, excludes canary treatment rows
  - delta guardrail fires when flash escalation exceeds baseline by > 5pp
  - delta guardrail does NOT fire when the baseline is thin (< 30)
  - absolute guardrails still fire
  - exit 0 + empty stdout when the window has no canary rows at all

## Out of scope

- Sticky-session splitting (explicitly declined by the user).
- Flipping `debugging`/`testing` onto flash.
- Any change to the nightly `router-lanes-nightly` digest (already carries the
  canary section; it will simply report flash-only once control traffic ends).

## Rollback

Monitor is read-only; revert by `git revert` of the monitor commit. No live
routing behaviour is touched by this change.
