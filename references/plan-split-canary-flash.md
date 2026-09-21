# Plan — split canary: serve a fraction of tool-bearing traffic from flash

## Goal

Move a bounded, reversible slice of tool-bearing `normal_chat` traffic from
`deepseek-v4-pro` (tier 8) to `deepseek-v4.1-flash` (tier 3) as a **served
split**, so flash *replaces* pro on that slice (saving ~67% on those calls)
instead of running alongside it (the shadow, which costs 2× per request).

Evidence already in hand (shadow, closed 2026-09-20):
- Flash streamed: 99.65% success, ~91% cache rate, **$0.00195/req** vs pro **$0.00585/req**.
- Tool-set agreement vs pro: **54%** (flash diverges ~46%, mostly `terminal` vs pro's `read_file`).

So flash is **cheaper and reliably emits valid tool calls**, but it is **not
tool-choice-interchangeable with pro**. The canary is how we measure whether
the divergence actually degrades user-visible outcomes, not whether it exists.

## Non-negotiable design constraints

1. **Served split, never wholesale.** Start 10%, ramp only on evidence.
2. **Pro is the immediate quality fallback.** A flash miss must escalate to
   `deepseek-v4-pro`, NOT to the highest-tier tool model (currently `gpt-5.5`).
3. **Escalation must not loop back to flash.** A failed flash call must never
   be re-served by flash.
4. **Hard failures still escalate** (malformed tool call, empty content,
   degeneration) — already implemented; unchanged.
5. **Everything observable** for outcome comparison: flash vs pro cohorts
   tagged by arm, so error/escalation/turn counts are comparable.

## Known hard problem (must be flagged, not hidden)

The incoming `tools` array is the **full Hermes toolset** — every request offers
the same tools and the model *chooses*. Therefore:

- We **cannot** scope the canary by "which tool is offered" or "read-only vs
  mutating tool". The choice is only known after flash responds.
- Risk must be scoped by **task_type** (available pre-routing), **complexity**,
  and **context size**, and by **escalation on failure** — not by tool.

The genuinely hard residual risk: flash picks a *valid but wrong* tool (e.g.
`terminal` where pro picks `read_file`), and that is **not detectable
pre-emptively** — only by outcome (a tool error, a corrective user turn, a
revert). This is the whole point of a bounded canary: measure that rate at
small scale before scaling it.

## Scope decision (for approval)

Proposed first slice (conservative):
- `workload = normal_chat`, `requires_tools = true`, `incumbent = deepseek-v4-pro`.
- `percent = 10`.
- **Exclude by task_type**: `debugging`, `coding`, `testing` (mutation-heavy).
- **Include**: `planning`, `research`, `qa`, and any task_type not in the
  exclude set.
- Arm assignment: stable hash of `session_id` (existing `select_arm`), so a
  session never flaps mid-conversation.

This needs a new match field `exclude_task_types` on `Experiment` + a task_type
plumbed into `pick_experiment`, because the existing match is workload/tier/tools/incumbent
only and cannot express "not these task types".

## Code changes

1. **`traffic_split.py`**
   - Add `match_exclude_task_types: Tuple[str, ...]` to `Experiment`.
   - Parse it in `build_experiment`; honour it in `_matches`.
   - Thread `task_type` through `pick_experiment` / `_matches`.

2. **`router.py`**
   - Add `deepseek-v4.1-flash` to `TOOL_CAPABLE_MODELS` (verified 2026-09-20:
     streamed shadow emits clean structured tool_calls).
   - **Fix escalation target**: for tool-bearing escalation, when the failed
     model is tier < pro's tier, escalate to pro (cheapest higher tool-capable
     model), never above pro — so a flash miss lands on pro, not gpt-5.5.
     Preserve the existing "most capable" behaviour for the gpt-5.5→glm-5.3
     downgrade path.

3. **`biggie_llm_endpoint.py`**
   - `apply_experiment` split path already serves the candidate and swaps the
     backend. Verify the `provider` field on the split experiment resolves the
     candidate backend (flash is `:cloud`-suffixed).
   - No change expected beyond confirming the split arm reaches flash via
     `provider: ollama-cloud`.

4. **`experiments.yaml`**
   - Add the split experiment, `enabled: false` (go/no-go is explicit).

## Tests (RED first, per convention)

- `test_traffic_split.py`: `exclude_task_types` parse + match behaviour;
  `stream` flag already covered.
- `test_tool_ladder.py` (new or extend): flash→pro escalation target; flash
  must not re-loop; pro stays on the ladder.

## Rollout

1. Land code + tests green; experiment `enabled: false`.
2. Restart `biggie-llm-endpoint` (NOT the gateway).
3. Approve → set `enabled: true` at 10%.
4. Watch flash-vs-pro cohort outcome deltas for 2–3 days; raise percent only on
   clean evidence.

## Open questions for the user

- Confirm the 10% start and the task-type exclusions.
- Confirm pro (not gpt-5.5) as the flash-miss fallback.
- Confirm we accept the residual "valid-but-wrong tool" risk at 10% scale.
