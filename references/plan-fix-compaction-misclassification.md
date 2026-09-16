# Plan — fix the compaction misclassification that strips tools from agent turns

Status: implementing (2026-09-16)
Thread: 56467

## Problem (measured + reproduced, not hypothetical)

A tool-bearing agent turn in a long Hermes session gets its tool schemas stripped
and is routed to a summariser-grade, non-tool-capable model — which then
**fabricates** markup like `<terminal>`/`<skill_view>` that never executes. The
user hit this: "it worked this morning at 8, now it can't call a skill."

### Root cause

`detect_workload_type()` (biggie_llm_endpoint.py) scans **every message in the
history** for 14 loose phrases (e.g. `"context compaction"`, `"compressed
summary"`, `"preserve key facts"`). If ANY message contains one, the whole request
is labelled `session_compression`, and `_normalize_compression_request()` strips
`tools` + `tool_choice`. Every long Hermes session carries a compaction handoff
(e.g. `[CONTEXT COMPACTION — REFERENCE ONLY]`) somewhere in its history, so once a
thread compacts, **every subsequent agent turn** is misclassified → tools
stripped → dispatched to a non-tool model.

### Live reproduction (control vs treatment, same tools array)

- Plain agent history → `deepseek-v4-pro`, `requires_tools=1`, `normal_chat`
- History containing `[CONTEXT COMPACTION …]` → `deepseek-v4.1-flash`, `rt=0`, `session_compression`
- History containing `"compressed summary"` → `deepseek-v4.1-flash`, same

Corroborating: `deepseek-v4-flash` logged **208 of 210** `session_compression`
rows with `saw_tool_calls=1` — a *summariser* should never call a tool; those are
misclassified agent turns.

### Timeline (why "worked this morning")

B29 (`9310faa`, Sep 14 13:09) introduced the strip; it went live at the
**Sep 14 20:25 endpoint restart**. `requires_tools=1` ran ~4,594–11,659/day
through Sep 14, then collapsed to **2** on Sep 15 — a cliff at deployment, not
decay. It is per-session: a fresh thread works until it compacts, then breaks
for good.

## Design decisions (with evidence)

1. **Scan only the LAST user message for text markers, not the whole history.**
   A real compaction request is a single synthesized user message (the prompt
   from `agent/context_compressor.py`). The marker phrases live in the *instruction*
   of that one message. Agent turns that merely reference a past compaction do not.

2. **Add the genuine first-time compaction marker.** The real first-time prompt —
   `"Create a structured checkpoint summary for the conversation after earlier
   turns are compacted…"` — matches **none** of the current 14 markers, so today it
   is (correctly for now) misclassified as `normal_chat` (a false negative B29
   never caught). Add `"checkpoint summary"` / `"structured checkpoint summary"`
   so fresh compactions stay classified.

3. **Tool-bearing requests are never demoted by text markers alone.** If the body
   carries a `tools` array and there is no *explicit* compression metadata/header,
   text markers cannot override it — only explicit metadata can. This is the
   belt-and-suspenders guard for the user's exact failing case, and it cannot
   resurrect the B29 defect (a genuine compaction request carries no `tools`).

## Acceptance criteria

- Ordinary agent turn (tools present, history contains compaction handoff) →
  stays `normal_chat`, `requires_tools=1`, dispatched to a tool-capable model.
- Genuine compaction (first-time AND iterative prompts) → still `session_compression`,
  tools stripped (they inherited parent schemas it can't use).
- Explicit metadata/header still beats everything.
- Full suite green; non-vacuity proven for each batch.

## Implementation (TDD batch-by-batch, run_tests.sh conventions apply)

### Batch 1 — scope the scan + add the first-time marker
- `detect_workload_type`: marker scan over `messages[-1]` (last user message) only
  — keep explicit metadata/header path as-is (checked first).
- Add `"checkpoint summary"` (covers `"structured checkpoint summary"`) to markers.
- Tests `tests/test_batch7_compaction_scope.py`: control (plain history), treatment
  (history with every marker phrase, last msg clean) → `normal_chat`; genuine
  first-time + iterative prompts → `session_compression`.

### Batch 2 — tool-bearing veto
- After classification, if the ORIGINAL body has `tools`/`tools_in` and the label
  came from text markers (not explicit metadata), force `normal_chat`.
- Tests: tool-bearing + marker in history → `normal_chat` + tools preserved;
  no-tools compaction → still `session_compression`.

## Restore discipline
- No hand-filling of governed evidence. Each batch: RED → GREEN → non-vacuity
  probe → restore byte-identical → full suite → commit. Plan doc updated at the end.

## Status
NOT deployed. Requires endpoint restart to go live (`sudo systemctl restart
biggie-llm-endpoint`) — user signs off, gateway already-current.
