# B27 — Session-compression tool-call and channel trace

## Goal

Prove the live channel split on one production compaction and stop tool-call-only
responses from being returned as successful compaction summaries.

## Evidence

- Production `router_logs`: 30,932 non-shadow `session_compression` rows have
  `saw_content=0, saw_tool_calls=1`.
- On 2026-09-13, 273 of those rows carry 94,405 billed output tokens.
- `_preflight_openai_stream` currently treats a structured tool-call delta as
  status `ok` for every workload, including session compression.
- The downstream quality probe accumulates only `delta.content`.
- The endpoint already documents reasoning-only empty completions and enforces a
  16,384-token output floor, so output budget alone does not close this path.

## Scope

1. Add a request-correlated, one-shot trace for a production session compaction.
   Record only bounded output-channel tails and counts — never source prompts,
   credentials, or full responses.
2. Track content chars, reasoning chars, tool-call deltas, usage tokens and finish
   reason across the complete SSE stream.
3. For `session_compression`, do not accept a tool-call-only stream as a valid
   summary. Return preflight status `empty` so existing escalation/fallback logic
   can choose another backend before committing a downstream response.
4. Keep ordinary agent/tool workloads unchanged.
5. Add regression tests, run focused tests, full suite and preflight.
6. Deploy with a detached endpoint-only restart; verify health and gateway PID,
   then read the single trace and disable one-shot tracing.

## Safety

- Endpoint only; never restart `hermes-gateway.service`.
- Trace one request only and bound each captured tail to 400 characters.
- Do not log source messages, full generated output, or secrets.
- Do not run rollups/backfills against the live database.
