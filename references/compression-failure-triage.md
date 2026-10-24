# Triage: `Shortening the conversation history failed`

The user-facing warning is the i18n key `gateway.compress.hygiene_failed`, emitted by
`gateway/run_turn_hygiene.py` when the compressor aborts and keeps the transcript as-is.

**It is not one bug.** Most of the time spent on it is spent diagnosing the wrong cause.

## 0. Resolve the warning to its emitter before anything else

```bash
grep -n "hygiene_failed" ~/.hermes/hermes-agent/locales/en.yaml   # -> gateway.compress.hygiene_failed
grep -rn "hygiene_failed" ~/.hermes/hermes-agent/ | grep -v locales # -> gateway/run_turn_hygiene.py
```

This works for any Hermes user-facing warning: the rendered English string is a locale value and
appears in exactly one place in the source tree. Guessing from the wording costs more than the grep.

## 1. Bucket by cause — never count "compression failures" as one number

```bash
sudo journalctl -u hermes-gateway --since "-14 days" --no-pager -o short-iso \
  | grep -E "Session hygiene compression aborted:|Failed to generate context summary:" \
  | sed -E 's/[0-9]+/N/g' | sort | uniq -c | sort -rn | head -20
```

The same warning covers at least three unrelated fixes:

| Reason text | Real cause | Where to look |
|---|---|---|
| `summary was truncated (finish_reason=length)` | summariser ran out of **output** budget | §2 |
| `Connection error` | the endpoint is **down** | `systemctl is-active biggie-llm-endpoint` |
| `all_models_exhausted` / rate-limit body | router availability, not Hermes | `router_state.db` `model_state` |

Check endpoint liveness FIRST. A dead endpoint reproduces every other symptom for as long as it is
down, so diagnosing a budget problem while the unit is inactive is wasted work.

## 2. Prove truncation by arithmetic, not by eye

```sql
SELECT finish_reason, COUNT(*), AVG(output_tokens), MIN(output_tokens), MAX(output_tokens)
FROM router_logs WHERE workload_type='session_compression' GROUP BY finish_reason;
```

If the `length` rows' `output_tokens` cluster tightly at the endpoint's own computed floor, the budget
is the cause. Compute that floor from `_compression_output_floor()` in `biggie_llm_endpoint.py`:
linear from `SESSION_COMPRESSION_MIN_MAX_TOKENS` (16384) to `..._MAX_MAX_TOKENS` (32768) across
`..._INPUT_FOR_MAX` (300000) input tokens, clamped.

**An exact match between the computed cap and the observed output is proof of budget exhaustion, not a
model defect.** Worked example: failing rows averaged context 24,643 → floor
`16384 + 16384*(24643/300000) = 17,730`; observed average failing output was 17,730.

## 3. Pitfalls that cost the most time

- **The applied-floor log line prints the module CONSTANT, not the computed value.**
  `logger.info("... floor applied: %s", SESSION_COMPRESSION_MIN_MAX_TOKENS)` reports `16384` on every
  request regardless of what was actually written to `body["max_tokens"]`. It cannot be used as
  evidence for or against the floor. Read `context_tokens` / `output_tokens` from `router_logs` instead.
  General rule: a log line that interpolates a settings constant rather than the value the code just
  computed is decoration, not instrumentation.
- **The endpoint deliberately does NOT escalate on `finish_reason=length`.** `_response_has_empty_content()`
  treats `length` as "the caller's `max_tokens` was too small" and lets it through — correct for a
  client-supplied budget. For `session_compression` the endpoint sets that budget itself, so a too-small
  self-imposed floor becomes a silent truncation that surfaces to the user as failed compaction instead
  of as a retry on the next rung. Fix the floor, or escalate on truncation for this workload only.
- **Distrust the router's `context_tokens` as the session's real size.** It is the endpoint's own
  preflight estimate and can sit far below what Hermes reports for the same compaction (~19–54K recorded
  while the gateway was compacting at 124K+). A floor scaled off an under-estimate under-provisions
  exactly the large sessions that need the room.
- **A rising truncation RATE, not a rising count, is the signal.** Counts track traffic. Bucket per week
  (`substr(timestamp,1,10)` → ISO week) and compare shares; a rate climbing while volume is flat means
  the *serving model* changed — typically a cheaper reasoning-heavy rung becoming the summariser, whose
  hidden reasoning trace is spent before any summary text is emitted.
- **A reasoning-heavy model can legitimately need more output than its context size suggests.**
  Successful compression calls here average ~13K output tokens against ~19K context, so an output ratio
  near 1:1 is normal for this workload. Size an output floor against measured output, never against a
  fraction of input.

## 4. Endpoint death masquerading as a compression bug

- **`Restart=on-failure` does not recover a CLEAN stop.** A SIGTERM the process handles and exits 0 from
  is a successful deactivation: systemd logs `Deactivated successfully` and does not restart. The tell is
  the *absence* of a `Stopped <unit>` line (a real `systemctl stop` logs both).
- **A foreground tool call killed mid-restart can leave the unit dead indefinitely.** Verify with
  `systemctl is-active`, never infer liveness from the restart policy. Prefer `Restart=always` for a unit
  whose absence breaks every session on the host.
- Confirm the endpoint recovered with `/health` (`status`, `build.git_sha`, `stale`), not with a port
  check — the health endpoint answers correctly and cheaply, and `stale: false` is what proves the
  running code is the current code.
