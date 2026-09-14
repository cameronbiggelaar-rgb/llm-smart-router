# Plan — B30: Codex unsupported-parameter strip, deterministic-4xx breaker, retry amplification

Status: **implemented + verified** (2026-09-14). Router-side only; no HEI contact.

## Defect (proven, not inferred)

`scripts/biggie_llm_endpoint.py`, `proxy_to_backend()`, the `openai-codex`
branch (~line 2039) forwarded sampling parameters to the ChatGPT Codex
`/responses` endpoint:

```python
for param in ("temperature", "top_p", "stop", "frequency_penalty", "presence_penalty"):
    if param in request_body:
        body[param] = request_body[param]
```

Codex rejects **all sampling parameters** (and `max_tokens` /
`max_output_tokens`) with `400 Unsupported parameter: <name>`.

Live probe against `https://chatgpt.com/backend-api/codex/responses`, model
`gpt-5.5`, `stream: true`:

| case | result |
|---|---|
| baseline, no sampling params | **200** |
| `temperature` | 400 Unsupported parameter: temperature |
| `top_p` | 400 Unsupported parameter: top_p |
| `stop` | 400 Unsupported parameter: stop |
| `frequency_penalty` | 400 Unsupported parameter: frequency_penalty |
| `presence_penalty` | 400 Unsupported parameter: presence_penalty |
| `max_tokens` | 400 Unsupported parameter: max_tokens |
| `max_output_tokens` | 400 Unsupported parameter: max_output_tokens |

Model names are fine: `gpt-5.5`, `gpt-5.6-sol`, `gpt-5.6-luna`,
`gpt-5.6-terra` all returned 200 as plain strings. (`gpt-5.1-codex` is genuinely
unsupported on a ChatGPT account — unrelated.)

### Amplification (fix 3) — measured, not assumed

One client request to our own endpoint (`POST /v1/chat/completions`, model
`gpt-5.5`, with `temperature`) produced **18 upstream Codex 400s** in a ~2
minute window, walking the escalation ladder and re-trying codex models that
reject deterministically. The client saw a timeout.

Root cause of the amplification: a deterministic 4xx is classified as a generic
`error`, which (a) increments `consecutive_failures` and can trip the breaker,
and (b) is retried on escalation, where the *same* unsupported params are
re-sent to the *same* rejecting endpoint.

## Fixes

**Fix 1 — strip unsupported params for Codex.** Remove the forwarding loop from
the `openai-codex` branch entirely and define a module-level denylist constant
so the intent is explicit and testable. Codex accepts no sampling knobs; the
correct body is `model` + `input` + `store` + `stream` (+ `tools`).

**Fix 2 — deterministic 4xx must not hot-loop the breaker.** Add
`_is_deterministic_client_error(text)` recognising the `Unsupported parameter`
/ malformed-request class. In `_handle_backend_status`, a deterministic 4xx
must NOT feed the ordinary escalation path as a retryable error; it is a
code/schema fault.

Implementation note: the strip (fix 1) removes the *cause*, so with fix 1 in
place the Codex branch cannot emit these params again. Fix 2 is defence in
depth — it prevents any *future* schema mismatch from becoming a retry storm.

**Fix 3 — an expired breaker must re-arm with a FRESH failure budget.**
(Original scoping called this "covered by fix 2"; that was WRONG. Live
evidence during B30 found a separate, independent defect.)

`is_model_available()` restored `available = True` when the circuit-breaker or
rate-limit cooldown expired, but never reset `consecutive_failures`. The
poisoned counter therefore survived recovery, so a SINGLE fresh failure
immediately satisfied `consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD` (3)
and re-opened the breaker.

Observed live: `gpt-5.5` at **4,046 consecutive failures** while its breaker
"cycled" open every 30 minutes. Net effect: a broken model received exactly
ONE attempt per 30-minute window and the counter grew without bound. This is
what actually sustained the amplification — fix 2 stops a single request from
walking the ladder, fix 3 stops the model from being permanently poisoned.

Both expiry paths (rate limit and circuit breaker) now reset
`consecutive_failures` to 0 so recovery is genuine. `mark_available()` remains
the explicit manual reset.

## Non-goals

- No changes to HEI (another thread owns that repo).
- No change to the account/cap behaviour (429 handling is correct and separate).
- No change to `temperature` forwarding for Ollama / OpenAI-compatible backends
  — those genuinely accept it (sites 2207, 2255, 2504, 2636 stay).

## Verification

- `tests/test_b30_codex_param_strip.py` — fix 1 static + behavioural.
- `tests/test_b30_deterministic_4xx_breaker.py` — fix 2 classification.
- `tests/test_b30_breaker_rearm_budget.py` — fix 3 re-arm budget.
- Full router suite green: **551 passed**.
- Live (post-restart, PID 1976698): the exact reproduction that previously
  produced 18 upstream 400s and a 180s client timeout now returns
  **HTTP 200 in 1.9s**, both streaming and non-streaming, with
  **0** `Unsupported parameter` 400s and **0** ERROR lines in the window.

### Measured before/after

| signal | before | after |
|---|---|---|
| client request carrying `temperature`, model gpt-5.5 | 180s timeout, 18 upstream 400s | HTTP 200 in 1.9s |
| codex `Unsupported parameter` 400s (20 min window) | 165 | 0 |
| escalation events (20 min window) | 290 | 0 |
| upstream codex successes in window | 0 | 7 |
| gpt-5.5 consecutive_failures | 4,046 (never reset) | resets to 0 on expiry |

