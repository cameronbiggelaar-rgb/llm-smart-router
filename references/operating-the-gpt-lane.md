# Operating the GPT (openai-codex) lane

Topical depth for day-to-day ops against the governed ChatGPT plan — checking
live budget, refreshing the credential, and answering "how much would have hit
GPT". Not the routing architecture (see the design doc for that).

## Check live budget / usage

The single source of truth is the account's own `/usage` verdict, surfaced via
`gpt_budget.fetch_reading()`. It reports two windows plus the account's
`allowed` flag and `banked_resets`.

```python
import sys; sys.path.insert(0, '<repo>/scripts')
import gpt_budget as gb
r = gb.fetch_reading(use_cache=False)   # force a live read
print(r.source, r.allowed)               # 'live', True/False
print(r.primary)   # 5h  window: used_percent, reset_after_seconds, reset_at
print(r.secondary) # 7d  window: same fields (this is the one that caused past outages)
print(r.banked_resets)
```

Run the probe under the **hermes-agent venv**, not bare `python3`:
`~/.hermes/hermes-agent/venv/bin/python`. Bare `python3` fails on
`import agent.account_usage` because it lacks `ruamel` (`hermes_yaml` dep) —
that is an interpreter-selection issue, not a broken import. The live endpoint
reads usage through the same `agent.account_usage` path, so the running process
is unaffected; only ad-hoc probes need the venv.

## Refresh the credential

Access tokens are short-lived (~1 week); the endpoint does NOT auto-refresh —
it only warns on token age and marks the credential expired on a 401. Refresh
is a manual step.

1. Try refresh first (redeems the existing `refresh_token`, no browser):
   `hermes auth refresh openai-codex <label>`  — `<label>` (e.g.
   `openai-codex-oauth-1`) is required when the pool holds more than one entry.
2. If refresh fails (refresh token also expired/revoked), re-add fresh with the
   device-code flow: `hermes auth add openai-codex --type oauth` — prints a URL
   + code the user approves in a browser, blocks until approved. (`--browser`
   switches to a PKCE flow on localhost:1455; only if device-code misbehaves.)
3. Verify: `hermes auth list openai-codex` and `hermes auth status openai-codex`.

The pool is priority-ordered: **priority 0 is tried first and is the credential
that actually serves traffic** — refresh that one; lower-priority entries only
matter if the primary fails. Check token expiry by decoding the JWT `exp` claim
(base64url-decode the middle `.`-segment of `access_token`) rather than trusting
`last_status`.

## Answering "how many requests would have hit GPT"

Under `explicit_only: true`, entitlement is the explicit ask naming a plan
model — the architectural *class* alone no longer grants it. So the count of
requests that would have reached GPT is the count that **named** a plan model:

```sql
SELECT requested_model, COUNT(*) FROM router_logs
 WHERE timestamp >= '<when>' AND requested_model LIKE '%gpt%'
   OR requested_model LIKE '%astra%' OR requested_model LIKE '%codex%';
```

Also check the force_model lane (`routing_reason LIKE '%forced model%'`) grouped
by `requested_model` — explicit overrides to non-GPT models (deepseek/glm) do
not touch GPT and should be excluded from the estimate. Zero GPT-named requests
means a cold lane, not a backed-up one: refreshing the key would route nothing.

## The router log undercounts plan spend

`router_logs` only records traffic that passes *through* the endpoint. Anything
that hits the Codex backend **directly** — the `astra-review` runner (`POST
chatgpt.com/backend-api/codex/responses` with the OAuth bearer), any standalone
script that talks to the account without going via `biggie_llm_endpoint` — draws
on the ChatGPT plan cap but never writes a `router_logs` row. So "how much
ChatGPT budget did we use" has two non-equivalent answers:

- **Authoritative total**: the account `/usage` reading (`fetch_reading`) —
  counts *all* consumers of the plan, router and direct alike.
- **Router's share only**: `router_logs` GPT-row counts — tells you what the
  routing system spent, not what the plan spent.

When the account shows a non-trivial `used_percent` (e.g. 58% of the 5h window)
while `router_logs` has zero GPT rows, that gap is **direct consumers**, not a
measurement error. Reconcile by identifying what bypasses the endpoint (the
`astra-review` runner is the common one) rather than suspecting the router's
observability. The 5h window (`primary.reset_after_seconds`) is the immediate
answer to "when does budget return"; the 7d window (`secondary`) is the
weekly-budget horizon.
