# B33 — route-blind charge in the anchored preflight estimator (IMPLEMENTED)

Status: **implemented, tested, committed** in the CORE repo (`/home/hermes/.hermes/hermes-agent`).
Scope note: B33 turned out to be a **core** change, not a router-skill change. The
`llm-smart-router` skill holds the diagnosis; the fix and its tests live in core.

## Root cause

`anchored_context_tokens(messages, anchor)` takes a keyword-only
`charge_stale_thinking: bool = True`. Three **gating** call sites omitted it, so
they charged stale reasoning text that the active route strips before sending.

Measured (reasoning-heavy fixture, real `capture_usage_anchor`):

| figure | tokens |
|---|---|
| blind (`charge_stale_thinking` defaulted True) | **230,269** |
| route-correct (`charge_stale_thinking=False`) | **155,912** |
| **inflation** | **74,357** |
| live trigger | **180,000** |

The inflated figure **straddles** the live threshold — so a paid compaction fired
on phantom tokens (the observed `session_compression` churn: 88 rows/40 min, 84
on one key).

`stale_thinking_reaches_wire(api_mode, provider, model, base_url)`:
- `False` for our chat-completions route → stale reasoning never reaches the wire
- `True` for `deepseek-reasoner` (echo route) → **a blanket `False` would be wrong**

So the fix is route-aware, not a constant.

## The fix — three sites, no new machinery

`_agent_stale_thinking_on_wire(agent)` already existed in `agent/turn_context.py`
(defensive: returns `True` when route facts are unavailable). Each site now passes
`charge_stale_thinking=_agent_stale_thinking_on_wire(agent)`:

| site | file | function | enclosing branch |
|---|---|---|---|
| A | `agent/turn_context.py` | `_preflight_request_tokens` | the **anchored** branch |
| B | `agent/turn_preflight.py` | `compress_after_tool_results` | post-tool compaction trigger |
| C | `agent/turn_request_assembly.py` | `assemble_api_request` | mid-turn pre-API pressure |

## The trap that made this look already-fixed

`_preflight_request_tokens` **already had** a route-aware call — on its
**fallback** branch (`estimate_request_tokens_rough`). The anchored branch
**above** it was blind. So:

```
git show HEAD~1:agent/turn_context.py | grep -c charge_stale_thinking=_agent_stale_thinking_on_wire
# -> 1   (the fallback)
# after the fix -> 2
```

A whole-file count of 1 before AND after means "not reverted" only if you assume
the pre-fix count is 0. **Count per branch, and read the pre-fix count from git.**

## Coverage

- `tests/agent/test_b33_anchored_preflight_route_aware.py` — 7 tests (site A:
  arithmetic, decision flip across the live threshold, route controls).
- `tests/agent/test_b33_sites_bc_route_aware.py` — 4 tests (sites B and C, via a
  spy on the real callable each module binds).
- **Patch target matters:** `turn_preflight` imports `anchored_context_tokens`
  *inside the function* → patch `agent.usage_anchor`; `turn_request_assembly`
  binds it at *module scope* → patch `agent.turn_request_assembly`. Patching the
  wrong one passes silently (the spy records nothing).

### Non-vacuity

Reverting each site **alone** to its pre-fix revision must produce RED:

```
A turn_context._preflight_request_tokens   -> CAUGHT (RED, 2 failed)
B turn_preflight.compress_after_tool_results -> CAUGHT (RED, 1 failed)
C turn_request_assembly.assemble_api_request -> CAUGHT (RED, 1 failed)
pristine: 11 passed / 0 failed
```

The first suite covered **only site A** — B and C reverted alone left it fully
green. That is why the second test file exists.

## Regression attribution

Full `tests/agent` + `tests/run_agent`: **9,129 passed, 19 failed**.
Pre-change baseline `422bc9bde9`: **18 failed** on the same files.

The single delta — `test_codex_aux_timeout_fd_ownership` (fd/timing) — was
**6/6 GREEN** run serially 3× on each head: a load flake. The other 18
(Anthropic/Bedrock/vision/wire clients) fail identically pre- and post-change.

**Zero regressions introduced.**

## Commits (core)

- `63cf719039` — the fix + site-A tests
- `d0d6cbd3a4` — sites B and C coverage
- `6e03bd13ac` — the systemd detector fix (separate concern, see below)

## Related fix in the same session (item 3, NOT a stale unit)

The "Stale systemd unit" warning was a **detector bug**:

`_systemd_timeout_stop_us` probes `--user` first. For a unit in the **system**
scope that scope still answers `rc=0` — with `LoadState=not-found` and systemd's
compiled-in default `TimeoutStopUSec=1min 30s`. The check then compared **90s
against a 210s requirement** and warned on every start, advising
`hermes gateway install --force` — which would have **rewritten a correct unit**.

Truth: unit file `TimeoutStopSec=210` (since 2026-09-13 09:35), `systemctl show`
(system) `3min 30s` / `LoadState=loaded` / `NeedDaemonReload=no`.

Fix (`6e03bd13ac`): also request `--property=LoadState` and only trust a scope
whose `LoadState == "loaded"`. Live verdict flips to
`{"timeout_stop_sec": 210.0, "expected_min": 210.0, "mismatch": False}`.
Tests: `tests/gateway/test_b33_timeoutstopsec_scope_ownership.py` (5). Non-vacuity:
revert alone → 5 pass becomes 3 fail.

## Still open (not fixed here)

- **B35 plugin is correct but NOT ACTIVE** — `session_id_source='cache_key'` rows
  remain 0. Needs one out-of-process gateway restart.
- **Codex account cap** — `resets_at` 2026-09-19 13:37Z (~99h / 4.13 days from
  2026-09-15 10:23Z). Holds 5 models. Breaker opened 1800s after 1155 consecutive
  failures. Not router-fixable.
- **Local Ollama saturated, not broken** — single slot (`-np 1`, `ctx=4096`)
  mid-generation; a 4-token probe returned HTTP 200 in 23.2s given a 150s budget.
