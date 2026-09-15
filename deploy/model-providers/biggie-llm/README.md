# Deploy: model-provider profile for the biggie route (B35 / Phase 3.2)

Makes Hermes' chat-completions transport send a session-scoped
`prompt_cache_key` for `biggie-llm`, so the router stops deriving a
system-prompt-hash session key.

## Install

```sh
mkdir -p ~/.hermes/plugins/model-providers/biggie-llm
cp deploy/model-providers/biggie-llm/__init__.py \
   ~/.hermes/plugins/model-providers/biggie-llm/__init__.py
```

## Activation requires a GATEWAY restart

Provider discovery is **once per process** (`providers/__init__.py::_discover_providers`
sets `_discovered = True` and never re-runs). A long-lived gateway will NOT pick
this up. Restart `hermes-gateway.service` out-of-process (the gateway cannot
restart itself).

Blast radius: a gateway restart drops every active session on the host. Check
for in-flight sibling threads and uncommitted work first.

## Verify

```sh
cd ~/.hermes/hermes-agent && python3 - <<'PY'
from providers import get_provider_profile, _discover_providers
_discover_providers()
for n in ("biggie-llm", "custom:biggie-llm"):
    p = get_provider_profile(n)
    print(n, "->", getattr(p, "name", None), getattr(p, "supports_prompt_cache_key", None))
PY
```

Both names must resolve to `custom:biggie-llm` with `True`. Then confirm live
rows carry `session_id_source='cache_key'`:

```sql
select session_id_source, count(*) from router_logs
where timestamp >= '<restart-cut>' group by 1;
```

## Pitfalls

- **Register BOTH names.** `agent.provider` may be the bare configured name
  (`biggie-llm`) or the `custom:`-prefixed id. A bare name that misses the
  registry resolves to `None` (not the generic `custom` profile), silently
  disabling the flag — so the `aliases=("biggie-llm",)` entry is load-bearing.
- A plugin that raises is skipped with a warning, not fatal.
