# B35 — Phase 3.2: put a real session identity on the chat-completions wire

**Status:** mechanism PROVEN (read-only probe). Implementation not yet written.
**Author:** assistant, thread 56467
**Depends on:** B32 (`session_id_source`), B34 (`_SESSION_ID_ENFORCEABLE_SOURCES`)
**Scope:** `~/.hermes/plugins/model-providers/<name>/` (user plugin) + optional
router-side acceptance test. **No core patch required.**

## 1. Why this matters (it is not cosmetic)

B34 batch 2 made the compaction guard refuse to cap a non-discriminating key —
correct, but it left the guard **inert on the chat-completions path**, which is
the path biggie-router actually serves:

- the derived key (`pfx:<sha256>`) collides across sessions sharing an opening
  wrapper (measured: one derived key covered 4 distinct state.db sessions);
- the chat-completions transport sends **no session id**, so `session_id_source`
  resolves to `derived` or `none`, both non-enforceable;
- therefore `session_compression_overflow` **no longer fires** for that traffic.

Phase 3.2 restores enforceability *and* real per-session attribution at the
source, instead of band-aiding the symptom.

## 2. What already exists in core (recon, read-only)

Core has complete `prompt_cache_key` machinery:

- `agent/transports/chat_completions.py::_add_prompt_cache_key()` adds a
  content-addressed key **iff** `supports_prompt_cache_key` is true;
- the flag is gated on either `params["supports_prompt_cache_key"]` **or**
  `_is_openai_api_base_url(base_url)` (exact `api.openai.com` only — our
  `127.0.0.1:8080` is not it);
- the profile path reads `getattr(profile, "supports_prompt_cache_key", False)`;
- `providers/__init__.py` resolves a provider name to a `ProviderProfile`, and
  **user plugins at `$HERMES_HOME/plugins/model-providers/<name>/` override
  bundled profiles (last-writer-wins)**;
- the transport receives the profile as `provider_profile=` (from
  `agent/chat_completion_helpers.py`), so a registered profile is honoured.

Resolution detail that matters: `get_provider_profile()` maps an unknown
`custom:<name>` to the generic **`custom`** profile
(`plugins/model-providers/custom/__init__.py`), whose
`supports_prompt_cache_key` is **False**. That is exactly why nothing is sent
today.

## 3. Proof the mechanism works (executed, not inferred)

A temporary user plugin registered `custom:biggie-llm` with the flag set, then
the REAL transport built kwargs for two different sessions:

```
custom:biggie-llm: resolved='custom:biggie-llm' flag=True
   keys=['pck_f8ff38fc50157e1d7a1129a0', 'pck_77d1e7c5e8b345704003b656']
   emitted=2/2 distinct=2 -> SESSION-SCOPED
custom:probe-off: resolved='custom:probe-off' flag=False
   keys=[None, None]  -> control: without the flag nothing is sent
grown-convo key for same session: pck_f8ff38fc50157e1d7a1129a0  (turn-stable)
```

So the key is **session-scoped** (two sessions differ) and **turn-stable** (a
grown conversation reuses it) — the two properties a session identity needs.

**Gotcha found during the probe:** plugin discovery **skips directories starting
with `_` or `.`**. A probe named `_probe32` was silently ignored and the lookup
fell back to the generic profile (flag False), which looks exactly like "the
mechanism does not work". Use a normal directory name.

## 4. Implementation

### 4.1 Create the user plugin (no core edit)

`~/.hermes/plugins/model-providers/biggie-llm/__init__.py`:

```python
"""biggie-llm route profile: declare prompt_cache_key support.

Hermes already derives a session-scoped prompt_cache_key, but only for
providers whose profile sets supports_prompt_cache_key. The bare custom:<name>
fallback resolves to the generic `custom` profile, which leaves the flag False,
so the router receives no session identity at all and every per-session number
is a prefix-hash approximation (B34). This profile only flips that flag; all
other wire behaviour is inherited unchanged.
"""
from providers import register_provider
from providers.base import ProviderProfile

biggie_llm = ProviderProfile(
    name="custom:biggie-llm",
    base_url="http://127.0.0.1:8080/v1",
    supports_prompt_cache_key=True,
)
register_provider(biggie_llm)
```

Notes:
- The name must match how the route resolves (`custom:<entry name>`), i.e.
  `custom:biggie-llm` for `custom_providers: [{name: biggie-llm}]`.
- Only the flag is set. No `default_headers`, no
  `fixed_temperature`, no `extra_body` — inheriting everything else keeps the
  change minimal and reversible (delete the directory).
- **Do NOT** copy the generic `CustomProfile` methods. If a future core change
  alters `custom`'s Ollama/`think` handling, an override that reimplements it
  would silently freeze the old behaviour.

### 4.2 What the router then sees

`prompt_cache_key = pck_<24 hex>` on every chat-completions request. B34 already
resolves that field as source `cache_key`, and
`_SESSION_ID_ENFORCEABLE_SOURCES = ("header", "body", "cache_key")` — so the
guard becomes enforceable again with **no router change**.

Verify with B32's own instrumentation: after the restart,
`session_id_source='cache_key'` should appear on new rows, and
`session_compression_overflow` should start firing on genuine runaways.

### 4.3 Ordering constraint (important)

The key is derived from `cache_scope_id or session_id`, where `cache_scope_id`
is the compression-lineage ROOT. That is deliberate (survives compression
rotation), but it means the router's `cache_key` identifies a **lineage**, not a
single physical session. Two consequences:

1. Good: the guard's cap survives compaction rotation, so it cannot be evaded
   by rotating the context.
2. To watch: distinct-session counts must be interpreted at lineage granularity.
   Do not claim "N sessions" from `cache_key` rows without saying "lineages".

## 5. Risks

| risk | assessment |
|---|---|
| Unknown top-level field rejected by the router | The router is ours; `prompt_cache_key` is already parsed by B34. Confirmed accepted. |
| Declaring support against a non-supporting endpoint | Not applicable — we control both ends. The flag is *about* this endpoint's tolerance. |
| Profile override drifting from the generic `custom` profile | Mitigated by setting ONLY the flag (§4.1). |
| Cache-bucket collision across sessions | The key is session-scoped (proven, §3). |
| Plugin silently ignored (the `_`-prefix trap) | Verify by resolving the profile and asserting the flag is True, not by assuming the file exists. |

## 6. Verification plan

1. Resolve `get_provider_profile("custom:biggie-llm")` and assert
   `supports_prompt_cache_key is True` (catches the `_`-prefix trap).
2. Build kwargs via the real transport for two sessions; assert both emit a key
   and the keys differ.
3. Assert a grown conversation for one session reuses its key (turn-stable).
4. Post-restart, confirm live rows carry `session_id_source='cache_key'`.
5. Confirm `session_compression_overflow` re-fires on a genuine runaway, and
   that a healthy session is no longer relabelled.

## 7. Open questions

1. Should the plugin also be committed to the skill repo as a reference artefact,
   or is `$HERMES_HOME/plugins/` the only home? (It is machine config, not repo
   code — but the *reasoning* belongs in `references/`.)
2. Does declaring the flag change **cache billing** behaviour? The key already
   influences provider-side caching on capable endpoints; here it adds a field
   the router itself consumes. Measure `cached_tokens` before/after anyway.
3. Lineage vs physical-session granularity for the guard cap (§4.3) — confirm
   that is the intended semantic.
