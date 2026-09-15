"""B35 (Phase 3.2): let the biggie route carry a SESSION identity on the wire.

WHY THIS EXISTS
---------------
Hermes' chat-completions transport can send a session-scoped
``prompt_cache_key``, but only when the resolved provider profile sets
``supports_prompt_cache_key``. Without it the router sees no session id and has
to DERIVE one from the prompt prefix (``session_id_source='derived'``), which
collides across sessions and cannot be capped per session (B34).

WHY THE PROFILE IS *OVERRIDDEN*, NOT ADDED
------------------------------------------
The transport resolves the profile with ``get_provider_profile(agent.provider)``
(agent/chat_completion_helpers.py:1321 and :2045). For a ``custom_providers``
route ``agent.provider`` is the literal string ``"custom"`` — proven by
``chat_completion_helpers.py:1351`` (``is_custom_provider=agent.provider ==
"custom"``), by the gateway journal (``provider=custom model=biggie-router
base_url=http://127.0.0.1:8080/v1``) and by ``state.db`` (every biggie-router
session has ``billing_provider='custom'``).

So registering a NEW name (``custom:biggie-llm`` / ``biggie-llm``) is never
reached by that lookup, and the bundled generic ``custom`` profile wins with
``supports_prompt_cache_key=False``. That was the B35 defect that survived a
gateway restart. ``register_provider`` is last-writer-wins, and user plugins are
discovered AFTER bundled ones, so re-registering the SAME name ``"custom"``
replaces it.

WHY IT SUBCLASSES
-----------------
The bundled profile is a ``CustomProfile`` with real behaviour
(``build_api_kwargs_extras`` for Ollama ``think``/``num_ctx``/``reasoning_effort``,
``fetch_models``). A fresh ``ProviderProfile`` would silently drop all of it
(verified: a constructed profile reset ``aliases`` and flipped
``supports_health_check``). Subclassing keeps every inherited field and method;
only ``supports_prompt_cache_key`` changes.

SCOPE / SAFETY
--------------
``custom`` also carries the aliases ``ollama``, ``local``, ``vllm``,
``llamacpp``, ``llama.cpp``, ``llama-cpp``, so this flag applies to those routes
too. Core warns that some OpenAI-compatible endpoints REJECT unknown top-level
fields. Verified against the live local Ollama
(``http://127.0.0.1:11434``) with a real request: control (no field) HTTP 200 and
with ``prompt_cache_key`` HTTP 200 — it is accepted. If a future endpoint
rejects it, narrow this override rather than shipping it for every alias.
"""

from providers import register_provider
from providers.base import ProviderProfile


class BiggieCustomProfile(ProviderProfile):
    """The generic custom profile, plus prompt-cache-key support.

    Inheritance supplies every behaviour; this class adds exactly one field so a
    diff against the bundled profile shows one line.
    """

    supports_prompt_cache_key: bool = True


def _build_profile():
    """Clone the bundled ``custom`` profile, preserving its class and methods.

    Imported lazily so this module never hard-depends on the bundled plugin
    having been imported first. ``providers`` guarantees bundled discovery has
    run by the time a user plugin is loaded, but we re-check defensively.
    """
    from providers import get_provider_profile, _discover_providers

    _discover_providers()
    base = get_provider_profile("custom")
    if base is None:  # pragma: no cover - bundled profile always present
        return None

    # Prefer a subclass of the live object's class: that keeps any method the
    # bundled plugin defines AND makes the flag part of the class, not an
    # instance attribute that a future dataclass comparison would miss.
    cls = type(base)

    class _Derived(cls):  # type: ignore[misc, valid-type]
        supports_prompt_cache_key = True

    _Derived.__name__ = "BiggieCustomProfile"
    _Derived.__qualname__ = "BiggieCustomProfile"

    # dataclasses.replace would also work (verified) but constructing from the
    # field values keeps every OTHER field byte-identical and is explicit.
    import dataclasses
    values = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    # The field must be set on the INSTANCE too: ``dataclasses.fields(base)``
    # yields ``supports_prompt_cache_key=False`` from the bundled instance, and
    # passing that explicitly overrides the subclass's class attribute. Without
    # this line the clone silently keeps the old flag (caught by the
    # acceptance test, which asserted the flag is True and saw False).
    values["supports_prompt_cache_key"] = True
    return _Derived(**values)


profile = _build_profile()
if profile is not None:
    register_provider(profile)
